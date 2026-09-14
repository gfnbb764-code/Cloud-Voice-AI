# voice.py
# ============================================================
# Cloud Voice AI — Discord Voice System
# Supports:
# - Discord DAVE voice receive
# - PCM capture
# - Gemini STT
# - Gemini AI
# - Gemini TTS
# - Characters
# - Per-character voice
# - Per-character speech speed
# - Manual speech speed
# ============================================================

from __future__ import annotations

import asyncio
import audioop
import logging
import math
import struct
import time
import wave
from dataclasses import dataclass, field
from typing import Any

import discord
from discord.ext import voice_recv

from config import (
    ALLOW_VOICE_CHANGE,
    AUTO_LEAVE_DELAY_SECONDS,
    AUTO_LEAVE_EMPTY_CHANNEL,
    DEFAULT_AUDIO_MIME_TYPE,
    DEFAULT_GEMINI_VOICE,
    DEFAULT_SPEECH_SPEED,
    DISCORD_CHANNELS,
    DISCORD_SAMPLE_RATE,
    DISCORD_SAMPLE_WIDTH,
    GEMINI_INPUT_CHANNELS,
    GEMINI_INPUT_SAMPLE_RATE,
    GEMINI_TTS_CHANNELS,
    GEMINI_TTS_SAMPLE_RATE,
    GEMINI_VOICES,
    MAX_AUDIO_BUFFER_BYTES,
    MAX_CONCURRENT_AI_REQUESTS,
    MAX_GUILD_VOICE_SESSIONS,
    MAX_RECORDING_SECONDS,
    MIN_AUDIO_SECONDS,
    SILENCE_TIMEOUT_SECONDS,
    is_valid_voice,
    normalize_speech_speed,
    normalize_voice_name,
)
from gemini import GeminiEngine

logger = logging.getLogger(__name__)


# ============================================================
# AUDIO HELPERS
# ============================================================

def _read_int16(
    pcm: bytes,
) -> list[int]:

    if not pcm:
        return []

    sample_count = len(pcm) // 2

    return list(
        struct.unpack(
            f"<{sample_count}h",
            pcm[: sample_count * 2],
        )
    )


def _write_int16(
    samples: list[int],
) -> bytes:

    if not samples:
        return b""

    clipped = [
        max(-32768, min(32767, int(sample)))
        for sample in samples
    ]

    return struct.pack(
        f"<{len(clipped)}h",
        *clipped,
    )


def pcm_rms(
    pcm: bytes,
) -> float:

    samples = _read_int16(pcm)

    if not samples:
        return 0.0

    square_sum = sum(
        sample * sample
        for sample in samples
    )

    return math.sqrt(
        square_sum / len(samples)
    )


def pcm_peak(
    pcm: bytes,
) -> int:

    samples = _read_int16(pcm)

    if not samples:
        return 0

    return max(
        abs(sample)
        for sample in samples
    )


def normalize_pcm_volume(
    pcm: bytes,
    target_peak: int = 26000,
) -> bytes:

    if not pcm:
        return b""

    peak = pcm_peak(pcm)

    if peak <= 0:
        return pcm

    if peak >= target_peak:
        return pcm

    gain = target_peak / peak

    # Prevent extreme amplification of very quiet audio.
    gain = min(gain, 4.0)

    samples = _read_int16(pcm)

    amplified = [
        int(sample * gain)
        for sample in samples
    ]

    return _write_int16(amplified)


def pcm_stereo_48k_to_mono_16k(
    pcm: bytes,
) -> bytes:

    if not pcm:
        return b""

    # Discord receive:
    # 48 kHz / stereo / signed 16-bit
    mono = audioop.tomono(
        pcm,
        DISCORD_SAMPLE_WIDTH,
        1.0,
        1.0,
    )

    # Convert 48 kHz mono -> 16 kHz mono.
    converted, _ = audioop.ratecv(
        mono,
        DISCORD_SAMPLE_WIDTH,
        GEMINI_INPUT_CHANNELS,
        DISCORD_SAMPLE_RATE,
        GEMINI_INPUT_SAMPLE_RATE,
        None,
    )

    return converted


def pcm_to_wav(
    pcm: bytes,
    *,
    sample_rate: int,
    channels: int,
    sample_width: int,
) -> bytes:

    output = bytearray()

    with wave.open(
        __import__("io").BytesIO(),
        "wb",
    ) as wav:
        pass

    import io

    buffer = io.BytesIO()

    with wave.open(
        buffer,
        "wb",
    ) as wav:

        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)

    return buffer.getvalue()


def tts_pcm_to_discord_pcm(
    pcm: bytes,
) -> bytes:

    if not pcm:
        return b""

    # Gemini TTS:
    # 24 kHz / mono / signed 16-bit
    #
    # Discord playback:
    # 48 kHz / stereo / signed 16-bit

    converted, _ = audioop.ratecv(
        pcm,
        DISCORD_SAMPLE_WIDTH,
        GEMINI_TTS_CHANNELS,
        GEMINI_TTS_SAMPLE_RATE,
        DISCORD_SAMPLE_RATE,
        None,
    )

    stereo = audioop.tostereo(
        converted,
        DISCORD_SAMPLE_WIDTH,
        1.0,
        1.0,
    )

    return stereo


def pcm_duration_seconds(
    pcm: bytes,
    *,
    sample_rate: int,
    channels: int,
    sample_width: int,
) -> float:

    bytes_per_second = (
        sample_rate
        * channels
        * sample_width
    )

    if bytes_per_second <= 0:
        return 0.0

    return len(pcm) / bytes_per_second


# ============================================================
# USER AUDIO STATE
# ============================================================

@dataclass
class UserAudioState:

    user_id: int

    username: str

    chunks: list[bytes] = field(
        default_factory=list
    )

    total_bytes: int = 0

    started_at: float = field(
        default_factory=time.monotonic
    )

    last_audio_at: float = field(
        default_factory=time.monotonic
    )

    processing: bool = False

    def add(
        self,
        pcm: bytes,
    ) -> None:

        if not pcm:
            return

        self.chunks.append(pcm)

        self.total_bytes += len(pcm)

        self.last_audio_at = (
            time.monotonic()
        )

    def build(
        self,
    ) -> bytes:

        return b"".join(
            self.chunks
        )

    def duration(self) -> float:

        return pcm_duration_seconds(
            self.build(),
            sample_rate=DISCORD_SAMPLE_RATE,
            channels=DISCORD_CHANNELS,
            sample_width=DISCORD_SAMPLE_WIDTH,
        )

    def clear(self) -> None:

        self.chunks.clear()
        self.total_bytes = 0
        self.started_at = time.monotonic()
        self.last_audio_at = time.monotonic()


# ============================================================
# VOICE AI SINK
# ============================================================

class VoiceAISink(
    voice_recv.AudioSink
):

    def __init__(
        self,
        session: "VoiceSession",
    ) -> None:

        super().__init__()

        self.session = session

        self.users: dict[
            int,
            UserAudioState,
        ] = {}

        self.loop = asyncio.get_running_loop()

        self._watchdog_task: asyncio.Task | None = None

        self._closed = False

        self._semaphore = asyncio.Semaphore(
            MAX_CONCURRENT_AI_REQUESTS
        )

    # ========================================================
    # DISCORD RECEIVE CONFIG
    # ========================================================

    def wants_opus(
        self,
    ) -> bool:

        # We want decoded PCM.
        return False

    # ========================================================
    # START
    # ========================================================

    def start_watchdog(
        self,
    ) -> None:

        if self._watchdog_task is not None:
            return

        self._watchdog_task = asyncio.create_task(
            self._watchdog()
        )

    # ========================================================
    # RECEIVE
    # ========================================================

    def write(
        self,
        user: Any,
        data: Any,
    ) -> None:

        if self._closed:
            return

        pcm = getattr(
            data,
            "pcm",
            None,
        )

        if not pcm:
            return

        if len(pcm) <= 0:
            return

        try:
            user_id = int(
                getattr(
                    user,
                    "id",
                    0,
                )
            )
        except Exception:
            user_id = 0

        if user_id <= 0:
            return

        username = getattr(
            user,
            "display_name",
            None,
        ) or getattr(
            user,
            "name",
            None,
        ) or f"User-{user_id}"

        try:
            self.loop.call_soon_threadsafe(
                lambda: asyncio.create_task(
                    self._handle_pcm(
                        user_id,
                        str(username),
                        pcm,
                    )
                )
            )

        except Exception:
            logger.exception(
                "Failed to schedule received audio"
            )

    # ========================================================
    # HANDLE PCM
    # ========================================================

    async def _handle_pcm(
        self,
        user_id: int,
        username: str,
        pcm: bytes,
    ) -> None:

        if self._closed:
            return

        state = self.users.get(
            user_id
        )

        if state is None:

            state = UserAudioState(
                user_id=user_id,
                username=username,
            )

            self.users[user_id] = state

        state.username = username

        if (
            state.total_bytes
            + len(pcm)
            > MAX_AUDIO_BUFFER_BYTES
        ):

            await self._flush_user(
                user_id
            )

            state = UserAudioState(
                user_id=user_id,
                username=username,
            )

            self.users[user_id] = state

        state.add(pcm)

        duration = state.duration()

        elapsed = (
            time.monotonic()
            - state.started_at
        )

        if duration < MIN_AUDIO_SECONDS:
            return

        if (
            duration >= MAX_RECORDING_SECONDS
            or elapsed >= MAX_RECORDING_SECONDS
        ):
            await self._flush_user(
                user_id
            )

    # ========================================================
    # WATCHDOG
    # ========================================================

    async def _watchdog(
        self,
    ) -> None:

        try:

            while not self._closed:

                await asyncio.sleep(
                    0.25
                )

                now = time.monotonic()

                for user_id, state in list(
                    self.users.items()
                ):

                    if state.processing:
                        continue

                    if not state.chunks:
                        continue

                    duration = state.duration()

                    if duration < MIN_AUDIO_SECONDS:
                        continue

                    silence_for = (
                        now
                        - state.last_audio_at
                    )

                    if (
                        silence_for
                        >= SILENCE_TIMEOUT_SECONDS
                    ):
                        await self._flush_user(
                            user_id
                        )

        except asyncio.CancelledError:
            return

        except Exception:
            logger.exception(
                "Voice watchdog failed"
            )

    # ========================================================
    # FLUSH USER
    # ========================================================

    async def _flush_user(
        self,
        user_id: int,
    ) -> None:

        state = self.users.get(
            user_id
        )

        if state is None:
            return

        if state.processing:
            return

        pcm = state.build()

        if not pcm:
            state.clear()
            return

        duration = state.duration()

        if duration < MIN_AUDIO_SECONDS:
            state.clear()
            return

        state.processing = True

        state.clear()

        asyncio.create_task(
            self._process_user_audio(
                user_id=user_id,
                username=state.username,
                pcm=pcm,
                duration=duration,
                state=state,
            )
        )

    # ========================================================
    # PROCESS USER AUDIO
    # ========================================================

    async def _process_user_audio(
        self,
        *,
        user_id: int,
        username: str,
        pcm: bytes,
        duration: float,
        state: UserAudioState,
    ) -> None:

        async with self._semaphore:

            try:

                logger.info(
                    "Processing voice | "
                    "user=%s | duration=%.2fs | bytes=%s | rms=%.1f | peak=%s",
                    username,
                    duration,
                    len(pcm),
                    pcm_rms(pcm),
                    pcm_peak(pcm),
                )

                normalized = normalize_pcm_volume(
                    pcm
                )

                mono_16k = (
                    pcm_stereo_48k_to_mono_16k(
                        normalized
                    )
                )

                if not mono_16k:
                    return

                wav = pcm_to_wav(
                    mono_16k,
                    sample_rate=GEMINI_INPUT_SAMPLE_RATE,
                    channels=GEMINI_INPUT_CHANNELS,
                    sample_width=DISCORD_SAMPLE_WIDTH,
                )

                await self.session.process_voice(
                    audio=wav,
                    username=username,
                    user_id=user_id,
                    mime_type=DEFAULT_AUDIO_MIME_TYPE,
                )

            except Exception:
                logger.exception(
                    "Failed to process voice audio | user=%s",
                    username,
                )

            finally:
                state.processing = False

    # ========================================================
    # CLEANUP
    # ========================================================

    def cleanup(
        self,
    ) -> None:

        self._closed = True

        if self._watchdog_task:

            self._watchdog_task.cancel()

            self._watchdog_task = None

        self.users.clear()

        try:
            super().cleanup()
        except Exception:
            pass


# ============================================================
# VOICE SESSION
# ============================================================

class VoiceSession:

    def __init__(
        self,
        *,
        guild: discord.Guild,
        channel: discord.VoiceChannel,
        voice: str = DEFAULT_GEMINI_VOICE,
        speed: float = DEFAULT_SPEECH_SPEED,
        character: Any | None = None,
    ) -> None:

        self.guild = guild
        self.channel = channel

        self.voice_client: Any | None = None

        self.engine = GeminiEngine()

        self.voice = normalize_voice_name(
            voice
        )

        if not is_valid_voice(
            self.voice
        ):
            self.voice = DEFAULT_GEMINI_VOICE

        self.engine.set_voice(
            self.voice
        )

        self.speed = normalize_speech_speed(
            speed
        )

        self.character = character

        if self.character is not None:
            self._apply_character()

        self.sink: VoiceAISink | None = None

        self.connected_at = time.monotonic()

        self._disconnect_task: asyncio.Task | None = None

    # ========================================================
    # CHARACTER
    # ========================================================

    def _apply_character(
        self,
    ) -> None:

        if self.character is None:
            return

        character_voice = getattr(
            self.character,
            "voice",
            None,
        )

        character_speed = getattr(
            self.character,
            "speed",
            None,
        )

        if character_voice:
            normalized = normalize_voice_name(
                character_voice
            )

            if is_valid_voice(
                normalized
            ):
                self.voice = normalized
                self.engine.set_voice(
                    normalized
                )

        if character_speed is not None:

            try:
                self.speed = normalize_speech_speed(
                    float(character_speed)
                )
            except (
                TypeError,
                ValueError,
            ):
                pass

    def set_character(
        self,
        character: Any | None,
    ) -> None:

        self.character = character

        if character is None:
            self.voice = DEFAULT_GEMINI_VOICE
            self.engine.set_voice(
                self.voice
            )
            self.speed = DEFAULT_SPEECH_SPEED
            return

        self._apply_character()

    def clear_character(
        self,
    ) -> None:

        self.set_character(None)

    # ========================================================
    # CONNECT
    # ========================================================

    async def connect(
        self,
    ) -> Any:

        if self.voice_client is not None:
            return self.voice_client

        logger.info(
            "Connecting to voice channel | guild=%s | channel=%s",
            self.guild.id,
            self.channel.name,
        )

        self.voice_client = await self.channel.connect(
            cls=voice_recv.VoiceRecvClient
        )

        self.sink = VoiceAISink(
            self
        )

        self.voice_client.listen(
            self.sink
        )

        self.sink.start_watchdog()

        logger.info(
            "Voice receive sink started | guild=%s",
            self.guild.id,
        )

        logger.info(
            "Voice AI listening | guild=%s | channel=%s",
            self.guild.id,
            self.channel.name,
        )

        return self.voice_client

    # ========================================================
    # PROCESS VOICE
    # ========================================================

    async def process_voice(
        self,
        *,
        audio: bytes,
        username: str,
        user_id: int | None = None,
        mime_type: str = DEFAULT_AUDIO_MIME_TYPE,
    ) -> dict[str, Any]:

        if not audio:
            return {
                "success": False,
                "error": "No audio received.",
            }

        # Character controls the voice and speed.
        if self.character is not None:
            self._apply_character()

        result = await self.engine.process_voice(
            audio=audio,
            username=username,
            voice=self.voice,
            speed=self.speed,
            character=self.character,
            mime_type=mime_type,
        )

        if not result.get("success"):

            logger.warning(
                "Voice processing failed | user=%s | error=%s",
                username,
                result.get("error"),
            )

            return result

        transcript = result.get(
            "transcript",
            "",
        )

        response = result.get(
            "response",
            "",
        )

        logger.info(
            "Voice AI result | "
            "user=%s | transcript=%s | response=%s",
            username,
            transcript,
            response,
        )

        speech = result.get(
            "audio",
            b"",
        )

        if speech:

            await self.play_tts(
                speech
            )

        return result

    # ========================================================
    # PLAY TTS
    # ========================================================

    async def play_tts(
        self,
        pcm: bytes,
    ) -> None:

        if not pcm:
            return

        if (
            self.voice_client is None
            or not self.voice_client.is_connected()
        ):
            return

        discord_pcm = tts_pcm_to_discord_pcm(
            pcm
        )

        if not discord_pcm:
            return

        source = discord.PCMAudio(
            __import__("io").BytesIO(
                discord_pcm
            )
        )

        # Wait for previous response to finish.
        while self.voice_client.is_playing():

            await asyncio.sleep(
                0.05
            )

        done = asyncio.Event()

        def after(
            error: Exception | None,
        ) -> None:

            if error:
                logger.error(
                    "TTS playback error: %s",
                    error,
                )

            try:
                asyncio.run_coroutine_threadsafe(
                    self._set_event(done),
                    asyncio.get_running_loop(),
                )
            except Exception:
                pass

        try:

            self.voice_client.play(
                source,
                after=after,
            )

        except Exception:
            logger.exception(
                "Failed to start TTS playback"
            )

            return

        # Polling is more reliable across
        # Discord voice client implementations.
        while self.voice_client.is_playing():

            await asyncio.sleep(
                0.05
            )

    async def _set_event(
        self,
        event: asyncio.Event,
    ) -> None:

        event.set()

    # ========================================================
    # VOICE
    # ========================================================

    def set_voice(
        self,
        voice: str,
    ) -> str:

        if not ALLOW_VOICE_CHANGE:
            raise RuntimeError(
                "Voice changing is disabled."
            )

        normalized = normalize_voice_name(
            voice
        )

        if not is_valid_voice(
            normalized
        ):
            raise ValueError(
                f"Invalid voice: {voice}"
            )

        self.voice = normalized

        self.engine.set_voice(
            normalized
        )

        # If a character is active, update its
        # runtime voice as well.
        if self.character is not None:

            try:
                setattr(
                    self.character,
                    "voice",
                    normalized,
                )
            except Exception:
                pass

        return normalized

    # ========================================================
    # SPEED
    # ========================================================

    def set_speed(
        self,
        speed: float,
        *,
        update_character: bool = True,
    ) -> float:

        normalized = normalize_speech_speed(
            speed
        )

        self.speed = normalized

        if (
            update_character
            and self.character is not None
        ):

            try:
                setattr(
                    self.character,
                    "speed",
                    normalized,
                )
            except Exception:
                pass

        return normalized

    # ========================================================
    # MEMORY
    # ========================================================

    def clear_memory(
        self,
    ) -> None:

        self.engine.clear_memory()

    def reset(
        self,
    ) -> None:

        self.engine.clear_memory()

        self.voice = DEFAULT_GEMINI_VOICE

        self.engine.set_voice(
            self.voice
        )

        self.speed = DEFAULT_SPEECH_SPEED

        self.character = None

    # ========================================================
    # DISCONNECT
    # ========================================================

    async def disconnect(
        self,
    ) -> None:

        if self.sink:

            self.sink.cleanup()
            self.sink = None

        if self.voice_client:

            try:
                await self.voice_client.disconnect(
                    force=True
                )
            except Exception:
                logger.exception(
                    "Failed to disconnect voice client"
                )

            self.voice_client = None

        await self.engine.close()

        logger.info(
            "Voice session disconnected | guild=%s",
            self.guild.id,
        )


# ============================================================
# VOICE SESSION MANAGER
# ============================================================

class VoiceSessionManager:

    def __init__(
        self,
    ) -> None:

        self.sessions: dict[
            int,
            VoiceSession,
        ] = {}

    # ========================================================
    # JOIN
    # ========================================================

    async def join(
        self,
        *,
        guild: discord.Guild,
        channel: discord.VoiceChannel,
        voice: str = DEFAULT_GEMINI_VOICE,
        speed: float = DEFAULT_SPEECH_SPEED,
        character: Any | None = None,
    ) -> VoiceSession:

        guild_id = guild.id

        existing = self.sessions.get(
            guild_id
        )

        if existing:

            if (
                existing.channel.id
                == channel.id
            ):
                return existing

            await existing.disconnect()

            self.sessions.pop(
                guild_id,
                None,
            )

        if (
            len(self.sessions)
            >= MAX_GUILD_VOICE_SESSIONS
        ):
            raise RuntimeError(
                "Maximum voice sessions reached."
            )

        session = VoiceSession(
            guild=guild,
            channel=channel,
            voice=voice,
            speed=speed,
            character=character,
        )

        await session.connect()

        self.sessions[guild_id] = session

        return session

    # ========================================================
    # LEAVE
    # ========================================================

    async def leave(
        self,
        guild_id: int,
    ) -> bool:

        session = self.sessions.pop(
            guild_id,
            None,
        )

        if session is None:
            return False

        await session.disconnect()

        return True

    # ========================================================
    # DISCONNECT ALL
    # ========================================================

    async def disconnect_all(
        self,
    ) -> None:

        sessions = list(
            self.sessions.values()
        )

        self.sessions.clear()

        await asyncio.gather(
            *(
                session.disconnect()
                for session in sessions
            ),
            return_exceptions=True,
        )

    # ========================================================
    # GET
    # ========================================================

    def get(
        self,
        guild_id: int,
    ) -> VoiceSession | None:

        return self.sessions.get(
            guild_id
        )

    # ========================================================
    # COUNT
    # ========================================================

    def count(self) -> int:
        return len(
            self.sessions
        )


# ============================================================
# VOICE LIST
# ============================================================

async def send_voice_list(
    interaction: discord.Interaction,
) -> None:

    embed = discord.Embed(
        title="🎙️ Gemini Voices",
        description=(
            f"Available voices: **{len(GEMINI_VOICES)}**"
        ),
        color=discord.Color.blurple(),
    )

    lines: list[str] = []

    for index, voice in enumerate(
        GEMINI_VOICES,
        start=1,
    ):

        lines.append(
            f"`{index:02}` • **{voice}**"
        )

    # Discord embed description has a character limit.
    description = "\n".join(lines)

    if len(description) > 4000:
        description = description[:3990] + "..."

    embed.description = description

    await interaction.response.send_message(
        embed=embed
    )


# ============================================================
# OPTIONAL VOICE CHANNEL MESSAGE
# ============================================================

async def send_voice_channel_message(
    session: VoiceSession,
    content: str,
) -> None:

    content = str(content).strip()

    if not content:
        return

    try:

        await session.channel.send(
            content
        )

    except Exception:
        logger.exception(
            "Failed to send voice channel message"
        )


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "VoiceSession",
    "VoiceSessionManager",
    "VoiceAISink",
    "UserAudioState",
    "send_voice_list",
    "send_voice_channel_message",
    "pcm_rms",
    "pcm_peak",
    "pcm_stereo_48k_to_mono_16k",
    "pcm_to_wav",
    "tts_pcm_to_discord_pcm",
    "pcm_duration_seconds",
]
