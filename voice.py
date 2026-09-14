# voice.py
# ============================================================
# Cloud Voice AI — Discord Voice Receive / Playback
# ============================================================

from __future__ import annotations

import asyncio
import audioop
import io
import logging
import math
import struct
import time
import wave
from typing import Any

import discord

from config import (
    ALLOW_VOICE_CHANGE,
    DEFAULT_AUDIO_MIME_TYPE,
    DEFAULT_GEMINI_VOICE,
    DEFAULT_SPEECH_SPEED,
    DISCORD_CHANNELS,
    DISCORD_SAMPLE_RATE,
    DISCORD_SAMPLE_WIDTH,
    GEMINI_INPUT_CHANNELS,
    GEMINI_INPUT_SAMPLE_RATE,
    GEMINI_TTS_CHANNELS,
    GEMINI_VOICES,
    MAX_AUDIO_BUFFER_BYTES,
    MAX_CONCURRENT_AI_REQUESTS,
    MAX_GUILD_VOICE_SESSIONS,
    MAX_RECORDING_SECONDS,
    MIN_AUDIO_SECONDS,
    is_valid_voice,
    normalize_speech_speed,
    normalize_voice_name,
)

from gemini import GeminiEngine


logger = logging.getLogger("voice")


# ============================================================
# AUDIO HELPERS
# ============================================================

def pcm_stereo_48k_to_mono_16k(
    pcm: bytes,
) -> bytes:

    if not pcm:
        return b""

    # Discord:
    # 48kHz / stereo / 16-bit
    #
    # Gemini:
    # 16kHz / mono / 16-bit

    mono, _ = audioop.tomono(
        pcm,
        DISCORD_SAMPLE_WIDTH,
        1.0,
        1.0,
    )

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

    if not pcm:
        return b""

    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(int(channels))
        wav.setsampwidth(int(sample_width))
        wav.setframerate(int(sample_rate))
        wav.writeframes(pcm)

    return buffer.getvalue()


def calculate_rms(
    pcm: bytes,
) -> float:

    if not pcm:
        return 0.0

    try:
        count = len(pcm) // 2

        if count <= 0:
            return 0.0

        samples = struct.unpack(
            f"<{count}h",
            pcm[: count * 2],
        )

        square_sum = sum(
            sample * sample
            for sample in samples
        )

        return math.sqrt(
            square_sum / count
        )

    except Exception:
        return 0.0


def calculate_peak(
    pcm: bytes,
) -> int:

    if not pcm:
        return 0

    try:
        count = len(pcm) // 2

        if count <= 0:
            return 0

        samples = struct.unpack(
            f"<{count}h",
            pcm[: count * 2],
        )

        return max(
            abs(sample)
            for sample in samples
        )

    except Exception:
        return 0


def pcm_duration_seconds(
    pcm: bytes,
    *,
    sample_rate: int = DISCORD_SAMPLE_RATE,
    channels: int = DISCORD_CHANNELS,
    sample_width: int = DISCORD_SAMPLE_WIDTH,
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
# VOICE SINK IMPORT
# ============================================================

try:
    from discord.ext import voice_recv

    AudioSink = voice_recv.AudioSink
    VoiceRecvClient = voice_recv.VoiceRecvClient

except ImportError:

    voice_recv = None

    class AudioSink:
        pass

    VoiceRecvClient = None


# ============================================================
# VOICE SINK
# ============================================================

class VoiceAISink(AudioSink):

    def __init__(
        self,
        session: "VoiceSession",
    ) -> None:

        super().__init__()

        self.session = session

        self.buffers: dict[
            int,
            bytearray,
        ] = {}

        self.user_names: dict[
            int,
            str,
        ] = {}

        self.last_audio_time: dict[
            int,
            float,
        ] = {}

        self.processing: set[
            int
        ] = set()

        self.closed = False

    # ========================================================
    # OPUS MODE
    # ========================================================

    def wants_opus(
        self,
    ) -> bool:
        """
        False means the sink wants decoded PCM.
        """

        return False

    # ========================================================
    # AUDIO RECEIVE
    # ========================================================

    def write(
        self,
        user: Any,
        data: Any,
    ) -> None:

        if self.closed:
            return

        if data is None:
            return

        try:

            pcm = getattr(
                data,
                "pcm",
                None,
            )

            if pcm is None:
                pcm = getattr(
                    data,
                    "data",
                    None,
                )

            if pcm is None:
                return

            if not isinstance(
                pcm,
                bytes,
            ):
                pcm = bytes(pcm)

            if not pcm:
                return

            user_id = int(
                getattr(
                    user,
                    "id",
                    0,
                )
            )

            if user_id <= 0:
                return

            username = str(
                getattr(
                    user,
                    "display_name",
                    None,
                )
                or getattr(
                    user,
                    "name",
                    f"User {user_id}",
                )
            )

            buffer = self.buffers.setdefault(
                user_id,
                bytearray(),
            )

            self.user_names[user_id] = username

            if (
                len(buffer)
                + len(pcm)
                > MAX_AUDIO_BUFFER_BYTES
            ):

                logger.warning(
                    "Audio buffer limit reached | user=%s",
                    username,
                )

                asyncio.create_task(
                    self._process_user_audio(
                        user_id
                    )
                )

                return

            buffer.extend(pcm)

            self.last_audio_time[user_id] = (
                time.monotonic()
            )

            duration = pcm_duration_seconds(
                bytes(buffer)
            )

            if duration >= MAX_RECORDING_SECONDS:

                asyncio.create_task(
                    self._process_user_audio(
                        user_id
                    )
                )

        except Exception:

            logger.exception(
                "Voice sink write failed"
            )

    # ========================================================
    # PROCESS AUDIO
    # ========================================================

    async def _process_user_audio(
        self,
        user_id: int,
    ) -> None:

        if user_id in self.processing:
            return

        self.processing.add(user_id)

        try:

            buffer = self.buffers.pop(
                user_id,
                None,
            )

            username = self.user_names.get(
                user_id,
                f"User {user_id}",
            )

            if not buffer:
                return

            pcm = bytes(buffer)

            duration = pcm_duration_seconds(
                pcm
            )

            if duration < MIN_AUDIO_SECONDS:

                logger.debug(
                    "Audio too short | user=%s | duration=%.2fs",
                    username,
                    duration,
                )

                return

            rms = calculate_rms(pcm)
            peak = calculate_peak(pcm)

            logger.info(
                (
                    "Processing voice | "
                    "user=%s | "
                    "duration=%.2fs | "
                    "bytes=%s | "
                    "rms=%.1f | "
                    "peak=%s"
                ),
                username,
                duration,
                len(pcm),
                rms,
                peak,
            )

            if peak <= 0:
                return

            normalized = (
                pcm_stereo_48k_to_mono_16k(
                    pcm
                )
            )

            if not normalized:
                return

            wav = pcm_to_wav(
                normalized,
                sample_rate=GEMINI_INPUT_SAMPLE_RATE,
                channels=GEMINI_INPUT_CHANNELS,
                sample_width=DISCORD_SAMPLE_WIDTH,
            )

            if not wav:
                return

            await self.session.process_voice(
                audio=wav,
                username=username,
                mime_type=DEFAULT_AUDIO_MIME_TYPE,
            )

        except Exception:

            logger.exception(
                "Failed to process voice audio"
            )

        finally:

            self.processing.discard(
                user_id
            )

    # ========================================================
    # CLEANUP
    # ========================================================

    def cleanup(
        self,
    ) -> None:

        self.closed = True

        self.buffers.clear()
        self.user_names.clear()
        self.last_audio_time.clear()
        self.processing.clear()

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
        voice_client: discord.VoiceClient,
        voice: str = DEFAULT_GEMINI_VOICE,
        speed: float = DEFAULT_SPEECH_SPEED,
        character: Any = None,
    ) -> None:

        self.guild = guild
        self.channel = channel
        self.voice_client = voice_client

        self.voice = normalize_voice_name(
            voice
        )

        self.speed = normalize_speech_speed(
            speed
        )

        self.character = character

        self.engine = GeminiEngine()

        self.engine.set_voice(
            self.voice
        )

        self.sink: VoiceAISink | None = None

        self.play_lock = asyncio.Lock()

        self.processing_semaphore = (
            asyncio.Semaphore(
                MAX_CONCURRENT_AI_REQUESTS
            )
        )

        self.closed = False

        self.started_at = time.monotonic()

        self.last_activity = (
            self.started_at
        )

    # ========================================================
    # START RECEIVE
    # ========================================================

    def start_receive(
        self,
    ) -> None:

        if self.closed:
            return

        if self.sink is not None:
            return

        if voice_recv is None:
            raise RuntimeError(
                "discord-ext-voice-recv-dave is not installed."
            )

        if not isinstance(
            self.voice_client,
            VoiceRecvClient,
        ):
            raise RuntimeError(
                "VoiceRecvClient is required for voice receive."
            )

        self.sink = VoiceAISink(
            self
        )

        try:

            self.voice_client.listen(
                self.sink
            )

            logger.info(
                "Voice receive sink started"
            )

            logger.info(
                "Voice AI listening"
            )

        except Exception:

            self.sink = None

            logger.exception(
                "Failed to start voice receive"
            )

            raise

    # ========================================================
    # PROCESS VOICE
    # ========================================================

    async def process_voice(
        self,
        *,
        audio: bytes,
        username: str,
        mime_type: str = DEFAULT_AUDIO_MIME_TYPE,
    ) -> None:

        if self.closed:
            return

        async with self.processing_semaphore:

            self.last_activity = (
                time.monotonic()
            )

            try:

                result = await self.engine.process_voice(
                    audio=audio,
                    username=username,
                    voice=self.voice,
                    speed=self.speed,
                    character=self.character,
                    mime_type=mime_type,
                )

                if not result:
                    return

                if not result.get(
                    "success",
                    False,
                ):

                    error = result.get(
                        "error",
                        "Unknown error",
                    )

                    logger.warning(
                        "Voice AI failed | user=%s | error=%s",
                        username,
                        error,
                    )

                    return

                transcript = result.get(
                    "transcript",
                    "",
                )

                response = result.get(
                    "response",
                    "",
                )

                audio_data = result.get(
                    "audio",
                    b"",
                )

                if transcript:

                    logger.info(
                        "STT | user=%s | text=%s",
                        username,
                        transcript,
                    )

                if response:

                    logger.info(
                        "AI response | text=%s",
                        response,
                    )

                if audio_data:

                    await self.play_tts(
                        audio_data
                    )

            except Exception:

                logger.exception(
                    "Voice processing failed"
                )

    # ========================================================
    # PLAY TTS
    # ========================================================

    async def play_tts(
        self,
        pcm: bytes,
    ) -> None:

        if self.closed:
            return

        if not pcm:
            return

        async with self.play_lock:

            if self.closed:
                return

            if not self.voice_client.is_connected():
                return

            loop = asyncio.get_running_loop()

            source = discord.PCMAudio(
                io.BytesIO(pcm)
            )

            finished = asyncio.Event()

            def after(
                error: Exception | None,
            ) -> None:

                if error:

                    logger.error(
                        "TTS playback error: %s",
                        error,
                    )

                loop.call_soon_threadsafe(
                    finished.set
                )

            try:

                self.voice_client.play(
                    source,
                    after=after,
                )

                await finished.wait()

            except Exception:

                logger.exception(
                    "Failed to play TTS"
                )

    # ========================================================
    # VOICE
    # ========================================================

    def set_voice(
        self,
        voice: str,
    ) -> str:

        if not ALLOW_VOICE_CHANGE:

            raise ValueError(
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

        if self.character is not None:

            try:
                self.character.voice = normalized
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
                self.character.speed = normalized
            except Exception:
                pass

        return normalized

    # ========================================================
    # CHARACTER
    # ========================================================

    def set_character(
        self,
        character: Any,
    ) -> None:

        self.character = character

        if character is None:
            return

        try:

            self.voice = normalize_voice_name(
                character.voice
            )

            self.speed = normalize_speech_speed(
                character.speed
            )

            self.engine.set_voice(
                self.voice
            )

        except Exception:

            logger.exception(
                "Failed to apply character"
            )

    def clear_character(
        self,
    ) -> None:

        self.character = None

        self.voice = normalize_voice_name(
            DEFAULT_GEMINI_VOICE
        )

        self.speed = normalize_speech_speed(
            DEFAULT_SPEECH_SPEED
        )

        self.engine.set_voice(
            self.voice
        )

    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
    ) -> None:

        self.engine.clear_memory()

        self.voice = normalize_voice_name(
            DEFAULT_GEMINI_VOICE
        )

        self.speed = normalize_speech_speed(
            DEFAULT_SPEECH_SPEED
        )

        self.character = None

        self.engine.set_voice(
            self.voice
        )

    # ========================================================
    # CLOSE
    # ========================================================

    async def close(
        self,
    ) -> None:

        if self.closed:
            return

        self.closed = True

        if self.sink:

            try:
                self.sink.cleanup()
            except Exception:
                pass

            self.sink = None

        try:

            if (
                hasattr(
                    self.voice_client,
                    "is_listening",
                )
                and self.voice_client.is_listening()
            ):

                self.voice_client.stop_listening()

        except Exception:

            logger.exception(
                "Failed to stop voice listening"
            )

        try:

            if self.voice_client.is_playing():
                self.voice_client.stop()

        except Exception:
            pass

        try:

            if self.voice_client.is_connected():

                await self.voice_client.disconnect(
                    force=True
                )

        except Exception:

            logger.exception(
                "Voice disconnect failed"
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

        self.lock = asyncio.Lock()

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
    # JOIN
    # ========================================================

    async def join(
        self,
        *,
        guild: discord.Guild,
        channel: discord.VoiceChannel,
        voice: str = DEFAULT_GEMINI_VOICE,
        speed: float = DEFAULT_SPEECH_SPEED,
        character: Any = None,
    ) -> VoiceSession:

        async with self.lock:

            existing = self.sessions.get(
                guild.id
            )

            if existing:

                if (
                    existing.channel.id
                    != channel.id
                ):

                    try:

                        await existing.voice_client.move_to(
                            channel
                        )

                        existing.channel = channel

                    except Exception:

                        await existing.close()

                        self.sessions.pop(
                            guild.id,
                            None,
                        )

                existing.set_voice(
                    voice
                )

                existing.set_speed(
                    speed
                )

                if character is not None:

                    existing.set_character(
                        character
                    )

                return existing

            if (
                len(self.sessions)
                >= MAX_GUILD_VOICE_SESSIONS
            ):

                raise RuntimeError(
                    "Maximum voice sessions reached."
                )

            normalized_voice = normalize_voice_name(
                voice
            )

            if not is_valid_voice(
                normalized_voice
            ):

                raise ValueError(
                    f"Invalid voice: {voice}"
                )

            normalized_speed = normalize_speech_speed(
                speed
            )

            if VoiceRecvClient is None:
                raise RuntimeError(
                    "VoiceRecvClient is unavailable. "
                    "Check discord-ext-voice-recv-dave installation."
                )

            voice_client = guild.voice_client

            if voice_client is None:

                voice_client = await channel.connect(
                    cls=VoiceRecvClient,
                    self_deaf=False,
                    self_mute=False,
                )

            else:

                if not isinstance(
                    voice_client,
                    VoiceRecvClient,
                ):

                    try:

                        await voice_client.disconnect(
                            force=True
                        )

                    except Exception:
                        pass

                    voice_client = await channel.connect(
                        cls=VoiceRecvClient,
                        self_deaf=False,
                        self_mute=False,
                    )

                elif (
                    voice_client.channel is None
                    or voice_client.channel.id
                    != channel.id
                ):

                    await voice_client.move_to(
                        channel
                    )

            session = VoiceSession(
                guild=guild,
                channel=channel,
                voice_client=voice_client,
                voice=normalized_voice,
                speed=normalized_speed,
                character=character,
            )

            self.sessions[
                guild.id
            ] = session

            try:

                session.start_receive()

            except Exception:

                self.sessions.pop(
                    guild.id,
                    None,
                )

                await session.close()

                raise

            logger.info(
                (
                    "Voice session created | "
                    "guild=%s | "
                    "channel=%s | "
                    "voice=%s | "
                    "speed=%.2f"
                ),
                guild.id,
                channel.name,
                session.voice,
                session.speed,
            )

            return session

    # ========================================================
    # LEAVE
    # ========================================================

    async def leave(
        self,
        guild_id: int,
    ) -> bool:

        async with self.lock:

            session = self.sessions.pop(
                guild_id,
                None,
            )

            if session is None:
                return False

            await session.close()

            logger.info(
                "Voice session closed | guild=%s",
                guild_id,
            )

            return True

    # ========================================================
    # DISCONNECT ALL
    # ========================================================

    async def disconnect_all(
        self,
    ) -> None:

        async with self.lock:

            sessions = list(
                self.sessions.items()
            )

            self.sessions.clear()

            for guild_id, session in sessions:

                try:

                    await session.close()

                except Exception:

                    logger.exception(
                        "Failed to close session | guild=%s",
                        guild_id,
                    )


# ============================================================
# VOICE LIST
# ============================================================

async def send_voice_list(
    interaction: discord.Interaction,
) -> None:

    voices = list(
        GEMINI_VOICES
    )

    chunks = []

    for index in range(
        0,
        len(voices),
        10,
    ):

        chunks.append(
            voices[
                index:index + 10
            ]
        )

    embed = discord.Embed(
        title="🎙️ Gemini Voices",
        description=(
            "يمكنك اختيار أي صوت من القائمة "
            "باستخدام `/setvoice` أو أثناء إنشاء الشخصية."
        ),
        color=discord.Color.blurple(),
    )

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):

        embed.add_field(
            name=f"Voices {index}",
            value="\n".join(
                f"`{voice}`"
                for voice in chunk
            ),
            inline=True,
        )

    await interaction.response.send_message(
        embed=embed
            )
