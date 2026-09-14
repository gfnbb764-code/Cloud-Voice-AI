# voice.py
# ============================================================
# Cloud Voice AI — Discord Voice Engine
# Full voice session / receive / Gemini / TTS system
# Compatible with main.py + config.py
# ============================================================

from __future__ import annotations

import asyncio
import io
import logging
import time
import wave
from collections import deque
from dataclasses import dataclass
from typing import Optional

import discord
from discord.ext import voice_recv

from config import (
    DEFAULT_GEMINI_VOICE,
    DISCORD_CHANNELS,
    DISCORD_SAMPLE_RATE,
    DISCORD_SAMPLE_WIDTH,
    GEMINI_INPUT_CHANNELS,
    GEMINI_INPUT_SAMPLE_RATE,
    GEMINI_INPUT_SAMPLE_WIDTH,
    GEMINI_TTS_CHANNELS,
    GEMINI_TTS_SAMPLE_RATE,
    GEMINI_TTS_SAMPLE_WIDTH,
    GEMINI_VOICES,
    MAX_CONCURRENT_AI_REQUESTS,
    MAX_MEMORY_MESSAGES,
    MAX_RECORDING_SECONDS,
    MIN_AUDIO_SECONDS,
    MEMORY_ENABLED,
    VOICE_SILENCE_TIMEOUT,
    normalize_voice_name,
    is_valid_voice,
)

from gemini import GeminiEngine


# ============================================================
# LOGGER
# ============================================================

logger = logging.getLogger("cloud_voice_ai.voice")


# ============================================================
# AUDIO CONSTANTS
# ============================================================

INPUT_RATE = GEMINI_INPUT_SAMPLE_RATE
INPUT_CHANNELS = GEMINI_INPUT_CHANNELS
INPUT_WIDTH = GEMINI_INPUT_SAMPLE_WIDTH

TTS_RATE = GEMINI_TTS_SAMPLE_RATE
TTS_CHANNELS = GEMINI_TTS_CHANNELS
TTS_WIDTH = GEMINI_TTS_SAMPLE_WIDTH

DISCORD_RATE = DISCORD_SAMPLE_RATE
DISCORD_CHANNEL_COUNT = DISCORD_CHANNELS
DISCORD_WIDTH = DISCORD_SAMPLE_WIDTH

MAX_BUFFER_SECONDS = MAX_RECORDING_SECONDS
SILENCE_TIMEOUT = VOICE_SILENCE_TIMEOUT
MIN_AUDIO_LENGTH = MIN_AUDIO_SECONDS


# ============================================================
# INTERNAL LIMITS
# ============================================================

MAX_BUFFER_BYTES = (
    DISCORD_RATE
    * DISCORD_CHANNEL_COUNT
    * DISCORD_WIDTH
    * int(MAX_BUFFER_SECONDS)
)

MAX_MEMORY = max(1, int(MAX_MEMORY_MESSAGES))

AI_SEMAPHORE_LIMIT = max(
    1,
    int(MAX_CONCURRENT_AI_REQUESTS),
)


# ============================================================
# AUDIO HELPERS
# ============================================================

def pcm_stereo_48k_to_mono_16k(
    pcm: bytes,
) -> bytes:
    """
    Convert Discord PCM:

        48000 Hz
        stereo
        signed 16-bit PCM

    into Gemini input:

        16000 Hz
        mono
        signed 16-bit PCM

    Discord -> Gemini.
    """

    if not pcm:
        return b""

    frame_width = DISCORD_WIDTH * DISCORD_CHANNEL_COUNT

    if frame_width <= 0:
        return b""

    usable_length = len(pcm) - (
        len(pcm) % frame_width
    )

    if usable_length <= 0:
        return b""

    pcm = pcm[:usable_length]

    output = bytearray()

    # Every 3 frames at 48kHz becomes 1 frame at 16kHz.
    step = 3

    frame_count = usable_length // frame_width

    for frame_index in range(
        0,
        frame_count,
        step,
    ):
        offset = frame_index * frame_width

        left = int.from_bytes(
            pcm[offset:offset + 2],
            byteorder="little",
            signed=True,
        )

        right = int.from_bytes(
            pcm[offset + 2:offset + 4],
            byteorder="little",
            signed=True,
        )

        mono = (left + right) // 2

        output.extend(
            int(mono).to_bytes(
                2,
                byteorder="little",
                signed=True,
            )
        )

    return bytes(output)


def pcm_to_wav(
    pcm: bytes,
    sample_rate: int,
    channels: int,
    sample_width: int,
) -> bytes:
    """
    Wrap raw PCM inside a WAV container.
    """

    buffer = io.BytesIO()

    with wave.open(
        buffer,
        "wb",
    ) as wav_file:

        wav_file.setnchannels(
            channels
        )

        wav_file.setsampwidth(
            sample_width
        )

        wav_file.setframerate(
            sample_rate
        )

        wav_file.writeframes(
            pcm
        )

    return buffer.getvalue()


def tts_pcm_to_discord_pcm(
    pcm: bytes,
) -> bytes:
    """
    Convert Gemini TTS raw PCM:

        24000 Hz
        mono
        16-bit

    into Discord playback PCM:

        48000 Hz
        stereo
        16-bit

    This uses simple sample duplication / nearest-neighbor
    upsampling because 24kHz -> 48kHz is exactly 2x.
    """

    if not pcm:
        return b""

    source_width = TTS_WIDTH

    if source_width != 2:
        logger.warning(
            "Unexpected TTS sample width: %s",
            source_width,
        )

    usable_length = len(pcm) - (
        len(pcm) % source_width
    )

    if usable_length <= 0:
        return b""

    pcm = pcm[:usable_length]

    output = bytearray()

    for offset in range(
        0,
        len(pcm),
        2,
    ):
        sample = pcm[
            offset:offset + 2
        ]

        # 24k -> 48k
        output.extend(sample)
        output.extend(sample)

        # Mono -> stereo
        output.extend(sample)
        output.extend(sample)

    return bytes(output)


def pcm_duration_seconds(
    pcm: bytes,
    sample_rate: int,
    channels: int,
    sample_width: int,
) -> float:
    """
    Calculate PCM duration.
    """

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
    """
    Per-user audio state.
    """

    user_id: int
    username: str

    buffer: bytearray

    started_at: float = 0.0
    last_packet_at: float = 0.0
    processing: bool = False

    def start(self) -> None:
        now = time.monotonic()

        if self.started_at <= 0:
            self.started_at = now

        self.last_packet_at = now

    def append(
        self,
        pcm: bytes,
    ) -> None:

        if not pcm:
            return

        if len(self.buffer) >= MAX_BUFFER_BYTES:
            return

        remaining = (
            MAX_BUFFER_BYTES
            - len(self.buffer)
        )

        self.buffer.extend(
            pcm[:remaining]
        )

        self.last_packet_at = (
            time.monotonic()
        )

    def elapsed(self) -> float:

        if self.started_at <= 0:
            return 0.0

        return (
            time.monotonic()
            - self.started_at
        )

    def silence_elapsed(self) -> float:

        if self.last_packet_at <= 0:
            return 0.0

        return (
            time.monotonic()
            - self.last_packet_at
        )

    def take_buffer(self) -> bytes:

        data = bytes(self.buffer)

        self.buffer.clear()

        self.started_at = 0.0
        self.last_packet_at = 0.0

        return data

    def clear(self) -> None:

        self.buffer.clear()

        self.started_at = 0.0
        self.last_packet_at = 0.0

        self.processing = False


# ============================================================
# VOICE AI SINK
# ============================================================

class VoiceAISink(voice_recv.AudioSink):
    """
    Discord voice receive sink.

    Important:
        wants_opus() returns False.

    The DAVE-aware voice-recv fork is responsible for
    decrypting and decoding Discord voice packets before
    this sink receives PCM.
    """

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

        self._lock = asyncio.Lock()

        self._closed = False

        self._watchdog_task: Optional[
            asyncio.Task
        ] = None

    # ========================================================
    # DISCORD VOICE-RECV
    # ========================================================

    def wants_opus(self) -> bool:
        """
        We want decoded PCM, not Opus packets.
        """

        return False

    def write(
        self,
        user,
        data,
    ) -> None:
        """
        Called by discord-ext-voice-recv when PCM
        audio is received.

        This callback is synchronous, so it schedules
        async processing on the event loop.
        """

        if self._closed:
            return

        if user is None:
            return

        try:

            user_id = int(
                getattr(
                    user,
                    "id",
                    0,
                )
            )

            username = str(
                getattr(
                    user,
                    "display_name",
                    None,
                )
                or getattr(
                    user,
                    "name",
                    "Unknown",
                )
            )

        except Exception:

            logger.exception(
                "Failed to identify voice user."
            )

            return

        if user_id <= 0:
            return

        pcm = getattr(
            data,
            "pcm",
            None,
        )

        if pcm is None:
            return

        if not isinstance(
            pcm,
            bytes,
        ):
            try:
                pcm = bytes(pcm)
            except Exception:
                return

        if not pcm:
            return

        loop = self.session.loop

        if loop is None:
            return

        try:

            loop.call_soon_threadsafe(
                self._handle_pcm_threadsafe,
                user_id,
                username,
                pcm,
            )

        except RuntimeError:

            logger.debug(
                "Voice event loop is unavailable."
            )

    def _handle_pcm_threadsafe(
        self,
        user_id: int,
        username: str,
        pcm: bytes,
    ) -> None:

        if self._closed:
            return

        task = asyncio.create_task(
            self._handle_pcm(
                user_id,
                username,
                pcm,
            )
        )

        task.add_done_callback(
            self._task_done
        )

    @staticmethod
    def _task_done(
        task: asyncio.Task,
    ) -> None:

        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception(
                "Unhandled voice sink task error."
            )

    async def _handle_pcm(
        self,
        user_id: int,
        username: str,
        pcm: bytes,
    ) -> None:

        if self._closed:
            return

        async with self._lock:

            state = self.users.get(
                user_id
            )

            if state is None:

                state = UserAudioState(
                    user_id=user_id,
                    username=username,
                    buffer=bytearray(),
                )

                self.users[
                    user_id
                ] = state

            state.username = username

            if not state.buffer:
                state.start()

            state.append(pcm)

            elapsed = state.elapsed()

            silence = (
                state.silence_elapsed()
            )

            should_process = (
                elapsed >= MAX_BUFFER_SECONDS
                or silence >= SILENCE_TIMEOUT
            )

            if (
                should_process
                and not state.processing
            ):

                state.processing = True

                audio = state.take_buffer()

                username_for_processing = (
                    state.username
                )

            else:

                return

        await self._process_user_audio(
            user_id=user_id,
            username=username_for_processing,
            pcm=audio,
        )

        async with self._lock:

            current = self.users.get(
                user_id
            )

            if current is not None:
                current.processing = False

    async def _process_user_audio(
        self,
        user_id: int,
        username: str,
        pcm: bytes,
    ) -> None:

        if self._closed:
            return

        duration = pcm_duration_seconds(
            pcm,
            DISCORD_RATE,
            DISCORD_CHANNEL_COUNT,
            DISCORD_WIDTH,
        )

        if duration < MIN_AUDIO_LENGTH:
            return

        try:

            gemini_pcm = (
                pcm_stereo_48k_to_mono_16k(
                    pcm
                )
            )

            if not gemini_pcm:
                return

            wav_audio = pcm_to_wav(
                gemini_pcm,
                INPUT_RATE,
                INPUT_CHANNELS,
                INPUT_WIDTH,
            )

            await self.session.process_voice(
                audio=wav_audio,
                username=username,
                mime_type="audio/wav",
            )

        except asyncio.CancelledError:

            raise

        except Exception:

            self.session.failed_requests += 1

            logger.exception(
                "Voice processing failed for %s (%s).",
                username,
                user_id,
            )

    def start_watchdog(
        self,
    ) -> None:

        if self._watchdog_task is not None:
            return

        self._watchdog_task = asyncio.create_task(
            self._watchdog()
        )

    async def _watchdog(self) -> None:

        try:

            while not self._closed:

                await asyncio.sleep(
                    0.25
                )

                await self._check_timeouts()

        except asyncio.CancelledError:

            pass

        except Exception:

            logger.exception(
                "Voice sink watchdog crashed."
            )

    async def _check_timeouts(
        self,
    ) -> None:

        to_process: list[
            tuple[int, str, bytes]
        ] = []

        async with self._lock:

            for user_id, state in list(
                self.users.items()
            ):

                if state.processing:
                    continue

                if not state.buffer:
                    continue

                elapsed = state.elapsed()

                silence = (
                    state.silence_elapsed()
                )

                if (
                    elapsed < MAX_BUFFER_SECONDS
                    and silence < SILENCE_TIMEOUT
                ):
                    continue

                state.processing = True

                audio = state.take_buffer()

                to_process.append(
                    (
                        user_id,
                        state.username,
                        audio,
                    )
                )

        for (
            user_id,
            username,
            audio,
        ) in to_process:

            await self._process_user_audio(
                user_id=user_id,
                username=username,
                pcm=audio,
            )

            async with self._lock:

                state = self.users.get(
                    user_id
                )

                if state is not None:
                    state.processing = False

    def cleanup(self) -> None:

        self._closed = True

        if self._watchdog_task:

            self._watchdog_task.cancel()

            self._watchdog_task = None

        for state in self.users.values():
            state.clear()

        self.users.clear()


# ============================================================
# VOICE SESSION
# ============================================================

class VoiceSession:
    """
    One guild voice AI session.
    """

    def __init__(
        self,
        guild: discord.Guild,
        channel: discord.VoiceChannel,
        voice: str = DEFAULT_GEMINI_VOICE,
    ) -> None:

        self.guild = guild
        self.channel = channel

        self.voice = (
            normalize_voice_name(voice)
            if is_valid_voice(voice)
            else DEFAULT_GEMINI_VOICE
        )

        self.loop = (
            asyncio.get_running_loop()
        )

        self.voice_client: Optional[
            voice_recv.VoiceRecvClient
        ] = None

        self.sink: Optional[
            VoiceAISink
        ] = None

        self.engine = GeminiEngine(
            voice=self.voice
        )

        self._closed = False

        self._processing_lock = asyncio.Lock()

        self._playback_lock = asyncio.Lock()

        self._ai_semaphore = asyncio.Semaphore(
            AI_SEMAPHORE_LIMIT
        )

        self.processed_requests = 0

        self.failed_requests = 0

        self.created_at = time.time()

        self.last_activity_at = (
            time.time()
        )

    # ========================================================
    # PROPERTIES
    # ========================================================

    @property
    def memory_count(self) -> int:

        if not MEMORY_ENABLED:
            return 0

        try:

            return len(
                self.engine.memory
            )

        except Exception:

            return 0

    # ========================================================
    # CONNECTION
    # ========================================================

    async def connect(self) -> None:
        """
        Connect to the configured Discord voice channel.
        """

        if self._closed:
            raise RuntimeError(
                "Voice session is closed."
            )

        if self.voice_client is not None:
            if self.voice_client.is_connected():
                return

        logger.info(
            "Connecting voice session in guild %s to %s.",
            self.guild.id,
            self.channel.name,
        )

        connected = await self.channel.connect(
            cls=voice_recv.VoiceRecvClient,
            reconnect=True,
        )

        self.voice_client = connected

        self.sink = VoiceAISink(
            self
        )

        self.voice_client.listen(
            self.sink
        )

        self.sink.start_watchdog()

        self.last_activity_at = time.time()

        logger.info(
            "Voice receive sink started for guild %s.",
            self.guild.id,
        )

        logger.info(
            "Voice AI listening in guild %s.",
            self.guild.id,
        )

    # ========================================================
    # VOICE PROCESSING
    # ========================================================

    async def process_voice(
        self,
        audio: bytes,
        username: str,
        mime_type: str = "audio/wav",
    ) -> None:
        """
        Full pipeline:

            Discord PCM
                ↓
            WAV
                ↓
            Gemini STT
                ↓
            Gemini Chat
                ↓
            Gemini TTS
                ↓
            Discord PCM
                ↓
            Voice playback
        """

        if self._closed:
            return

        if not audio:
            return

        self.last_activity_at = time.time()

        async with self._ai_semaphore:

            try:

                result = await asyncio.to_thread(
                    self.engine.process_voice,
                    audio=audio,
                    username=username,
                    mime_type=mime_type,
                )

                if not result:
                    return

                transcript = result.get(
                    "transcript",
                    "",
                )

                response_text = result.get(
                    "response",
                    "",
                )

                audio_bytes = result.get(
                    "audio",
                    b"",
                )

                if transcript:
                    logger.info(
                        "STT [%s]: %s",
                        username,
                        transcript[:300],
                    )

                if response_text:
                    logger.info(
                        "AI [%s]: %s",
                        username,
                        response_text[:300],
                    )

                if not audio_bytes:
                    self.processed_requests += 1
                    return

                await self.play_tts(
                    audio_bytes
                )

                self.processed_requests += 1

            except asyncio.CancelledError:

                raise

            except Exception:

                self.failed_requests += 1

                logger.exception(
                    "Gemini voice pipeline failed."
                )

    # ========================================================
    # PLAYBACK
    # ========================================================

    async def play_tts(
        self,
        audio_bytes: bytes,
    ) -> None:
        """
        Play Gemini TTS audio into Discord.
        """

        if self._closed:
            return

        if self.voice_client is None:
            return

        if not self.voice_client.is_connected():
            return

        if not audio_bytes:
            return

        discord_pcm = (
            tts_pcm_to_discord_pcm(
                audio_bytes
            )
        )

        if not discord_pcm:
            return

        source = discord.PCMAudio(
            io.BytesIO(
                discord_pcm
            )
        )

        async with self._playback_lock:

            if self._closed:
                return

            done = asyncio.Event()

            def after_playback(
                error: Optional[Exception],
            ) -> None:

                if error:

                    logger.error(
                        "Voice playback error: %s",
                        error,
                    )

                try:

                    self.loop.call_soon_threadsafe(
                        done.set
                    )

                except Exception:
                    pass

            try:

                self.voice_client.play(
                    source,
                    after=after_playback,
                )

            except Exception:

                logger.exception(
                    "Failed to start voice playback."
                )

                return

            try:

                await asyncio.wait_for(
                    done.wait(),
                    timeout=60.0,
                )

            except asyncio.TimeoutError:

                logger.warning(
                    "Voice playback timed out."
                )

                try:

                    if self.voice_client.is_playing():
                        self.voice_client.stop()

                except Exception:
                    pass

    # ========================================================
    # VOICE SETTING
    # ========================================================

    def set_voice(
        self,
        voice: str,
    ) -> None:

        if not is_valid_voice(voice):
            raise ValueError(
                f"Invalid Gemini voice: {voice}"
            )

        selected = normalize_voice_name(
            voice
        )

        self.voice = selected

        try:

            self.engine.set_voice(
                selected
            )

        except AttributeError:

            # Compatibility with GeminiEngine
            # versions that expose the voice
            # attribute directly.
            self.engine.voice = selected

        logger.info(
            "Voice changed to %s in guild %s.",
            selected,
            self.guild.id,
        )

    # ========================================================
    # MEMORY
    # ========================================================

    def clear_memory(self) -> None:

        try:

            self.engine.clear_memory()

        except AttributeError:

            try:

                self.engine.memory.clear()

            except Exception:

                logger.exception(
                    "Failed to clear Gemini memory."
                )

    def reset(self) -> None:

        self.clear_memory()

        self.processed_requests = 0
        self.failed_requests = 0

        self.last_activity_at = time.time()

        logger.info(
            "Voice session reset in guild %s.",
            self.guild.id,
        )

    # ========================================================
    # DISCONNECT
    # ========================================================

    async def disconnect(self) -> None:

        if self._closed:
            return

        self._closed = True

        logger.info(
            "Disconnecting voice session for guild %s.",
            self.guild.id,
        )

        if self.sink is not None:

            try:
                self.sink.cleanup()
            except Exception:
                logger.exception(
                    "Failed to clean voice sink."
                )

            self.sink = None

        if self.voice_client is not None:

            try:

                if self.voice_client.is_playing():
                    self.voice_client.stop()

            except Exception:
                pass

            try:

                await self.voice_client.disconnect(
                    force=True
                )

            except Exception:

                logger.exception(
                    "Failed to disconnect voice client."
                )

            self.voice_client = None

        logger.info(
            "Voice session disconnected for guild %s.",
            self.guild.id,
        )


# ============================================================
# VOICE SESSION MANAGER
# ============================================================

class VoiceSessionManager:
    """
    Manages one VoiceSession per guild.
    """

    def __init__(self) -> None:

        self.sessions: dict[
            int,
            VoiceSession,
        ] = {}

        self._lock = asyncio.Lock()

    # ========================================================
    # COUNT
    # ========================================================

    @property
    def count(self) -> int:
        return len(
            self.sessions
        )

    # ========================================================
    # GET
    # ========================================================

    def get(
        self,
        guild_id: int,
    ) -> Optional[VoiceSession]:

        return self.sessions.get(
            guild_id
        )

    # ========================================================
    # JOIN
    # ========================================================

    async def join(
        self,
        guild: discord.Guild,
        channel: discord.VoiceChannel,
        voice: str = DEFAULT_GEMINI_VOICE,
    ) -> VoiceSession:

        async with self._lock:

            existing = self.sessions.get(
                guild.id
            )

            if existing is not None:

                if (
                    existing.channel.id
                    == channel.id
                ):

                    if (
                        existing.voice
                        != normalize_voice_name(
                            voice
                        )
                    ):

                        existing.set_voice(
                            voice
                        )

                    return existing

                await existing.disconnect()

                self.sessions.pop(
                    guild.id,
                    None,
                )

            session = VoiceSession(
                guild=guild,
                channel=channel,
                voice=voice,
            )

            try:

                await session.connect()

            except Exception:

                await session.disconnect()

                raise

            self.sessions[
                guild.id
            ] = session

            logger.info(
                "Voice session created for guild %s.",
                guild.id,
            )

            return session

    # ========================================================
    # LEAVE
    # ========================================================

    async def leave(
        self,
        guild_id: int,
    ) -> bool:

        async with self._lock:

            session = self.sessions.pop(
                guild_id,
                None,
            )

            if session is None:
                return False

            await session.disconnect()

            logger.info(
                "Voice session removed for guild %s.",
                guild_id,
            )

            return True

    # ========================================================
    # DISCONNECT ALL
    # ========================================================

    async def disconnect_all(self) -> None:

        async with self._lock:

            sessions = list(
                self.sessions.values()
            )

            self.sessions.clear()

        if not sessions:
            return

        await asyncio.gather(
            *(
                session.disconnect()
                for session in sessions
            ),
            return_exceptions=True,
        )

        logger.info(
            "Disconnected %d voice session(s).",
            len(sessions),
        )


# ============================================================
# VOICE LIST
# ============================================================

async def send_voice_list(
    interaction: discord.Interaction,
) -> None:
    """
    Send the complete Gemini voice list.

    This function is imported directly by main.py.
    """

    voices = list(
        GEMINI_VOICES
    )

    if not voices:

        await interaction.followup.send(
            "❌ لا توجد أصوات Gemini متاحة حاليًا.",
            ephemeral=True,
        )

        return

    # Discord embed description limit is 4096.
    # Keep enough room for formatting.
    chunks: list[str] = []

    current: list[str] = []
    current_length = 0

    for index, voice in enumerate(
        voices,
        start=1,
    ):

        line = (
            f"`{index:02d}` — **{voice}**"
        )

        if (
            current
            and current_length
            + len(line)
            + 1
            > 3800
        ):

            chunks.append(
                "\n".join(current)
            )

            current = []
            current_length = 0

        current.append(line)

        current_length += (
            len(line) + 1
        )

    if current:
        chunks.append(
            "\n".join(current)
        )

    first = True

    for chunk_index, chunk in enumerate(
        chunks,
        start=1,
    ):

        title = (
            "🎙️ Gemini Voices"
            if len(chunks) == 1
            else f"🎙️ Gemini Voices — {chunk_index}/{len(chunks)}"
        )

        embed = discord.Embed(
            title=title,
            description=chunk,
            color=discord.Color.blurple(),
        )

        if first:

            embed.add_field(
                name="⭐ Default",
                value=(
                    f"`{DEFAULT_GEMINI_VOICE}`"
                ),
                inline=True,
            )

            embed.add_field(
                name="🎤 Available",
                value=f"`{len(voices)}`",
                inline=True,
            )

            first = False

        await interaction.followup.send(
            embed=embed,
            ephemeral=True,
        )


# ============================================================
# SIMPLE CHANNEL MESSAGE HELPER
# ============================================================

async def send_voice_channel_message(
    channel: discord.TextChannel,
    content: str,
) -> None:
    """
    Optional helper for future voice events.
    """

    if not content:
        return

    try:

        await channel.send(
            content
        )

    except Exception:

        logger.exception(
            "Failed to send voice channel message."
        )


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "VoiceSession",
    "VoiceSessionManager",
    "VoiceAISink",
    "send_voice_list",
    "send_voice_channel_message",
]
