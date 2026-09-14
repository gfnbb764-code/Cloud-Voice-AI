# voice.py
# ============================================================
# Cloud Voice AI — Discord Voice Engine
# Robust audio capture / buffering / voice AI / TTS
# ============================================================

from __future__ import annotations

import asyncio
import io
import logging
import math
import time
import wave
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
# AUDIO SETTINGS
# ============================================================

DISCORD_RATE = DISCORD_SAMPLE_RATE
DISCORD_CHANNEL_COUNT = DISCORD_CHANNELS
DISCORD_WIDTH = DISCORD_SAMPLE_WIDTH

INPUT_RATE = GEMINI_INPUT_SAMPLE_RATE
INPUT_CHANNELS = GEMINI_INPUT_CHANNELS
INPUT_WIDTH = GEMINI_INPUT_SAMPLE_WIDTH

TTS_RATE = GEMINI_TTS_SAMPLE_RATE
TTS_CHANNELS = GEMINI_TTS_CHANNELS
TTS_WIDTH = GEMINI_TTS_SAMPLE_WIDTH


# ============================================================
# BUFFER SETTINGS
# ============================================================

MAX_BUFFER_SECONDS = max(
    1.0,
    float(MAX_RECORDING_SECONDS),
)

SILENCE_TIMEOUT = max(
    0.35,
    float(VOICE_SILENCE_TIMEOUT),
)

MIN_AUDIO_LENGTH = max(
    0.05,
    float(MIN_AUDIO_SECONDS),
)

MAX_BUFFER_BYTES = int(
    DISCORD_RATE
    * DISCORD_CHANNEL_COUNT
    * DISCORD_WIDTH
    * MAX_BUFFER_SECONDS
)

MAX_MEMORY = max(
    1,
    int(MAX_MEMORY_MESSAGES),
)

AI_SEMAPHORE_LIMIT = max(
    1,
    int(MAX_CONCURRENT_AI_REQUESTS),
)


# ============================================================
# AUDIO QUALITY SETTINGS
# ============================================================

# Diagnostic only. Never reject audio based on RMS.
QUIET_RMS = 120.0

# Maximum amplification allowed.
MAX_GAIN = 5.0

# Target RMS after optional normalization.
TARGET_RMS = 5000.0

# Headroom to reduce clipping.
CLIP_LIMIT = 30000


# ============================================================
# PCM HELPERS
# ============================================================

def _read_int16(
    pcm: bytes,
    offset: int,
) -> int:
    return int.from_bytes(
        pcm[offset:offset + 2],
        byteorder="little",
        signed=True,
    )


def _write_int16(
    value: int,
) -> bytes:
    value = max(
        -32768,
        min(32767, int(value)),
    )

    return value.to_bytes(
        2,
        byteorder="little",
        signed=True,
    )


def pcm_rms(
    pcm: bytes,
) -> float:
    """
    Calculate RMS level of signed 16-bit PCM.
    Diagnostic only.
    """

    if not pcm:
        return 0.0

    usable = len(pcm) - (
        len(pcm) % 2
    )

    if usable <= 0:
        return 0.0

    total = 0.0
    count = 0

    for offset in range(
        0,
        usable,
        2,
    ):
        sample = _read_int16(
            pcm,
            offset,
        )

        total += (
            float(sample)
            * float(sample)
        )

        count += 1

    if count <= 0:
        return 0.0

    return math.sqrt(
        total / count
    )


def pcm_peak(
    pcm: bytes,
) -> int:
    """
    Calculate absolute PCM peak.
    """

    if not pcm:
        return 0

    usable = len(pcm) - (
        len(pcm) % 2
    )

    peak = 0

    for offset in range(
        0,
        usable,
        2,
    ):
        value = abs(
            _read_int16(
                pcm,
                offset,
            )
        )

        if value > peak:
            peak = value

    return peak


def normalize_pcm_volume(
    pcm: bytes,
) -> bytes:
    """
    Raise quiet PCM conservatively without
    aggressively modifying already-loud audio.
    """

    if not pcm:
        return b""

    rms = pcm_rms(
        pcm
    )

    if rms <= 0:
        return pcm

    peak = pcm_peak(
        pcm
    )

    if peak <= 0:
        return pcm

    if rms >= TARGET_RMS:
        return pcm

    desired_gain = (
        TARGET_RMS / rms
    )

    gain = min(
        MAX_GAIN,
        max(1.0, desired_gain),
    )

    if peak * gain > CLIP_LIMIT:
        gain = (
            CLIP_LIMIT / peak
        )

    if gain <= 1.01:
        return pcm

    output = bytearray(
        len(pcm)
    )

    usable = len(pcm) - (
        len(pcm) % 2
    )

    for offset in range(
        0,
        usable,
        2,
    ):
        sample = _read_int16(
            pcm,
            offset,
        )

        amplified = int(
            sample * gain
        )

        output[
            offset:
            offset + 2
        ] = _write_int16(
            amplified
        )

    if usable < len(pcm):
        output[
            usable:
        ] = pcm[
            usable:
        ]

    return bytes(output)


def pcm_stereo_48k_to_mono_16k(
    pcm: bytes,
) -> bytes:
    """
    Convert Discord PCM:

        48000 Hz
        stereo
        signed 16-bit

    into:

        16000 Hz
        mono
        signed 16-bit

    Uses averaging over each group of 3 frames
    instead of simply discarding 2 out of 3 frames.
    """

    if not pcm:
        return b""

    frame_width = (
        DISCORD_WIDTH
        * DISCORD_CHANNEL_COUNT
    )

    if frame_width <= 0:
        return b""

    usable_length = len(pcm) - (
        len(pcm) % frame_width
    )

    if usable_length <= 0:
        return b""

    pcm = pcm[:usable_length]

    frame_count = (
        len(pcm)
        // frame_width
    )

    output = bytearray()

    # 48000 -> 16000 = exactly 3:1
    output_frame_count = frame_count // 3

    for output_index in range(
        output_frame_count
    ):
        base_frame = (
            output_index * 3
        )

        total = 0

        for offset_frame in range(3):
            frame_index = (
                base_frame
                + offset_frame
            )

            offset = (
                frame_index
                * frame_width
            )

            if DISCORD_CHANNEL_COUNT >= 2:

                left = _read_int16(
                    pcm,
                    offset,
                )

                right = _read_int16(
                    pcm,
                    offset + 2,
                )

                mono = (
                    left + right
                ) // 2

            else:

                mono = _read_int16(
                    pcm,
                    offset,
                )

            total += mono

        averaged = (
            total // 3
        )

        output.extend(
            _write_int16(
                averaged
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
    Wrap raw PCM in a WAV container.
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
    Gemini TTS:

        24000 Hz
        mono
        16-bit

    Discord:

        48000 Hz
        stereo
        16-bit
    """

    if not pcm:
        return b""

    usable = len(pcm) - (
        len(pcm) % 2
    )

    if usable <= 0:
        return b""

    pcm = pcm[:usable]

    output = bytearray()

    for offset in range(
        0,
        len(pcm),
        2,
    ):
        sample = pcm[
            offset:
            offset + 2
        ]

        # 24k -> 48k:
        # duplicate each sample twice.
        #
        # Mono -> stereo:
        # duplicate each frame L/R.
        output.extend(sample)
        output.extend(sample)
        output.extend(sample)
        output.extend(sample)

    return bytes(output)


def pcm_duration_seconds(
    pcm: bytes,
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

    return (
        len(pcm)
        / bytes_per_second
    )


# ============================================================
# USER AUDIO STATE
# ============================================================

@dataclass
class UserAudioState:
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

        remaining = (
            MAX_BUFFER_BYTES
            - len(self.buffer)
        )

        if remaining <= 0:
            return

        self.buffer.extend(
            pcm[:remaining]
        )

        self.last_packet_at = (
            time.monotonic()
        )

    def elapsed(
        self,
    ) -> float:

        if self.started_at <= 0:
            return 0.0

        return (
            time.monotonic()
            - self.started_at
        )

    def silence_elapsed(
        self,
    ) -> float:

        if self.last_packet_at <= 0:
            return 0.0

        return (
            time.monotonic()
            - self.last_packet_at
        )

    def take_buffer(
        self,
    ) -> bytes:

        data = bytes(
            self.buffer
        )

        self.buffer.clear()

        self.started_at = 0.0
        self.last_packet_at = 0.0

        return data

    def clear(
        self,
    ) -> None:

        self.buffer.clear()

        self.started_at = 0.0
        self.last_packet_at = 0.0

        self.processing = False


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

        self._lock = asyncio.Lock()

        self._closed = False

        self._watchdog_task: Optional[
            asyncio.Task
        ] = None

    # ========================================================
    # VOICE RECV
    # ========================================================

    def wants_opus(
        self,
    ) -> bool:
        """
        Ask discord-ext-voice-recv for decoded PCM.
        """

        return False

    def write(
        self,
        user,
        data,
    ) -> None:

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
                "Could not identify voice user."
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

        try:

            if not isinstance(
                pcm,
                bytes,
            ):
                pcm = bytes(pcm)

        except Exception:

            return

        if not pcm:
            return

        try:

            loop = self.session.loop

            loop.call_soon_threadsafe(
                self._handle_pcm_threadsafe,
                user_id,
                username,
                pcm,
            )

        except RuntimeError:

            logger.debug(
                "Voice loop unavailable."
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

    # ========================================================
    # AUDIO COLLECTION
    # ========================================================

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

            state.append(
                pcm
            )

            elapsed = (
                state.elapsed()
            )

            silence = (
                state.silence_elapsed()
            )

            # Only time-based buffering.
            # Do not reject quiet or distorted speech.
            should_process = (
                elapsed >= MAX_BUFFER_SECONDS
                or silence >= SILENCE_TIMEOUT
            )

            if (
                not should_process
                or state.processing
            ):
                return

            state.processing = True

            audio = state.take_buffer()

            process_username = (
                state.username
            )

        try:

            await self._process_user_audio(
                user_id=user_id,
                username=process_username,
                pcm=audio,
            )

        finally:

            async with self._lock:

                current = self.users.get(
                    user_id
                )

                if current is not None:
                    current.processing = False

    # ========================================================
    # PROCESS AUDIO
    # ========================================================

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

        raw_rms = pcm_rms(
            pcm
        )

        raw_peak = pcm_peak(
            pcm
        )

        logger.info(
            "Voice audio from %s | "
            "duration=%.2fs | "
            "bytes=%d | "
            "rms=%.1f | "
            "peak=%d",
            username,
            duration,
            len(pcm),
            raw_rms,
            raw_peak,
        )

        try:

            # ------------------------------------------------
            # 1. Discord 48k stereo -> Gemini 16k mono
            # ------------------------------------------------

            gemini_pcm = (
                pcm_stereo_48k_to_mono_16k(
                    pcm
                )
            )

            if not gemini_pcm:
                return

            # ------------------------------------------------
            # 2. Normalize quiet audio
            # ------------------------------------------------

            before_rms = pcm_rms(
                gemini_pcm
            )

            boosted_pcm = (
                normalize_pcm_volume(
                    gemini_pcm
                )
            )

            after_rms = pcm_rms(
                boosted_pcm
            )

            if before_rms != after_rms:

                logger.info(
                    "Voice normalization for %s | "
                    "rms %.1f -> %.1f",
                    username,
                    before_rms,
                    after_rms,
                )

            # ------------------------------------------------
            # 3. Gemini audio diagnostics
            # ------------------------------------------------

            logger.info(
                "Gemini audio for %s | "
                "duration=%.2fs | "
                "bytes=%d | "
                "rms=%.1f | "
                "peak=%d",
                username,
                pcm_duration_seconds(
                    boosted_pcm,
                    INPUT_RATE,
                    INPUT_CHANNELS,
                    INPUT_WIDTH,
                ),
                len(boosted_pcm),
                pcm_rms(boosted_pcm),
                pcm_peak(boosted_pcm),
            )

            # ------------------------------------------------
            # 4. WAV for Gemini STT
            # ------------------------------------------------

            wav_audio = pcm_to_wav(
                boosted_pcm,
                INPUT_RATE,
                INPUT_CHANNELS,
                INPUT_WIDTH,
            )

            logger.info(
                "Gemini WAV ready for %s | bytes=%d",
                username,
                len(wav_audio),
            )

            # ------------------------------------------------
            # 5. Send audio to Gemini
            # ------------------------------------------------

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

    # ========================================================
    # WATCHDOG
    # ========================================================

    def start_watchdog(
        self,
    ) -> None:

        if self._watchdog_task is not None:
            return

        self._watchdog_task = (
            asyncio.create_task(
                self._watchdog()
            )
        )

    async def _watchdog(
        self,
    ) -> None:

        try:

            while not self._closed:

                await asyncio.sleep(
                    0.20
                )

                await self._check_timeouts()

        except asyncio.CancelledError:
            pass

        except Exception:

            logger.exception(
                "Voice watchdog crashed."
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

                elapsed = (
                    state.elapsed()
                )

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

            try:

                await self._process_user_audio(
                    user_id=user_id,
                    username=username,
                    pcm=audio,
                )

            finally:

                async with self._lock:

                    state = self.users.get(
                        user_id
                    )

                    if state is not None:
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

        for state in self.users.values():
            state.clear()

        self.users.clear()


# ============================================================
# VOICE SESSION
# ============================================================

class VoiceSession:

    def __init__(
        self,
        guild: discord.Guild,
        channel: discord.VoiceChannel,
        voice: str = DEFAULT_GEMINI_VOICE,
    ) -> None:

        self.guild = guild
        self.channel = channel

        if is_valid_voice(voice):

            self.voice = (
                normalize_voice_name(
                    voice
                )
            )

        else:

            self.voice = (
                DEFAULT_GEMINI_VOICE
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

        self.engine = GeminiEngine()

        self.engine.set_voice(
            self.voice
        )

        self._closed = False

        self._playback_lock = (
            asyncio.Lock()
        )

        self._ai_semaphore = (
            asyncio.Semaphore(
                AI_SEMAPHORE_LIMIT
            )
        )

        self.processed_requests = 0
        self.failed_requests = 0

        self.created_at = time.time()

        self.last_activity_at = (
            time.time()
        )

    # ========================================================
    # MEMORY
    # ========================================================

    @property
    def memory_count(
        self,
    ) -> int:

        if not MEMORY_ENABLED:
            return 0

        try:

            return self.engine.memory_size()

        except Exception:

            try:
                return len(
                    self.engine.memory
                )
            except Exception:
                return 0

    # ========================================================
    # CONNECT
    # ========================================================

    async def connect(
        self,
    ) -> None:

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

        self.last_activity_at = (
            time.time()
        )

        logger.info(
            "Voice receive sink started for guild %s.",
            self.guild.id,
        )

        logger.info(
            "Voice AI listening in guild %s.",
            self.guild.id,
        )

        logger.info(
            "Voice session created for guild %s.",
            self.guild.id,
        )

    # ========================================================
    # GEMINI VOICE PIPELINE
    # ========================================================

    async def process_voice(
        self,
        audio: bytes,
        username: str,
        mime_type: str = "audio/wav",
    ) -> None:

        if self._closed:
            return

        if not audio:
            return

        self.last_activity_at = (
            time.time()
        )

        async with self._ai_semaphore:

            try:

                result = await self.engine.process_voice(
                    audio=audio,
                    username=username,
                    voice=self.voice,
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

                error = result.get(
                    "error"
                )

                if transcript:

                    logger.info(
                        "STT [%s]: %s",
                        username,
                        transcript[:500],
                    )

                if response_text:

                    logger.info(
                        "AI [%s]: %s",
                        username,
                        response_text[:500],
                    )

                if error:

                    logger.warning(
                        "Voice pipeline returned an error "
                        "for %s: %s",
                        username,
                        error,
                    )

                if not audio_bytes:

                    if error:
                        self.failed_requests += 1
                    else:
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
    # TTS PLAYBACK
    # ========================================================

    async def play_tts(
        self,
        audio_bytes: bytes,
    ) -> None:

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

            if self.voice_client.is_playing():

                try:
                    self.voice_client.stop()
                except Exception:
                    pass

            finished = asyncio.Event()

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
                        finished.set
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
                    finished.wait(),
                    timeout=90.0,
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
    # VOICE SELECTION
    # ========================================================

    def set_voice(
        self,
        voice: str,
    ) -> None:

        if not is_valid_voice(voice):

            raise ValueError(
                f"Invalid Gemini voice: {voice}"
            )

        selected = (
            normalize_voice_name(
                voice
            )
        )

        self.voice = selected

        self.engine.set_voice(
            selected
        )

        logger.info(
            "Voice changed to %s in guild %s.",
            selected,
            self.guild.id,
        )

    # ========================================================
    # MEMORY
    # ========================================================

    def clear_memory(
        self,
    ) -> None:

        try:

            self.engine.clear_memory()

        except Exception:

            logger.exception(
                "Failed to clear Gemini memory."
            )

    def reset(
        self,
    ) -> None:

        self.clear_memory()

        self.processed_requests = 0
        self.failed_requests = 0

        self.last_activity_at = (
            time.time()
        )

        logger.info(
            "Voice session reset in guild %s.",
            self.guild.id,
        )

    # ========================================================
    # DISCONNECT
    # ========================================================

    async def disconnect(
        self,
    ) -> None:

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

        close_method = getattr(
            self.engine,
            "close",
            None,
        )

        if close_method is not None:

            try:

                result = close_method()

                if asyncio.iscoroutine(
                    result
                ):
                    await result

            except Exception:

                logger.debug(
                    "Gemini engine close failed.",
                    exc_info=True,
                )

        logger.info(
            "Voice session disconnected for guild %s.",
            self.guild.id,
        )


# ============================================================
# SESSION MANAGER
# ============================================================

class VoiceSessionManager:

    def __init__(
        self,
    ) -> None:

        self.sessions: dict[
            int,
            VoiceSession,
        ] = {}

        self._lock = asyncio.Lock()

    @property
    def count(
        self,
    ) -> int:

        return len(
            self.sessions
        )

    def get(
        self,
        guild_id: int,
    ) -> Optional[VoiceSession]:

        return self.sessions.get(
            guild_id
        )

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

                    selected = (
                        normalize_voice_name(
                            voice
                        )
                    )

                    if (
                        existing.voice
                        != selected
                    ):

                        existing.set_voice(
                            selected
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

    async def disconnect_all(
        self,
    ) -> None:

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

    voices = list(
        GEMINI_VOICES
    )

    if not voices:

        await interaction.followup.send(
            "❌ لا توجد أصوات Gemini متاحة حاليًا.",
            ephemeral=True,
        )

        return

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
            len(line)
            + 1
        )

    if current:

        chunks.append(
            "\n".join(current)
        )

    for chunk_index, chunk in enumerate(
        chunks,
        start=1,
    ):

        title = (
            "🎙️ Gemini Voices"
            if len(chunks) == 1
            else (
                "🎙️ Gemini Voices — "
                f"{chunk_index}/{len(chunks)}"
            )
        )

        embed = discord.Embed(
            title=title,
            description=chunk,
            color=discord.Color.blurple(),
        )

        if chunk_index == 1:

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

        await interaction.followup.send(
            embed=embed,
            ephemeral=True,
        )


# ============================================================
# OPTIONAL TEXT CHANNEL HELPER
# ============================================================

async def send_voice_channel_message(
    channel: discord.TextChannel,
    content: str,
) -> None:

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
# EXPORTS
# ============================================================

__all__ = [
    "VoiceSession",
    "VoiceSessionManager",
    "VoiceAISink",
    "send_voice_list",
    "send_voice_channel_message",
    ]
