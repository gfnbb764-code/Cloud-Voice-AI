# voice.py
# ============================================================
# Cloud Voice AI — Discord Voice System
#
# Pipeline:
# Discord receive
#     ↓
# Local faster-whisper STT
#     ↓
# Groq Chat
#     ↓
# Groq Orpheus Arabic Saudi TTS
#     ↓
# Discord playback
#
# Important:
# - Keeps a tiny pre-roll before speech starts.
# - Silence is NOT added to the speech buffer.
# - Bot audio is ignored.
# - Voice receive is disabled while TTS is playing.
# - Buffers are cleared between turns.
# - STT is processed locally by faster-whisper.
# - Groq is used only for Chat + TTS.
# ============================================================

from __future__ import annotations

import asyncio
import audioop
import io
import logging
import time
import wave
from collections import deque
from typing import Any, Mapping

import discord
from discord.ext import voice_recv

from config import (
    ALLOW_VOICE_CHANGE,
    DEFAULT_GEMINI_VOICE,
    DEFAULT_SPEECH_SPEED,
    DISCORD_SAMPLE_RATE,
    DISCORD_SAMPLE_WIDTH,
    GEMINI_INPUT_CHANNELS,
    GEMINI_INPUT_SAMPLE_RATE,
    GEMINI_TTS_CHANNELS,
    GEMINI_TTS_SAMPLE_RATE,
    GEMINI_VOICES,
    MAX_AUDIO_BUFFER_BYTES,
    MAX_RECORDING_SECONDS,
    MIN_AUDIO_SECONDS,
    SILENCE_TIMEOUT_SECONDS,
    is_valid_voice,
    normalize_speech_speed,
    normalize_voice_name,
)

from gemini import GeminiEngine


logger = logging.getLogger("voice")


# ============================================================
# LOGGER FILTERING
# ============================================================

for noisy_logger_name in (
    "discord.ext.voice_recv.gateway",
    "discord.ext.voice_recv.reader",
    "discord.ext.voice_recv.rtp",
):
    noisy_logger = logging.getLogger(
        noisy_logger_name
    )
    noisy_logger.setLevel(
        logging.WARNING
    )


# Opus packet-loss warnings are not fatal.
logging.getLogger(
    "discord.ext.voice_recv.opus"
).setLevel(
    logging.ERROR
)


# ============================================================
# AUDIO SETTINGS
# ============================================================

SILENCE_RMS_THRESHOLD = 500

# Keep a tiny amount of recent audio before speech starts.
#
# Discord receive audio is normally:
#   48000 Hz
#   stereo
#   16-bit PCM
#
# This gives faster-whisper a little context before the
# detected speech begins, which can help preserve the first
# syllables of short phrases such as:
#
#   "السلام عليكم"
#   "تسمعني؟"
#   "هلا"
#   "الو"
#
# Silence is still NOT added to the actual speech buffer.
PRE_ROLL_SECONDS = 0.16

PRE_ROLL_MAX_BYTES = int(
    DISCORD_SAMPLE_RATE
    * 2
    * DISCORD_SAMPLE_WIDTH
    * PRE_ROLL_SECONDS
)


# ============================================================
# HELPERS
# ============================================================

def pcm_to_wav(
    pcm: bytes,
    sample_rate: int = GEMINI_INPUT_SAMPLE_RATE,
    channels: int = GEMINI_INPUT_CHANNELS,
    sample_width: int = DISCORD_SAMPLE_WIDTH,
) -> bytes:
    """
    Convert raw PCM into a WAV container.
    """

    if not pcm:
        return b""

    buffer = io.BytesIO()

    with wave.open(
        buffer,
        "wb",
    ) as wav:

        wav.setnchannels(
            channels
        )

        wav.setsampwidth(
            sample_width
        )

        wav.setframerate(
            sample_rate
        )

        wav.writeframes(
            pcm
        )

    return buffer.getvalue()


def resample_pcm(
    pcm: bytes,
    source_rate: int,
    target_rate: int,
    source_channels: int,
    target_channels: int,
    sample_width: int = 2,
) -> bytes:
    """
    Convert PCM sample rate and channel count.
    """

    if not pcm:
        return b""

    result = pcm

    # --------------------------------------------------------
    # CHANNEL CONVERSION
    # --------------------------------------------------------

    if source_channels != target_channels:

        if (
            source_channels == 2
            and target_channels == 1
        ):

            result = audioop.tomono(
                result,
                sample_width,
                0.5,
                0.5,
            )

        elif (
            source_channels == 1
            and target_channels == 2
        ):

            result = audioop.tostereo(
                result,
                sample_width,
                1.0,
                1.0,
            )

        else:

            raise ValueError(
                "Unsupported channel conversion: "
                f"{source_channels} -> {target_channels}"
            )

    # --------------------------------------------------------
    # SAMPLE RATE CONVERSION
    # --------------------------------------------------------

    if source_rate != target_rate:

        result, _ = audioop.ratecv(
            result,
            sample_width,
            target_channels,
            source_rate,
            target_rate,
            None,
        )

    return result


def calculate_rms(
    pcm: bytes,
    sample_width: int = 2,
) -> float:
    """
    Calculate RMS volume.
    """

    if not pcm:
        return 0.0

    try:

        return float(
            audioop.rms(
                pcm,
                sample_width,
            )
        )

    except Exception:

        return 0.0


def calculate_peak(
    pcm: bytes,
    sample_width: int = 2,
) -> int:
    """
    Calculate peak volume.
    """

    if not pcm:
        return 0

    try:

        return int(
            audioop.max(
                pcm,
                sample_width,
            )
        )

    except Exception:

        return 0


def pcm_duration(
    pcm: bytes,
    sample_rate: int,
    channels: int,
    sample_width: int,
) -> float:
    """
    Calculate PCM duration in seconds.
    """

    bytes_per_second = (
        sample_rate
        * channels
        * sample_width
    )

    if bytes_per_second <= 0:
        return 0.0

    return len(pcm) / bytes_per_second


def _character_value(
    character: Any | None,
    key: str,
    default: Any = None,
) -> Any:
    """
    Supports both:
    - Character objects
    - dict-like characters
    """

    if character is None:
        return default

    if hasattr(
        character,
        key,
    ):

        return getattr(
            character,
            key,
            default,
        )

    if isinstance(
        character,
        Mapping,
    ):

        return character.get(
            key,
            default,
        )

    return default


# ============================================================
# DISCORD PCM SOURCE
# ============================================================

class PCMSource(
    discord.AudioSource
):
    """
    Raw PCM source for Discord.

    Expected:
        48000 Hz
        stereo
        16-bit
    """

    FRAME_DURATION = 0.02

    def __init__(
        self,
        pcm: bytes,
        sample_rate: int = DISCORD_SAMPLE_RATE,
        channels: int = 2,
        sample_width: int = 2,
    ):

        self.pcm = pcm
        self.position = 0

        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width

        self.frame_size = int(
            self.sample_rate
            * self.channels
            * self.sample_width
            * self.FRAME_DURATION
        )

        self.frame_size = max(
            1,
            self.frame_size,
        )

    def read(self) -> bytes:
        """
        Return exactly one 20ms frame.
        """

        if self.position >= len(
            self.pcm
        ):

            return b""

        frame = self.pcm[
            self.position:
            self.position + self.frame_size
        ]

        self.position += len(
            frame
        )

        if len(frame) < self.frame_size:

            frame += b"\x00" * (
                self.frame_size
                - len(frame)
            )

        return frame

    def is_opus(self) -> bool:
        return False

    def cleanup(self) -> None:

        self.pcm = b""
        self.position = 0


# ============================================================
# VOICE AI SINK
# ============================================================

class VoiceAISink(
    voice_recv.AudioSink
):
    """
    Receives decoded Discord PCM,
    detects speech,
    buffers speech,
    and sends completed speech
    to the local faster-whisper STT pipeline.

    A tiny rolling pre-roll is maintained for each user.
    Only when speech is detected is that recent audio attached
    to the actual speech buffer.
    """

    def __init__(
        self,
        session: "VoiceSession",
    ):

        super().__init__()

        self.session = session

        # Event loop used by thread-safe callbacks.
        self.loop = (
            asyncio.get_running_loop()
        )

        # ----------------------------------------------------
        # Actual speech buffers.
        # ----------------------------------------------------

        self.buffers: dict[
            int,
            bytearray,
        ] = {}

        # ----------------------------------------------------
        # Tiny rolling pre-roll buffers.
        #
        # This stores only recent audio and is never treated
        # as speech unless a speech frame is detected.
        # ----------------------------------------------------

        self.pre_roll: dict[
            int,
            deque[bytes],
        ] = {}

        # ----------------------------------------------------
        # Users currently inside a speech segment.
        # ----------------------------------------------------

        self.speech_active: set[int] = set()

        self.last_speech_time: dict[
            int,
            float,
        ] = {}

        self.processing: set[int] = set()
        self.scheduled: set[int] = set()

        self.closed = False

        self._silence_task = (
            asyncio.create_task(
                self._silence_monitor()
            )
        )

        logger.info(
            "VoiceAISink initialized | "
            "threshold=%s | pre_roll=%.2fs",
            SILENCE_RMS_THRESHOLD,
            PRE_ROLL_SECONDS,
        )

    # ========================================================
    # RECEIVE CONFIG
    # ========================================================

    def wants_opus(self) -> bool:
        """
        False = receive decoded PCM.
        """

        return False

    # ========================================================
    # RECEIVE AUDIO
    # ========================================================

    def write(
        self,
        user: Any,
        data: Any,
    ) -> None:
        """
        Synchronous audio callback.

        Flow:

            Discord PCM
                ↓
            pre-roll
                ↓
            speech detection
                ↓
            actual speech buffer
                ↓
            silence timeout
                ↓
            WAV
                ↓
            local faster-whisper
        """

        if self.closed:
            return

        # ----------------------------------------------------
        # Do not capture anything while the bot is speaking.
        #
        # This prevents TTS echo from returning to STT.
        # ----------------------------------------------------

        if self.session.is_speaking:
            return

        try:

            if user is None:
                return

            user_id = getattr(
                user,
                "id",
                None,
            )

            if user_id is None:
                return

            # ------------------------------------------------
            # Ignore the bot's own Discord user.
            # ------------------------------------------------

            bot_user = self.session.bot.user

            if (
                bot_user is not None
                and getattr(
                    bot_user,
                    "id",
                    None,
                ) == user_id
            ):

                return

            # ------------------------------------------------
            # EXTRACT PCM
            # ------------------------------------------------

            pcm = getattr(
                data,
                "pcm",
                None,
            )

            if not pcm:

                raw_data = getattr(
                    data,
                    "data",
                    None,
                )

                if isinstance(
                    raw_data,
                    bytes,
                ):

                    pcm = raw_data

            if not pcm:
                return

            if not isinstance(
                pcm,
                bytes,
            ):

                pcm = bytes(
                    pcm
                )

            if not pcm:
                return

            # ------------------------------------------------
            # SPEECH DETECTION
            # ------------------------------------------------

            rms = calculate_rms(
                pcm
            )

            peak = calculate_peak(
                pcm
            )

            is_speech = (
                rms >= SILENCE_RMS_THRESHOLD
            )

            # ------------------------------------------------
            # PRE-ROLL
            #
            # Keep only a very small amount of recent audio.
            #
            # This includes silence, but that silence is not
            # copied into the actual speech buffer unless it is
            # part of the short pre-roll immediately before
            # speech starts.
            # ------------------------------------------------

            pre_roll = self.pre_roll.setdefault(
                user_id,
                deque(),
            )

            pre_roll.append(
                pcm
            )

            total_pre_roll = sum(
                len(chunk)
                for chunk in pre_roll
            )

            while (
                total_pre_roll
                > PRE_ROLL_MAX_BYTES
                and pre_roll
            ):

                removed = pre_roll.popleft()

                total_pre_roll -= len(
                    removed
                )

            # ------------------------------------------------
            # SILENCE
            #
            # Keep it only inside pre-roll.
            # Do NOT add it to actual speech buffer.
            # ------------------------------------------------

            if not is_speech:
                return

            # ------------------------------------------------
            # SPEECH JUST STARTED
            # ------------------------------------------------

            if user_id not in self.speech_active:

                buffer = (
                    self.buffers.setdefault(
                        user_id,
                        bytearray(),
                    )
                )

                previous_frames = list(
                    pre_roll
                )

                # The final frame is the current frame.
                # Add only the older pre-roll frames here.
                if len(previous_frames) > 1:

                    for chunk in previous_frames[:-1]:

                        if (
                            len(buffer)
                            + len(chunk)
                            <= MAX_AUDIO_BUFFER_BYTES
                        ):

                            buffer.extend(
                                chunk
                            )

                self.speech_active.add(
                    user_id
                )

                logger.debug(
                    "Speech started | user=%s | "
                    "pre_roll=%.2fs | rms=%.1f | peak=%s",
                    user_id,
                    PRE_ROLL_SECONDS,
                    rms,
                    peak,
                )

            # ------------------------------------------------
            # BUFFER CURRENT SPEECH FRAME
            # ------------------------------------------------

            buffer = (
                self.buffers.setdefault(
                    user_id,
                    bytearray(),
                )
            )

            if (
                len(buffer)
                + len(pcm)
                > MAX_AUDIO_BUFFER_BYTES
            ):

                remaining = max(
                    0,
                    MAX_AUDIO_BUFFER_BYTES
                    - len(buffer),
                )

                if remaining > 0:

                    buffer.extend(
                        pcm[:remaining]
                    )

            else:

                buffer.extend(
                    pcm
                )

            # ------------------------------------------------
            # SPEECH TIMER
            # ------------------------------------------------

            self.last_speech_time[
                user_id
            ] = time.monotonic()

            duration = pcm_duration(
                bytes(buffer),
                DISCORD_SAMPLE_RATE,
                2,
                DISCORD_SAMPLE_WIDTH,
            )

            logger.debug(
                "Audio received | user=%s | "
                "bytes=%s | buffer=%s | duration=%.2fs | "
                "rms=%.1f | peak=%s | speech=%s",
                user_id,
                len(pcm),
                len(buffer),
                duration,
                rms,
                peak,
                is_speech,
            )

            # ------------------------------------------------
            # MAXIMUM RECORDING
            # ------------------------------------------------

            if (
                duration
                >= MAX_RECORDING_SECONDS
            ):

                logger.info(
                    "Maximum recording duration reached | "
                    "user=%s | duration=%.2fs",
                    user_id,
                    duration,
                )

                self._schedule_process(
                    user_id
                )

        except Exception:

            logger.exception(
                "Voice sink write failed"
            )

    # ========================================================
    # THREAD-SAFE SCHEDULER
    # ========================================================

    def _schedule_process(
        self,
        user_id: int,
    ) -> None:

        if self.closed:
            return

        if self.session.is_speaking:
            return

        if user_id in self.processing:
            return

        if user_id in self.scheduled:
            return

        self.scheduled.add(
            user_id
        )

        def schedule() -> None:

            if self.closed:

                self.scheduled.discard(
                    user_id
                )

                return

            if self.session.is_speaking:

                self.scheduled.discard(
                    user_id
                )

                return

            if user_id in self.processing:

                self.scheduled.discard(
                    user_id
                )

                return

            asyncio.create_task(
                self._process_user_audio(
                    user_id
                )
            )

        try:

            self.loop.call_soon_threadsafe(
                schedule
            )

        except RuntimeError:

            self.scheduled.discard(
                user_id
            )

    # ========================================================
    # SILENCE MONITOR
    # ========================================================

    async def _silence_monitor(
        self,
    ) -> None:

        try:

            while not self.closed:

                await asyncio.sleep(
                    0.10
                )

                if self.session.is_speaking:
                    continue

                now = time.monotonic()

                for (
                    user_id,
                    last_time,
                ) in list(
                    self.last_speech_time.items()
                ):

                    if user_id in self.processing:
                        continue

                    if user_id in self.scheduled:
                        continue

                    buffer = self.buffers.get(
                        user_id
                    )

                    if not buffer:
                        continue

                    duration = pcm_duration(
                        bytes(buffer),
                        DISCORD_SAMPLE_RATE,
                        2,
                        DISCORD_SAMPLE_WIDTH,
                    )

                    silence = (
                        now - last_time
                    )

                    if (
                        silence
                        >= SILENCE_TIMEOUT_SECONDS
                        and duration
                        >= MIN_AUDIO_SECONDS
                    ):

                        logger.info(
                            "Speech ended | user=%s | "
                            "silence=%.2fs | duration=%.2fs",
                            user_id,
                            silence,
                            duration,
                        )

                        self._schedule_process(
                            user_id
                        )

        except asyncio.CancelledError:

            pass

        except Exception:

            logger.exception(
                "Silence monitor crashed"
            )

    # ========================================================
    # PROCESS AUDIO
    # ========================================================

    async def _process_user_audio(
        self,
        user_id: int,
    ) -> None:

        self.scheduled.discard(
            user_id
        )

        if self.closed:
            return

        if self.session.is_speaking:
            return

        if user_id in self.processing:
            return

        self.processing.add(
            user_id
        )

        try:

            # ------------------------------------------------
            # TAKE BUFFER
            # ------------------------------------------------

            buffer = self.buffers.pop(
                user_id,
                None,
            )

            self.last_speech_time.pop(
                user_id,
                None,
            )

            self.speech_active.discard(
                user_id
            )

            # Start next turn cleanly.
            self.pre_roll.pop(
                user_id,
                None,
            )

            if not buffer:
                return

            pcm = bytes(
                buffer
            )

            duration = pcm_duration(
                pcm,
                DISCORD_SAMPLE_RATE,
                2,
                DISCORD_SAMPLE_WIDTH,
            )

            if (
                duration
                < MIN_AUDIO_SECONDS
            ):

                logger.debug(
                    "Audio too short | user=%s | duration=%.2f",
                    user_id,
                    duration,
                )

                return

            logger.info(
                "Processing voice | user=%s | "
                "duration=%.2fs | bytes=%s | "
                "rms=%.1f | peak=%s",
                user_id,
                duration,
                len(pcm),
                calculate_rms(
                    pcm
                ),
                calculate_peak(
                    pcm
                ),
            )

            # ------------------------------------------------
            # DISCORD PCM
            #
            # 48kHz stereo
            #
            # ↓
            #
            # STT PCM
            #
            # 16kHz mono
            # ------------------------------------------------

            stt_pcm = resample_pcm(
                pcm,
                source_rate=DISCORD_SAMPLE_RATE,
                target_rate=GEMINI_INPUT_SAMPLE_RATE,
                source_channels=2,
                target_channels=GEMINI_INPUT_CHANNELS,
                sample_width=DISCORD_SAMPLE_WIDTH,
            )

            if not stt_pcm:

                logger.warning(
                    "STT PCM conversion returned empty audio | "
                    "user=%s",
                    user_id,
                )

                return

            # ------------------------------------------------
            # PCM -> WAV
            # ------------------------------------------------

            wav_data = pcm_to_wav(
                stt_pcm,
                sample_rate=GEMINI_INPUT_SAMPLE_RATE,
                channels=GEMINI_INPUT_CHANNELS,
                sample_width=DISCORD_SAMPLE_WIDTH,
            )

            if not wav_data:

                logger.warning(
                    "WAV conversion returned empty audio | "
                    "user=%s",
                    user_id,
                )

                return

            # ------------------------------------------------
            # USERNAME
            # ------------------------------------------------

            username = (
                f"User {user_id}"
            )

            try:

                member = (
                    self.session.guild.get_member(
                        user_id
                    )
                )

                if member is not None:

                    username = (
                        member.display_name
                        or member.name
                    )

            except Exception:

                logger.debug(
                    "Could not resolve member name | "
                    "user=%s",
                    user_id,
                    exc_info=True,
                )

            # ------------------------------------------------
            # CHARACTER
            # ------------------------------------------------

            character = getattr(
                self.session,
                "character",
                None,
            )

            # ------------------------------------------------
            # VOICE
            # ------------------------------------------------

            voice_name = normalize_voice_name(
                getattr(
                    self.session,
                    "voice_name",
                    DEFAULT_GEMINI_VOICE,
                )
            )

            if not is_valid_voice(
                voice_name
            ):

                voice_name = (
                    DEFAULT_GEMINI_VOICE
                )

            # ------------------------------------------------
            # SPEED
            # ------------------------------------------------

            speed = normalize_speech_speed(
                getattr(
                    self.session,
                    "speech_speed",
                    DEFAULT_SPEECH_SPEED,
                )
            )

            # ------------------------------------------------
            # AI PIPELINE
            #
            # gemini.py keeps the compatibility class name
            # GeminiEngine, but STT is now local faster-whisper.
            # ------------------------------------------------

            result = (
                await self.session.engine.process_voice(
                    audio=wav_data,
                    username=username,
                    voice=voice_name,
                    speed=speed,
                    character=character,
                )
            )

            if not result:

                logger.warning(
                    "AI returned no result | user=%s",
                    user_id,
                )

                return

            if not result.get(
                "success",
                False,
            ):

                logger.warning(
                    "Voice AI failed | user=%s | error=%s",
                    user_id,
                    result.get(
                        "error",
                        "unknown error",
                    ),
                )

                return

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            transcript = str(
                result.get(
                    "transcript",
                    "",
                )
                or ""
            ).strip()

            response_text = str(
                result.get(
                    "response",
                    "",
                )
                or ""
            ).strip()

            audio = result.get(
                "audio"
            )

            logger.info(
                "Voice AI complete | user=%s | "
                "transcript=%r | response=%r",
                user_id,
                transcript,
                response_text,
            )

            # ------------------------------------------------
            # PLAY GROQ ORPHEUS AUDIO
            # ------------------------------------------------

            if audio:

                await self.session.play_tts(
                    audio
                )

            else:

                logger.warning(
                    "AI returned no TTS audio | "
                    "user=%s",
                    user_id,
                )

        except asyncio.CancelledError:

            raise

        except Exception:

            logger.exception(
                "Voice processing failed | user=%s",
                user_id,
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

        if self._silence_task:

            self._silence_task.cancel()

            self._silence_task = None

        self.buffers.clear()
        self.pre_roll.clear()
        self.last_speech_time.clear()
        self.processing.clear()
        self.scheduled.clear()
        self.speech_active.clear()

        logger.info(
            "Voice receive sink cleaned up"
        )


# ============================================================
# VOICE SESSION
# ============================================================

class VoiceSession:

    def __init__(
        self,
        bot: discord.Client,
        guild: discord.Guild,
        channel: discord.VoiceChannel,
        voice_client: voice_recv.VoiceRecvClient,
        *,
        voice: str = DEFAULT_GEMINI_VOICE,
        speed: float = DEFAULT_SPEECH_SPEED,
        character: Any | None = None,
    ):

        self.bot = bot
        self.guild = guild
        self.channel = channel
        self.voice_client = voice_client

        self.engine = GeminiEngine()

        self.voice_name = normalize_voice_name(
            voice
        )

        if not is_valid_voice(
            self.voice_name
        ):

            self.voice_name = (
                DEFAULT_GEMINI_VOICE
            )

        self.speech_speed = (
            normalize_speech_speed(
                speed
            )
        )

        self.character = character

        self.sink: VoiceAISink | None = None

        self.started_at = (
            time.monotonic()
        )

        self._play_lock = (
            asyncio.Lock()
        )

        # True while the bot is speaking.
        #
        # The receive sink uses this to completely ignore
        # incoming audio and prevent the TTS voice from
        # feeding itself back into STT.
        self.is_speaking = False

        self._closed = False

    # ========================================================
    # PROPERTIES
    # ========================================================

    @property
    def voice(
        self,
    ) -> str:

        return self.voice_name

    @property
    def speed(
        self,
    ) -> float:

        return self.speech_speed

    # ========================================================
    # START RECEIVE
    # ========================================================

    def start_receive(
        self,
    ) -> None:

        if self._closed:

            raise RuntimeError(
                "Voice session is closed."
            )

        if not isinstance(
            self.voice_client,
            voice_recv.VoiceRecvClient,
        ):

            raise RuntimeError(
                "VoiceRecvClient is required "
                "for voice receive."
            )

        if self.sink is not None:
            return

        self.sink = VoiceAISink(
            self
        )

        self.voice_client.listen(
            self.sink
        )

        logger.info(
            "Voice receive sink started"
        )

        logger.info(
            "Voice AI listening"
        )

    # ========================================================
    # CHANGE VOICE
    # ========================================================

    def set_voice(
        self,
        voice: str,
    ) -> str:

        if not ALLOW_VOICE_CHANGE:

            raise RuntimeError(
                "Voice changing is disabled."
            )

        normalized = (
            normalize_voice_name(
                voice
            )
        )

        if not is_valid_voice(
            normalized
        ):

            raise ValueError(
                f"Invalid voice: {voice}"
            )

        self.voice_name = (
            normalized
        )

        try:

            self.engine.set_voice(
                normalized
            )

        except Exception:

            # Keep session state valid even if the engine
            # does not expose voice configuration.
            pass

        logger.info(
            "Voice changed | guild=%s | voice=%s",
            self.guild.id,
            normalized,
        )

        return normalized

    # ========================================================
    # CHANGE SPEED
    # ========================================================

    def set_speed(
        self,
        speed: float,
    ) -> float:

        self.speech_speed = (
            normalize_speech_speed(
                speed
            )
        )

        logger.info(
            "Speech speed changed | guild=%s | speed=%.2f",
            self.guild.id,
            self.speech_speed,
        )

        return self.speech_speed

    # ========================================================
    # CHANGE CHARACTER
    # ========================================================

    def set_character(
        self,
        character: Any | None,
    ) -> None:

        self.character = character

        if character:

            name = _character_value(
                character,
                "name",
                "Unknown",
            )

            logger.info(
                "Character changed | guild=%s | character=%s",
                self.guild.id,
                name,
            )

        else:

            logger.info(
                "Character cleared | guild=%s",
                self.guild.id,
            )

    # ========================================================
    # PLAY TTS
    # ========================================================

    async def play_tts(
        self,
        audio: bytes,
    ) -> None:

        if self._closed:
            return

        if not audio:

            logger.warning(
                "TTS audio is empty"
            )

            return

        async with self._play_lock:

            if self._closed:
                return

            # ------------------------------------------------
            # Disable voice capture while TTS is playing.
            # ------------------------------------------------

            self.is_speaking = True

            if self.sink is not None:

                self.sink.buffers.clear()
                self.sink.pre_roll.clear()
                self.sink.last_speech_time.clear()
                self.sink.speech_active.clear()

            try:

                # ------------------------------------------------
                # Wait for previous audio, if any.
                # ------------------------------------------------

                while (
                    self.voice_client.is_playing()
                ):

                    if self._closed:
                        return

                    await asyncio.sleep(
                        0.05
                    )

                # ------------------------------------------------
                # Groq Orpheus:
                #
                # 24kHz mono 16-bit
                #
                # ↓
                #
                # Discord:
                #
                # 48kHz stereo 16-bit
                # ------------------------------------------------

                discord_pcm = resample_pcm(
                    audio,
                    source_rate=(
                        GEMINI_TTS_SAMPLE_RATE
                    ),
                    target_rate=(
                        DISCORD_SAMPLE_RATE
                    ),
                    source_channels=(
                        GEMINI_TTS_CHANNELS
                    ),
                    target_channels=2,
                    sample_width=2,
                )

                if not discord_pcm:

                    logger.warning(
                        "Converted TTS audio is empty"
                    )

                    return

                # ------------------------------------------------
                # Discord audio source.
                # ------------------------------------------------

                source = PCMSource(
                    discord_pcm,
                    sample_rate=(
                        DISCORD_SAMPLE_RATE
                    ),
                    channels=2,
                    sample_width=2,
                )

                loop = (
                    asyncio.get_running_loop()
                )

                finished = (
                    loop.create_future()
                )

                # ------------------------------------------------
                # Discord playback callback.
                # ------------------------------------------------

                def after_playback(
                    error: Exception | None,
                ) -> None:

                    def finish() -> None:

                        if finished.done():
                            return

                        if error:

                            finished.set_exception(
                                error
                            )

                        else:

                            finished.set_result(
                                None
                            )

                    try:

                        loop.call_soon_threadsafe(
                            finish
                        )

                    except RuntimeError:

                        pass

                logger.info(
                    "Playing TTS | Groq Orpheus | bytes=%s",
                    len(discord_pcm),
                )

                try:

                    self.voice_client.play(
                        source,
                        after=after_playback,
                    )

                except Exception:

                    source.cleanup()

                    raise

                try:

                    await finished

                    logger.info(
                        "TTS playback finished"
                    )

                except asyncio.CancelledError:

                    try:

                        if (
                            self.voice_client.is_playing()
                        ):

                            self.voice_client.stop()

                    except Exception:

                        pass

                    raise

                except Exception:

                    logger.exception(
                        "TTS playback failed"
                    )

                finally:

                    source.cleanup()

            finally:

                # ------------------------------------------------
                # Re-enable receive after TTS.
                # ------------------------------------------------

                self.is_speaking = False

                if self.sink is not None:

                    self.sink.buffers.clear()
                    self.sink.pre_roll.clear()
                    self.sink.last_speech_time.clear()
                    self.sink.speech_active.clear()

    # ========================================================
    # CLOSE
    # ========================================================

    async def close(
        self,
    ) -> None:

        if self._closed:
            return

        self._closed = True
        self.is_speaking = True

        try:

            if self.sink:

                self.sink.cleanup()

        except Exception:

            logger.exception(
                "Failed to cleanup voice sink"
            )

        self.sink = None

        try:

            if (
                self.voice_client.is_playing()
            ):

                self.voice_client.stop()

        except Exception:

            pass

        try:

            await self.engine.close()

        except Exception:

            logger.exception(
                "Failed to close AI engine"
            )

        logger.info(
            "Voice session closed | guild=%s",
            self.guild.id,
        )


# ============================================================
# VOICE SESSION MANAGER
# ============================================================

class VoiceSessionManager:

    def __init__(
        self,
        bot: discord.Client,
    ):

        self.bot = bot

        self.sessions: dict[
            int,
            VoiceSession,
        ] = {}

        self._lock = (
            asyncio.Lock()
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
    # JOIN
    # ========================================================

    async def join(
        self,
        channel: discord.VoiceChannel,
        *,
        voice: str = DEFAULT_GEMINI_VOICE,
        speed: float = DEFAULT_SPEECH_SPEED,
        character: Any | None = None,
    ) -> VoiceSession:

        guild = channel.guild

        async with self._lock:

            # ------------------------------------------------
            # Existing Cloud Voice AI session
            # ------------------------------------------------

            existing = self.sessions.get(
                guild.id
            )

            if existing:

                # Already in requested channel.
                if (
                    existing.channel.id
                    == channel.id
                ):

                    return existing

                # Close old session before creating a new one.
                self.sessions.pop(
                    guild.id,
                    None,
                )

                try:

                    await existing.close()

                except Exception:

                    logger.exception(
                        "Failed to close existing "
                        "voice session"
                    )

                try:

                    if (
                        existing.voice_client
                        .is_connected()
                    ):

                        await (
                            existing.voice_client.disconnect(
                                force=True
                            )
                        )

                except Exception:

                    logger.exception(
                        "Failed to disconnect existing "
                        "voice client"
                    )

            # ------------------------------------------------
            # Existing Discord voice client
            # ------------------------------------------------

            existing_client = (
                guild.voice_client
            )

            if existing_client:

                # Our system requires VoiceRecvClient.
                if not isinstance(
                    existing_client,
                    voice_recv.VoiceRecvClient,
                ):

                    try:

                        await (
                            existing_client.disconnect(
                                force=True
                            )
                        )

                    except Exception:

                        logger.exception(
                            "Failed to disconnect old "
                            "voice client"
                        )

            # ------------------------------------------------
            # CONNECT
            # ------------------------------------------------

            logger.info(
                "Connecting VoiceRecvClient | "
                "guild=%s | channel=%s",
                guild.id,
                channel.name,
            )

            voice_client = (
                await channel.connect(
                    cls=(
                        voice_recv.VoiceRecvClient
                    ),
                    self_deaf=False,
                    self_mute=False,
                )
            )

            if not isinstance(
                voice_client,
                voice_recv.VoiceRecvClient,
            ):

                try:

                    await voice_client.disconnect(
                        force=True
                    )

                except Exception:

                    pass

                raise RuntimeError(
                    "Discord did not create "
                    "VoiceRecvClient."
                )

            # ------------------------------------------------
            # CREATE SESSION
            # ------------------------------------------------

            session = VoiceSession(
                self.bot,
                guild,
                channel,
                voice_client,
                voice=voice,
                speed=speed,
                character=character,
            )

            self.sessions[
                guild.id
            ] = session

            # ------------------------------------------------
            # START RECEIVE
            # ------------------------------------------------

            try:

                session.start_receive()

            except Exception:

                self.sessions.pop(
                    guild.id,
                    None,
                )

                try:

                    await voice_client.disconnect(
                        force=True
                    )

                except Exception:

                    pass

                raise

            logger.info(
                "Voice session created | "
                "guild=%s | channel=%s | "
                "voice=%s | speed=%.2f",
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

        async with self._lock:

            session = self.sessions.pop(
                guild_id,
                None,
            )

            if not session:
                return False

            try:

                await session.close()

            except Exception:

                logger.exception(
                    "Failed to close voice session"
                )

            try:

                if (
                    session.voice_client
                    .is_connected()
                ):

                    await (
                        session.voice_client.disconnect(
                            force=True
                        )
                    )

            except Exception:

                logger.exception(
                    "Failed to disconnect voice client"
                )

            return True

    # ========================================================
    # CLOSE ALL
    # ========================================================

    async def close_all(
        self,
    ) -> None:

        guild_ids = list(
            self.sessions.keys()
        )

        for guild_id in guild_ids:

            try:

                await self.leave(
                    guild_id
                )

            except Exception:

                logger.exception(
                    "Failed to close guild "
                    "voice session | guild=%s",
                    guild_id,
                )

    # ========================================================
    # STATS
    # ========================================================

    def stats(
        self,
    ) -> dict[str, int]:

        return {
            "active_sessions": len(
                self.sessions
            ),
            "max_sessions": 0,
        }


# ============================================================
# VOICE LIST
# ============================================================

def get_voice_names() -> list[str]:

    return list(
        GEMINI_VOICES
    )


def format_voice_list() -> str:

    voices = (
        get_voice_names()
    )

    lines = [
        f"🎙️ **Available voices ({len(voices)})**",
        "",
    ]

    for index, voice in enumerate(
        voices,
        start=1,
    ):

        lines.append(
            f"`{index:02}` • **{voice}**"
        )

    return "\n".join(
        lines
    )


async def send_voice_list(
    interaction: discord.Interaction,
) -> None:

    await interaction.response.send_message(
        format_voice_list(),
        ephemeral=True,
    )
