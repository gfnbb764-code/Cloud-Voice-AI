# voice.py
# ============================================================
# Cloud Voice AI — Discord Voice System
# Discord receive + Gemini STT/AI/TTS
# ============================================================

from __future__ import annotations

import asyncio
import audioop
import io
import logging
import time
import wave
from typing import Any

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
# AUDIO SETTINGS
# ============================================================

# Discord can continue sending PCM frames even when the user
# is not actively speaking. Therefore silence detection must
# be based on volume, not simply packet arrival time.
#
# 500 is intentionally low enough to detect normal speech while
# filtering normal near-silent Discord PCM noise.
SILENCE_RMS_THRESHOLD = 500


# ============================================================
# HELPERS
# ============================================================

def pcm_to_wav(
    pcm: bytes,
    sample_rate: int = GEMINI_INPUT_SAMPLE_RATE,
    channels: int = GEMINI_INPUT_CHANNELS,
    sample_width: int = DISCORD_SAMPLE_WIDTH,
) -> bytes:
    """Convert raw PCM to WAV."""

    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)

    return buffer.getvalue()


def resample_pcm(
    pcm: bytes,
    source_rate: int,
    target_rate: int,
    source_channels: int,
    target_channels: int,
    sample_width: int = 2,
) -> bytes:
    """Convert PCM sample rate/channels."""

    if not pcm:
        return b""

    result = pcm

    if source_channels != target_channels:
        if source_channels == 2 and target_channels == 1:
            result = audioop.tomono(
                result,
                sample_width,
                0.5,
                0.5,
            )

        elif source_channels == 1 and target_channels == 2:
            result = audioop.tostereo(
                result,
                sample_width,
                1.0,
                1.0,
            )

        else:
            raise ValueError(
                f"Unsupported channel conversion: "
                f"{source_channels} -> {target_channels}"
            )

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
    """Calculate RMS volume."""

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
    """Calculate peak volume."""

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
    """Calculate PCM duration."""

    bytes_per_second = (
        sample_rate
        * channels
        * sample_width
    )

    if bytes_per_second <= 0:
        return 0.0

    return len(pcm) / bytes_per_second


# ============================================================
# PCM DISCORD AUDIO SOURCE
# ============================================================

class PCMSource(discord.AudioSource):
    """
    Plays raw PCM through Discord.

    Expected:
        48kHz
        stereo
        16-bit
    """

    FRAME_BYTES = 3840  # 20ms @ 48kHz stereo 16-bit

    def __init__(
        self,
        pcm: bytes,
        sample_rate: int = GEMINI_TTS_SAMPLE_RATE,
        channels: int = GEMINI_TTS_CHANNELS,
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
            * 0.02
        )

        self.frame_size = max(
            1,
            self.frame_size,
        )

    def read(self) -> bytes:
        """Return the next 20ms frame."""

        if self.position >= len(self.pcm):
            return b""

        frame = self.pcm[
            self.position:
            self.position + self.frame_size
        ]

        self.position += len(frame)

        # Discord expects exactly one 20ms frame.
        if len(frame) < self.frame_size:
            frame += b"\x00" * (
                self.frame_size - len(frame)
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

class VoiceAISink(voice_recv.AudioSink):
    """
    Receives decoded PCM from Discord.

    Important:
    - wants_opus() returns False.
    - The fork therefore provides decoded PCM.
    - Audio is buffered per user.
    - Speech is processed after actual silence.
    """

    def __init__(
        self,
        session: "VoiceSession",
    ):
        super().__init__()

        self.session = session

        # The sink callback can run outside the normal asyncio
        # task context, so capture the loop here.
        self.loop = asyncio.get_running_loop()

        self.buffers: dict[int, bytearray] = {}

        # Time of the last NON-SILENT audio frame.
        self.last_speech_time: dict[int, float] = {}

        self.processing: set[int] = set()
        self.scheduled: set[int] = set()

        self.closed = False

        self._silence_task = asyncio.create_task(
            self._silence_monitor()
        )

        logger.info(
            "VoiceAISink initialized"
        )

    # ========================================================
    # REQUIRED BY discord-ext-voice-recv
    # ========================================================

    def wants_opus(self) -> bool:
        """
        False = give us decoded PCM instead of Opus.
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
        """
        Called by discord-ext-voice-recv when audio arrives.

        This function must stay synchronous.
        """

        if self.closed:
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
            # Get PCM
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
                pcm = bytes(pcm)

            if not pcm:
                return

            # ------------------------------------------------
            # Calculate volume BEFORE buffering.
            # ------------------------------------------------

            rms = calculate_rms(pcm)
            peak = calculate_peak(pcm)

            is_speech = (
                rms >= SILENCE_RMS_THRESHOLD
            )

            # ------------------------------------------------
            # Buffer audio.
            #
            # We intentionally keep the silence after speech
            # because Gemini can benefit from the natural end
            # of the utterance.
            # ------------------------------------------------

            buffer = self.buffers.setdefault(
                user_id,
                bytearray(),
            )

            if (
                len(buffer) + len(pcm)
                > MAX_AUDIO_BUFFER_BYTES
            ):
                remaining = max(
                    0,
                    MAX_AUDIO_BUFFER_BYTES
                    - len(buffer),
                )

                if remaining:
                    buffer.extend(
                        pcm[:remaining]
                    )
            else:
                buffer.extend(pcm)

            # ------------------------------------------------
            # ONLY speech updates the silence timer.
            #
            # This is the important fix.
            # ------------------------------------------------

            if is_speech:
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
                "Audio received | user=%s | bytes=%s | "
                "buffer=%s | duration=%.2fs | "
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
            # Maximum recording duration fallback.
            # ------------------------------------------------

            if duration >= MAX_RECORDING_SECONDS:
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
    # THREAD-SAFE PROCESS SCHEDULER
    # ========================================================

    def _schedule_process(
        self,
        user_id: int,
    ) -> None:
        """
        Safely schedule processing on the asyncio event loop.

        The receive sink callback may execute outside the
        normal asyncio task context.
        """

        if self.closed:
            return

        if user_id in self.processing:
            return

        if user_id in self.scheduled:
            return

        self.scheduled.add(user_id)

        def schedule() -> None:
            if self.closed:
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
    # SILENCE DETECTION
    # ========================================================

    async def _silence_monitor(
        self,
    ) -> None:
        """
        Detect when a user stops talking.

        Only NON-SILENT audio updates last_speech_time.
        Therefore Discord's continuing silent PCM packets
        cannot keep the recording alive forever.
        """

        try:
            while not self.closed:

                await asyncio.sleep(
                    0.10
                )

                now = time.monotonic()

                for user_id, last_time in list(
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
    # PROCESS USER AUDIO
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

        if user_id in self.processing:
            return

        self.processing.add(
            user_id
        )

        try:
            # ------------------------------------------------
            # Take current buffer.
            # ------------------------------------------------

            buffer = self.buffers.pop(
                user_id,
                None,
            )

            self.last_speech_time.pop(
                user_id,
                None,
            )

            if not buffer:
                return

            pcm = bytes(buffer)

            duration = pcm_duration(
                pcm,
                DISCORD_SAMPLE_RATE,
                2,
                DISCORD_SAMPLE_WIDTH,
            )

            if duration < MIN_AUDIO_SECONDS:
                logger.debug(
                    "Audio too short | user=%s | "
                    "duration=%.2f",
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
                calculate_rms(pcm),
                calculate_peak(pcm),
            )

            # ------------------------------------------------
            # Convert Discord audio:
            #
            # 48kHz stereo
            # ->
            # 16kHz mono
            # ------------------------------------------------

            gemini_pcm = resample_pcm(
                pcm,
                source_rate=DISCORD_SAMPLE_RATE,
                target_rate=GEMINI_INPUT_SAMPLE_RATE,
                source_channels=2,
                target_channels=GEMINI_INPUT_CHANNELS,
                sample_width=DISCORD_SAMPLE_WIDTH,
            )

            wav_data = pcm_to_wav(
                gemini_pcm,
                sample_rate=GEMINI_INPUT_SAMPLE_RATE,
                channels=GEMINI_INPUT_CHANNELS,
                sample_width=DISCORD_SAMPLE_WIDTH,
            )

            # ------------------------------------------------
            # User information.
            # ------------------------------------------------

            username = f"User {user_id}"

            try:
                guild = self.session.guild

                member = guild.get_member(
                    user_id
                )

                if member is not None:
                    username = (
                        member.display_name
                        or member.name
                    )

            except Exception:
                pass

            # ------------------------------------------------
            # Character.
            # ------------------------------------------------

            character = None

            try:
                character = self.session.character

            except Exception:
                character = None

            # ------------------------------------------------
            # Voice + speed.
            # ------------------------------------------------

            voice_name = normalize_voice_name(
                getattr(
                    self.session,
                    "voice_name",
                    DEFAULT_GEMINI_VOICE,
                )
            )

            speed = normalize_speech_speed(
                getattr(
                    self.session,
                    "speech_speed",
                    DEFAULT_SPEECH_SPEED,
                )
            )

            # ------------------------------------------------
            # AI processing.
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
                    "Gemini returned no result | user=%s",
                    user_id,
                )
                return

            if not result.get(
                "success"
            ):
                logger.warning(
                    "Voice AI failed | user=%s | error=%s",
                    user_id,
                    result.get("error"),
                )
                return

            transcript = result.get(
                "transcript",
                "",
            )

            response_text = result.get(
                "response",
                "",
            )

            audio = result.get(
                "audio",
            )

            logger.info(
                "Voice AI complete | user=%s | "
                "transcript=%r | response=%r",
                user_id,
                transcript,
                response_text,
            )

            # ------------------------------------------------
            # Play TTS.
            # ------------------------------------------------

            if audio:
                await self.session.play_tts(
                    audio
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

            # ------------------------------------------------
            # If the user started talking again while AI was
            # processing, keep the new buffer alive.
            # ------------------------------------------------

            if (
                not self.closed
                and user_id in self.buffers
                and self.buffers[user_id]
            ):
                self.last_speech_time.setdefault(
                    user_id,
                    time.monotonic(),
                )

    # ========================================================
    # CLEANUP
    # ========================================================

    def cleanup(self) -> None:
        self.closed = True

        if self._silence_task:
            self._silence_task.cancel()
            self._silence_task = None

        self.buffers.clear()
        self.last_speech_time.clear()
        self.processing.clear()
        self.scheduled.clear()

        logger.info(
            "Voice receive sink cleaned up"
        )


# ============================================================
# VOICE SESSION
# ============================================================

class VoiceSession:
    """One AI voice session."""

    def __init__(
        self,
        bot: discord.Client,
        guild: discord.Guild,
        channel: discord.VoiceChannel,
        voice_client: voice_recv.VoiceRecvClient,
        *,
        voice: str = DEFAULT_GEMINI_VOICE,
        speed: float = DEFAULT_SPEECH_SPEED,
        character: dict[str, Any] | None = None,
    ):
        self.bot = bot
        self.guild = guild
        self.channel = channel
        self.voice_client = voice_client

        self.engine = GeminiEngine()

        self.voice_name = normalize_voice_name(
            voice
        )

        self.speech_speed = normalize_speech_speed(
            speed
        )

        self.character = character

        self.sink: VoiceAISink | None = None

        self.started_at = time.monotonic()

        self._play_lock = asyncio.Lock()
        self._closed = False

    # ========================================================
    # PROPERTIES
    # ========================================================

    @property
    def voice(self) -> str:
        return self.voice_name

    @property
    def speed(self) -> float:
        return self.speech_speed

    # ========================================================
    # START RECEIVE
    # ========================================================

    def start_receive(self) -> None:

        if self._closed:
            raise RuntimeError(
                "Voice session is closed."
            )

        if not isinstance(
            self.voice_client,
            voice_recv.VoiceRecvClient,
        ):
            raise RuntimeError(
                "VoiceRecvClient is required for "
                "voice receive."
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

        normalized = normalize_voice_name(
            voice
        )

        if not is_valid_voice(
            normalized
        ):
            raise ValueError(
                f"Invalid voice: {voice}"
            )

        self.voice_name = normalized

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
        character: dict[str, Any] | None,
    ) -> None:

        self.character = character

        if character:
            name = character.get(
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

            # Wait for previous audio.
            while self.voice_client.is_playing():
                await asyncio.sleep(
                    0.05
                )

            # ------------------------------------------------
            # Gemini TTS:
            #
            # 24kHz mono 16-bit
            #
            # Discord:
            #
            # 48kHz stereo 16-bit
            # ------------------------------------------------

            discord_pcm = resample_pcm(
                audio,
                source_rate=GEMINI_TTS_SAMPLE_RATE,
                target_rate=DISCORD_SAMPLE_RATE,
                source_channels=GEMINI_TTS_CHANNELS,
                target_channels=2,
                sample_width=2,
            )

            if not discord_pcm:
                logger.warning(
                    "Converted TTS audio is empty"
                )
                return

            source = PCMSource(
                discord_pcm,
                sample_rate=DISCORD_SAMPLE_RATE,
                channels=2,
                sample_width=2,
            )

            loop = asyncio.get_running_loop()

            finished = loop.create_future()

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
                "Playing TTS | voice=%s | speed=%.2f | bytes=%s",
                self.voice_name,
                self.speech_speed,
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

            except asyncio.CancelledError:
                try:
                    if self.voice_client.is_playing():
                        self.voice_client.stop()
                except Exception:
                    pass

                raise

            except Exception:
                logger.exception(
                    "TTS playback failed"
                )

    # ========================================================
    # CLOSE
    # ========================================================

    async def close(self) -> None:

        if self._closed:
            return

        self._closed = True

        try:
            if self.sink:
                self.sink.cleanup()

        except Exception:
            logger.exception(
                "Failed to cleanup voice sink"
            )

        self.sink = None

        try:
            if self.voice_client.is_playing():
                self.voice_client.stop()

        except Exception:
            pass

        logger.info(
            "Voice session closed | guild=%s",
            self.guild.id,
        )


# ============================================================
# VOICE SESSION MANAGER
# ============================================================

class VoiceSessionManager:
    """Manage active AI voice sessions."""

    def __init__(
        self,
        bot: discord.Client,
    ):
        self.bot = bot

        self.sessions: dict[
            int,
            VoiceSession,
        ] = {}

        self._lock = asyncio.Lock()

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
        character: dict[str, Any] | None = None,
    ) -> VoiceSession:

        guild = channel.guild

        # ----------------------------------------------------
        # Avoid nested self._lock -> leave() deadlock.
        # ----------------------------------------------------

        async with self._lock:

            existing = self.sessions.get(
                guild.id
            )

            if existing:
                if existing.channel.id == channel.id:
                    return existing

                self.sessions.pop(
                    guild.id,
                    None,
                )

                try:
                    await existing.close()
                except Exception:
                    logger.exception(
                        "Failed to close existing voice session"
                    )

                try:
                    if existing.voice_client.is_connected():
                        await existing.voice_client.disconnect(
                            force=True
                        )
                except Exception:
                    logger.exception(
                        "Failed to disconnect existing voice client"
                    )

            # ------------------------------------------------
            # Existing Discord voice client.
            # ------------------------------------------------

            existing_client = guild.voice_client

            if existing_client:

                if not isinstance(
                    existing_client,
                    voice_recv.VoiceRecvClient,
                ):
                    try:
                        await existing_client.disconnect(
                            force=True
                        )
                    except Exception:
                        logger.exception(
                            "Failed to disconnect "
                            "old voice client"
                        )

            # ------------------------------------------------
            # Connect using VoiceRecvClient.
            # ------------------------------------------------

            logger.info(
                "Connecting VoiceRecvClient | guild=%s | channel=%s",
                guild.id,
                channel.name,
            )

            voice_client = await channel.connect(
                cls=voice_recv.VoiceRecvClient,
                self_deaf=False,
                self_mute=False,
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
            # Create session.
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
                "Voice session created | guild=%s | "
                "channel=%s | voice=%s | speed=%.2f",
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
                if session.voice_client.is_connected():
                    await session.voice_client.disconnect(
                        force=True
                    )

            except Exception:
                logger.exception(
                    "Failed to disconnect voice client"
                )

            return True

    # ========================================================
    # LEAVE ALL
    # ========================================================

    async def close_all(self) -> None:

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
                    "Failed to close guild voice session | guild=%s",
                    guild_id,
                )

    # ========================================================
    # STATS
    # ========================================================

    def stats(self) -> dict[str, int]:

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
    """Return available Gemini voices."""

    return list(
        GEMINI_VOICES
    )


def format_voice_list() -> str:

    voices = get_voice_names()

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

    return "\n".join(lines)


async def send_voice_list(
    interaction: discord.Interaction,
) -> None:

    await interaction.response.send_message(
        format_voice_list(),
        ephemeral=True,
            )
