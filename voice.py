from __future__ import annotations

import asyncio
import io
import logging
import struct
import time
import wave
from dataclasses import dataclass
from typing import Optional

import discord
from discord.ext import commands, voice_recv

from config import (
    GEMINI_INPUT_CHANNELS,
    GEMINI_INPUT_SAMPLE_RATE,
    GEMINI_INPUT_SAMPLE_WIDTH,
    GEMINI_TTS_CHANNELS,
    GEMINI_TTS_SAMPLE_RATE,
    GEMINI_TTS_SAMPLE_WIDTH,
    DISCORD_PLAYBACK_CHANNELS,
    DISCORD_PLAYBACK_SAMPLE_RATE,
    DISCORD_PLAYBACK_SAMPLE_WIDTH,
    VOICE_MAX_BUFFER_SECONDS,
    VOICE_SILENCE_SECONDS,
    VOICE_MIN_AUDIO_SECONDS,
)

from gemini import GeminiEngine


logger = logging.getLogger(__name__)


# ============================================================
# AUDIO CONSTANTS
# ============================================================

INPUT_RATE = GEMINI_INPUT_SAMPLE_RATE
INPUT_CHANNELS = GEMINI_INPUT_CHANNELS
INPUT_WIDTH = GEMINI_INPUT_SAMPLE_WIDTH

TTS_RATE = GEMINI_TTS_SAMPLE_RATE
TTS_CHANNELS = GEMINI_TTS_CHANNELS
TTS_WIDTH = GEMINI_TTS_SAMPLE_WIDTH

DISCORD_RATE = DISCORD_PLAYBACK_SAMPLE_RATE
DISCORD_CHANNELS = DISCORD_PLAYBACK_CHANNELS
DISCORD_WIDTH = DISCORD_PLAYBACK_SAMPLE_WIDTH


# ============================================================
# AUDIO HELPERS
# ============================================================

def pcm_48k_stereo_to_16k_mono(pcm: bytes) -> bytes:
    """
    Convert Discord PCM:
        48 kHz / stereo / 16-bit

    Into Gemini input:
        16 kHz / mono / 16-bit
    """

    if not pcm:
        return b""

    frame_size = DISCORD_CHANNELS * DISCORD_WIDTH

    if frame_size <= 0:
        return b""

    usable = len(pcm) - (len(pcm) % frame_size)

    if usable <= 0:
        return b""

    pcm = pcm[:usable]

    output = bytearray()

    # 48k -> 16k = every third frame.
    # Stereo -> mono = average L/R.
    for offset in range(0, len(pcm), frame_size * 3):
        if offset + frame_size > len(pcm):
            break

        left, right = struct.unpack_from("<hh", pcm, offset)

        mono = (left + right) // 2

        output.extend(struct.pack("<h", mono))

    return bytes(output)


def pcm_to_wav(
    pcm: bytes,
    *,
    sample_rate: int,
    channels: int,
    sample_width: int,
) -> bytes:
    """
    Wrap raw PCM in a WAV container.
    """

    if not pcm:
        return b""

    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)

    return buffer.getvalue()


def tts_pcm_to_discord_pcm(pcm: bytes) -> bytes:
    """
    Convert Gemini TTS PCM:

        24 kHz / mono / 16-bit

    Into Discord playback PCM:

        48 kHz / stereo / 16-bit
    """

    if not pcm:
        return b""

    if TTS_WIDTH != 2:
        raise ValueError("Only 16-bit TTS PCM is supported.")

    usable = len(pcm) - (len(pcm) % 2)

    if usable <= 0:
        return b""

    pcm = pcm[:usable]

    output = bytearray()

    for offset in range(0, len(pcm), 2):
        sample = struct.unpack_from("<h", pcm, offset)[0]

        # 24k -> 48k
        # Duplicate each sample once.
        output.extend(struct.pack("<hh", sample, sample))
        output.extend(struct.pack("<hh", sample, sample))

    return bytes(output)


def pcm_is_silent(
    pcm: bytes,
    *,
    threshold: int = 500,
) -> bool:
    """
    Lightweight silence detector for 16-bit PCM.
    """

    if not pcm:
        return True

    usable = len(pcm) - (len(pcm) % 2)

    if usable <= 0:
        return True

    sample_count = usable // 2

    # Don't inspect every single sample for performance.
    step = max(1, sample_count // 2000)

    for index in range(0, sample_count, step):
        offset = index * 2
        sample = struct.unpack_from("<h", pcm, offset)[0]

        if abs(sample) > threshold:
            return False

    return True


# ============================================================
# USER AUDIO STATE
# ============================================================

@dataclass
class UserAudioState:
    user_id: int

    username: str

    pcm_chunks: list[bytes]

    started_at: float

    last_audio_at: float

    processing: bool = False

    def __post_init__(self) -> None:
        if not self.pcm_chunks:
            self.pcm_chunks = []

    @property
    def duration(self) -> float:
        return max(0.0, time.monotonic() - self.started_at)

    @property
    def audio_bytes(self) -> int:
        return sum(len(chunk) for chunk in self.pcm_chunks)

    def append(self, pcm: bytes) -> None:
        if not pcm:
            return

        self.pcm_chunks.append(pcm)
        self.last_audio_at = time.monotonic()

    def build_audio(self) -> bytes:
        if not self.pcm_chunks:
            return b""

        return b"".join(self.pcm_chunks)

    def clear(self) -> None:
        self.pcm_chunks.clear()


# ============================================================
# VOICE RECEIVE SINK
# ============================================================

class VoiceAISink(voice_recv.AudioSink):
    """
    Receives decoded Discord PCM and forwards it to VoiceSession.

    Important:
    wants_opus() is False because the AI pipeline expects PCM.
    """

    def __init__(self, session: "VoiceSession"):
        super().__init__()

        self.session = session
        self._closed = False

    def wants_opus(self) -> bool:
        return False

    def write(
        self,
        user: discord.User | discord.Member | None,
        data: voice_recv.VoiceData,
    ) -> None:

        if self._closed:
            return

        if user is None:
            return

        pcm = getattr(data, "pcm", None)

        if not pcm:
            return

        try:
            self.session.receive_pcm(user, pcm)

        except Exception:
            logger.exception(
                "VoiceAISink failed while receiving audio from %s",
                getattr(user, "id", "unknown"),
            )

    def cleanup(self) -> None:
        self._closed = True


# ============================================================
# VOICE SESSION
# ============================================================

class VoiceSession:
    """
    Controls one active AI voice session.
    """

    def __init__(
        self,
        bot: commands.Bot,
        voice_client: voice_recv.VoiceRecvClient,
        *,
        guild_id: int,
    ):
        self.bot = bot
        self.voice_client = voice_client
        self.guild_id = guild_id

        self.engine = GeminiEngine()

        self.states: dict[int, UserAudioState] = {}

        self.processing_users: set[int] = set()

        self.processing_semaphore = asyncio.Semaphore(2)

        self.sink = VoiceAISink(self)

        self._monitor_task: Optional[asyncio.Task] = None

        self._closed = False

        self._started = False

    # --------------------------------------------------------
    # START
    # --------------------------------------------------------

    def start(self) -> None:
        if self._started:
            return

        self._started = True

        self.voice_client.listen(
            self.sink,
            after=self._receive_finished,
        )

        self._monitor_task = asyncio.create_task(
            self._monitor_loop(),
            name=f"voice-monitor-{self.guild_id}",
        )

        logger.info("Voice receive sink started")
        logger.info("Voice AI listening")
        logger.info("Voice session created")

    # --------------------------------------------------------
    # RECEIVE CALLBACK
    # --------------------------------------------------------

    def receive_pcm(
        self,
        user: discord.User | discord.Member,
        pcm: bytes,
    ) -> None:

        if self._closed:
            return

        user_id = user.id

        if user_id in self.processing_users:
            return

        now = time.monotonic()

        state = self.states.get(user_id)

        if state is None:
            state = UserAudioState(
                user_id=user_id,
                username=getattr(user, "display_name", None)
                or getattr(user, "name", "User"),
                pcm_chunks=[],
                started_at=now,
                last_audio_at=now,
            )

            self.states[user_id] = state

        # Convert Discord PCM to Gemini PCM.
        gemini_pcm = pcm_48k_stereo_to_16k_mono(pcm)

        if not gemini_pcm:
            return

        # Ignore chunks that are completely silent.
        if pcm_is_silent(gemini_pcm):
            return

        state.append(gemini_pcm)

    # --------------------------------------------------------
    # MONITOR
    # --------------------------------------------------------

    async def _monitor_loop(self) -> None:
        try:
            while not self._closed:

                await asyncio.sleep(0.10)

                now = time.monotonic()

                candidates: list[tuple[UserAudioState, bytes]] = []

                for user_id, state in list(self.states.items()):

                    if state.processing:
                        continue

                    if not state.pcm_chunks:
                        continue

                    elapsed_since_audio = now - state.last_audio_at

                    should_flush = (
                        elapsed_since_audio >= VOICE_SILENCE_SECONDS
                        or state.duration >= VOICE_MAX_BUFFER_SECONDS
                    )

                    if not should_flush:
                        continue

                    audio = state.build_audio()

                    state.clear()

                    if not audio:
                        continue

                    duration = len(audio) / (
                        INPUT_RATE * INPUT_CHANNELS * INPUT_WIDTH
                    )

                    if duration < VOICE_MIN_AUDIO_SECONDS:
                        continue

                    state.processing = True

                    self.processing_users.add(user_id)

                    candidates.append((state, audio))

                for state, audio in candidates:
                    asyncio.create_task(
                        self._process_user_audio(
                            state,
                            audio,
                        ),
                        name=f"voice-ai-{state.user_id}",
                    )

        except asyncio.CancelledError:
            pass

        except Exception:
            logger.exception("Voice monitor crashed")

    # --------------------------------------------------------
    # PROCESS USER
    # --------------------------------------------------------

    async def _process_user_audio(
        self,
        state: UserAudioState,
        audio: bytes,
    ) -> None:

        try:
            async with self.processing_semaphore:

                logger.info(
                    "Processing voice from %s (%d bytes)",
                    state.username,
                    len(audio),
                )

                wav_audio = pcm_to_wav(
                    audio,
                    sample_rate=INPUT_RATE,
                    channels=INPUT_CHANNELS,
                    sample_width=INPUT_WIDTH,
                )

                if not wav_audio:
                    return

                response = await self.engine.process_voice(
                    audio=wav_audio,
                    username=state.username,
                    mime_type="audio/wav",
                )

                if not response:
                    return

                tts_pcm = response

                discord_pcm = tts_pcm_to_discord_pcm(tts_pcm)

                if not discord_pcm:
                    return

                await self._play_pcm(discord_pcm)

        except asyncio.CancelledError:
            raise

        except Exception:
            logger.exception(
                "Voice AI processing failed for %s",
                state.username,
            )

        finally:
            state.processing = False
            self.processing_users.discard(state.user_id)

    # --------------------------------------------------------
    # PLAY TTS
    # --------------------------------------------------------

    async def _play_pcm(self, pcm: bytes) -> None:

        if self._closed:
            return

        if not pcm:
            return

        if not self.voice_client.is_connected():
            return

        # Wait for an existing response to finish.
        while self.voice_client.is_playing():

            if self._closed:
                return

            await asyncio.sleep(0.05)

        source = discord.PCMAudio(
            io.BytesIO(pcm),
        )

        finished = asyncio.Event()

        def after(error: Optional[Exception]) -> None:

            if error:
                logger.error(
                    "Voice playback error: %s",
                    error,
                )

            try:
                self.bot.loop.call_soon_threadsafe(
                    finished.set,
                )
            except RuntimeError:
                pass

        try:
            self.voice_client.play(
                source,
                after=after,
            )

        except Exception:
            logger.exception("Failed to start voice playback")
            return

        await finished.wait()

    # --------------------------------------------------------
    # RECEIVE FINISHED
    # --------------------------------------------------------

    def _receive_finished(
        self,
        error: Optional[Exception],
    ) -> None:

        if error:
            logger.error(
                "Voice receive stopped with error: %r",
                error,
            )

    # --------------------------------------------------------
    # STOP
    # --------------------------------------------------------

    async def stop(self) -> None:

        if self._closed:
            return

        self._closed = True

        if self._monitor_task:
            self._monitor_task.cancel()

            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("Voice monitor shutdown error")

            self._monitor_task = None

        try:
            self.sink.cleanup()
        except Exception:
            logger.exception("Voice sink cleanup failed")

        self.states.clear()
        self.processing_users.clear()

        logger.info("Voice session stopped")


# ============================================================
# VOICE SESSION MANAGER
# ============================================================

class VoiceSessionManager:
    """
    Manages one VoiceSession per guild.
    """

    def __init__(self, bot: commands.Bot):
        self.bot = bot

        self.sessions: dict[int, VoiceSession] = {}

    # --------------------------------------------------------
    # JOIN
    # --------------------------------------------------------

    async def join(
        self,
        channel: discord.VoiceChannel | discord.StageChannel,
    ) -> VoiceSession:

        guild_id = channel.guild.id

        existing = self.sessions.get(guild_id)

        if existing:
            if existing.voice_client.channel == channel:
                return existing

            await self.leave(guild_id)

        logger.info(
            "Connecting to voice channel %s",
            channel.name,
        )

        voice_client = await channel.connect(
            cls=voice_recv.VoiceRecvClient,
            reconnect=True,
        )

        session = VoiceSession(
            self.bot,
            voice_client,
            guild_id=guild_id,
        )

        self.sessions[guild_id] = session

        session.start()

        return session

    # --------------------------------------------------------
    # LEAVE
    # --------------------------------------------------------

    async def leave(
        self,
        guild_id: int,
    ) -> None:

        session = self.sessions.pop(
            guild_id,
            None,
        )

        if session:
            await session.stop()

            vc = session.voice_client

            if vc.is_connected():
                try:
                    await vc.disconnect(force=True)
                except Exception:
                    logger.exception(
                        "Failed to disconnect voice client",
                    )

        logger.info(
            "Voice session removed for guild %s",
            guild_id,
        )

    # --------------------------------------------------------
    # GET
    # --------------------------------------------------------

    def get(
        self,
        guild_id: int,
    ) -> Optional[VoiceSession]:

        return self.sessions.get(guild_id)

    # --------------------------------------------------------
    # LEAVE ALL
    # --------------------------------------------------------

    async def close(self) -> None:

        for guild_id in list(self.sessions):
            await self.leave(guild_id)


# ============================================================
# VOICE COMMAND HELPERS
# ============================================================

async def send_voice_channel_message(
    interaction: discord.Interaction,
    content: str,
) -> None:

    if interaction.response.is_done():
        await interaction.followup.send(
            content,
            ephemeral=True,
        )
    else:
        await interaction.response.send_message(
            content,
            ephemeral=True,
        )            channel.name,
        )

        voice_client = await channel.connect(
            cls=voice_recv.VoiceRecvClient,
            reconnect=True,
        )

        session = VoiceSession(
            self.bot,
            voice_client,
            guild_id=guild_id,
        )

        self.sessions[guild_id] = session

        session.start()

        return session

    # --------------------------------------------------------
    # LEAVE
    # --------------------------------------------------------

    async def leave(
        self,
        guild_id: int,
    ) -> None:

        session = self.sessions.pop(
            guild_id,
            None,
        )

        if session:
            await session.stop()

            vc = session.voice_client

            if vc.is_connected():
                try:
                    await vc.disconnect(force=True)
                except Exception:
                    logger.exception(
                        "Failed to disconnect voice client",
                    )

        logger.info(
            "Voice session removed for guild %s",
            guild_id,
        )

    # --------------------------------------------------------
    # GET
    # --------------------------------------------------------

    def get(
        self,
        guild_id: int,
    ) -> Optional[VoiceSession]:

        return self.sessions.get(guild_id)

    # --------------------------------------------------------
    # LEAVE ALL
    # --------------------------------------------------------

    async def close(self) -> None:

        for guild_id in list(self.sessions):
            await self.leave(guild_id)


# ============================================================
# VOICE COMMAND HELPERS
# ============================================================

async def send_voice_channel_message(
    interaction: discord.Interaction,
    content: str,
) -> None:

    if interaction.response.is_done():
        await interaction.followup.send(
            content,
            ephemeral=True,
        )
    else:
        await interaction.response.send_message(
            content,
            ephemeral=True,
    )
