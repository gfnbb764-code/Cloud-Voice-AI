# voice.py
# ============================================================
# Cloud Voice AI — Discord Voice Engine
# ============================================================
# مسؤول عن:
# - دخول الروم الصوتي
# - استقبال صوت المستخدمين
# - تجميع PCM
# - تحويل Discord Audio إلى 16kHz Mono WAV
# - إرسال الصوت إلى Gemini
# - استقبال الرد الصوتي
# - تحويل Gemini TTS إلى Discord PCM
# - تشغيل الرد داخل الروم
# - إدارة الذاكرة والإحصائيات
# ============================================================

from __future__ import annotations

import asyncio
import io
import logging
import time
import wave
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

import discord
from discord.ext import commands, voice_recv

from config import (
    MAX_MEMORY_MESSAGES,
    DEFAULT_GEMINI_VOICE,
    GEMINI_VOICES,
)

from gemini import GeminiEngine


logger = logging.getLogger("cloud_voice_ai.voice")


# ============================================================
# AUDIO FORMAT
# ============================================================

# Discord voice receive/playback format
PCM_SAMPLE_RATE = 48000
PCM_CHANNELS = 2
PCM_SAMPLE_WIDTH = 2

# Gemini STT input
GEMINI_SAMPLE_RATE = 16000
GEMINI_CHANNELS = 1
GEMINI_SAMPLE_WIDTH = 2

# Gemini TTS output
TTS_SAMPLE_RATE = 24000
TTS_CHANNELS = 1
TTS_SAMPLE_WIDTH = 2


# ============================================================
# AUDIO LIMITS
# ============================================================

# لا نعتمد على متغيرات config القديمة.
SILENCE_TIMEOUT = 1.20
MIN_AUDIO_SECONDS = 0.35
MAX_AUDIO_SECONDS = 15.0

MAX_AUDIO_BUFFER_BYTES = (
    PCM_SAMPLE_RATE
    * PCM_CHANNELS
    * PCM_SAMPLE_WIDTH
    * int(MAX_AUDIO_SECONDS)
)

MAX_USERS_TRACKED = 50


# ============================================================
# HELPERS
# ============================================================

def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def pcm_duration(
    pcm: bytes,
    sample_rate: int = PCM_SAMPLE_RATE,
    channels: int = PCM_CHANNELS,
    sample_width: int = PCM_SAMPLE_WIDTH,
) -> float:
    if not pcm:
        return 0.0

    bytes_per_second = sample_rate * channels * sample_width

    if bytes_per_second <= 0:
        return 0.0

    return len(pcm) / bytes_per_second


def pcm_to_wav(pcm_data: bytes) -> bytes:
    """
    Discord:
        48000 Hz
        stereo
        signed 16-bit PCM

    Gemini STT:
        16000 Hz
        mono
        signed 16-bit PCM

    نحول الصوت يدويًا بدون audioop/audioop-lts
    حتى يشتغل على Python 3.11 في FadeHost.
    """

    if not pcm_data:
        raise ValueError("PCM data is empty.")

    # --------------------------------------------------------
    # 48k stereo -> 16k mono
    # --------------------------------------------------------

    source_frame_size = PCM_CHANNELS * PCM_SAMPLE_WIDTH

    usable_length = (
        len(pcm_data) // source_frame_size
    ) * source_frame_size

    pcm_data = pcm_data[:usable_length]

    if not pcm_data:
        raise ValueError("PCM data has no complete frames.")

    output = bytearray()

    # كل 3 frames من 48k = frame واحد تقريبًا من 16k
    for index in range(0, len(pcm_data), source_frame_size * 3):
        frame_block = pcm_data[
            index:index + source_frame_size * 3
        ]

        if len(frame_block) < source_frame_size:
            break

        # أخذ frame واحد من كل 3 frames
        frame = frame_block[:source_frame_size]

        left = int.from_bytes(
            frame[0:2],
            byteorder="little",
            signed=True,
        )

        right = int.from_bytes(
            frame[2:4],
            byteorder="little",
            signed=True,
        )

        mono = (left + right) // 2

        mono = int(
            clamp(
                mono,
                -32768,
                32767,
            )
        )

        output.extend(
            int(mono).to_bytes(
                2,
                byteorder="little",
                signed=True,
            )
        )

    if not output:
        raise ValueError("Audio conversion produced no data.")

    # --------------------------------------------------------
    # WAV wrapper
    # --------------------------------------------------------

    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(GEMINI_CHANNELS)
        wav.setsampwidth(GEMINI_SAMPLE_WIDTH)
        wav.setframerate(GEMINI_SAMPLE_RATE)
        wav.writeframes(bytes(output))

    return buffer.getvalue()


def tts_pcm_to_discord_pcm(tts_pcm: bytes) -> bytes:
    """
    Gemini TTS:
        24000 Hz
        mono
        16-bit PCM

    Discord playback:
        48000 Hz
        stereo
        16-bit PCM

    بدون audioop.
    """

    if not tts_pcm:
        raise ValueError("TTS PCM data is empty.")

    usable_length = (
        len(tts_pcm) // TTS_SAMPLE_WIDTH
    ) * TTS_SAMPLE_WIDTH

    tts_pcm = tts_pcm[:usable_length]

    if not tts_pcm:
        raise ValueError("Invalid TTS PCM data.")

    output = bytearray()

    # --------------------------------------------------------
    # 24k -> 48k
    # --------------------------------------------------------
    # نكرر كل sample مرتين.
    #
    # Mono -> Stereo:
    # نفس sample في Left + Right
    # --------------------------------------------------------

    for index in range(0, len(tts_pcm), 2):
        sample = tts_pcm[index:index + 2]

        if len(sample) < 2:
            break

        output.extend(sample)
        output.extend(sample)

        output.extend(sample)
        output.extend(sample)

    return bytes(output)


def normalize_voice_name(name: str) -> str:
    """
    يحاول إيجاد اسم الصوت الصحيح من القائمة.
    """

    if not name:
        return DEFAULT_GEMINI_VOICE

    value = str(name).strip().lower()

    for voice in GEMINI_VOICES:
        if voice.lower() == value:
            return voice

    # aliases شائعة
    aliases = {
        "default": DEFAULT_GEMINI_VOICE,
        "افتراضي": DEFAULT_GEMINI_VOICE,
        "kore": "Kore",
        "k": "Kore",
    }

    return aliases.get(
        value,
        DEFAULT_GEMINI_VOICE,
    )


# ============================================================
# USER AUDIO STATE
# ============================================================

@dataclass
class UserAudioState:
    user_id: int

    username: str = "Unknown"

    chunks: list[bytes] = field(
        default_factory=list
    )

    total_bytes: int = 0

    started_at: float = 0.0

    last_audio_at: float = 0.0

    processing: bool = False

    lock: asyncio.Lock = field(
        default_factory=asyncio.Lock
    )

    def start(self, username: str) -> None:
        self.username = username
        self.started_at = time.monotonic()
        self.last_audio_at = self.started_at
        self.total_bytes = 0
        self.chunks.clear()

    def append(self, pcm: bytes) -> None:
        if not pcm:
            return

        self.chunks.append(pcm)
        self.total_bytes += len(pcm)
        self.last_audio_at = time.monotonic()

    def build_audio(self) -> bytes:
        return b"".join(self.chunks)

    def clear(self) -> None:
        self.chunks.clear()
        self.total_bytes = 0
        self.started_at = 0.0
        self.last_audio_at = 0.0

    @property
    def duration(self) -> float:
        return pcm_duration(
            self.total_bytes.to_bytes(
                8,
                "little",
            )
            if False
            else b"",
        )


# ============================================================
# DISCORD AUDIO SINK
# ============================================================

class VoiceAISink(voice_recv.AudioSink):
    """
    يستقبل صوت المستخدمين من Discord.
    """

    def __init__(
        self,
        session: "VoiceSession",
    ):
        super().__init__()

        self.session = session

        self.users: dict[
            int,
            UserAudioState,
        ] = {}

        self.closed = False

        self._monitor_task: Optional[
            asyncio.Task
        ] = None

    # --------------------------------------------------------
    # Discord voice receive
    # --------------------------------------------------------

    def wants_opus(self) -> bool:
        return False

    def write(
        self,
        user: discord.User | discord.Member,
        data,
    ) -> None:

        if self.closed:
            return

        if user is None:
            return

        user_id = user.id

        # ----------------------------------------------
        # PCM packet
        # ----------------------------------------------

        pcm = getattr(data, "pcm", None)

        if not pcm:
            return

        if len(pcm) <= 0:
            return

        # ----------------------------------------------
        # Get/create user state
        # ----------------------------------------------

        state = self.users.get(user_id)

        if state is None:
            if len(self.users) >= MAX_USERS_TRACKED:
                return

            state = UserAudioState(
                user_id=user_id,
            )

            self.users[user_id] = state

        # ----------------------------------------------
        # Start new speech segment
        # ----------------------------------------------

        if not state.chunks:
            state.start(
                getattr(
                    user,
                    "display_name",
                    getattr(
                        user,
                        "name",
                        "Unknown",
                    ),
                )
            )

        # ----------------------------------------------
        # Prevent giant buffers
        # ----------------------------------------------

        if (
            state.total_bytes + len(pcm)
            > MAX_AUDIO_BUFFER_BYTES
        ):
            asyncio.create_task(
                self._flush_user(
                    user_id,
                    reason="max_buffer",
                )
            )

            return

        # ----------------------------------------------
        # Store PCM
        # ----------------------------------------------

        state.append(pcm)

        # ----------------------------------------------
        # Maximum duration
        # ----------------------------------------------

        duration = (
            state.total_bytes
            / (
                PCM_SAMPLE_RATE
                * PCM_CHANNELS
                * PCM_SAMPLE_WIDTH
            )
        )

        if duration >= MAX_AUDIO_SECONDS:
            asyncio.create_task(
                self._flush_user(
                    user_id,
                    reason="max_duration",
                )
            )

    # --------------------------------------------------------
    # Monitor
    # --------------------------------------------------------

    async def start_monitor(self) -> None:
        if self._monitor_task is not None:
            return

        self._monitor_task = asyncio.create_task(
            self._monitor_loop()
        )

    async def _monitor_loop(self) -> None:
        try:
            while not self.closed:
                await asyncio.sleep(0.15)

                now = time.monotonic()

                for user_id, state in list(
                    self.users.items()
                ):
                    if not state.chunks:
                        continue

                    if state.processing:
                        continue

                    silence = (
                        now - state.last_audio_at
                    )

                    if silence >= SILENCE_TIMEOUT:
                        await self._flush_user(
                            user_id,
                            reason="silence",
                        )

        except asyncio.CancelledError:
            pass

        except Exception:
            logger.exception(
                "Voice monitor crashed."
            )

    # --------------------------------------------------------
    # Flush user
    # --------------------------------------------------------

    async def _flush_user(
        self,
        user_id: int,
        reason: str = "unknown",
    ) -> None:

        state = self.users.get(user_id)

        if state is None:
            return

        async with state.lock:

            if state.processing:
                return

            if not state.chunks:
                return

            state.processing = True

            try:
                audio = state.build_audio()

                duration = (
                    len(audio)
                    / (
                        PCM_SAMPLE_RATE
                        * PCM_CHANNELS
                        * PCM_SAMPLE_WIDTH
                    )
                )

                logger.debug(
                    "Captured %.2fs from user %s (%s).",
                    duration,
                    state.username,
                    reason,
                )

                # تجاهل المقاطع القصيرة جدًا
                if duration < MIN_AUDIO_SECONDS:
                    state.clear()
                    return

                state.clear()

                await self.session.process_user_audio(
                    user_id=user_id,
                    username=state.username,
                    pcm=audio,
                )

            except Exception:
                logger.exception(
                    "Failed processing audio from user %s.",
                    user_id,
                )

            finally:
                state.processing = False

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    def cleanup(self) -> None:
        self.closed = True

        if self._monitor_task:
            self._monitor_task.cancel()

            self._monitor_task = None

        for state in self.users.values():
            state.clear()

        self.users.clear()


# ============================================================
# VOICE SESSION
# ============================================================

class VoiceSession:
    """
    جلسة Voice AI كاملة لسيرفر واحد.
    """

    def __init__(
        self,
        bot: commands.Bot,
        guild: discord.Guild,
    ):
        self.bot = bot
        self.guild = guild

        self.voice_client: Optional[
            voice_recv.VoiceRecvClient
        ] = None

        self.sink: Optional[
            VoiceAISink
        ] = None

        self.ai = GeminiEngine()

        self.voice_name = normalize_voice_name(
            DEFAULT_GEMINI_VOICE
        )

        self.memory: dict[
            int,
            deque,
        ] = defaultdict(
            lambda: deque(
                maxlen=MAX_MEMORY_MESSAGES
            )
        )

        self.processing_users: set[int] = set()

        self.started_at = time.monotonic()

        self.messages_processed = 0

        self.audio_processed = 0

        self.errors = 0

        self.total_response_time = 0.0

        self._play_lock = asyncio.Lock()

    # ========================================================
    # JOIN
    # ========================================================

    async def start(
        self,
        channel: discord.VoiceChannel,
    ) -> None:

        if self.voice_client is not None:
            if self.voice_client.is_connected():
                return

        logger.info(
            "Connecting to voice channel: %s",
            channel.name,
        )

        try:
            vc = await channel.connect(
                cls=voice_recv.VoiceRecvClient,
                reconnect=True,
            )

            self.voice_client = vc

            self.sink = VoiceAISink(self)

            vc.listen(self.sink)

            await self.sink.start_monitor()

            logger.info(
                "Connected to %s.",
                channel.name,
            )

        except Exception:
            self.voice_client = None
            self.sink = None

            logger.exception(
                "Failed to connect to voice channel."
            )

            raise

    # ========================================================
    # LEAVE
    # ========================================================

    async def stop(self) -> None:

        if self.sink is not None:
            try:
                self.sink.cleanup()
            except Exception:
                logger.exception(
                    "Sink cleanup failed."
                )

            self.sink = None

        if self.voice_client is not None:

            try:
                if self.voice_client.is_listening():
                    self.voice_client.stop_listening()
            except Exception:
                pass

            try:
                if self.voice_client.is_connected():
                    await self.voice_client.disconnect(
                        force=True
                    )
            except Exception:
                logger.exception(
                    "Voice disconnect failed."
                )

            self.voice_client = None

        logger.info(
            "Voice session stopped for guild %s.",
            self.guild.id,
        )

    # ========================================================
    # PROCESS USER AUDIO
    # ========================================================

    async def process_user_audio(
        self,
        user_id: int,
        username: str,
        pcm: bytes,
    ) -> None:

        if not pcm:
            return

        if user_id in self.processing_users:
            return

        self.processing_users.add(user_id)

        started = time.monotonic()

        try:

            logger.info(
                "Processing speech from %s.",
                username,
            )

            # ------------------------------------------------
            # Convert Discord PCM -> Gemini WAV
            # ------------------------------------------------

            wav_audio = await asyncio.to_thread(
                pcm_to_wav,
                pcm,
            )

            # ------------------------------------------------
            # Memory
            # ------------------------------------------------

            memory = list(
                self.memory[user_id]
            )

            # ------------------------------------------------
            # Gemini
            # ------------------------------------------------

            result = await self.ai.process_voice(
                audio=wav_audio,
                memory=memory,
                username=username,
                voice=self.voice_name,
            )

            if not result:
                return

            transcript = (
                result.get("transcript")
                or result.get("text")
                or ""
            )

            response_text = (
                result.get("response")
                or result.get("reply")
                or ""
            )

            audio_response = (
                result.get("audio")
                or result.get("tts")
                or result.get("audio_data")
            )

            # ------------------------------------------------
            # Update memory
            # ------------------------------------------------

            if transcript:
                self.memory[user_id].append(
                    {
                        "role": "user",
                        "content": transcript,
                    }
                )

            if response_text:
                self.memory[user_id].append(
                    {
                        "role": "assistant",
                        "content": response_text,
                    }
                )

            self.messages_processed += 1
            self.audio_processed += 1

            # ------------------------------------------------
            # Send text to Discord
            # ------------------------------------------------

            await self.send_text_response(
                username=username,
                transcript=transcript,
                response=response_text,
            )

            # ------------------------------------------------
            # Play TTS
            # ------------------------------------------------

            if audio_response:
                await self.play_audio(
                    audio_response
                )

            elapsed = (
                time.monotonic()
                - started
            )

            self.total_response_time += elapsed

            logger.info(
                "Processed %s in %.2fs.",
                username,
                elapsed,
            )

        except Exception:
            self.errors += 1

            logger.exception(
                "Voice AI processing failed for %s.",
                username,
            )

        finally:
            self.processing_users.discard(
                user_id
            )

    # ========================================================
    # TEXT RESPONSE
    # ========================================================

    async def send_text_response(
        self,
        username: str,
        transcript: str,
        response: str,
    ) -> None:

        if not response:
            return

        # نبحث عن channel نصي مناسب.
        channel = None

        for candidate in self.guild.text_channels:

            if candidate.permissions_for(
                self.guild.me
            ).send_messages:

                channel = candidate
                break

        if channel is None:
            return

        # لا نرسل التسجيل كاملًا إذا كان طويلًا.
        transcript_display = transcript[:500]

        content = (
            f"🎙️ **{discord.utils.escape_markdown(username)}**\n"
            f"> {transcript_display}\n\n"
            f"🤖 {response}"
        )

        try:
            await channel.send(
                content[:2000]
            )
        except Exception:
            logger.exception(
                "Failed sending text response."
            )

    # ========================================================
    # PLAY AUDIO
    # ========================================================

    async def play_audio(
        self,
        audio_data: bytes,
    ) -> None:

        if not audio_data:
            return

        if self.voice_client is None:
            return

        if not self.voice_client.is_connected():
            return

        async with self._play_lock:

            try:

                # ------------------------------------------------
                # Gemini TTS PCM -> Discord PCM
                # ------------------------------------------------

                discord_pcm = await asyncio.to_thread(
                    tts_pcm_to_discord_pcm,
                    audio_data,
                )

                if not discord_pcm:
                    return

                # ------------------------------------------------
                # Stop previous playback
                # ------------------------------------------------

                if self.voice_client.is_playing():
                    self.voice_client.stop()

                source = discord.PCMAudio(
                    io.BytesIO(
                        discord_pcm
                    ),
                    sample_width=2,
                )

                finished = asyncio.Event()

                def after_play(error):
                    if error:
                        logger.error(
                            "Discord playback error: %s",
                            error,
                        )

                    self.bot.loop.call_soon_threadsafe(
                        finished.set
                    )

                self.voice_client.play(
                    source,
                    after=after_play,
                )

                try:
                    await asyncio.wait_for(
                        finished.wait(),
                        timeout=60,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Audio playback timed out."
                    )

                    if self.voice_client.is_playing():
                        self.voice_client.stop()

            except Exception:
                self.errors += 1

                logger.exception(
                    "Failed playing TTS audio."
                )

    # ========================================================
    # VOICE SETTINGS
    # ========================================================

    def set_voice(
        self,
        voice_name: str,
    ) -> str:

        normalized = normalize_voice_name(
            voice_name
        )

        self.voice_name = normalized

        return normalized

    def get_voice(self) -> str:
        return self.voice_name

    def list_voices(self) -> list[str]:
        return list(GEMINI_VOICES)

    # ========================================================
    # MEMORY
    # ========================================================

    def clear_memory(
        self,
        user_id: Optional[int] = None,
    ) -> None:

        if user_id is None:
            self.memory.clear()

            try:
                self.ai.clear_memory()
            except Exception:
                pass

            return

        self.memory.pop(
            user_id,
            None,
        )

    def get_memory(
        self,
        user_id: int,
    ) -> list:

        return list(
            self.memory.get(
                user_id,
                [],
            )
        )

    # ========================================================
    # STATS
    # ========================================================

    @property
    def uptime(self) -> float:
        return (
            time.monotonic()
            - self.started_at
        )

    @property
    def average_response_time(
        self,
    ) -> float:

        if self.messages_processed <= 0:
            return 0.0

        return (
            self.total_response_time
            / self.messages_processed
        )

    def get_stats(self) -> dict:

        return {
            "guild_id": self.guild.id,
            "voice_connected": (
                self.voice_client is not None
                and self.voice_client.is_connected()
            ),
            "voice_channel": (
                self.voice_client.channel.name
                if (
                    self.voice_client
                    and self.voice_client.channel
                )
                else None
            ),
            "voice": self.voice_name,
            "messages_processed": (
                self.messages_processed
            ),
            "audio_processed": (
                self.audio_processed
            ),
            "errors": self.errors,
            "average_response_time": (
                round(
                    self.average_response_time,
                    2,
                )
            ),
            "uptime": round(
                self.uptime,
                1,
            ),
            "processing_users": len(
                self.processing_users
            ),
        }

    # ========================================================
    # CLOSE
    # ========================================================

    async def close(self) -> None:

        await self.stop()

        try:
            await self.ai.close()
        except Exception:
            logger.exception(
                "Gemini engine close failed."
            )

        self.memory.clear()
        self.processing_users.clear()


# ============================================================
# VOICE COMMAND HELPERS
# ============================================================

async def send_voice_list(
    interaction: discord.Interaction,
    voices: list[str],
) -> None:
    """
    يرسل قائمة الأصوات في Discord.
    """

    if not voices:
        await interaction.response.send_message(
            "❌ ما فيه أصوات متاحة حاليًا.",
            ephemeral=True,
        )
        return

    lines = [
        "🎙️ **Gemini Voices**",
        "",
    ]

    for index, voice in enumerate(
        voices,
        start=1,
    ):
        lines.append(
            f"`{index:02d}` • **{voice}**"
        )

    lines.extend(
        [
            "",
            "💡 استخدم `/setvoice` لاختيار الصوت.",
        ]
    )

    text = "\n".join(lines)

    # Discord limit
    if len(text) > 1900:
        text = text[:1890] + "\n…"

    await interaction.response.send_message(
        text,
        ephemeral=True,
    )


# ============================================================
# SESSION FACTORY
# ============================================================

class VoiceSessionManager:
    """
    يدير جلسات Voice AI لكل Guild.
    """

    def __init__(
        self,
        bot: commands.Bot,
    ):
        self.bot = bot

        self.sessions: dict[
            int,
            VoiceSession,
        ] = {}

    def get(
        self,
        guild_id: int,
    ) -> Optional[VoiceSession]:

        return self.sessions.get(
            guild_id
        )

    async def create(
        self,
        guild: discord.Guild,
        channel: discord.VoiceChannel,
    ) -> VoiceSession:

        existing = self.sessions.get(
            guild.id
        )

        if existing is not None:
            if (
                existing.voice_client
                and existing.voice_client.is_connected()
            ):
                return existing

            await existing.close()

            self.sessions.pop(
                guild.id,
                None,
            )

        session = VoiceSession(
            bot=self.bot,
            guild=guild,
        )

        await session.start(
            channel
        )

        self.sessions[guild.id] = session

        return session

    async def remove(
        self,
        guild_id: int,
    ) -> None:

        session = self.sessions.pop(
            guild_id,
            None,
        )

        if session is None:
            return

        await session.close()

    async def close_all(self) -> None:

        sessions = list(
            self.sessions.values()
        )

        self.sessions.clear()

        for session in sessions:
            try:
                await session.close()
            except Exception:
                logger.exception(
                    "Failed closing voice session."
                )


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "VoiceSession",
    "VoiceAISink",
    "VoiceSessionManager",
    "send_voice_list",
    "pcm_to_wav",
    "tts_pcm_to_discord_pcm",
    "normalize_voice_name",
]
