# ============================================================
# AI VOICE BOT
# voice.py
# ============================================================
#
# نظام الصوت الكامل للبوت
#
# المسؤوليات:
# - استقبال صوت Discord
# - إنشاء Audio Sink
# - تجميع الصوت لكل مستخدم
# - اكتشاف نهاية الكلام
# - حماية الذاكرة
# - تحويل صوت Discord إلى صيغة Gemini
# - إرسال الصوت إلى Gemini
# - تشغيل رد Gemini الصوتي
# - إدارة Voice Session
# - إدارة الذاكرة
# - تغيير صوت البوت
#
# ============================================================

from __future__ import annotations

import asyncio
import io
import logging
import time
import wave

from dataclasses import dataclass
from typing import Optional

import discord
from discord.ext import commands, voice_recv

from config import (
    VOICE_SILENCE_TIMEOUT,
    MIN_AUDIO_SECONDS,
    MAX_AUDIO_SECONDS,
    MAX_AUDIO_BUFFER_BYTES,
    MAX_MEMORY_MESSAGES,
    DEFAULT_GEMINI_VOICE,
    GEMINI_VOICES,
)

from gemini import GeminiEngine


# ============================================================
# LOGGER
# ============================================================

logger = logging.getLogger(
    "ai_voice_bot.voice"
)


# ============================================================
# DISCORD AUDIO FORMAT
# ============================================================

PCM_SAMPLE_RATE = 48000
PCM_CHANNELS = 2
PCM_SAMPLE_WIDTH = 2


# ============================================================
# GEMINI AUDIO FORMAT
# ============================================================

GEMINI_SAMPLE_RATE = 16000
GEMINI_CHANNELS = 1
GEMINI_SAMPLE_WIDTH = 2


# ============================================================
# GEMINI TTS AUDIO FORMAT
# ============================================================

TTS_SAMPLE_RATE = 24000
TTS_CHANNELS = 1
TTS_SAMPLE_WIDTH = 2


# ============================================================
# USER AUDIO BUFFER
# ============================================================

@dataclass
class UserAudioBuffer:
    """
    مخزن صوت مؤقت لمستخدم واحد.

    لا يتم حفظ الصوت بشكل دائم.
    """

    user_id: int

    username: str

    data: bytearray

    started_at: float

    last_packet_at: float

    packet_count: int = 0

    # --------------------------------------------------------
    # ADD
    # --------------------------------------------------------

    def add(
        self,
        pcm: bytes,
    ) -> None:

        if not pcm:
            return

        self.data.extend(
            pcm
        )

        self.last_packet_at = (
            time.monotonic()
        )

        self.packet_count += 1

    # --------------------------------------------------------
    # CLEAR
    # --------------------------------------------------------

    def clear(
        self,
    ) -> None:

        self.data.clear()

        now = time.monotonic()

        self.started_at = now

        self.last_packet_at = now

        self.packet_count = 0

    # --------------------------------------------------------
    # SIZE
    # --------------------------------------------------------

    def size(
        self,
    ) -> int:

        return len(
            self.data
        )

    # --------------------------------------------------------
    # DURATION
    # --------------------------------------------------------

    def duration(
        self,
    ) -> float:

        if not self.data:
            return 0.0

        bytes_per_second = (
            PCM_SAMPLE_RATE
            * PCM_CHANNELS
            * PCM_SAMPLE_WIDTH
        )

        return (
            len(self.data)
            / bytes_per_second
        )

    # --------------------------------------------------------
    # SILENCE
    # --------------------------------------------------------

    def silence_duration(
        self,
    ) -> float:

        return (
            time.monotonic()
            - self.last_packet_at
        )

    # --------------------------------------------------------
    # SNAPSHOT
    # --------------------------------------------------------

    def snapshot(
        self,
    ) -> bytes:

        return bytes(
            self.data
        )


# ============================================================
# AUDIO SINK
# ============================================================

class VoiceAISink(
    voice_recv.AudioSink
):
    """
    يستقبل PCM من Discord.

    Discord
        ↓
    VoiceAISink
        ↓
    UserAudioBuffer
        ↓
    VoiceSession
    """

    def __init__(
        self,
        session: "VoiceSession",
    ):

        super().__init__()

        self.session = session

        self.buffers: dict[
            int,
            UserAudioBuffer,
        ] = {}

        self.processing_users: set[
            int
        ] = set()

        self.closed = False

        self.monitor_task = (
            asyncio.create_task(
                self.monitor_buffers()
            )
        )

    # ========================================================
    # WANTS OPUS
    # ========================================================

    def wants_opus(
        self,
    ) -> bool:

        return False

    # ========================================================
    # WRITE
    # ========================================================

    def write(
        self,
        user,
        data,
    ) -> None:

        if self.closed:
            return

        if user is None:
            return

        # ----------------------------------------------------
        # تجاهل البوتات
        # ----------------------------------------------------

        if getattr(
            user,
            "bot",
            False,
        ):

            return

        # ----------------------------------------------------
        # الحصول على PCM
        # ----------------------------------------------------

        pcm = getattr(
            data,
            "pcm",
            None,
        )

        if not pcm:
            return

        user_id = int(
            user.id
        )

        username = getattr(
            user,
            "display_name",
            getattr(
                user,
                "name",
                str(user_id),
            ),
        )

        # ----------------------------------------------------
        # إنشاء Buffer
        # ----------------------------------------------------

        buffer = self.buffers.get(
            user_id
        )

        if buffer is None:

            now = time.monotonic()

            buffer = UserAudioBuffer(
                user_id=user_id,
                username=username,
                data=bytearray(),
                started_at=now,
                last_packet_at=now,
            )

            self.buffers[
                user_id
            ] = buffer

        # ----------------------------------------------------
        # حماية الذاكرة
        # ----------------------------------------------------

        if (
            buffer.size()
            + len(pcm)
            > MAX_AUDIO_BUFFER_BYTES
        ):

            logger.warning(
                "Audio buffer limit reached for %s",
                username,
            )

            asyncio.create_task(
                self.flush_user(
                    user_id
                )
            )

            return

        # ----------------------------------------------------
        # إضافة الصوت
        # ----------------------------------------------------

        buffer.add(
            pcm
        )

    # ========================================================
    # MONITOR
    # ========================================================

    async def monitor_buffers(
        self,
    ) -> None:

        try:

            while not self.closed:

                await asyncio.sleep(
                    0.20
                )

                now = (
                    time.monotonic()
                )

                for user_id, buffer in list(
                    self.buffers.items()
                ):

                    # ----------------------------------------
                    # لا تعالج نفس المستخدم مرتين
                    # ----------------------------------------

                    if (
                        user_id
                        in self.processing_users
                    ):

                        continue

                    duration = (
                        buffer.duration()
                    )

                    silence = (
                        now
                        - buffer.last_packet_at
                    )

                    # ----------------------------------------
                    # المقطع قصير جدًا
                    # ----------------------------------------

                    if (
                        duration
                        < MIN_AUDIO_SECONDS
                    ):

                        continue

                    # ----------------------------------------
                    # الحد الأقصى
                    # ----------------------------------------

                    if (
                        duration
                        >= MAX_AUDIO_SECONDS
                    ):

                        await self.flush_user(
                            user_id
                        )

                        continue

                    # ----------------------------------------
                    # نهاية الكلام
                    # ----------------------------------------

                    if (
                        silence
                        >= VOICE_SILENCE_TIMEOUT
                    ):

                        await self.flush_user(
                            user_id
                        )

        except asyncio.CancelledError:

            return

        except Exception:

            logger.exception(
                "Audio monitor crashed"
            )

    # ========================================================
    # FLUSH USER
    # ========================================================

    async def flush_user(
        self,
        user_id: int,
    ) -> None:

        buffer = self.buffers.get(
            user_id
        )

        if buffer is None:
            return

        if (
            user_id
            in self.processing_users
        ):

            return

        audio = (
            buffer.snapshot()
        )

        username = (
            buffer.username
        )

        buffer.clear()

        if not audio:
            return

        if (
            len(audio)
            < 1000
        ):

            return

        self.processing_users.add(
            user_id
        )

        try:

            await self.session.process_user_audio(
                user_id=user_id,
                username=username,
                pcm=audio,
            )

        except Exception:

            logger.exception(
                "Failed processing audio from %s",
                username,
            )

        finally:

            self.processing_users.discard(
                user_id
            )

    # ========================================================
    # CLEANUP
    # ========================================================

    def cleanup(
        self,
    ) -> None:

        if self.closed:
            return

        self.closed = True

        self.buffers.clear()

        if self.monitor_task:

            self.monitor_task.cancel()

            self.monitor_task = None


# ============================================================
# VOICE SESSION
# ============================================================

class VoiceSession:
    """
    جلسة AI Voice كاملة لسيرفر واحد.

    كل Guild له Session واحدة.
    """

    def __init__(
        self,
        bot,
        guild: discord.Guild,
        voice_client: voice_recv.VoiceRecvClient,
        text_channel: discord.abc.Messageable,
    ):

        self.bot = bot

        self.guild = guild

        self.voice_client = (
            voice_client
        )

        self.text_channel = (
            text_channel
        )

        # ----------------------------------------------------
        # Gemini
        # ----------------------------------------------------

        self.ai = GeminiEngine()

        # ----------------------------------------------------
        # Sink
        # ----------------------------------------------------

        self.sink: Optional[
            VoiceAISink
        ] = None

        # ----------------------------------------------------
        # Listening
        # ----------------------------------------------------

        self.listening = False

        # ----------------------------------------------------
        # Processing
        # ----------------------------------------------------

        self.processing = False

        self.processing_lock = (
            asyncio.Lock()
        )

        # ----------------------------------------------------
        # Memory
        # ----------------------------------------------------

        self.memory: list[
            dict
        ] = []

        # ----------------------------------------------------
        # Voice
        # ----------------------------------------------------

        self.voice_name = (
            DEFAULT_GEMINI_VOICE
        )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        self.messages_processed = 0

        self.audio_received = 0

        self.audio_responses = 0

        self.errors = 0

        # ----------------------------------------------------
        # Creation time
        # ----------------------------------------------------

        self.created_at = (
            time.monotonic()
        )

        # ----------------------------------------------------
        # Stop flag
        # ----------------------------------------------------

        self.stopping = False

    # ========================================================
    # START
    # ========================================================

    async def start(
        self,
    ) -> None:

        if self.stopping:
            return

        logger.info(
            "Starting VoiceSession for guild %s",
            self.guild.id,
        )

        # ----------------------------------------------------
        # إنشاء Sink
        # ----------------------------------------------------

        self.sink = VoiceAISink(
            self
        )

        # ----------------------------------------------------
        # بدء استقبال الصوت
        # ----------------------------------------------------

        self.voice_client.listen(
            self.sink
        )

        self.listening = True

        logger.info(
            "Voice listening started for guild %s",
            self.guild.id,
        )

    # ========================================================
    # PROCESS AUDIO
    # ========================================================

    async def process_user_audio(
        self,
        user_id: int,
        username: str,
        pcm: bytes,
    ) -> None:

        if self.stopping:
            return

        if not pcm:
            return

        self.audio_received += len(
            pcm
        )

        # ----------------------------------------------------
        # طلب AI واحد في نفس الوقت
        # ----------------------------------------------------

        async with self.processing_lock:

            self.processing = True

            try:

                # --------------------------------------------
                # تحويل Discord PCM
                # إلى WAV 16kHz Mono
                # --------------------------------------------

                wav_data = (
                    pcm_to_wav(
                        pcm
                    )
                )

                if not wav_data:
                    return

                # --------------------------------------------
                # إحصائيات
                # --------------------------------------------

                if hasattr(
                    self.bot,
                    "stats",
                ):

                    self.bot.stats[
                        "ai_requests"
                    ] += 1

                # --------------------------------------------
                # Gemini
                # --------------------------------------------

                result = (
                    await self.ai.process_voice(
                        audio=wav_data,
                        memory=self.memory,
                        username=username,
                        voice=self.voice_name,
                    )
                )

                if not result:
                    return

                text = result.get(
                    "text"
                )

                response = result.get(
                    "response"
                )

                audio = result.get(
                    "audio"
                )

                # --------------------------------------------
                # لا يوجد نص
                # --------------------------------------------

                if not text:
                    return

                logger.info(
                    "[%s] %s: %s",
                    username,
                    user_id,
                    text,
                )

                # --------------------------------------------
                # Memory
                # --------------------------------------------

                self.add_memory(
                    role="user",
                    content=text,
                    username=username,
                )

                if response:

                    self.add_memory(
                        role="assistant",
                        content=response,
                    )

                self.messages_processed += 1

                if hasattr(
                    self.bot,
                    "stats",
                ):

                    self.bot.stats[
                        "messages_processed"
                    ] += 1

                # --------------------------------------------
                # إرسال النص للشات
                # --------------------------------------------

                await self.send_transcript(
                    username=username,
                    text=text,
                    response=response,
                )

                # --------------------------------------------
                # تشغيل TTS
                # --------------------------------------------

                if audio:

                    await self.play_audio(
                        audio
                    )

                    self.audio_responses += 1

            except Exception:

                self.errors += 1

                if hasattr(
                    self.bot,
                    "stats",
                ):

                    self.bot.stats[
                        "errors"
                    ] += 1

                logger.exception(
                    "AI audio processing error"
                )

            finally:

                self.processing = False

    # ========================================================
    # SEND TRANSCRIPT
    # ========================================================

    async def send_transcript(
        self,
        username: str,
        text: str,
        response: Optional[str],
    ) -> None:

        if not self.text_channel:
            return

        try:

            await self.text_channel.send(
                f"🎤 **{username}:** {text}"
            )

            if response:

                await self.text_channel.send(
                    f"🤖 **AI:** {response}"
                )

        except Exception as error:

            logger.warning(
                "Could not send transcript: %s",
                error,
            )

    # ========================================================
    # PLAY AUDIO
    # ========================================================

    async def play_audio(
        self,
        audio: bytes,
    ) -> None:

        if self.stopping:
            return

        if not audio:
            return

        if not self.voice_client:
            return

        # ----------------------------------------------------
        # تحويل TTS من:
        #
        # 24kHz Mono
        #
        # إلى:
        #
        # 48kHz Stereo
        # ----------------------------------------------------

        discord_audio = (
            tts_pcm_to_discord_pcm(
                audio
            )
        )

        if not discord_audio:
            return

        # ----------------------------------------------------
        # الانتظار حتى ينتهي الصوت السابق
        # ----------------------------------------------------

        while (
            self.voice_client.is_playing()
            and not self.stopping
        ):

            await asyncio.sleep(
                0.10
            )

        if self.stopping:
            return

        # ----------------------------------------------------
        # Raw PCM
        # ----------------------------------------------------

        stream = io.BytesIO(
            discord_audio
        )

        source = discord.PCMAudio(
            stream,
            sample_width=2,
        )

        finished = (
            asyncio.Event()
        )

        loop = (
            asyncio.get_running_loop()
        )

        def after_play(
            error,
        ):

            if error:

                logger.error(
                    "Audio playback error: %s",
                    error,
                )

            loop.call_soon_threadsafe(
                finished.set
            )

        try:

            self.voice_client.play(
                source,
                after=after_play,
            )

            await finished.wait()

        except Exception:

            logger.exception(
                "Could not play AI audio"
            )

            try:

                source.cleanup()

            except Exception:
                pass

    # ========================================================
    # CHANGE VOICE
    # ========================================================

    def set_voice(
        self,
        voice_name: str,
    ) -> bool:

        voice_name = (
            voice_name.strip()
        )

        if not voice_name:
            return False

        if (
            voice_name
            not in GEMINI_VOICES
        ):

            return False

        self.voice_name = (
            voice_name
        )

        logger.info(
            "Guild %s changed AI voice to %s",
            self.guild.id,
            voice_name,
        )

        return True

    # ========================================================
    # GET VOICE
    # ========================================================

    def get_voice(
        self,
    ) -> str:

        return self.voice_name

    # ========================================================
    # LIST VOICES
    # ========================================================

    def list_voices(
        self,
    ) -> list[str]:

        return list(
            GEMINI_VOICES
        )

    # ========================================================
    # ADD MEMORY
    # ========================================================

    def add_memory(
        self,
        role: str,
        content: str,
        username: Optional[str] = None,
    ) -> None:

        if not content:
            return

        item = {
            "role": role,
            "content": content,
        }

        if username:

            item[
                "username"
            ] = username

        self.memory.append(
            item
        )

        if (
            len(self.memory)
            > MAX_MEMORY_MESSAGES
        ):

            overflow = (
                len(self.memory)
                - MAX_MEMORY_MESSAGES
            )

            del self.memory[
                0:overflow
            ]

    # ========================================================
    # CLEAR MEMORY
    # ========================================================

    def clear_memory(
        self,
    ) -> None:

        self.memory.clear()

        try:

            self.ai.clear_memory()

        except Exception:

            pass

    # ========================================================
    # MEMORY SIZE
    # ========================================================

    def memory_size(
        self,
    ) -> int:

        return len(
            self.memory
        )

    # ========================================================
    # RESET
    # ========================================================

    async def reset(
        self,
    ) -> None:

        self.clear_memory()

        try:

            await self.ai.reset()

        except Exception as error:

            logger.warning(
                "AI reset warning: %s",
                error,
            )

    # ========================================================
    # STOP
    # ========================================================

    async def stop(
        self,
    ) -> None:

        if self.stopping:
            return

        self.stopping = True

        self.listening = False

        logger.info(
            "Stopping VoiceSession for guild %s",
            self.guild.id,
        )

        # ----------------------------------------------------
        # إيقاف Sink
        # ----------------------------------------------------

        if self.sink:

            try:

                self.sink.cleanup()

            except Exception:

                pass

            self.sink = None

        # ----------------------------------------------------
        # إيقاف الاستماع
        # ----------------------------------------------------

        try:

            self.voice_client.stop_listening()

        except Exception:

            pass

        # ----------------------------------------------------
        # إيقاف Gemini
        # ----------------------------------------------------

        try:

            await self.ai.close()

        except Exception as error:

            logger.warning(
                "Gemini close error: %s",
                error,
            )

        # ----------------------------------------------------
        # تنظيف الذاكرة
        # ----------------------------------------------------

        self.memory.clear()

        logger.info(
            "VoiceSession stopped for guild %s",
            self.guild.id,
        )


# ============================================================
# PCM 48kHz Stereo
# ->
# PCM 16kHz Mono
# ============================================================

def downsample_pcm_48k_to_16k(
    pcm: bytes,
) -> bytes:

    """
    تحويل Discord PCM:

        48000Hz
        Stereo
        16-bit

    إلى:

        16000Hz
        Mono
        16-bit

    بدون NumPy أو audioop.
    """

    if not pcm:
        return b""

    frame_size = (
        PCM_CHANNELS
        * PCM_SAMPLE_WIDTH
    )

    if (
        len(pcm)
        < frame_size
    ):

        return b""

    total_frames = (
        len(pcm)
        // frame_size
    )

    target_frames = (
        total_frames // 3
    )

    output = bytearray(
        target_frames
        * GEMINI_SAMPLE_WIDTH
    )

    output_index = 0

    for index in range(
        target_frames
    ):

        source_frame = (
            index * 3
        )

        offset = (
            source_frame
            * frame_size
        )

        # ----------------------------------------------------
        # Left
        # ----------------------------------------------------

        left = int.from_bytes(
            pcm[
                offset:
                offset + 2
            ],
            byteorder="little",
            signed=True,
        )

        # ----------------------------------------------------
        # Right
        # ----------------------------------------------------

        right = int.from_bytes(
            pcm[
                offset + 2:
                offset + 4
            ],
            byteorder="little",
            signed=True,
        )

        # ----------------------------------------------------
        # Stereo -> Mono
        # ----------------------------------------------------

        mono = (
            left + right
        ) // 2

        mono = max(
            -32768,
            min(
                32767,
                mono,
            ),
        )

        output[
            output_index:
            output_index + 2
        ] = int(
            mono
        ).to_bytes(
            2,
            byteorder="little",
            signed=True,
        )

        output_index += 2

    return bytes(
        output
    )


# ============================================================
# PCM -> WAV
# ============================================================

def pcm_to_wav(
    pcm: bytes,
) -> bytes:

    """
    يحول Discord PCM إلى WAV مناسب لـ Gemini.

    Discord:
        48kHz Stereo PCM

    Gemini:
        16kHz Mono WAV
    """

    if not pcm:
        return b""

    gemini_pcm = (
        downsample_pcm_48k_to_16k(
            pcm
        )
    )

    if not gemini_pcm:
        return b""

    output = io.BytesIO()

    with wave.open(
        output,
        "wb",
    ) as wav:

        wav.setnchannels(
            GEMINI_CHANNELS
        )

        wav.setsampwidth(
            GEMINI_SAMPLE_WIDTH
        )

        wav.setframerate(
            GEMINI_SAMPLE_RATE
        )

        wav.writeframes(
            gemini_pcm
        )

    return output.getvalue()


# ============================================================
# TTS 24kHz Mono
# ->
# Discord 48kHz Stereo
# ============================================================

def tts_pcm_to_discord_pcm(
    pcm: bytes,
) -> bytes:

    """
    Gemini TTS:

        24000Hz
        Mono
        16-bit

    Discord:

        48000Hz
        Stereo
        16-bit

    نرفع معدل العينات من 24k إلى 48k
    بتكرار كل عينة مرتين،
    ثم نحول Mono إلى Stereo.
    """

    if not pcm:
        return b""

    frame_size = (
        TTS_CHANNELS
        * TTS_SAMPLE_WIDTH
    )

    if (
        len(pcm)
        < frame_size
    ):

        return b""

    sample_count = (
        len(pcm)
        // frame_size
    )

    output = bytearray(
        sample_count
        * 2
        * PCM_CHANNELS
        * PCM_SAMPLE_WIDTH
    )

    output_index = 0

    for index in range(
        sample_count
    ):

        offset = (
            index
            * frame_size
        )

        sample = int.from_bytes(
            pcm[
                offset:
                offset + 2
            ],
            byteorder="little",
            signed=True,
        )

        sample_bytes = int(
            sample
        ).to_bytes(
            2,
            byteorder="little",
            signed=True,
        )

        # ----------------------------------------------------
        # 24k -> 48k
        # كل عينة تصبح عينتين
        # ----------------------------------------------------

        for _ in range(2):

            # Left
            output[
                output_index:
                output_index + 2
            ] = sample_bytes

            output_index += 2

            # Right
            output[
                output_index:
                output_index + 2
            ] = sample_bytes

            output_index += 2

    return bytes(
        output
    )


# ============================================================
# VOICE COMMAND HELPERS
# ============================================================

async def send_voice_list(
    ctx: commands.Context,
    session: VoiceSession,
):

    voices = (
        session.list_voices()
    )

    if not voices:

        await ctx.send(
            "❌ ما فيه أصوات متاحة."
        )

        return

    lines = []

    for index, voice in enumerate(
        voices,
        start=1,
    ):

        current = (
            " ← الحالي"
            if voice
            == session.get_voice()
            else ""
        )

        lines.append(
            f"`{index}.` **{voice}**{current}"
        )

    embed = discord.Embed(
        title="🎙️ أصوات Gemini",
        description="\n".join(
            lines
        ),
        color=discord.Color.blurple(),
    )

    embed.set_footer(
        text=(
            "استخدم أمر /setvoice لاختيار الصوت"
        )
    )

    await ctx.send(
        embed=embed
    )
