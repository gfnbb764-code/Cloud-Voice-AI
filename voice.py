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
# - إدارة جلسات الصوت لكل Guild
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
    VOICE_ALIASES,
    VOICE_SILENCE_TIMEOUT,
    MIN_AUDIO_SECONDS,
    MAX_AUDIO_BUFFER_BYTES,
    MAX_RECORDING_SECONDS,
)

from gemini import GeminiEngine


logger = logging.getLogger("cloud_voice_ai.voice")


# ============================================================
# AUDIO FORMAT
# ============================================================

# Discord voice receive/playback:
# 48000 Hz / Stereo / Signed 16-bit PCM
PCM_SAMPLE_RATE = 48000
PCM_CHANNELS = 2
PCM_SAMPLE_WIDTH = 2

# Gemini STT:
# 16000 Hz / Mono / Signed 16-bit PCM
GEMINI_SAMPLE_RATE = 16000
GEMINI_CHANNELS = 1
GEMINI_SAMPLE_WIDTH = 2

# Gemini TTS:
# 24000 Hz / Mono / Signed 16-bit PCM
TTS_SAMPLE_RATE = 24000
TTS_CHANNELS = 1
TTS_SAMPLE_WIDTH = 2


# ============================================================
# AUDIO LIMITS
# ============================================================

SILENCE_TIMEOUT = max(
    0.2,
    float(VOICE_SILENCE_TIMEOUT),
)

MIN_AUDIO_SECONDS = max(
    0.05,
    float(MIN_AUDIO_SECONDS),
)

MAX_AUDIO_SECONDS = max(
    1.0,
    float(MAX_RECORDING_SECONDS),
)

MAX_AUDIO_BUFFER_BYTES = max(
    1,
    int(MAX_AUDIO_BUFFER_BYTES),
)

MAX_USERS_TRACKED = 50


# ============================================================
# HELPERS
# ============================================================

def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    return max(
        minimum,
        min(maximum, value),
    )


def pcm_duration(
    pcm: bytes,
    sample_rate: int = PCM_SAMPLE_RATE,
    channels: int = PCM_CHANNELS,
    sample_width: int = PCM_SAMPLE_WIDTH,
) -> float:
    """
    يحسب مدة PCM بالثواني.
    """

    if not pcm:
        return 0.0

    bytes_per_second = (
        sample_rate
        * channels
        * sample_width
    )

    if bytes_per_second <= 0:
        return 0.0

    return len(pcm) / bytes_per_second


def pcm_to_wav(
    pcm_data: bytes,
) -> bytes:
    """
    Discord:
        48000 Hz
        Stereo
        Signed 16-bit PCM

    Gemini:
        16000 Hz
        Mono
        Signed 16-bit PCM

    يتم التحويل يدويًا بدون audioop
    حتى لا نعتمد على audioop-lts.
    """

    if not pcm_data:
        raise ValueError(
            "PCM data is empty."
        )

    source_frame_size = (
        PCM_CHANNELS
        * PCM_SAMPLE_WIDTH
    )

    if source_frame_size <= 0:
        raise ValueError(
            "Invalid source audio format."
        )

    usable_length = (
        len(pcm_data)
        // source_frame_size
    ) * source_frame_size

    pcm_data = pcm_data[
        :usable_length
    ]

    if not pcm_data:
        raise ValueError(
            "PCM data has no complete frames."
        )

    # --------------------------------------------------------
    # 48k stereo -> 16k mono
    # --------------------------------------------------------
    #
    # نأخذ frame واحدًا من كل 3 frames.
    # هذا يعطي تقريبًا 16kHz من مصدر 48kHz.
    #
    # ويتم دمج Left + Right للحصول على Mono.
    # --------------------------------------------------------

    output = bytearray()

    step = (
        source_frame_size
        * 3
    )

    for index in range(
        0,
        len(pcm_data),
        step,
    ):
        block = pcm_data[
            index:index + step
        ]

        if len(block) < source_frame_size:
            break

        frame = block[
            :source_frame_size
        ]

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

        mono = (
            left + right
        ) // 2

        mono = int(
            clamp(
                mono,
                -32768,
                32767,
            )
        )

        output.extend(
            mono.to_bytes(
                2,
                byteorder="little",
                signed=True,
            )
        )

    if not output:
        raise ValueError(
            "Audio conversion produced no data."
        )

    # --------------------------------------------------------
    # WAV wrapper
    # --------------------------------------------------------

    buffer = io.BytesIO()

    with wave.open(
        buffer,
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
            bytes(output)
        )

    return buffer.getvalue()


def tts_pcm_to_discord_pcm(
    tts_pcm: bytes,
) -> bytes:
    """
    Gemini TTS:
        24000 Hz
        Mono
        Signed 16-bit PCM

    Discord:
        48000 Hz
        Stereo
        Signed 16-bit PCM

    يتم التحويل بدون audioop.
    """

    if not tts_pcm:
        raise ValueError(
            "TTS PCM data is empty."
        )

    usable_length = (
        len(tts_pcm)
        // TTS_SAMPLE_WIDTH
    ) * TTS_SAMPLE_WIDTH

    tts_pcm = tts_pcm[
        :usable_length
    ]

    if not tts_pcm:
        raise ValueError(
            "Invalid TTS PCM data."
        )

    output = bytearray()

    # --------------------------------------------------------
    # 24kHz -> 48kHz
    # --------------------------------------------------------
    #
    # تكرار كل sample مرتين.
    #
    # Mono -> Stereo:
    # نفس sample في Left و Right.
    # --------------------------------------------------------

    for index in range(
        0,
        len(tts_pcm),
        TTS_SAMPLE_WIDTH,
    ):

        sample = tts_pcm[
            index:index + TTS_SAMPLE_WIDTH
        ]

        if len(sample) < TTS_SAMPLE_WIDTH:
            break

        # أول sample
        output.extend(sample)
        output.extend(sample)

        # ثاني sample
        # بهذه الطريقة نحصل على 48kHz
        output.extend(sample)
        output.extend(sample)

    return bytes(output)


def normalize_voice_name(
    name: str,
) -> str:
    """
    تحويل اسم الصوت إلى الاسم الرسمي الموجود
    في قائمة Gemini.
    """

    if not name:
        return DEFAULT_GEMINI_VOICE

    value = str(
        name
    ).strip().lower()

    # تطابق مباشر
    for voice in GEMINI_VOICES:
        if voice.lower() == value:
            return voice

    # aliases من config
    alias = VOICE_ALIASES.get(
        value
    )

    if alias:
        return alias

    # fallback
    return DEFAULT_GEMINI_VOICE


def safe_display_name(
    user: discord.User | discord.Member | None,
) -> str:
    """
    الحصول على اسم آمن للعرض.
    """

    if user is None:
        return "Unknown"

    return (
        getattr(
            user,
            "display_name",
            None,
        )
        or getattr(
            user,
            "name",
            None,
        )
        or "Unknown"
    )


# ============================================================
# USER AUDIO STATE
# ============================================================

@dataclass
class UserAudioState:
    """
    حالة تسجيل مستخدم واحد.
    """

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

    def start(
        self,
        username: str,
    ) -> None:

        self.username = (
            username
            or "Unknown"
        )

        now = time.monotonic()

        self.started_at = now
        self.last_audio_at = now

        self.total_bytes = 0

        self.chunks.clear()

    def append(
        self,
        pcm: bytes,
    ) -> None:

        if not pcm:
            return

        self.chunks.append(
            pcm
        )

        self.total_bytes += len(
            pcm
        )

        self.last_audio_at = (
            time.monotonic()
        )

    def build_audio(self) -> bytes:
        return b"".join(
            self.chunks
        )

    def clear(self) -> None:

        self.chunks.clear()

        self.total_bytes = 0

        self.started_at = 0.0

        self.last_audio_at = 0.0

    @property
    def duration(self) -> float:
        """
        مدة التسجيل الحالية.
        """

        if self.total_bytes <= 0:
            return 0.0

        return (
            self.total_bytes
            / (
                PCM_SAMPLE_RATE
                * PCM_CHANNELS
                * PCM_SAMPLE_WIDTH
            )
        )


# ============================================================
# DISCORD AUDIO SINK
# ============================================================

class VoiceAISink(
    voice_recv.AudioSink
):
    """
    يستقبل صوت المستخدمين من Discord
    ويقسمه إلى مقاطع كلام.
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

    # ========================================================
    # DISCORD VOICE RECEIVE
    # ========================================================

    def wants_opus(self) -> bool:
        """
        نريد PCM بدل Opus.
        """

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

        if data is None:
            return

        user_id = user.id

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

        # ----------------------------------------------------
        # Get/Create state
        # ----------------------------------------------------

        state = self.users.get(
            user_id
        )

        if state is None:

            if (
                len(self.users)
                >= MAX_USERS_TRACKED
            ):
                logger.warning(
                    "Maximum tracked voice users reached."
                )
                return

            state = UserAudioState(
                user_id=user_id
            )

            self.users[
                user_id
            ] = state

        # ----------------------------------------------------
        # Ignore new audio while processing
        # ----------------------------------------------------

        if state.processing:
            return

        # ----------------------------------------------------
        # Start new speech segment
        # ----------------------------------------------------

        if not state.chunks:

            state.start(
                safe_display_name(
                    user
                )
            )

        # ----------------------------------------------------
        # Prevent giant buffers
        # ----------------------------------------------------

        if (
            state.total_bytes
            + len(pcm)
            > MAX_AUDIO_BUFFER_BYTES
        ):

            self._schedule_flush(
                user_id,
                "max_buffer",
            )

            return

        # ----------------------------------------------------
        # Store PCM
        # ----------------------------------------------------

        state.append(
            pcm
        )

        # ----------------------------------------------------
        # Maximum duration
        # ----------------------------------------------------

        if (
            state.duration
            >= MAX_AUDIO_SECONDS
        ):

            self._schedule_flush(
                user_id,
                "max_duration",
            )

    def _schedule_flush(
        self,
        user_id: int,
        reason: str,
    ) -> None:
        """
        تشغيل flush من Thread استقبال الصوت
        بدون انتظار coroutine.
        """

        try:

            loop = (
                self.session.bot.loop
            )

            if (
                loop is None
                or loop.is_closed()
            ):
                return

            asyncio.run_coroutine_threadsafe(
                self._flush_user(
                    user_id,
                    reason=reason,
                ),
                loop,
            )

        except Exception:
            logger.exception(
                "Failed scheduling audio flush."
            )

    # ========================================================
    # MONITOR
    # ========================================================

    async def start_monitor(
        self,
    ) -> None:

        if self.closed:
            return

        if (
            self._monitor_task is not None
            and not self._monitor_task.done()
        ):
            return

        self._monitor_task = (
            asyncio.create_task(
                self._monitor_loop()
            )
        )

    async def _monitor_loop(
        self,
    ) -> None:

        try:

            while not self.closed:

                await asyncio.sleep(
                    0.15
                )

                now = (
                    time.monotonic()
                )

                for (
                    user_id,
                    state,
                ) in list(
                    self.users.items()
                ):

                    if not state.chunks:
                        continue

                    if state.processing:
                        continue

                    silence = (
                        now
                        - state.last_audio_at
                    )

                    if (
                        silence
                        >= SILENCE_TIMEOUT
                    ):

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

    # ========================================================
    # FLUSH USER
    # ========================================================

    async def _flush_user(
        self,
        user_id: int,
        reason: str = "unknown",
    ) -> None:

        state = self.users.get(
            user_id
        )

        if state is None:
            return

        async with state.lock:

            if state.processing:
                return

            if not state.chunks:
                return

            state.processing = True

            try:

                audio = (
                    state.build_audio()
                )

                username = (
                    state.username
                )

                duration = (
                    len(audio)
                    / (
                        PCM_SAMPLE_RATE
                        * PCM_CHANNELS
                        * PCM_SAMPLE_WIDTH
                    )
                )

                logger.debug(
                    "Captured %.2fs from %s (%s).",
                    duration,
                    username,
                    reason,
                )

                # ------------------------------------------------
                # Clear state BEFORE processing
                # ------------------------------------------------

                state.clear()

                # ------------------------------------------------
                # Ignore very short audio
                # ------------------------------------------------

                if (
                    duration
                    < MIN_AUDIO_SECONDS
                ):
                    return

                # ------------------------------------------------
                # Send to VoiceSession
                # ------------------------------------------------

                await (
                    self.session
                    .process_user_audio(
                        user_id=user_id,
                        username=username,
                        pcm=audio,
                    )
                )

            except Exception:
                logger.exception(
                    "Failed processing audio from user %s.",
                    user_id,
                )

            finally:
                state.processing = False

    # ========================================================
    # CLEANUP
    # ========================================================

    def cleanup(self) -> None:

        if self.closed:
            return

        self.closed = True

        if (
            self._monitor_task
            is not None
        ):

            self._monitor_task.cancel()

            self._monitor_task = None

        for state in list(
            self.users.values()
        ):
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

        self.voice_name = (
            normalize_voice_name(
                DEFAULT_GEMINI_VOICE
            )
        )

        self.memory: dict[
            int,
            deque,
        ] = defaultdict(
            lambda: deque(
                maxlen=MAX_MEMORY_MESSAGES
            )
        )

        self.processing_users: set[
            int
        ] = set()

        self.started_at = (
            time.monotonic()
        )

        self.messages_processed = 0

        self.audio_processed = 0

        self.errors = 0

        self.total_response_time = 0.0

        self._play_lock = (
            asyncio.Lock()
        )

        self._processing_lock = (
            asyncio.Lock()
        )

    # ========================================================
    # JOIN
    # ========================================================

    async def start(
        self,
        channel: discord.VoiceChannel,
    ) -> None:

        if channel is None:
            raise ValueError(
                "Voice channel is required."
            )

        if (
            self.voice_client is not None
            and self.voice_client.is_connected()
        ):

            # إذا كان متصلًا بالفعل في نفس القناة
            if (
                self.voice_client.channel
                and self.voice_client.channel.id
                == channel.id
            ):
                return

            # إذا كان في قناة ثانية
            try:
                await self.voice_client.move_to(
                    channel
                )
            except Exception:
                logger.exception(
                    "Failed moving voice client."
                )
                raise

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

            self.sink = (
                VoiceAISink(self)
            )

            vc.listen(
                self.sink
            )

            await (
                self.sink.start_monitor()
            )

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

    async def stop(
        self,
    ) -> None:

        sink = self.sink

        if sink is not None:

            try:
                sink.cleanup()
            except Exception:
                logger.exception(
                    "Sink cleanup failed."
                )

            self.sink = None

        vc = self.voice_client

        if vc is not None:

            try:

                if vc.is_listening():
                    vc.stop_listening()

            except Exception:
                pass

            try:

                if vc.is_playing():
                    vc.stop()

            except Exception:
                pass

            try:

                if vc.is_connected():

                    await vc.disconnect(
                        force=True
                    )

            except Exception:
                logger.exception(
                    "Voice disconnect failed."
                )

            self.voice_client = None

        self.processing_users.clear()

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

        # ----------------------------------------------------
        # Prevent same user from creating parallel requests.
        # ----------------------------------------------------

        if user_id in self.processing_users:
            return

        self.processing_users.add(
            user_id
        )

        started = (
            time.monotonic()
        )

        try:

            logger.info(
                "Processing speech from %s.",
                username,
            )

            # ------------------------------------------------
            # Discord PCM -> Gemini WAV
            # ------------------------------------------------

            wav_audio = (
                await asyncio.to_thread(
                    pcm_to_wav,
                    pcm,
                )
            )

            # ------------------------------------------------
            # Memory
            # ------------------------------------------------

            memory = list(
                self.memory[
                    user_id
                ]
            )

            # ------------------------------------------------
            # Gemini
            # ------------------------------------------------

            result = (
                await self.ai.process_voice(
                    audio=wav_audio,
                    memory=memory,
                    username=username,
                    voice=self.voice_name,
                )
            )

            if not result:
                return

            # ------------------------------------------------
            # Extract result
            # ------------------------------------------------

            transcript = str(
                result.get(
                    "transcript"
                )
                or result.get(
                    "text"
                )
                or ""
            ).strip()

            response_text = str(
                result.get(
                    "response"
                )
                or result.get(
                    "reply"
                )
                or ""
            ).strip()

            audio_response = (
                result.get("audio")
                or result.get("tts")
                or result.get("audio_data")
            )

            # ------------------------------------------------
            # Memory
            # ------------------------------------------------

            if transcript:

                self.memory[
                    user_id
                ].append(
                    {
                        "role": "user",
                        "content": transcript,
                    }
                )

            if response_text:

                self.memory[
                    user_id
                ].append(
                    {
                        "role": "assistant",
                        "content": response_text,
                    }
                )

            self.messages_processed += 1
            self.audio_processed += 1

            # ------------------------------------------------
            # Text response
            # ------------------------------------------------

            await (
                self.send_text_response(
                    username=username,
                    transcript=transcript,
                    response=response_text,
                )
            )

            # ------------------------------------------------
            # TTS
            # ------------------------------------------------

            if audio_response:

                await self.play_audio(
                    audio_response
                )

            elapsed = (
                time.monotonic()
                - started
            )

            self.total_response_time += (
                elapsed
            )

            logger.info(
                "Processed %s in %.2fs.",
                username,
                elapsed,
            )

        except asyncio.CancelledError:
            raise

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

        channel = None

        # ----------------------------------------------------
        # اختيار أول Text Channel يمكن للبوت الإرسال فيه.
        # ----------------------------------------------------

        me = self.guild.me

        for candidate in (
            self.guild.text_channels
        ):

            try:

                permissions = (
                    candidate.permissions_for(
                        me
                    )
                )

                if permissions.send_messages:
                    channel = candidate
                    break

            except Exception:
                continue

        if channel is None:
            return

        # ----------------------------------------------------
        # Discord message limit
        # ----------------------------------------------------

        transcript_display = (
            transcript[:500]
            if transcript
            else "لم يتم التعرف على الكلام."
        )

        username_display = (
            discord.utils.escape_markdown(
                username
            )
        )

        content = (
            f"🎙️ **{username_display}**\n"
            f"> {transcript_display}\n\n"
            f"🤖 {response}"
        )

        content = content[
            :2000
        ]

        try:

            await channel.send(
                content
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
                # Gemini TTS -> Discord PCM
                # ------------------------------------------------

                discord_pcm = (
                    await asyncio.to_thread(
                        tts_pcm_to_discord_pcm,
                        audio_data,
                    )
                )

                if not discord_pcm:
                    return

                # ------------------------------------------------
                # Stop previous playback
                # ------------------------------------------------

                if (
                    self.voice_client.is_playing()
                ):

                    self.voice_client.stop()

                # ------------------------------------------------
                # Discord PCM source
                # ------------------------------------------------

                source = discord.PCMAudio(
                    io.BytesIO(
                        discord_pcm
                    ),
                    sample_width=2,
                )

                finished = (
                    asyncio.Event()
                )

                loop = (
                    self.bot.loop
                )

                def after_play(
                    error,
                ):

                    if error:

                        logger.error(
                            "Discord playback error: %s",
                            error,
                        )

                    try:

                        loop.call_soon_threadsafe(
                            finished.set
                        )

                    except Exception:
                        pass

                self.voice_client.play(
                    source,
                    after=after_play,
                )

                # ------------------------------------------------
                # Wait until playback finishes
                # ------------------------------------------------

                try:

                    await asyncio.wait_for(
                        finished.wait(),
                        timeout=60,
                    )

                except asyncio.TimeoutError:

                    logger.warning(
                        "Audio playback timed out."
                    )

                    if (
                        self.voice_client
                        and self.voice_client.is_playing()
                    ):

                        self.voice_client.stop()

            except asyncio.CancelledError:
                raise

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

        normalized = (
            normalize_voice_name(
                voice_name
            )
        )

        self.voice_name = (
            normalized
        )

        return normalized

    def get_voice(self) -> str:
        return self.voice_name

    def list_voices(self) -> list[str]:
        return list(
            GEMINI_VOICES
        )

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

        # إذا كان GeminiEngine يحتفظ بذاكرة
        # مستقلة، نحاول مسحها أيضًا إن كانت API تدعم ذلك.
        try:

            clear_user = getattr(
                self.ai,
                "clear_user_memory",
                None,
            )

            if clear_user is not None:

                result = clear_user(
                    user_id
                )

                if asyncio.iscoroutine(
                    result
                ):
                    asyncio.create_task(
                        result
                    )

        except Exception:
            logger.debug(
                "User memory cleanup unavailable.",
                exc_info=True,
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
    def uptime(
        self,
    ) -> float:

        return (
            time.monotonic()
            - self.started_at
        )

    @property
    def average_response_time(
        self,
    ) -> float:

        if (
            self.messages_processed
            <= 0
        ):
            return 0.0

        return (
            self.total_response_time
            / self.messages_processed
        )

    def get_stats(
        self,
    ) -> dict:

        vc = (
            self.voice_client
        )

        return {
            "guild_id": self.guild.id,

            "voice_connected": (
                vc is not None
                and vc.is_connected()
            ),

            "voice_channel": (
                vc.channel.name
                if (
                    vc is not None
                    and vc.channel is not None
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

            "memory_users": len(
                self.memory
            ),
        }

    # ========================================================
    # CLOSE
    # ========================================================

    async def close(
        self,
    ) -> None:

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
    إرسال قائمة الأصوات في Discord.
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

    text = "\n".join(
        lines
    )

    # Discord message limit
    if len(text) > 1900:

        text = (
            text[:1890]
            + "\n…"
        )

    await interaction.response.send_message(
        text,
        ephemeral=True,
    )


# ============================================================
# SESSION MANAGER
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

        self._lock = (
            asyncio.Lock()
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
    # CREATE
    # ========================================================

    async def create(
        self,
        guild: discord.Guild,
        channel: discord.VoiceChannel,
    ) -> VoiceSession:

        async with self._lock:

            existing = (
                self.sessions.get(
                    guild.id
                )
            )

            if existing is not None:

                if (
                    existing.voice_client
                    and existing.voice_client.is_connected()
                ):

                    # إذا كان في نفس القناة
                    if (
                        existing.voice_client.channel
                        and existing.voice_client.channel.id
                        == channel.id
                    ):
                        return existing

                    # نقل البوت لقناة أخرى
                    await existing.start(
                        channel
                    )

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

            try:

                await session.start(
                    channel
                )

            except Exception:

                try:
                    await session.close()
                except Exception:
                    pass

                raise

            self.sessions[
                guild.id
            ] = session

            return session

    # ========================================================
    # REMOVE
    # ========================================================

    async def remove(
        self,
        guild_id: int,
    ) -> None:

        async with self._lock:

            session = (
                self.sessions.pop(
                    guild_id,
                    None,
                )
            )

        if session is None:
            return

        try:

            await session.close()

        except Exception:

            logger.exception(
                "Failed closing voice session."
            )

    # ========================================================
    # CLOSE ALL
    # ========================================================

    async def close_all(
        self,
    ) -> None:

        async with self._lock:

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
    "pcm_duration",
    ]
