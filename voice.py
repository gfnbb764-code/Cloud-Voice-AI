# voice.py
# ============================================================
# Cloud Voice AI — Discord Voice System
# Voice receiving + STT + Gemini + TTS + playback
# ============================================================

from __future__ import annotations

import asyncio
import io
import logging
import struct
import time
import wave
from dataclasses import dataclass, field
from typing import Optional

import discord
from discord.ext import commands
from discord.ext import voice_recv

from config import (
    MAX_MEMORY_MESSAGES,
    DEFAULT_GEMINI_VOICE,
    GEMINI_VOICES,
    VOICE_SILENCE_TIMEOUT,
    MIN_AUDIO_SECONDS,
    MAX_RECORDING_SECONDS,
    MAX_AUDIO_BUFFER_BYTES,
    MAX_CONCURRENT_AI_REQUESTS,
    GEMINI_INPUT_SAMPLE_RATE,
    GEMINI_INPUT_CHANNELS,
    GEMINI_INPUT_SAMPLE_WIDTH,
    GEMINI_TTS_SAMPLE_RATE,
    GEMINI_TTS_CHANNELS,
    GEMINI_TTS_SAMPLE_WIDTH,
    DISCORD_SAMPLE_RATE,
    DISCORD_CHANNELS,
    DISCORD_SAMPLE_WIDTH,
    normalize_voice_name,
)

from gemini import GeminiEngine


# ============================================================
# LOGGER
# ============================================================

logger = logging.getLogger("cloud_voice_ai.voice")


# ============================================================
# AUDIO CONSTANTS
# ============================================================

PCM_SAMPLE_RATE = DISCORD_SAMPLE_RATE
PCM_CHANNELS = DISCORD_CHANNELS
PCM_SAMPLE_WIDTH = DISCORD_SAMPLE_WIDTH

GEMINI_SAMPLE_RATE = GEMINI_INPUT_SAMPLE_RATE
GEMINI_CHANNELS = GEMINI_INPUT_CHANNELS
GEMINI_SAMPLE_WIDTH = GEMINI_INPUT_SAMPLE_WIDTH

TTS_SAMPLE_RATE = GEMINI_TTS_SAMPLE_RATE
TTS_CHANNELS = GEMINI_TTS_CHANNELS
TTS_SAMPLE_WIDTH = GEMINI_TTS_SAMPLE_WIDTH

DISCORD_PLAYBACK_SAMPLE_RATE = 48000
DISCORD_PLAYBACK_CHANNELS = 2
DISCORD_PLAYBACK_SAMPLE_WIDTH = 2


# ============================================================
# AUDIO LIMITS
# ============================================================

SILENCE_TIMEOUT = max(
    0.2,
    float(VOICE_SILENCE_TIMEOUT),
)

MIN_AUDIO_LENGTH = max(
    0.05,
    float(MIN_AUDIO_SECONDS),
)

MAX_AUDIO_LENGTH = max(
    1.0,
    float(MAX_RECORDING_SECONDS),
)

MAX_BUFFER_SIZE = max(
    4096,
    int(MAX_AUDIO_BUFFER_BYTES),
)


# ============================================================
# AUDIO HELPERS
# ============================================================

def pcm_duration(
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

    return len(pcm) / bytes_per_second


def pcm_48k_stereo_to_16k_mono(
    pcm: bytes,
) -> bytes:
    """
    Discord 48 kHz stereo 16-bit PCM
    -> Gemini 16 kHz mono 16-bit PCM.
    """

    if not pcm:
        return b""

    frame_size = (
        PCM_CHANNELS
        * PCM_SAMPLE_WIDTH
    )

    if frame_size <= 0:
        return b""

    usable_length = (
        len(pcm) // frame_size
    ) * frame_size

    pcm = pcm[:usable_length]

    if not pcm:
        return b""

    output = bytearray()

    sample_count = (
        len(pcm) // frame_size
    )

    # 48 kHz -> 16 kHz.
    # Take one frame out of every three.
    for frame_index in range(
        0,
        sample_count,
        3,
    ):
        offset = (
            frame_index * frame_size
        )

        left_bytes = pcm[
            offset:
            offset + 2
        ]

        right_bytes = pcm[
            offset + 2:
            offset + 4
        ]

        if len(left_bytes) < 2:
            break

        left = struct.unpack(
            "<h",
            left_bytes,
        )[0]

        if len(right_bytes) >= 2:
            right = struct.unpack(
                "<h",
                right_bytes,
            )[0]

            mono = (
                int(left)
                + int(right)
            ) // 2
        else:
            mono = left

        mono = max(
            -32768,
            min(
                32767,
                mono,
            ),
        )

        output.extend(
            struct.pack(
                "<h",
                mono,
            )
        )

    return bytes(output)


def pcm_to_wav(
    pcm: bytes,
) -> bytes:
    """
    Discord PCM -> WAV suitable for Gemini STT.
    """

    mono_pcm = (
        pcm_48k_stereo_to_16k_mono(
            pcm
        )
    )

    if not mono_pcm:
        return b""

    buffer = io.BytesIO()

    with wave.open(
        buffer,
        "wb",
    ) as wav_file:

        wav_file.setnchannels(
            GEMINI_CHANNELS
        )

        wav_file.setsampwidth(
            GEMINI_SAMPLE_WIDTH
        )

        wav_file.setframerate(
            GEMINI_SAMPLE_RATE
        )

        wav_file.writeframes(
            mono_pcm
        )

    return buffer.getvalue()


def tts_pcm_to_discord_pcm(
    pcm: bytes,
) -> bytes:
    """
    Gemini TTS:
        24 kHz mono 16-bit PCM

    Discord:
        48 kHz stereo 16-bit PCM
    """

    if not pcm:
        return b""

    if TTS_SAMPLE_WIDTH != 2:
        raise ValueError(
            "Only 16-bit TTS PCM is supported."
        )

    usable_length = (
        len(pcm) // 2
    ) * 2

    pcm = pcm[:usable_length]

    if not pcm:
        return b""

    output = bytearray()

    # 24 kHz -> 48 kHz:
    # duplicate each sample once.
    for offset in range(
        0,
        len(pcm),
        2,
    ):
        sample = pcm[
            offset:
            offset + 2
        ]

        if len(sample) < 2:
            break

        # Left + right at original rate.
        stereo_frame = (
            sample + sample
        )

        # Duplicate frame for 2x sample rate.
        output.extend(
            stereo_frame
        )
        output.extend(
            stereo_frame
        )

    return bytes(output)


def pcm_is_silent(
    pcm: bytes,
    threshold: int = 450,
) -> bool:
    """
    Lightweight average-amplitude silence detector.
    """

    if not pcm:
        return True

    usable_length = (
        len(pcm) // 2
    ) * 2

    if usable_length <= 0:
        return True

    pcm = pcm[:usable_length]

    sample_count = len(pcm) // 2

    if sample_count <= 0:
        return True

    total = 0

    for offset in range(
        0,
        len(pcm),
        2,
    ):
        sample = struct.unpack(
            "<h",
            pcm[
                offset:
                offset + 2
            ],
        )[0]

        total += abs(sample)

    average = (
        total / sample_count
    )

    return average < threshold


def split_text(
    text: str,
    limit: int = 1900,
) -> list[str]:
    text = str(text or "").strip()

    if not text:
        return []

    if len(text) <= limit:
        return [text]

    chunks: list[str] = []

    while len(text) > limit:

        split_at = text.rfind(
            " ",
            0,
            limit,
        )

        if split_at < 1:
            split_at = limit

        chunks.append(
            text[:split_at].strip()
        )

        text = text[
            split_at:
        ].strip()

    if text:
        chunks.append(text)

    return chunks


# ============================================================
# USER AUDIO STATE
# ============================================================

@dataclass
class UserAudioState:
    user_id: int

    buffer: bytearray = field(
        default_factory=bytearray
    )

    started_at: Optional[float] = None

    last_audio_at: Optional[float] = None

    processing: bool = False

    lock: asyncio.Lock = field(
        default_factory=asyncio.Lock
    )

    def add_audio(
        self,
        pcm: bytes,
    ) -> bool:

        if not pcm:
            return False

        now = time.monotonic()

        if self.started_at is None:
            self.started_at = now

        self.last_audio_at = now

        remaining = (
            MAX_BUFFER_SIZE
            - len(self.buffer)
        )

        if remaining <= 0:
            return True

        if len(pcm) > remaining:
            self.buffer.extend(
                pcm[:remaining]
            )
        else:
            self.buffer.extend(
                pcm
            )

        return True

    def duration(self) -> float:
        return pcm_duration(
            bytes(self.buffer),
            PCM_SAMPLE_RATE,
            PCM_CHANNELS,
            PCM_SAMPLE_WIDTH,
        )

    def silence_duration(self) -> float:
        if self.last_audio_at is None:
            return 0.0

        return max(
            0.0,
            time.monotonic()
            - self.last_audio_at,
        )

    def reset(self) -> None:
        self.buffer.clear()
        self.started_at = None
        self.last_audio_at = None
        self.processing = False


# ============================================================
# VOICE AUDIO SINK
# ============================================================

class VoiceAISink(voice_recv.AudioSink):
    """
    Receives decoded Discord PCM and groups it by user.

    Important:
    The sink itself cannot catch an Opus decoder exception
    that happens inside discord-ext-voice-recv before write().
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

        self.closed = False

        self._monitor_task: Optional[
            asyncio.Task
        ] = None

        self._state_lock = asyncio.Lock()

    def wants_opus(self) -> bool:
        # False = library decodes Opus to PCM for us.
        return False

    def start(self) -> None:
        if self._monitor_task is None:
            self._monitor_task = (
                asyncio.create_task(
                    self._monitor_loop()
                )
            )

        logger.info(
            "Voice receive sink started for guild %s.",
            self.session.guild.id,
        )

    def write(
        self,
        user: discord.Member | discord.User | None,
        data,
    ) -> None:

        if self.closed:
            return

        if user is None:
            return

        if user.id == self.session.bot_user_id:
            return

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

        # Keep the real-time callback lightweight.
        asyncio.create_task(
            self._handle_audio(
                user,
                pcm,
            )
        )

    async def _handle_audio(
        self,
        user: discord.Member | discord.User,
        pcm: bytes,
    ) -> None:

        if self.closed:
            return

        async with self._state_lock:

            state = self.users.get(
                user.id
            )

            if state is None:
                state = UserAudioState(
                    user_id=user.id
                )

                self.users[
                    user.id
                ] = state

            if state.processing:
                return

            state.add_audio(
                pcm
            )

    async def _monitor_loop(
        self,
    ) -> None:

        try:
            while not self.closed:

                await asyncio.sleep(
                    0.10
                )

                await self._check_users()

        except asyncio.CancelledError:
            return

        except Exception:
            logger.exception(
                "Voice sink monitor crashed."
            )

    async def _check_users(
        self,
    ) -> None:

        now = time.monotonic()

        candidates: list[
            tuple[
                discord.Member,
                bytes,
            ]
        ] = []

        async with self._state_lock:

            for user_id, state in list(
                self.users.items()
            ):

                if state.processing:
                    continue

                if not state.buffer:
                    continue

                duration = state.duration()

                silence = (
                    now
                    - (
                        state.last_audio_at
                        or now
                    )
                )

                should_flush = (
                    duration >= MAX_AUDIO_LENGTH
                    or (
                        duration >= MIN_AUDIO_LENGTH
                        and silence >= SILENCE_TIMEOUT
                    )
                )

                if not should_flush:
                    continue

                audio = bytes(
                    state.buffer
                )

                state.reset()

                member = (
                    self.session.guild.get_member(
                        user_id
                    )
                )

                if member is None:
                    continue

                candidates.append(
                    (
                        member,
                        audio,
                    )
                )

        for member, audio in candidates:

            asyncio.create_task(
                self.session.process_audio(
                    member,
                    audio,
                )
            )

    def cleanup(self) -> None:

        if self.closed:
            return

        self.closed = True

        if self._monitor_task is not None:
            self._monitor_task.cancel()
            self._monitor_task = None

        for state in self.users.values():
            state.reset()

        self.users.clear()

        logger.info(
            "Voice receive sink cleaned up."
        )


# ============================================================
# VOICE SESSION
# ============================================================

class VoiceSession:
    """
    One active Voice AI session per Discord guild.
    """

    def __init__(
        self,
        bot: commands.Bot,
        guild: discord.Guild,
        voice_client: voice_recv.VoiceRecvClient,
        channel: discord.VoiceChannel,
        voice: str = DEFAULT_GEMINI_VOICE,
    ) -> None:

        self.bot = bot
        self.guild = guild
        self.voice_client = voice_client
        self.channel = channel

        self.voice = normalize_voice_name(
            voice
        )

        if self.voice not in GEMINI_VOICES:
            self.voice = normalize_voice_name(
                DEFAULT_GEMINI_VOICE
            )

        self.engine = GeminiEngine()

        self.memory: list[
            dict[str, str]
        ] = []

        self.started_at = time.monotonic()

        self.processed_requests = 0
        self.failed_requests = 0
        self.received_audio = 0
        self.played_responses = 0

        self.processing_users: set[int] = set()

        self.request_semaphore = asyncio.Semaphore(
            max(
                1,
                MAX_CONCURRENT_AI_REQUESTS,
            )
        )

        self.sink: Optional[
            VoiceAISink
        ] = None

        self._closed = False
        self._play_lock = asyncio.Lock()
        self._session_lock = asyncio.Lock()

    @property
    def bot_user_id(self) -> int:
        if self.bot.user is None:
            return 0

        return self.bot.user.id

    @property
    def memory_count(self) -> int:
        return len(self.memory)

    @property
    def uptime(self) -> float:
        return max(
            0.0,
            time.monotonic()
            - self.started_at,
        )

    @property
    def is_connected(self) -> bool:
        try:
            connected = (
                self.voice_client.is_connected()
            )
        except Exception:
            connected = False

        return (
            not self._closed
            and connected
        )

    async def start(
        self,
    ) -> None:

        if self._closed:
            raise RuntimeError(
                "Voice session is already closed."
            )

        if self.sink is not None:
            return

        sink = VoiceAISink(
            self
        )

        try:

            self.voice_client.listen(
                sink
            )

            sink.start()

            self.sink = sink

            logger.info(
                "Voice AI listening in %s / %s.",
                self.guild.name,
                self.channel.name,
            )

        except Exception:

            sink.cleanup()

            self.sink = None

            raise

    async def process_audio(
        self,
        member: discord.Member,
        pcm: bytes,
    ) -> None:

        user_id = member.id

        if self._closed:
            return

        if not pcm:
            return

        duration = pcm_duration(
            pcm,
            PCM_SAMPLE_RATE,
            PCM_CHANNELS,
            PCM_SAMPLE_WIDTH,
        )

        if duration < MIN_AUDIO_LENGTH:
            return

        if user_id in self.processing_users:
            return

        self.processing_users.add(
            user_id
        )

        self.received_audio += 1

        try:

            async with self.request_semaphore:

                await self._process_audio_locked(
                    member,
                    pcm,
                )

        except asyncio.CancelledError:
            raise

        except Exception:

            self.failed_requests += 1

            logger.exception(
                "Failed processing audio from %s.",
                member,
            )

        finally:

            self.processing_users.discard(
                user_id
            )

    async def _process_audio_locked(
        self,
        member: discord.Member,
        pcm: bytes,
    ) -> None:

        if self._closed:
            return

        if pcm_is_silent(
            pcm
        ):
            logger.debug(
                "Ignoring silent audio from %s.",
                member,
            )
            return

        wav_data = pcm_to_wav(
            pcm
        )

        if not wav_data:
            return

        username = (
            member.display_name
            or member.name
            or f"User {member.id}"
        )

        result = await self.engine.process_voice(
            audio=wav_data,
            memory=self.memory,
            username=username,
            voice=self.voice,
        )

        if not result:
            return

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

        if transcript:

            self._add_memory(
                "user",
                f"{username}: {transcript}",
            )

            logger.info(
                "[%s] %s: %s",
                self.guild.name,
                username,
                transcript,
            )

        if response_text:

            self._add_memory(
                "assistant",
                response_text,
            )

            logger.info(
                "[%s] AI: %s",
                self.guild.name,
                response_text,
            )

            await self._send_text_log(
                member,
                transcript,
                response_text,
            )

        if audio:
            await self.play_tts(
                audio
            )

        if (
            transcript
            or response_text
            or audio
        ):
            self.processed_requests += 1

    def _add_memory(
        self,
        role: str,
        content: str,
    ) -> None:

        if not content:
            return

        self.memory.append(
            {
                "role": role,
                "content": content,
            }
        )

        if len(self.memory) > MAX_MEMORY_MESSAGES:

            del self.memory[
                : len(self.memory)
                - MAX_MEMORY_MESSAGES
            ]

    def clear_memory(
        self,
    ) -> None:

        self.memory.clear()

        try:
            self.engine.clear_memory()
        except Exception:
            pass

        logger.info(
            "Memory cleared for guild %s.",
            self.guild.id,
        )

    def reset(
        self,
    ) -> None:

        self.clear_memory()

        self.processed_requests = 0
        self.failed_requests = 0
        self.received_audio = 0
        self.played_responses = 0

        self.processing_users.clear()

    def set_voice(
        self,
        voice: str,
    ) -> str:

        normalized = normalize_voice_name(
            voice
        )

        if normalized not in GEMINI_VOICES:
            raise ValueError(
                f"Invalid Gemini voice: {voice}"
            )

        self.voice = normalized

        logger.info(
            "Voice changed to %s in guild %s.",
            normalized,
            self.guild.id,
        )

        return normalized

    async def play_tts(
        self,
        audio: bytes,
    ) -> None:

        if self._closed:
            return

        if not audio:
            return

        playback_pcm = (
            tts_pcm_to_discord_pcm(
                audio
            )
        )

        if not playback_pcm:
            return

        source = discord.PCMAudio(
            io.BytesIO(
                playback_pcm
            )
        )

        async with self._play_lock:

            if self._closed:
                return

            if not self.voice_client.is_connected():
                return

            # Do not overlap an existing response.
            if self.voice_client.is_playing():
                self.voice_client.stop()

            finished = asyncio.Event()

            def after_playback(
                error: Optional[Exception],
            ) -> None:

                if error:
                    logger.error(
                        "Discord playback error: %s",
                        error,
                    )

                try:
                    self.bot.loop.call_soon_threadsafe(
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
                    "Failed to start TTS playback."
                )

                return

            self.played_responses += 1

            try:

                await asyncio.wait_for(
                    finished.wait(),
                    timeout=120.0,
                )

            except asyncio.TimeoutError:

                logger.warning(
                    "TTS playback timed out."
                )

                try:
                    if self.voice_client.is_playing():
                        self.voice_client.stop()
                except Exception:
                    pass

    async def _send_text_log(
        self,
        member: discord.Member,
        transcript: str,
        response: str,
    ) -> None:

        try:

            channel = None

            preferred = (
                self.guild.system_channel
            )

            if (
                preferred is not None
                and isinstance(
                    preferred,
                    discord.TextChannel,
                )
            ):
                channel = preferred

            if channel is None:

                me = self.guild.me

                for candidate in self.guild.text_channels:

                    if me is None:
                        continue

                    permissions = (
                        candidate.permissions_for(
                            me
                        )
                    )

                    if (
                        permissions.send_messages
                        and permissions.embed_links
                    ):

                        channel = candidate
                        break

            if channel is None:
                return

            safe_response = (
                response[:1800]
                if len(response) > 1800
                else response
            )

            embed = discord.Embed(
                description=safe_response,
                color=discord.Color.blurple(),
            )

            embed.set_author(
                name=f"🎙️ {member.display_name}"
            )

            if transcript:

                embed.add_field(
                    name="🗣️ Heard",
                    value=transcript[:1000],
                    inline=False,
                )

            embed.add_field(
                name="🤖 AI",
                value=safe_response,
                inline=False,
            )

            await channel.send(
                embed=embed
            )

        except Exception:

            logger.debug(
                "Could not send voice text log.",
                exc_info=True,
            )

    async def stop(
        self,
    ) -> None:

        async with self._session_lock:

            if self._closed:
                return

            self._closed = True

            if self.sink is not None:

                try:
                    self.sink.cleanup()
                except Exception:
                    logger.exception(
                        "Failed to clean voice sink."
                    )

                try:
                    self.voice_client.stop_listening()
                except Exception:
                    pass

                self.sink = None

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
                    "Failed to disconnect voice client."
                )

            try:
                await self.engine.close()
            except Exception:
                logger.debug(
                    "Gemini engine close failed.",
                    exc_info=True,
                )

            self.memory.clear()
            self.processing_users.clear()

            logger.info(
                "Voice session stopped for guild %s.",
                self.guild.id,
            )


# ============================================================
# VOICE SESSION MANAGER
# ============================================================

class VoiceSessionManager:
    """
    Maintains one VoiceSession per Discord guild.
    """

    def __init__(
        self,
    ) -> None:

        self.sessions: dict[
            int,
            VoiceSession,
        ] = {}

        self._lock = asyncio.Lock()

    @property
    def count(self) -> int:
        return len(self.sessions)

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
                    existing.is_connected
                    and existing.channel.id
                    == channel.id
                ):

                    existing.set_voice(
                        voice
                    )

                    return existing

                await existing.stop()

                self.sessions.pop(
                    guild.id,
                    None,
                )

            voice_client = guild.voice_client

            if voice_client is not None:

                if isinstance(
                    voice_client,
                    voice_recv.VoiceRecvClient,
                ):

                    try:

                        if (
                            voice_client.channel
                            and voice_client.channel.id
                            != channel.id
                        ):

                            await voice_client.move_to(
                                channel
                            )

                    except Exception:

                        logger.exception(
                            "Failed to move existing voice client."
                        )

                else:

                    try:
                        await voice_client.disconnect(
                            force=True
                        )
                    except Exception:
                        pass

                    voice_client = None

            if voice_client is None:

                logger.info(
                    "Connecting to voice channel %s in guild %s.",
                    channel.name,
                    guild.id,
                )

                voice_client = await channel.connect(
                    cls=voice_recv.VoiceRecvClient
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
                    "Discord did not create a VoiceRecvClient."
                )

            bot = guild._state._get_client()

            session = VoiceSession(
                bot=bot,
                guild=guild,
                voice_client=voice_client,
                channel=channel,
                voice=voice,
            )

            try:

                await session.start()

            except Exception:

                try:
                    await session.stop()
                except Exception:
                    pass

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

            try:
                await session.stop()
            except Exception:
                logger.exception(
                    "Failed to stop voice session."
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

            for session in sessions:

                try:
                    await session.stop()
                except Exception:
                    logger.exception(
                        "Failed to stop session during shutdown."
                    )


# ============================================================
# VOICE LIST
# ============================================================

async def send_voice_list(
    target,
) -> None:

    embed = discord.Embed(
        title="🎙️ Gemini Voice List",
        description=(
            f"عدد الأصوات المتاحة: **{len(GEMINI_VOICES)}**\n"
            f"الصوت الافتراضي: **{DEFAULT_GEMINI_VOICE}**"
        ),
        color=discord.Color.blurple(),
    )

    lines: list[str] = []

    default_voice = normalize_voice_name(
        DEFAULT_GEMINI_VOICE
    )

    for index, voice in enumerate(
        GEMINI_VOICES,
        start=1,
    ):

        marker = (
            " ⭐"
            if default_voice == voice
            else ""
        )

        lines.append(
            f"`{index:02d}` • **{voice}**{marker}"
        )

    for chunk_start in range(
        0,
        len(lines),
        15,
    ):

        chunk = lines[
            chunk_start:
            chunk_start + 15
        ]

        embed.add_field(
            name=(
                f"Voices "
                f"{chunk_start + 1}-"
                f"{chunk_start + len(chunk)}"
            ),
            value="\n".join(chunk),
            inline=True,
        )

    if hasattr(target, "followup"):

        await target.followup.send(
            embed=embed,
            ephemeral=True,
        )

    elif hasattr(target, "response"):

        await target.response.send_message(
            embed=embed,
            ephemeral=True,
        )

    else:

        await target.send(
            embed=embed
        )


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "VoiceAISink",
    "VoiceSession",
    "VoiceSessionManager",
    "UserAudioState",
    "pcm_to_wav",
    "tts_pcm_to_discord_pcm",
    "pcm_duration",
    "pcm_is_silent",
    "send_voice_list",
]
