# gemini.py
# ============================================================
# Cloud Voice AI
#
# Pipeline:
# Discord Voice PCM / WAV
#     ↓
# Local faster-whisper STT
#     ↓
# Groq GPT-OSS 120B
#     ↓
# Groq Orpheus Arabic Saudi TTS
#     ↓
# PCM 24kHz Mono
#     ↓
# Discord
#
# STT:
#     faster-whisper (LOCAL)
#
# Chat:
#     Groq GPT-OSS 120B
#
# TTS:
#     Groq Orpheus Arabic Saudi
#
# Gemini / Claude / Piper are completely removed.
# Groq Whisper STT is completely removed.
# ============================================================

from __future__ import annotations

import asyncio
import audioop
import logging
import os
import tempfile
import wave
from collections import deque
from pathlib import Path
from typing import Any, Iterable

from groq import Groq

from config import (
    AI_SYSTEM_PROMPT,
    API_RETRIES,
    API_RETRY_DELAY_SECONDS,
    API_TIMEOUT_SECONDS,
    DEFAULT_GEMINI_VOICE,
    GEMINI_VOICES,
    MEMORY_ENABLED,
    MAX_MEMORY_MESSAGES,
    normalize_speech_speed,
    normalize_voice_name,
)

logger = logging.getLogger(__name__)


# ============================================================
# ENVIRONMENT
# ============================================================

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY",
    "",
).strip()


# ============================================================
# MODELS
# ============================================================

# ------------------------------------------------------------
# Local faster-whisper model.
#
# Recommended for a normal CPU server:
#     small
#
# Other possible values:
#     base
#     medium
#     large-v3
#
# "small" is selected by default because the goal is to keep
# the voice bot reasonably light and responsive.
# ------------------------------------------------------------

FASTER_WHISPER_MODEL = (
    os.getenv(
        "FASTER_WHISPER_MODEL",
        "small",
    ).strip()
    or "small"
)

FASTER_WHISPER_DEVICE = (
    os.getenv(
        "FASTER_WHISPER_DEVICE",
        "cpu",
    ).strip().lower()
    or "cpu"
)

FASTER_WHISPER_COMPUTE_TYPE = (
    os.getenv(
        "FASTER_WHISPER_COMPUTE_TYPE",
        "int8",
    ).strip().lower()
    or "int8"
)

# Number of CPU threads used by CTranslate2.
#
# 0 = let the library decide.
# On shared hosting, a modest number is usually better than
# aggressively consuming every CPU thread.
try:

    FASTER_WHISPER_CPU_THREADS = max(
        int(
            os.getenv(
                "FASTER_WHISPER_CPU_THREADS",
                "4",
            )
        ),
        0,
    )

except (TypeError, ValueError):

    FASTER_WHISPER_CPU_THREADS = 4


# ------------------------------------------------------------
# Beam search.
#
# A moderate beam size keeps recognition quality decent
# without making short voice messages unnecessarily heavy.
# ------------------------------------------------------------

try:

    FASTER_WHISPER_BEAM_SIZE = max(
        int(
            os.getenv(
                "FASTER_WHISPER_BEAM_SIZE",
                "5",
            )
        ),
        1,
    )

except (TypeError, ValueError):

    FASTER_WHISPER_BEAM_SIZE = 5


# ------------------------------------------------------------
# Voice recognition settings.
# ------------------------------------------------------------

FASTER_WHISPER_LANGUAGE = (
    os.getenv(
        "FASTER_WHISPER_LANGUAGE",
        "ar",
    ).strip().lower()
    or "ar"
)

# Keep VAD enabled so pure silence/noise is less likely to
# become text.
FASTER_WHISPER_VAD = (
    os.getenv(
        "FASTER_WHISPER_VAD",
        "true",
    ).strip().lower()
    not in {
        "0",
        "false",
        "no",
        "off",
    }
)

# 500 ms minimum silence duration for the VAD stage.
try:

    FASTER_WHISPER_MIN_SILENCE_MS = max(
        int(
            os.getenv(
                "FASTER_WHISPER_MIN_SILENCE_MS",
                "500",
            )
        ),
        0,
    )

except (TypeError, ValueError):

    FASTER_WHISPER_MIN_SILENCE_MS = 500


# ------------------------------------------------------------
# Do not condition transcription on previous text.
#
# This is particularly useful in a voice assistant because
# every Discord speech turn should be treated as its own
# independent utterance rather than continuing a hallucinated
# phrase from a previous segment.
# ------------------------------------------------------------

FASTER_WHISPER_CONDITION_ON_PREVIOUS_TEXT = (
    os.getenv(
        "FASTER_WHISPER_CONDITION_ON_PREVIOUS_TEXT",
        "false",
    ).strip().lower()
    not in {
        "0",
        "false",
        "no",
        "off",
    }
)


# ------------------------------------------------------------
# Groq chat model.
# ------------------------------------------------------------

GROQ_CHAT_MODEL = (
    os.getenv(
        "GROQ_CHAT_MODEL",
        "openai/gpt-oss-120b",
    ).strip()
    or "openai/gpt-oss-120b"
)


# ------------------------------------------------------------
# Groq TTS model.
# ------------------------------------------------------------

GROQ_TTS_MODEL = (
    os.getenv(
        "GROQ_TTS_MODEL",
        "canopylabs/orpheus-arabic-saudi",
    ).strip()
    or "canopylabs/orpheus-arabic-saudi"
)

GROQ_TTS_VOICE = (
    os.getenv(
        "GROQ_TTS_VOICE",
        "fahad",
    ).strip().lower()
    or "fahad"
)

GROQ_REASONING_EFFORT = (
    os.getenv(
        "GROQ_REASONING_EFFORT",
        "low",
    ).strip().lower()
    or "low"
)


# ============================================================
# VOICE MAP
# ============================================================

GROQ_TTS_VOICE_MAP: dict[str, str] = {
    "Kore": "fahad",
    "Puck": "abdullah",
    "Charon": "sultan",
    "Fenrir": "abdullah",
    "Aoede": "noura",
    "Leda": "lulwa",
    "Orus": "sultan",
    "Zephyr": "aisha",
}

VALID_GROQ_TTS_VOICES = {
    "abdullah",
    "fahad",
    "sultan",
    "lulwa",
    "noura",
    "aisha",
}


# ============================================================
# AUDIO
# ============================================================

DEFAULT_AUDIO_MIME_TYPE = "audio/wav"

TTS_OUTPUT_SAMPLE_RATE = 24000
TTS_OUTPUT_CHANNELS = 1
TTS_OUTPUT_SAMPLE_WIDTH = 2

MAX_TRANSCRIPT_LENGTH = 2500
MAX_RESPONSE_LENGTH = 1800
MAX_TTS_CHARS = 200

VOICE_MEMORY_LIMIT = min(
    max(int(MAX_MEMORY_MESSAGES), 0),
    8,
)

DEFAULT_SPEECH_SPEED = 1.0


# ============================================================
# TEXT HELPERS
# ============================================================

def _clean_text(
    value: Any,
) -> str:

    if value is None:
        return ""

    return str(value).strip()


def _normalize_transcript(
    text: str,
) -> str:
    """
    Minimal transcript cleanup.

    IMPORTANT:
    This is deliberately NOT an STT correction engine.

    faster-whisper is now responsible for transcription.
    We only clean surrounding whitespace/control artifacts.
    """

    text = _clean_text(
        text
    )

    if not text:
        return ""

    # Remove zero-width/control formatting characters commonly
    # introduced by text serialization.
    for character in (
        "\u200b",
        "\u200c",
        "\u200d",
        "\ufeff",
    ):

        text = text.replace(
            character,
            "",
        )

    # Normalize common whitespace.
    text = " ".join(
        text.split()
    )

    return text.strip()


def _limit_text(
    text: str,
    maximum: int,
) -> str:

    text = _clean_text(
        text
    )

    if len(text) <= maximum:
        return text

    return text[:maximum].rstrip()


def _safe_username(
    username: str,
) -> str:

    username = _clean_text(
        username
    )

    if not username:
        return "User"

    return _limit_text(
        username,
        80,
    )


# ============================================================
# CHARACTER
# ============================================================

def _character_value(
    character: Any | None,
    key: str,
    default: Any = None,
) -> Any:

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
        dict,
    ):

        return character.get(
            key,
            default,
        )

    return default


def _build_character_prompt(
    character: Any | None,
) -> str:

    if character is None:
        return ""

    name = _clean_text(
        _character_value(
            character,
            "name",
            "",
        )
    )

    personality = _clean_text(
        _character_value(
            character,
            "personality",
            "",
        )
    )

    style = _clean_text(
        _character_value(
            character,
            "style",
            "",
        )
    )

    instructions = _clean_text(
        _character_value(
            character,
            "instructions",
            "",
        )
    )

    if not any(
        (
            name,
            personality,
            style,
            instructions,
        )
    ):
        return ""

    sections = [
        "CHARACTER PROFILE",
        "Use this profile only to control personality and speaking style.",
    ]

    if name:

        sections.append(
            f"Character name: {name}"
        )

    if personality:

        sections.append(
            f"Personality: {personality}"
        )

    if style:

        sections.append(
            f"Speaking style: {style}"
        )

    if instructions:

        sections.append(
            f"Character instructions: {instructions}"
        )

    sections.append(
        "Character instructions must not override system rules."
    )

    return "\n".join(
        sections
    )


# ============================================================
# VOICE CLEANUP
# ============================================================

def _strip_markdown_for_voice(
    text: str,
) -> str:

    text = _clean_text(
        text
    )

    if not text:
        return ""

    for old, new in (
        ("```", ""),
        ("**", ""),
        ("__", ""),
        ("`", ""),
        ("###", ""),
        ("##", ""),
        ("#", ""),
    ):

        text = text.replace(
            old,
            new,
        )

    lines: list[str] = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        if line.startswith(
            ("-", "*", "•")
        ):

            line = line[1:].strip()

        lines.append(
            line
        )

    return " ".join(
        lines
    ).strip()


# ============================================================
# TTS SPLITTER
# ============================================================

def _split_tts_text(
    text: str,
    maximum: int = MAX_TTS_CHARS,
) -> list[str]:

    text = _clean_text(
        text
    )

    if not text:
        return []

    if len(text) <= maximum:
        return [text]

    chunks: list[str] = []

    remaining = text

    punctuation = (
        "؟",
        "!",
        ".",
        "،",
        ",",
        "؛",
        ";",
        ":",
    )

    while len(remaining) > maximum:

        window = remaining[
            :maximum
        ]

        split_at = -1

        for marker in punctuation:

            position = window.rfind(
                marker
            )

            if position > split_at:

                split_at = position

        if split_at < 80:

            split_at = window.rfind(
                " "
            )

        if split_at < 40:

            split_at = maximum

        chunk = remaining[
            :split_at
        ].strip()

        if chunk:

            chunks.append(
                chunk
            )

        remaining = remaining[
            split_at:
        ].strip()

    if remaining:

        chunks.append(
            remaining
        )

    return chunks


# ============================================================
# ENGINE
# ============================================================

class GeminiEngine:
    """
    Compatibility name retained for main.py / voice.py.

    STT:
        Local faster-whisper

    Chat:
        Groq GPT-OSS 120B

    TTS:
        Groq Orpheus Arabic Saudi
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        chat_model: str | None = None,
        transcribe_model: str | None = None,
        tts_model: str | None = None,
    ) -> None:

        # ----------------------------------------------------
        # GROQ
        # ----------------------------------------------------

        self.groq_api_key = (
            api_key or GROQ_API_KEY
        ).strip()

        if not self.groq_api_key:

            raise RuntimeError(
                "GROQ_API_KEY is missing."
            )

        self.chat_model = (
            chat_model
            or GROQ_CHAT_MODEL
        )

        self.tts_model = (
            tts_model
            or GROQ_TTS_MODEL
        )

        # ----------------------------------------------------
        # STT MODEL
        #
        # Keep the old argument for compatibility with
        # main.py / voice.py, but the actual STT backend is now
        # faster-whisper.
        # ----------------------------------------------------

        self.transcribe_model = (
            transcribe_model
            or FASTER_WHISPER_MODEL
        )

        # ----------------------------------------------------
        # GROQ CLIENT
        # ----------------------------------------------------

        self.groq = Groq(
            api_key=self.groq_api_key
        )

        # ----------------------------------------------------
        # LOCAL WHISPER MODEL
        #
        # We load it lazily so the Discord bot can initialize
        # without blocking startup for model download/loading.
        # ----------------------------------------------------

        self._whisper_model = None
        self._whisper_lock = (
            asyncio.Lock()
        )

        # ----------------------------------------------------
        # VOICE
        # ----------------------------------------------------

        self.current_voice = (
            normalize_voice_name(
                DEFAULT_GEMINI_VOICE
            )
        )

        # ----------------------------------------------------
        # MEMORY
        # ----------------------------------------------------

        self._memory: deque[
            dict[str, str]
        ] = deque(
            maxlen=VOICE_MEMORY_LIMIT
        )

        # ----------------------------------------------------
        # STATS
        # ----------------------------------------------------

        self.processed_requests = 0
        self.failed_requests = 0
        self.quota_exhausted = False

    # ========================================================
    # WHISPER MODEL
    # ========================================================

    async def _get_whisper_model(self):
        """
        Lazily load faster-whisper.

        The first voice message may take longer because the
        model can need to download and initialize.
        """

        if self._whisper_model is not None:

            return self._whisper_model

        async with self._whisper_lock:

            if self._whisper_model is not None:

                return self._whisper_model

            logger.info(
                "Loading local faster-whisper | "
                "model=%s | device=%s | compute_type=%s | "
                "cpu_threads=%s",
                self.transcribe_model,
                FASTER_WHISPER_DEVICE,
                FASTER_WHISPER_COMPUTE_TYPE,
                FASTER_WHISPER_CPU_THREADS,
            )

            try:

                from faster_whisper import (
                    WhisperModel,
                )

            except ImportError as error:

                raise RuntimeError(
                    "faster-whisper is not installed. "
                    "Add 'faster-whisper' to requirements.txt."
                ) from error

            def load_model():
                return WhisperModel(
                    self.transcribe_model,
                    device=FASTER_WHISPER_DEVICE,
                    compute_type=FASTER_WHISPER_COMPUTE_TYPE,
                    cpu_threads=(
                        FASTER_WHISPER_CPU_THREADS
                    ),
                    num_workers=1,
                )

            try:

                self._whisper_model = (
                    await asyncio.to_thread(
                        load_model
                    )
                )

            except Exception as error:

                logger.exception(
                    "Failed to load faster-whisper model"
                )

                raise RuntimeError(
                    "Could not load faster-whisper model "
                    f"'{self.transcribe_model}': {error}"
                ) from error

            logger.info(
                "Local faster-whisper loaded | "
                "model=%s | device=%s | compute_type=%s",
                self.transcribe_model,
                FASTER_WHISPER_DEVICE,
                FASTER_WHISPER_COMPUTE_TYPE,
            )

            return self._whisper_model

    # ========================================================
    # VOICE
    # ========================================================

    @property
    def voice(
        self,
    ) -> str:

        return self.current_voice

    def set_voice(
        self,
        voice: str,
    ) -> str:

        normalized = (
            normalize_voice_name(
                voice
            )
        )

        if normalized not in GEMINI_VOICES:

            raise ValueError(
                f"Invalid voice: {voice}"
            )

        self.current_voice = (
            normalized
        )

        return normalized

    def _get_tts_voice(
        self,
    ) -> str:

        mapped = (
            GROQ_TTS_VOICE_MAP.get(
                self.current_voice
            )
        )

        if mapped in VALID_GROQ_TTS_VOICES:

            return mapped

        if (
            GROQ_TTS_VOICE
            in VALID_GROQ_TTS_VOICES
        ):

            return GROQ_TTS_VOICE

        return "fahad"

    # ========================================================
    # ERROR HELPERS
    # ========================================================

    @staticmethod
    def _error_text(
        error: Exception,
    ) -> str:

        try:

            return str(
                error
            ).lower()

        except Exception:

            return ""

    @classmethod
    def _is_quota_error(
        cls,
        error: Exception,
    ) -> bool:

        text = cls._error_text(
            error
        )

        return any(
            marker in text
            for marker in (
                "rate limit",
                "rate_limit",
                "quota",
                "too many requests",
                "429",
            )
        )

    @classmethod
    def _is_retryable_error(
        cls,
        error: Exception,
    ) -> bool:

        if cls._is_quota_error(
            error
        ):
            return False

        text = cls._error_text(
            error
        )

        return any(
            marker in text
            for marker in (
                "timeout",
                "timed out",
                "temporarily unavailable",
                "service unavailable",
                "internal server error",
                "500",
                "502",
                "503",
                "504",
            )
        )

    async def _with_retry(
        self,
        operation,
        *,
        operation_name: str,
    ):

        last_error: Exception | None = None

        attempts = max(
            0,
            int(API_RETRIES),
        )

        for attempt in range(
            attempts + 1
        ):

            try:

                return await asyncio.wait_for(
                    asyncio.to_thread(
                        operation
                    ),
                    timeout=API_TIMEOUT_SECONDS,
                )

            except Exception as error:

                last_error = error

                if self._is_quota_error(
                    error
                ):

                    self.quota_exhausted = True

                    logger.error(
                        "%s stopped: rate limit/quota.",
                        operation_name,
                    )

                    raise

                if (
                    attempt >= attempts
                    or not self._is_retryable_error(
                        error
                    )
                ):

                    break

                delay = (
                    float(
                        API_RETRY_DELAY_SECONDS
                    )
                    * float(
                        attempt + 1
                    )
                )

                logger.warning(
                    "%s failed; retrying in %.1fs: %s",
                    operation_name,
                    delay,
                    error,
                )

                await asyncio.sleep(
                    delay
                )

        self.failed_requests += 1

        if last_error:

            raise last_error

        raise RuntimeError(
            f"{operation_name} failed."
        )

    # ========================================================
    # AUDIO PREPARATION
    # ========================================================

    @staticmethod
    def _ensure_wav_file(
        audio: bytes,
    ) -> str:

        if not audio:

            raise ValueError(
                "Empty audio."
            )

        temp_path: str | None = None

        try:

            # ------------------------------------------------
            # Already WAV
            # ------------------------------------------------

            if audio.startswith(
                b"RIFF"
            ):

                with tempfile.NamedTemporaryFile(
                    suffix=".wav",
                    delete=False,
                ) as temp_file:

                    temp_path = (
                        temp_file.name
                    )

                    temp_file.write(
                        audio
                    )

                return temp_path

            # ------------------------------------------------
            # Raw Discord PCM:
            #
            # 48kHz
            # stereo
            # 16-bit
            #
            # ↓
            #
            # 16kHz
            # mono
            # 16-bit
            # ------------------------------------------------

            pcm = audio

            pcm = audioop.tomono(
                pcm,
                2,
                0.5,
                0.5,
            )

            pcm, _ = audioop.ratecv(
                pcm,
                2,
                1,
                48000,
                16000,
                None,
            )

            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False,
            ) as temp_file:

                temp_path = (
                    temp_file.name
                )

            with wave.open(
                temp_path,
                "wb",
            ) as wav_file:

                wav_file.setnchannels(
                    1
                )

                wav_file.setsampwidth(
                    2
                )

                wav_file.setframerate(
                    16000
                )

                wav_file.writeframes(
                    pcm
                )

            return temp_path

        except Exception:

            if temp_path:

                try:

                    Path(
                        temp_path
                    ).unlink(
                        missing_ok=True
                    )

                except Exception:

                    pass

            raise

    # ========================================================
    # LOCAL FASTER-WHISPER STT
    # ========================================================

    def _transcribe_local_sync(
        self,
        model,
        wav_path: str,
    ) -> str:

        # ----------------------------------------------------
        # VAD SETTINGS
        # ----------------------------------------------------

        vad_parameters = None

        if FASTER_WHISPER_VAD:

            vad_parameters = {
                "min_silence_duration_ms": (
                    FASTER_WHISPER_MIN_SILENCE_MS
                ),
            }

        # ----------------------------------------------------
        # TRANSCRIBE
        # ----------------------------------------------------

        kwargs: dict[str, Any] = {
            "language": FASTER_WHISPER_LANGUAGE,
            "beam_size": FASTER_WHISPER_BEAM_SIZE,
            "condition_on_previous_text": (
                FASTER_WHISPER_CONDITION_ON_PREVIOUS_TEXT
            ),
            "vad_filter": (
                FASTER_WHISPER_VAD
            ),
            "temperature": 0.0,
        }

        if vad_parameters is not None:

            kwargs[
                "vad_parameters"
            ] = vad_parameters

        segments, info = (
            model.transcribe(
                wav_path,
                **kwargs,
            )
        )

        # faster-whisper returns a generator.
        # We MUST iterate it for actual inference to happen.
        segment_list = list(
            segments
        )

        parts: list[str] = []

        for segment in segment_list:

            segment_text = (
                _normalize_transcript(
                    getattr(
                        segment,
                        "text",
                        "",
                    )
                )
            )

            if not segment_text:
                continue

            parts.append(
                segment_text
            )

        transcript = _normalize_transcript(
            " ".join(parts)
        )

        # ----------------------------------------------------
        # LANGUAGE INFO
        # ----------------------------------------------------

        detected_language = getattr(
            info,
            "language",
            None,
        )

        try:

            language_probability = float(
                getattr(
                    info,
                    "language_probability",
                    0.0,
                )
                or 0.0
            )

        except Exception:

            language_probability = 0.0

        logger.info(
            "faster-whisper STT | "
            "model=%s | detected_language=%s | "
            "language_probability=%.3f | "
            "segments=%d | transcript=%r",
            self.transcribe_model,
            detected_language or FASTER_WHISPER_LANGUAGE,
            language_probability,
            len(segment_list),
            transcript,
        )

        return transcript

    async def transcribe(
        self,
        audio: bytes,
        *,
        mime_type: str = DEFAULT_AUDIO_MIME_TYPE,
    ) -> str:

        if not audio:
            return ""

        _ = mime_type

        wav_path = (
            self._ensure_wav_file(
                audio
            )
        )

        try:

            model = (
                await self._get_whisper_model()
            )

            transcript = await asyncio.wait_for(
                asyncio.to_thread(
                    self._transcribe_local_sync,
                    model,
                    wav_path,
                ),
                timeout=API_TIMEOUT_SECONDS,
            )

            transcript = _limit_text(
                transcript,
                MAX_TRANSCRIPT_LENGTH,
            )

            if not transcript:

                logger.warning(
                    "faster-whisper returned no transcript."
                )

                return ""

            logger.info(
                "Local STT transcript | transcript=%r",
                transcript,
            )

            return transcript

        finally:

            try:

                Path(
                    wav_path
                ).unlink(
                    missing_ok=True
                )

            except Exception:

                logger.warning(
                    "Failed to remove STT temp file: %s",
                    wav_path,
                )

    # ========================================================
    # MEMORY
    # ========================================================

    def add_memory(
        self,
        role: str,
        content: str,
    ) -> None:

        if not MEMORY_ENABLED:
            return

        if VOICE_MEMORY_LIMIT <= 0:
            return

        content = _clean_text(
            content
        )

        if not content:
            return

        if role not in {
            "user",
            "assistant",
        }:

            role = "user"

        self._memory.append(
            {
                "role": role,
                "content": content,
            }
        )

    def add_user_message(
        self,
        content: str,
    ) -> None:

        self.add_memory(
            "user",
            content,
        )

    def add_assistant_message(
        self,
        content: str,
    ) -> None:

        self.add_memory(
            "assistant",
            content,
        )

    def clear_memory(
        self,
    ) -> None:

        self._memory.clear()

    def reset_memory(
        self,
    ) -> None:

        self.clear_memory()

    def get_memory(
        self,
    ) -> list[dict[str, str]]:

        return [
            dict(item)
            for item in self._memory
        ]

    @property
    def memory(
        self,
    ) -> list[dict[str, str]]:

        return self.get_memory()

    def memory_size(
        self,
    ) -> int:

        return len(
            self._memory
        )

    # ========================================================
    # GROQ CHAT
    # ========================================================

    def _build_chat_messages(
        self,
        text: str,
        username: str,
        memory: Iterable[
            dict[str, str]
        ] | None = None,
    ) -> list[dict[str, str]]:

        messages: list[
            dict[str, str]
        ] = []

        source_memory = (
            list(memory)
            if memory is not None
            else self.get_memory()
        )

        for item in source_memory:

            role = item.get(
                "role",
                "user",
            )

            content = _clean_text(
                item.get(
                    "content",
                    "",
                )
            )

            if not content:
                continue

            if role not in {
                "user",
                "assistant",
            }:

                role = "user"

            messages.append(
                {
                    "role": role,
                    "content": _limit_text(
                        content,
                        800,
                    ),
                }
            )

        messages.append(
            {
                "role": "user",
                "content": (
                    f"{_safe_username(username)} said:\n"
                    f"{_limit_text(text, MAX_TRANSCRIPT_LENGTH)}"
                ),
            }
        )

        return messages

    def _generate_response_sync(
        self,
        messages: list[
            dict[str, str]
        ],
        system_prompt: str,
    ):

        kwargs: dict[str, Any] = {
            "model": self.chat_model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                *messages,
            ],
            "max_completion_tokens": min(
                int(
                    os.getenv(
                        "GROQ_MAX_OUTPUT_TOKENS",
                        "350",
                    )
                ),
                1000,
            ),
            "temperature": float(
                os.getenv(
                    "GROQ_TEMPERATURE",
                    "0.65",
                )
            ),
        }

        if self.chat_model.startswith(
            "openai/gpt-oss"
        ):

            kwargs[
                "reasoning_effort"
            ] = GROQ_REASONING_EFFORT

            kwargs[
                "include_reasoning"
            ] = False

        return self.groq.chat.completions.create(
            **kwargs
        )

    async def generate_response(
        self,
        text: str,
        *,
        username: str = "User",
        memory: Iterable[
            dict[str, str]
        ] | None = None,
        character: Any | None = None,
        system_prompt: str | None = None,
    ) -> str:

        text = _clean_text(
            text
        )

        if not text:
            return ""

        base_prompt = (
            system_prompt
            if system_prompt is not None
            else AI_SYSTEM_PROMPT
        )

        character_prompt = (
            _build_character_prompt(
                character
            )
        )

        final_system_prompt = (
            f"{base_prompt}\n\n"
            f"{character_prompt}\n\n"
            "VOICE MODE:\n"
            "You are speaking inside a Discord voice channel.\n"
            "Answer naturally and directly.\n"
            "Usually answer in 1-3 short sentences.\n"
            "Match the user's language.\n"
            "If the user speaks Arabic, answer in Arabic.\n"
            "Use Saudi/Gulf Arabic naturally when appropriate.\n"
            "Do not use markdown.\n"
            "Do not use code blocks.\n"
            "Do not invent what the user said.\n"
            "Return only the final answer intended for the user.\n"
            "Do not output analysis, reasoning, or hidden thoughts."
        )

        messages = self._build_chat_messages(
            text,
            username,
            memory,
        )

        response = await self._with_retry(
            lambda: self._generate_response_sync(
                messages,
                final_system_prompt,
            ),
            operation_name="Groq Chat",
        )

        answer = ""

        try:

            if (
                response.choices
                and response.choices[0].message
            ):

                answer = _clean_text(
                    getattr(
                        response.choices[0].message,
                        "content",
                        "",
                    )
                )

        except Exception:

            logger.exception(
                "Failed to extract Groq response"
            )

        answer = _limit_text(
            answer,
            MAX_RESPONSE_LENGTH,
        )

        answer = _strip_markdown_for_voice(
            answer
        )

        logger.info(
            "AI response | model=%s | response=%s",
            self.chat_model,
            answer or "<empty>",
        )

        return answer

    # ========================================================
    # GROQ TTS
    # ========================================================

    def _generate_tts_chunk_sync(
        self,
        text: str,
        voice: str,
        speed: float,
    ) -> bytes:

        text = _clean_text(
            text
        )

        if not text:
            return b""

        text = text[
            :MAX_TTS_CHARS
        ]

        response = (
            self.groq.audio.speech.create(
                model=self.tts_model,
                voice=voice,
                input=text,
                response_format="wav",
            )
        )

        # Current Groq TTS endpoint does not
        # use this value directly.
        _ = speed

        temp_path: str | None = None

        try:

            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False,
            ) as temp_file:

                temp_path = (
                    temp_file.name
                )

            response.write_to_file(
                temp_path
            )

            with wave.open(
                temp_path,
                "rb",
            ) as wav_file:

                channels = (
                    wav_file.getnchannels()
                )

                sample_width = (
                    wav_file.getsampwidth()
                )

                sample_rate = (
                    wav_file.getframerate()
                )

                frame_count = (
                    wav_file.getnframes()
                )

                if frame_count <= 0:
                    return b""

                pcm = wav_file.readframes(
                    frame_count
                )

            if sample_width != 2:

                pcm = audioop.lin2lin(
                    pcm,
                    sample_width,
                    2,
                )

            if channels > 1:

                if channels == 2:

                    pcm = audioop.tomono(
                        pcm,
                        2,
                        0.5,
                        0.5,
                    )

                else:

                    pcm = audioop.tomono(
                        pcm,
                        2,
                        1.0,
                        0.0,
                    )

            if (
                sample_rate
                != TTS_OUTPUT_SAMPLE_RATE
            ):

                pcm, _ = audioop.ratecv(
                    pcm,
                    2,
                    1,
                    sample_rate,
                    TTS_OUTPUT_SAMPLE_RATE,
                    None,
                )

            return pcm

        finally:

            if temp_path:

                try:

                    Path(
                        temp_path
                    ).unlink(
                        missing_ok=True
                    )

                except Exception:

                    logger.warning(
                        "Failed to remove TTS temp file: %s",
                        temp_path,
                    )

    async def generate_speech(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float = DEFAULT_SPEECH_SPEED,
    ) -> bytes:

        text = _clean_text(
            text
        )

        if not text:
            return b""

        selected_voice = (
            self.set_voice(
                voice
            )
            if voice
            else self.voice
        )

        selected_speed = (
            normalize_speech_speed(
                speed
            )
        )

        groq_voice = (
            self._get_tts_voice()
        )

        logger.info(
            "Groq TTS | model=%s | "
            "voice=%s | speed=%.2f",
            self.tts_model,
            groq_voice,
            selected_speed,
        )

        chunks = _split_tts_text(
            text,
            MAX_TTS_CHARS,
        )

        if not chunks:
            return b""

        output_parts: list[
            bytes
        ] = []

        started = (
            asyncio.get_running_loop().time()
        )

        for index, chunk in enumerate(
            chunks,
            start=1,
        ):

            logger.info(
                "Groq TTS chunk %d/%d | chars=%d",
                index,
                len(chunks),
                len(chunk),
            )

            audio = await self._with_retry(
                lambda chunk=chunk:
                    self._generate_tts_chunk_sync(
                        chunk,
                        groq_voice,
                        selected_speed,
                    ),
                operation_name=(
                    f"Groq TTS chunk "
                    f"{index}/{len(chunks)}"
                ),
            )

            if audio:

                output_parts.append(
                    audio
                )

        final_audio = b"".join(
            output_parts
        )

        elapsed = (
            asyncio.get_running_loop().time()
            - started
        )

        logger.info(
            "Groq TTS generated | "
            "chunks=%d | bytes=%d | %.2fs",
            len(chunks),
            len(final_audio),
            elapsed,
        )

        return final_audio

    # ========================================================
    # COMPLETE PIPELINE
    # ========================================================

    async def process_voice(
        self,
        audio: bytes,
        *,
        username: str = "User",
        voice: str | None = None,
        speed: float = DEFAULT_SPEECH_SPEED,
        character: Any | None = None,
        system_prompt: str | None = None,
        mime_type: str = DEFAULT_AUDIO_MIME_TYPE,
        memory: Iterable[
            dict[str, str]
        ] | None = None,
    ) -> dict[str, Any]:

        result: dict[str, Any] = {
            "success": False,
            "transcript": "",
            "response": "",
            "audio": b"",
            "voice": (
                normalize_voice_name(
                    voice
                )
                if voice
                else self.voice
            ),
            "speed": (
                normalize_speech_speed(
                    speed
                )
            ),
            "character": _character_value(
                character,
                "name",
                None,
            ),
            "tts_model": self.tts_model,
            "stt_model": self.transcribe_model,
            "error": None,
        }

        if not audio:

            result["error"] = (
                "No audio received."
            )

            return result

        try:

            # ------------------------------------------------
            # STT
            # ------------------------------------------------

            transcript = await self.transcribe(
                audio,
                mime_type=mime_type,
            )

            if not transcript:

                result["error"] = (
                    "Speech rejected by local STT."
                )

                return result

            result["transcript"] = (
                transcript
            )

            # ------------------------------------------------
            # CHAT
            # ------------------------------------------------

            response = (
                await self.generate_response(
                    transcript,
                    username=username,
                    memory=memory,
                    character=character,
                    system_prompt=system_prompt,
                )
            )

            if not response:

                result["error"] = (
                    "Groq returned an empty response."
                )

                return result

            result["response"] = (
                response
            )

            # ------------------------------------------------
            # MEMORY
            # ------------------------------------------------

            self.add_user_message(
                f"{_safe_username(username)}: "
                f"{transcript}"
            )

            self.add_assistant_message(
                response
            )

            # ------------------------------------------------
            # TTS
            # ------------------------------------------------

            selected_voice = (
                self.set_voice(
                    voice
                )
                if voice
                else self.voice
            )

            result["voice"] = (
                selected_voice
            )

            speech = (
                await self.generate_speech(
                    response,
                    voice=selected_voice,
                    speed=speed,
                )
            )

            if not speech:

                result["error"] = (
                    "Groq TTS returned no audio."
                )

                return result

            result["audio"] = speech
            result["success"] = True

            self.processed_requests += 1

            logger.info(
                "Voice pipeline completed | "
                "user=%s | stt=%s | chat=%s | "
                "tts=%s | voice=%s",
                username,
                self.transcribe_model,
                self.chat_model,
                self.tts_model,
                self._get_tts_voice(),
            )

            return result

        except Exception as error:

            self.failed_requests += 1

            result["error"] = str(
                error
            )

            if self._is_quota_error(
                error
            ):

                self.quota_exhausted = True

                logger.error(
                    "API rate limit/quota reached."
                )

            else:

                logger.exception(
                    "Voice pipeline failed"
                )

            return result

    # ========================================================
    # STATS
    # ========================================================

    def get_stats(
        self,
    ) -> dict[str, Any]:

        return {
            "voice": self.voice,
            "memory_size": self.memory_size(),
            "memory_limit": VOICE_MEMORY_LIMIT,
            "memory_enabled": MEMORY_ENABLED,
            "processed_requests": self.processed_requests,
            "failed_requests": self.failed_requests,
            "quota_exhausted": self.quota_exhausted,

            # ------------------------------------------------
            # CHAT
            # ------------------------------------------------

            "chat_provider": "Groq",
            "chat_model": self.chat_model,

            # ------------------------------------------------
            # STT
            # ------------------------------------------------

            "transcribe_provider": "faster-whisper",
            "transcribe_model": self.transcribe_model,
            "stt_language": FASTER_WHISPER_LANGUAGE,
            "stt_device": FASTER_WHISPER_DEVICE,
            "stt_compute_type": (
                FASTER_WHISPER_COMPUTE_TYPE
            ),
            "stt_cpu_threads": (
                FASTER_WHISPER_CPU_THREADS
            ),
            "stt_beam_size": (
                FASTER_WHISPER_BEAM_SIZE
            ),
            "stt_vad": FASTER_WHISPER_VAD,
            "stt_filter": "none",

            # ------------------------------------------------
            # TTS
            # ------------------------------------------------

            "tts_provider": "Groq",
            "tts_model": self.tts_model,
            "tts_local": False,
            "tts_voice": self._get_tts_voice(),
        }

    # ========================================================
    # CLOSE
    # ========================================================

    async def close(
        self,
    ) -> None:

        # faster-whisper model is managed in-process by
        # CTranslate2. There is no network client that needs
        # an explicit async close here.

        self._whisper_model = None
        self.groq = None


# ============================================================
# FACTORY
# ============================================================

def create_gemini_engine() -> GeminiEngine:

    return GeminiEngine()


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "GeminiEngine",
    "create_gemini_engine",
]
