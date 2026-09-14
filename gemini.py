# gemini.py
# ============================================================
# Cloud Voice AI
#
# Pipeline:
# Discord PCM
#     ↓
# Groq Whisper Large V3
#     ↓
# Strong Arabic STT Filter
#     ↓
# Groq GPT-OSS 120B
#     ↓
# Groq Orpheus Arabic Saudi TTS
#     ↓
# PCM 24kHz Mono
#     ↓
# Discord
#
# Gemini removed completely.
# Claude removed completely.
# Piper removed completely.
# ============================================================

from __future__ import annotations

import asyncio
import audioop
import logging
import os
import re
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

GROQ_STT_MODEL = os.getenv(
    "GROQ_STT_MODEL",
    "whisper-large-v3",
).strip() or "whisper-large-v3"

GROQ_CHAT_MODEL = os.getenv(
    "GROQ_CHAT_MODEL",
    "openai/gpt-oss-120b",
).strip() or "openai/gpt-oss-120b"

GROQ_TTS_MODEL = os.getenv(
    "GROQ_TTS_MODEL",
    "canopylabs/orpheus-arabic-saudi",
).strip() or "canopylabs/orpheus-arabic-saudi"

GROQ_TTS_VOICE = os.getenv(
    "GROQ_TTS_VOICE",
    "fahad",
).strip().lower() or "fahad"

GROQ_REASONING_EFFORT = os.getenv(
    "GROQ_REASONING_EFFORT",
    "low",
).strip().lower() or "low"


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
# STRONG ARABIC STT FILTER
# ============================================================

_ARABIC_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]"
)

_LATIN_RE = re.compile(
    r"[A-Za-zÀ-ÖØ-öø-ÿ]"
)

_BAD_FOREIGN_CHARS_RE = re.compile(
    r"[ðþæœÐÞÆŒ]"
)

_WORD_RE = re.compile(
    r"\S+"
)

_REPEAT_PHRASE_RE = re.compile(
    r"\b(.{1,24})\s+\1\s+\1\b",
    re.IGNORECASE,
)

_NOISE_WORDS = {
    "uh",
    "um",
    "umm",
    "hmm",
    "hm",
    "er",
    "eh",
    "mmm",
    "mhm",
}

_KNOWN_HALLUCINATIONS = (
    "thank you for watching",
    "thanks for watching",
    "subscribe",
    "subtitles by",
    "subtitle by",
    "amara",
    "copyright",
    "please subscribe",
)


def _arabic_ratio(text: str) -> float:
    letters = re.findall(
        r"[^\W\d_]",
        text,
        re.UNICODE,
    )

    if not letters:
        return 0.0

    return len(
        _ARABIC_RE.findall(text)
    ) / max(len(letters), 1)


def _latin_ratio(text: str) -> float:
    letters = re.findall(
        r"[^\W\d_]",
        text,
        re.UNICODE,
    )

    if not letters:
        return 0.0

    return len(
        _LATIN_RE.findall(text)
    ) / max(len(letters), 1)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""

    return str(value).strip()


def _normalize_transcript(text: str) -> str:
    text = _clean_text(text)

    if not text:
        return ""

    text = text.replace("\u200b", "")
    text = text.replace("\u200c", "")
    text = text.replace("\u200d", "")

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def _looks_like_bad_transcript(text: str) -> bool:
    text = _normalize_transcript(text)

    if not text:
        return True

    lowered = text.lower()

    if lowered in _NOISE_WORDS:
        return True

    if any(
        phrase in lowered
        for phrase in _KNOWN_HALLUCINATIONS
    ):
        return True

    if _BAD_FOREIGN_CHARS_RE.search(text):
        return True

    arabic_chars = len(
        _ARABIC_RE.findall(text)
    )

    if arabic_chars < 2:
        return True

    arabic_ratio = _arabic_ratio(text)
    latin_ratio = _latin_ratio(text)

    if (
        arabic_ratio < 0.45
        and latin_ratio > 0.35
    ):
        return True

    if _REPEAT_PHRASE_RE.search(text):
        return True

    words = _WORD_RE.findall(text)

    if len(words) >= 5:
        counts: dict[str, int] = {}

        for word in words:
            key = word.strip(
                "،؛,.!?؟:()[]{}\"'`"
            )

            if key:
                counts[key] = (
                    counts.get(key, 0) + 1
                )

        highest = max(
            counts.values(),
            default=0,
        )

        if (
            highest >= 4
            and highest / len(words) >= 0.60
        ):
            return True

    return False


def _filter_transcript(text: str) -> str:
    text = _normalize_transcript(text)

    if _looks_like_bad_transcript(text):
        logger.warning(
            "STT rejected | transcript=%r",
            text,
        )
        return ""

    text = re.sub(
        r"([،؛,.!?؟])\1+",
        r"\1",
        text,
    )

    return text.strip()


# ============================================================
# GENERAL HELPERS
# ============================================================

def _limit_text(
    text: str,
    maximum: int,
) -> str:
    text = _clean_text(text)

    if len(text) <= maximum:
        return text

    return text[:maximum].rstrip()


def _safe_username(
    username: str,
) -> str:
    username = _clean_text(username)

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

    if hasattr(character, key):
        return getattr(
            character,
            key,
            default,
        )

    if isinstance(character, dict):
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

    return "\n".join(sections)


# ============================================================
# VOICE CLEANUP
# ============================================================

def _strip_markdown_for_voice(
    text: str,
) -> str:

    text = _clean_text(text)

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

        lines.append(line)

    return " ".join(lines).strip()


# ============================================================
# TTS SPLITTER
# ============================================================

def _split_tts_text(
    text: str,
    maximum: int = MAX_TTS_CHARS,
) -> list[str]:

    text = _clean_text(text)

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

        window = remaining[:maximum]
        split_at = -1

        for marker in punctuation:
            position = window.rfind(marker)

            if position > split_at:
                split_at = position

        if split_at < 80:
            split_at = window.rfind(" ")

        if split_at < 40:
            split_at = maximum

        chunk = remaining[:split_at].strip()

        if chunk:
            chunks.append(chunk)

        remaining = remaining[split_at:].strip()

    if remaining:
        chunks.append(remaining)

    return chunks


# ============================================================
# ENGINE
# ============================================================

class GeminiEngine:
    """
    Compatibility name retained for main.py / voice.py.

    STT:
        Groq Whisper Large V3

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

        # api_key is retained for compatibility.
        self.groq_api_key = (
            api_key or GROQ_API_KEY
        ).strip()

        if not self.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is missing."
            )

        self.transcribe_model = (
            transcribe_model
            or GROQ_STT_MODEL
        )

        self.chat_model = (
            chat_model
            or GROQ_CHAT_MODEL
        )

        self.tts_model = (
            tts_model
            or GROQ_TTS_MODEL
        )

        self.groq = Groq(
            api_key=self.groq_api_key
        )

        self.current_voice = normalize_voice_name(
            DEFAULT_GEMINI_VOICE
        )

        self._memory: deque[
            dict[str, str]
        ] = deque(
            maxlen=VOICE_MEMORY_LIMIT
        )

        self.processed_requests = 0
        self.failed_requests = 0
        self.quota_exhausted = False

    # ========================================================
    # VOICE
    # ========================================================

    @property
    def voice(self) -> str:
        return self.current_voice

    def set_voice(
        self,
        voice: str,
    ) -> str:

        normalized = normalize_voice_name(
            voice
        )

        if normalized not in GEMINI_VOICES:
            raise ValueError(
                f"Invalid voice: {voice}"
            )

        self.current_voice = normalized

        return normalized

    def _get_tts_voice(self) -> str:

        mapped = GROQ_TTS_VOICE_MAP.get(
            self.current_voice
        )

        if mapped in VALID_GROQ_TTS_VOICES:
            return mapped

        if GROQ_TTS_VOICE in VALID_GROQ_TTS_VOICES:
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
            return str(error).lower()
        except Exception:
            return ""

    @classmethod
    def _is_quota_error(
        cls,
        error: Exception,
    ) -> bool:

        text = cls._error_text(error)

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

        if cls._is_quota_error(error):
            return False

        text = cls._error_text(error)

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
                    asyncio.to_thread(operation),
                    timeout=API_TIMEOUT_SECONDS,
                )

            except Exception as error:

                last_error = error

                if self._is_quota_error(error):

                    self.quota_exhausted = True

                    logger.error(
                        "%s stopped: rate limit/quota.",
                        operation_name,
                    )

                    raise

                if (
                    attempt >= attempts
                    or not self._is_retryable_error(error)
                ):
                    break

                delay = (
                    float(API_RETRY_DELAY_SECONDS)
                    * float(attempt + 1)
                )

                logger.warning(
                    "%s failed; retrying in %.1fs: %s",
                    operation_name,
                    delay,
                    error,
                )

                await asyncio.sleep(delay)

        self.failed_requests += 1

        if last_error:
            raise last_error

        raise RuntimeError(
            f"{operation_name} failed."
        )

    # ========================================================
    # DISCORD PCM -> WAV
    # ========================================================

    @staticmethod
    def _discord_pcm_to_wav(
        audio: bytes,
    ) -> str:

        if not audio:
            raise ValueError(
                "Empty audio."
            )

        # Discord PCM = 48kHz / stereo / signed 16-bit.
        pcm = audio

        # Stereo -> mono.
        pcm = audioop.tomono(
            pcm,
            2,
            0.5,
            0.5,
        )

        # 48kHz -> 16kHz.
        pcm, _ = audioop.ratecv(
            pcm,
            2,
            1,
            48000,
            16000,
            None,
        )

        temp_path: str | None = None

        try:

            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False,
            ) as temp_file:

                temp_path = temp_file.name

            with wave.open(
                temp_path,
                "wb",
            ) as wav_file:

                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(pcm)

            return temp_path

        except Exception:

            if temp_path:
                try:
                    Path(temp_path).unlink(
                        missing_ok=True
                    )
                except Exception:
                    pass

            raise

    # ========================================================
    # GROQ STT
    # ========================================================

    def _transcribe_sync(
        self,
        audio: bytes,
    ):

        wav_path = self._discord_pcm_to_wav(
            audio
        )

        try:

            return self.groq.audio.transcriptions.create(
                file=(
                    "discord_audio.wav",
                    open(wav_path, "rb"),
                ),
                model=self.transcribe_model,
                language="ar",
                temperature=0.0,
                prompt=(
                    "تفريغ كلام عربي باللهجة السعودية "
                    "والعربية العامية. "
                    "اكتب الكلام كما نُطق بالعربية. "
                    "لا تترجم الكلام. "
                    "لا تكتب العربية بأحرف لاتينية. "
                    "لا تخمن كلامًا غير مسموع. "
                    "إذا كان الصوت غير واضح فلا تضف كلامًا من عندك. "
                    "أسماء وكلمات مهمة: "
                    "ديسكورد، دردشة، مكالمة، بوت، "
                    "ذكاء اصطناعي، جيميناي، جروك، "
                    "فويس، السعودية، الأحساء، الهفوف، "
                    "كم عمرك، وش عمرك، من أنت، وش اسمك، "
                    "كيف حالك، وش تسوي، وش الأخبار."
                ),
                response_format="verbose_json",
                timestamp_granularities=[
                    "segment",
                ],
            )

        finally:

            try:
                Path(wav_path).unlink(
                    missing_ok=True
                )
            except Exception:
                pass

    async def transcribe(
        self,
        audio: bytes,
        *,
        mime_type: str = DEFAULT_AUDIO_MIME_TYPE,
    ) -> str:

        if not audio:
            return ""

        _ = mime_type

        response = await self._with_retry(
            lambda: self._transcribe_sync(audio),
            operation_name="Groq STT",
        )

        raw_text = _normalize_transcript(
            getattr(
                response,
                "text",
                "",
            )
        )

        if not raw_text:
            logger.warning(
                "Groq STT returned no transcript."
            )
            return ""

        # Use segment confidence information when available.
        segments = getattr(
            response,
            "segments",
            None,
        ) or []

        valid_segments: list[str] = []

        for segment in segments:

            segment_text = _normalize_transcript(
                getattr(
                    segment,
                    "text",
                    "",
                )
            )

            if not segment_text:
                continue

            no_speech_prob = float(
                getattr(
                    segment,
                    "no_speech_prob",
                    0.0,
                )
                or 0.0
            )

            avg_logprob = float(
                getattr(
                    segment,
                    "avg_logprob",
                    0.0,
                )
                or 0.0
            )

            if no_speech_prob >= 0.75:
                logger.warning(
                    "Rejected STT segment | "
                    "no_speech_prob=%.2f | text=%r",
                    no_speech_prob,
                    segment_text,
                )
                continue

            if avg_logprob < -1.8:
                logger.warning(
                    "Rejected weak STT segment | "
                    "avg_logprob=%.2f | text=%r",
                    avg_logprob,
                    segment_text,
                )
                continue

            valid_segments.append(
                segment_text
            )

        candidate_text = (
            " ".join(valid_segments)
            if valid_segments
            else raw_text
        )

        filtered_text = _filter_transcript(
            candidate_text
        )

        filtered_text = _limit_text(
            filtered_text,
            MAX_TRANSCRIPT_LENGTH,
        )

        logger.info(
            "Groq STT | model=%s | language=ar | "
            "filtered=true | transcript=%s",
            self.transcribe_model,
            filtered_text or "<rejected>",
        )

        return filtered_text

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

        content = _clean_text(content)

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

    def clear_memory(self) -> None:
        self._memory.clear()

    def reset_memory(self) -> None:
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

    def memory_size(self) -> int:
        return len(self._memory)

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
        messages: list[dict[str, str]],
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
            kwargs["reasoning_effort"] = (
                GROQ_REASONING_EFFORT
            )
            kwargs["include_reasoning"] = False

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

        text = _clean_text(text)

        if not text:
            return ""

        base_prompt = (
            system_prompt
            if system_prompt is not None
            else AI_SYSTEM_PROMPT
        )

        character_prompt = _build_character_prompt(
            character
        )

        final_system_prompt = (
            f"{base_prompt}\n\n"
            f"{character_prompt}\n\n"
            "VOICE MODE:\n"
            "You are speaking inside a Discord voice channel.\n"
            "Keep replies concise and natural.\n"
            "Usually answer in 1-3 short sentences.\n"
            "Match the user's language.\n"
            "If the user speaks Arabic, respond in Arabic.\n"
            "Use Saudi/Gulf Arabic naturally when appropriate.\n"
            "Do not use markdown.\n"
            "Do not use code blocks.\n"
            "Do not write long explanations unless explicitly asked.\n"
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

        text = _clean_text(text)

        if not text:
            return b""

        text = text[:MAX_TTS_CHARS]

        response = self.groq.audio.speech.create(
            model=self.tts_model,
            voice=voice,
            input=text,
            response_format="wav",
        )

        _ = speed

        temp_path: str | None = None

        try:

            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False,
            ) as temp_file:

                temp_path = temp_file.name

            response.write_to_file(
                temp_path
            )

            with wave.open(
                temp_path,
                "rb",
            ) as wav_file:

                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                sample_rate = wav_file.getframerate()
                frame_count = wav_file.getnframes()

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

            if sample_rate != TTS_OUTPUT_SAMPLE_RATE:

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
                    Path(temp_path).unlink(
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

        text = _clean_text(text)

        if not text:
            return b""

        selected_voice = (
            self.set_voice(voice)
            if voice
            else self.voice
        )

        selected_speed = normalize_speech_speed(
            speed
        )

        groq_voice = self._get_tts_voice()

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

        output_parts: list[bytes] = []

        started = asyncio.get_running_loop().time()

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
                output_parts.append(audio)

        final_audio = b"".join(output_parts)

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
                normalize_voice_name(voice)
                if voice
                else self.voice
            ),
            "speed": normalize_speech_speed(speed),
            "character": _character_value(
                character,
                "name",
                None,
            ),
            "tts_model": self.tts_model,
            "error": None,
        }

        if not audio:
            result["error"] = "No audio received."
            return result

        try:

            # ------------------------------
            # Groq STT
            # ------------------------------

            transcript = await self.transcribe(
                audio,
                mime_type=mime_type,
            )

            if not transcript:

                result["error"] = (
                    "Speech rejected by STT filter."
                )

                return result

            result["transcript"] = transcript

            # ------------------------------
            # Groq Chat
            # ------------------------------

            response = await self.generate_response(
                transcript,
                username=username,
                memory=memory,
                character=character,
                system_prompt=system_prompt,
            )

            if not response:

                result["error"] = (
                    "Groq returned an empty response."
                )

                return result

            result["response"] = response

            # ------------------------------
            # Memory
            # ------------------------------

            self.add_user_message(
                f"{_safe_username(username)}: {transcript}"
            )

            self.add_assistant_message(
                response
            )

            # ------------------------------
            # TTS
            # ------------------------------

            selected_voice = (
                self.set_voice(voice)
                if voice
                else self.voice
            )

            result["voice"] = selected_voice

            speech = await self.generate_speech(
                response,
                voice=selected_voice,
                speed=speed,
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
            result["error"] = str(error)

            if self._is_quota_error(error):

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
            "chat_provider": "Groq",
            "chat_model": self.chat_model,
            "transcribe_provider": "Groq",
            "transcribe_model": self.transcribe_model,
            "stt_language": "ar",
            "stt_filter": "strong",
            "tts_provider": "Groq",
            "tts_model": self.tts_model,
            "tts_local": False,
            "tts_voice": self._get_tts_voice(),
        }

    # ========================================================
    # CLOSE
    # ========================================================

    async def close(self) -> None:
        self.groq = None


# ============================================================
# FACTORY
# ============================================================

def create_gemini_engine() -> GeminiEngine:
    return GeminiEngine()


__all__ = [
    "GeminiEngine",
    "create_gemini_engine",
]
