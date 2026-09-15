# gemini.py
# ============================================================
# Cloud Voice AI
#
# Pipeline:
# Discord Voice PCM / WAV
#     ↓
# Gladia STT
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
#     Gladia
#
# Chat:
#     Groq GPT-OSS 120B
#
# TTS:
#     Groq Orpheus Arabic Saudi
#
# Gemini / Claude / Piper are completely removed.
# Groq Whisper STT is completely removed.
# faster-whisper is completely removed.
#
# IMPORTANT:
# - No Gladia SDK is required.
# - Gladia STT is accessed directly over HTTPS.
# - This keeps the installation lightweight for small hosts.
# ============================================================

from __future__ import annotations

import asyncio
import audioop
import json
import logging
import os
import tempfile
import urllib.error
import urllib.request
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

GLADIA_API_KEY = os.getenv(
    "GLADIA_API_KEY",
    "",
).strip()


# ============================================================
# MODELS
# ============================================================

GLADIA_STT_MODEL = (
    os.getenv(
        "GLADIA_STT_MODEL",
        "solaria-1",
    ).strip()
    or "solaria-1"
)

GLADIA_STT_LANGUAGE = (
    os.getenv(
        "GLADIA_STT_LANGUAGE",
        "ar",
    ).strip()
    or "ar"
)

GLADIA_STT_ENDPOINT = (
    os.getenv(
        "GLADIA_STT_ENDPOINT",
        "https://api.gladia.io/v2/pre-recorded",
    ).strip()
    or "https://api.gladia.io/v2/pre-recorded"
)

GLADIA_UPLOAD_ENDPOINT = (
    os.getenv(
        "GLADIA_UPLOAD_ENDPOINT",
        "https://api.gladia.io/v2/upload",
    ).strip()
    or "https://api.gladia.io/v2/upload"
)

GLADIA_POLL_INTERVAL = 0.25

try:
    GLADIA_MAX_POLL_SECONDS = max(
        float(
            os.getenv(
                "GLADIA_MAX_POLL_SECONDS",
                "30",
            )
        ),
        5.0,
    )
except (TypeError, ValueError):
    GLADIA_MAX_POLL_SECONDS = 30.0


# ------------------------------------------------------------
# Gladia options.
#
# Keep the processing intentionally simple.
# ------------------------------------------------------------

GLADIA_DIARIZATION = (
    os.getenv(
        "GLADIA_DIARIZATION",
        "false",
    ).strip().lower()
    not in {
        "0",
        "false",
        "no",
        "off",
    }
)

GLADIA_SUBTITLES = (
    os.getenv(
        "GLADIA_SUBTITLES",
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

    return str(
        value
    ).strip()


def _normalize_transcript(
    text: str,
) -> str:
    """
    Very light transcript cleanup.

    Gladia is responsible for the speech recognition.
    We do not attempt to rewrite or guess words here.
    """

    text = _clean_text(
        text
    )

    if not text:
        return ""

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

    return text[
        :maximum
    ].rstrip()


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

            line = line[
                1:
            ].strip()

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
        Gladia

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

        # ----------------------------------------------------
        # GLADIA
        # ----------------------------------------------------

        self.gladia_api_key = (
            GLADIA_API_KEY
        ).strip()

        if not self.gladia_api_key:

            raise RuntimeError(
                "GLADIA_API_KEY is missing."
            )

        # ----------------------------------------------------
        # MODELS
        # ----------------------------------------------------

        self.chat_model = (
            chat_model
            or GROQ_CHAT_MODEL
        )

        self.tts_model = (
            tts_model
            or GROQ_TTS_MODEL
        )

        self.transcribe_model = (
            transcribe_model
            or GLADIA_STT_MODEL
        )

        # ----------------------------------------------------
        # CLIENT
        # ----------------------------------------------------

        self.groq = Groq(
            api_key=self.groq_api_key
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
                "credit",
                "credits",
                "payment required",
                "insufficient balance",
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
                "bad gateway",
                "gateway timeout",
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
    # AUDIO
    # ========================================================

    @staticmethod
    def _ensure_wav_bytes(
        audio: bytes,
    ) -> bytes:

        if not audio:

            raise ValueError(
                "Empty audio."
            )

        # ----------------------------------------------------
        # Already WAV.
        # ----------------------------------------------------

        if audio.startswith(
            b"RIFF"
        ):

            return audio

        # ----------------------------------------------------
        # Raw Discord PCM:
        #
        # 48kHz stereo 16-bit
        #
        # ↓
        #
        # 16kHz mono 16-bit
        # ----------------------------------------------------

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

        import io

        buffer = io.BytesIO()

        with wave.open(
            buffer,
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

        return buffer.getvalue()

    # ========================================================
    # GLADIA HTTP HELPER
    # ========================================================

    @staticmethod
    def _json_request(
        url: str,
        *,
        method: str,
        headers: dict[str, str],
        data: bytes | None = None,
    ) -> dict[str, Any]:

        request = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers=headers,
        )

        try:

            with urllib.request.urlopen(
                request,
                timeout=API_TIMEOUT_SECONDS,
            ) as response:

                body = response.read()

        except urllib.error.HTTPError as error:

            try:

                error_body = (
                    error.read()
                    .decode(
                        "utf-8",
                        errors="replace",
                    )
                    .strip()
                )

            except Exception:

                error_body = ""

            if error_body:

                raise RuntimeError(
                    f"Gladia HTTP {error.code}: "
                    f"{error_body}"
                ) from error

            raise RuntimeError(
                f"Gladia HTTP {error.code}: "
                f"{error.reason}"
            ) from error

        except urllib.error.URLError as error:

            raise RuntimeError(
                "Gladia connection failed: "
                f"{error.reason}"
            ) from error

        except TimeoutError as error:

            raise RuntimeError(
                "Gladia request timed out."
            ) from error

        try:

            decoded = body.decode(
                "utf-8"
            )

            payload = json.loads(
                decoded
            )

        except Exception as error:

            raise RuntimeError(
                "Gladia returned invalid JSON."
            ) from error

        if not isinstance(
            payload,
            dict,
        ):

            raise RuntimeError(
                "Gladia returned an unexpected response."
            )

        return payload

    # ========================================================
    # GLADIA PRE-RECORDED STT
    # ========================================================

    def _upload_to_gladia_sync(
        self,
        wav_data: bytes,
    ) -> str:

        # ----------------------------------------------------
        # Upload WAV to Gladia.
        # ----------------------------------------------------

        headers = {
            "x-gladia-key": (
                self.gladia_api_key
            ),
            "Content-Type": "audio/wav",
            "Accept": "application/json",
            "User-Agent": "CloudVoiceAI/1.0",
        }

        request = urllib.request.Request(
            GLADIA_UPLOAD_ENDPOINT,
            data=wav_data,
            method="POST",
            headers=headers,
        )

        try:

            with urllib.request.urlopen(
                request,
                timeout=API_TIMEOUT_SECONDS,
            ) as response:

                body = response.read()

        except urllib.error.HTTPError as error:

            try:

                error_body = (
                    error.read()
                    .decode(
                        "utf-8",
                        errors="replace",
                    )
                    .strip()
                )

            except Exception:

                error_body = ""

            if error_body:

                raise RuntimeError(
                    f"Gladia upload HTTP "
                    f"{error.code}: {error_body}"
                ) from error

            raise RuntimeError(
                f"Gladia upload HTTP "
                f"{error.code}: {error.reason}"
            ) from error

        except urllib.error.URLError as error:

            raise RuntimeError(
                "Gladia upload connection failed: "
                f"{error.reason}"
            ) from error

        except TimeoutError as error:

            raise RuntimeError(
                "Gladia upload request timed out."
            ) from error

        try:

            payload = json.loads(
                body.decode(
                    "utf-8"
                )
            )

        except Exception as error:

            raise RuntimeError(
                "Gladia upload returned invalid JSON."
            ) from error

        if not isinstance(
            payload,
            dict,
        ):

            raise RuntimeError(
                "Gladia upload returned an unexpected response."
            )

        audio_url = (
            payload.get(
                "audio_url"
            )
            or payload.get(
                "url"
            )
        )

        if not audio_url:

            raise RuntimeError(
                "Gladia upload did not return an audio URL."
            )

        return str(
            audio_url
        )

    def _start_gladia_transcription_sync(
        self,
        audio_url: str,
    ) -> dict[str, Any]:

        # ----------------------------------------------------
        # Create a pre-recorded transcription job.
        # ----------------------------------------------------

        payload: dict[str, Any] = {
            "audio_url": audio_url,
            "language_config": {
                "languages": [
                    GLADIA_STT_LANGUAGE,
                ],
            },
        }

        if GLADIA_DIARIZATION:

            payload[
                "diarization"
            ] = True

        if GLADIA_SUBTITLES:

            payload[
                "subtitles"
            ] = True

        request_data = json.dumps(
            payload,
            ensure_ascii=False,
        ).encode(
            "utf-8"
        )

        headers = {
            "x-gladia-key": (
                self.gladia_api_key
            ),
            "Content-Type": (
                "application/json"
            ),
            "Accept": "application/json",
            "User-Agent": "CloudVoiceAI/1.0",
        }

        result = self._json_request(
            GLADIA_STT_ENDPOINT,
            method="POST",
            headers=headers,
            data=request_data,
        )

        return result

    def _get_gladia_result_sync(
        self,
        result_url: str,
    ) -> dict[str, Any]:

        headers = {
            "x-gladia-key": (
                self.gladia_api_key
            ),
            "Accept": "application/json",
            "User-Agent": "CloudVoiceAI/1.0",
        }

        return self._json_request(
            result_url,
            method="GET",
            headers=headers,
        )

    @staticmethod
    def _extract_gladia_transcript(
        payload: dict[str, Any],
    ) -> str:

        # ----------------------------------------------------
        # Common v2 response locations.
        # ----------------------------------------------------

        candidates: list[Any] = []

        result = payload.get(
            "result"
        )

        if isinstance(
            result,
            dict,
        ):

            candidates.extend(
                (
                    result.get(
                        "transcription",
                        {},
                    ),
                    result.get(
                        "transcript",
                        "",
                    ),
                )
            )

        candidates.extend(
            (
                payload.get(
                    "transcription",
                    {},
                ),
                payload.get(
                    "transcript",
                    "",
                ),
            )
        )

        # ----------------------------------------------------
        # Search likely nested transcript fields.
        # ----------------------------------------------------

        for candidate in candidates:

            if isinstance(
                candidate,
                str,
            ):

                text = _normalize_transcript(
                    candidate
                )

                if text:

                    return text

            if isinstance(
                candidate,
                dict,
            ):

                for key in (
                    "full_transcript",
                    "text",
                    "transcript",
                ):

                    value = candidate.get(
                        key
                    )

                    if isinstance(
                        value,
                        str,
                    ):

                        text = _normalize_transcript(
                            value
                        )

                        if text:

                            return text

        # ----------------------------------------------------
        # Recursive fallback for unexpected response shapes.
        # ----------------------------------------------------

        def find_text(
            value: Any,
        ) -> str:

            if isinstance(
                value,
                dict,
            ):

                for key in (
                    "full_transcript",
                    "transcript",
                    "text",
                ):

                    item = value.get(
                        key
                    )

                    if isinstance(
                        item,
                        str,
                    ):

                        normalized = (
                            _normalize_transcript(
                                item
                            )
                        )

                        if normalized:

                            return normalized

                for item in value.values():

                    found = find_text(
                        item
                    )

                    if found:

                        return found

            elif isinstance(
                value,
                list,
            ):

                for item in value:

                    found = find_text(
                        item
                    )

                    if found:

                        return found

            return ""

        return find_text(
            payload
        )

    def _transcribe_gladia_sync(
        self,
        wav_data: bytes,
    ) -> str:

        logger.info(
            "Gladia STT upload | model=%s | "
            "language=%s | bytes=%d",
            self.transcribe_model,
            GLADIA_STT_LANGUAGE,
            len(wav_data),
        )

        audio_url = (
            self._upload_to_gladia_sync(
                wav_data
            )
        )

        logger.debug(
            "Gladia upload complete"
        )

        result = (
            self._start_gladia_transcription_sync(
                audio_url
            )
        )

        # ----------------------------------------------------
        # Depending on the API response, a polling URL may
        # be returned.
        # ----------------------------------------------------

        result_url = (
            result.get(
                "result_url"
            )
            or result.get(
                "url"
            )
            or result.get(
                "result",
                {},
            ).get(
                "url",
            )
            if isinstance(
                result.get(
                    "result",
                    {},
                ),
                dict,
            )
            else None
        )

        transcript = (
            self._extract_gladia_transcript(
                result
            )
        )

        if transcript:

            return transcript

        if not result_url:

            status = str(
                result.get(
                    "status",
                    "",
                )
                or ""
            ).lower()

            result_id = (
                result.get(
                    "id"
                )
            )

            if result_id:

                result_url = (
                    f"{GLADIA_STT_ENDPOINT}/"
                    f"{result_id}"
                )

            elif status in {
                "done",
                "completed",
                "success",
            }:

                raise RuntimeError(
                    "Gladia finished but no transcript "
                    "was found in the response."
                )

        if not result_url:

            raise RuntimeError(
                "Gladia did not provide a transcription result URL."
            )

        started = (
            asyncio.get_event_loop_policy()
            .time()
            if hasattr(
                asyncio.get_event_loop_policy(),
                "time",
            )
            else 0.0
        )

        # ----------------------------------------------------
        # Poll until final result.
        #
        # This function runs in a worker thread, so ordinary
        # time.monotonic is preferable here.
        # ----------------------------------------------------

        import time as _time

        polling_started = _time.monotonic()

        while (
            _time.monotonic()
            - polling_started
            < GLADIA_MAX_POLL_SECONDS
        ):

            current = (
                self._get_gladia_result_sync(
                    str(result_url)
                )
            )

            transcript = (
                self._extract_gladia_transcript(
                    current
                )
            )

            if transcript:

                status = str(
                    current.get(
                        "status",
                        "",
                    )
                    or ""
                ).lower()

                if status in {
                    "",
                    "done",
                    "completed",
                    "success",
                    "finished",
                }:

                    return transcript

            status = str(
                current.get(
                    "status",
                    "",
                )
                or ""
            ).lower()

            if status in {
                "error",
                "failed",
                "failure",
                "cancelled",
            }:

                raise RuntimeError(
                    "Gladia transcription failed: "
                    f"{current}"
                )

            _time.sleep(
                GLADIA_POLL_INTERVAL
            )

        raise RuntimeError(
            "Gladia transcription timed out."
        )

    async def transcribe(
        self,
        audio: bytes,
        *,
        mime_type: str = DEFAULT_AUDIO_MIME_TYPE,
    ) -> str:

        if not audio:
            return ""

        _ = mime_type

        wav_data = (
            await asyncio.to_thread(
                self._ensure_wav_bytes,
                audio,
            )
        )

        transcript = await self._with_retry(
            lambda: self._transcribe_gladia_sync(
                wav_data
            ),
            operation_name="Gladia STT",
        )

        transcript = _limit_text(
            _normalize_transcript(
                transcript
            ),
            MAX_TRANSCRIPT_LENGTH,
        )

        if not transcript:

            logger.warning(
                "Gladia STT returned empty transcript."
            )

            return ""

        logger.info(
            "Gladia STT FINAL | "
            "model=%s | language=%s | transcript=%r",
            self.transcribe_model,
            GLADIA_STT_LANGUAGE,
            transcript,
        )

        return transcript

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
                    "Speech rejected by STT."
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
                    "API rate limit/quota/credit limit reached."
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

            "processed_requests": (
                self.processed_requests
            ),

            "failed_requests": (
                self.failed_requests
            ),

            "quota_exhausted": (
                self.quota_exhausted
            ),

            # ------------------------------------------------
            # CHAT
            # ------------------------------------------------

            "chat_provider": "Groq",
            "chat_model": self.chat_model,

            # ------------------------------------------------
            # STT
            # ------------------------------------------------

            "transcribe_provider": "Gladia",
            "transcribe_model": (
                self.transcribe_model
            ),
            "stt_language": (
                GLADIA_STT_LANGUAGE
            ),
            "stt_filter": "none",
            "stt_diarization": (
                GLADIA_DIARIZATION
            ),
            "stt_subtitles": (
                GLADIA_SUBTITLES
            ),

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
