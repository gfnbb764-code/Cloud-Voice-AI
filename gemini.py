# gemini.py
# ============================================================
# Cloud Voice AI — Gemini Flash-Lite + Local Piper
#
# Pipeline:
# Discord PCM
#     ↓
# Gemini 3.5 Flash-Lite (audio understanding / STT)
#     ↓
# Gemini 3.5 Flash-Lite (AI)
#     ↓
# Local Piper TTS
#     ↓
# Discord PCM
# ============================================================

from __future__ import annotations

import asyncio
import audioop
import logging
from collections import deque
from pathlib import Path
from typing import Any, Iterable

from google import genai
from google.genai import types

from piper import PiperVoice
from piper.config import SynthesisConfig
from piper.download_voices import download_voice

from config import (
    AI_SYSTEM_PROMPT,
    API_RETRIES,
    API_RETRY_DELAY_SECONDS,
    API_TIMEOUT_SECONDS,
    DEFAULT_GEMINI_VOICE,
    GEMINI_API_KEY,
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_TEMPERATURE,
    GEMINI_VOICES,
    MEMORY_ENABLED,
    MAX_MEMORY_MESSAGES,
    normalize_speech_speed,
    normalize_voice_name,
)


logger = logging.getLogger(__name__)


# ============================================================
# MODELS
# ============================================================

# IMPORTANT:
# Do NOT use gemini-3.5-transcribe.
#
# Gemini 3.5 Flash-Lite accepts audio input and is used for
# both transcription and response generation.
#
# This prevents the bot from consuming the dedicated
# gemini-3.5-transcribe daily quota.
# ============================================================

LIGHT_MODEL = "gemini-3.5-flash-lite"

# TTS is NOT Gemini.
# It is 100% local Piper.
LOCAL_TTS_NAME = "Piper/ar_JO-kareem-low"


# ============================================================
# AUDIO
# ============================================================

DEFAULT_AUDIO_MIME_TYPE = "audio/wav"

# Piper output returned to voice.py:
# 24kHz mono 16-bit PCM
PIPER_OUTPUT_SAMPLE_RATE = 24000
PIPER_OUTPUT_CHANNELS = 1

MAX_TRANSCRIPT_LENGTH = 2500
MAX_RESPONSE_LENGTH = 1800

VOICE_MEMORY_LIMIT = min(
    max(int(MAX_MEMORY_MESSAGES), 0),
    8,
)

DEFAULT_SPEECH_SPEED = 1.0


# ============================================================
# PIPER
# ============================================================

PIPER_VOICE_MODEL = "ar_JO-kareem-low"
PIPER_MODEL_DIR = Path("piper_models")

_PIPER_VOICE: PiperVoice | None = None
_PIPER_LOAD_LOCK: asyncio.Lock | None = None


def _clean_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip()

    if not text:
        return ""

    return text


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
# CHARACTER HELPERS
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
        "Character instructions must never override system rules, "
        "security requirements, privacy, or authorization rules."
    )

    return "\n".join(
        sections
    )


# ============================================================
# RESPONSE EXTRACTION
# ============================================================

def _extract_response_text(
    response: Any,
) -> str:

    results: list[str] = []

    try:

        candidates = (
            getattr(
                response,
                "candidates",
                None,
            )
            or []
        )

        for candidate in candidates:

            content = getattr(
                candidate,
                "content",
                None,
            )

            if content is None:
                continue

            parts = (
                getattr(
                    content,
                    "parts",
                    None,
                )
                or []
            )

            for part in parts:

                text = getattr(
                    part,
                    "text",
                    None,
                )

                if text:

                    cleaned = _clean_text(
                        text
                    )

                    if (
                        cleaned
                        and cleaned not in results
                    ):
                        results.append(
                            cleaned
                        )

    except Exception:
        logger.exception(
            "Failed to extract Gemini response"
        )

    if results:
        return "\n".join(
            results
        ).strip()

    try:

        fallback = getattr(
            response,
            "text",
            None,
        )

        if fallback:
            return _clean_text(
                fallback
            )

    except Exception:
        pass

    return ""


# ============================================================
# ERROR HELPERS
# ============================================================

def _error_text(
    error: Exception,
) -> str:

    try:
        return str(error).lower()
    except Exception:
        return ""


def _is_quota_error(
    error: Exception,
) -> bool:

    text = _error_text(
        error
    )

    markers = (
        "resource_exhausted",
        "quota exceeded",
        "free_tier",
        "quota_value",
        "generativelanguage.googleapis.com",
    )

    return (
        "429" in text
        and any(
            marker in text
            for marker in markers
        )
    )


def _is_retryable_error(
    error: Exception,
) -> bool:

    if _is_quota_error(
        error
    ):
        return False

    text = _error_text(
        error
    )

    if "429" in text:
        return False

    retry_markers = (
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

    return any(
        marker in text
        for marker in retry_markers
    )


# ============================================================
# VOICE TEXT CLEANUP
# ============================================================

def _strip_markdown_for_voice(
    text: str,
) -> str:

    text = _clean_text(
        text
    )

    if not text:
        return ""

    replacements = (
        ("```", ""),
        ("**", ""),
        ("__", ""),
        ("`", ""),
        ("###", ""),
        ("##", ""),
        ("#", ""),
    )

    for old, new in replacements:
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
            (
                "-",
                "*",
                "•",
            )
        ):
            line = line[1:].strip()

        lines.append(
            line
        )

    return " ".join(
        lines
    ).strip()


# ============================================================
# PIPER MODEL
# ============================================================

def _piper_paths() -> tuple[Path, Path]:

    PIPER_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        PIPER_MODEL_DIR
        / f"{PIPER_VOICE_MODEL}.onnx"
    )

    config_path = (
        PIPER_MODEL_DIR
        / f"{PIPER_VOICE_MODEL}.onnx.json"
    )

    return (
        model_path,
        config_path,
    )


def _ensure_piper_model_sync() -> Path:

    model_path, config_path = (
        _piper_paths()
    )

    if (
        not model_path.exists()
        or not config_path.exists()
    ):

        logger.info(
            "Downloading Piper voice | model=%s",
            PIPER_VOICE_MODEL,
        )

        download_voice(
            PIPER_VOICE_MODEL,
            PIPER_MODEL_DIR,
        )

    if not model_path.exists():
        raise RuntimeError(
            f"Piper model missing: {model_path}"
        )

    if not config_path.exists():
        raise RuntimeError(
            f"Piper config missing: {config_path}"
        )

    return model_path


def _load_piper_sync() -> PiperVoice:

    model_path = (
        _ensure_piper_model_sync()
    )

    logger.info(
        "Loading local Piper voice | model=%s",
        PIPER_VOICE_MODEL,
    )

    voice = PiperVoice.load(
        str(model_path)
    )

    logger.info(
        "Piper voice loaded | model=%s | sample_rate=%s",
        PIPER_VOICE_MODEL,
        voice.config.sample_rate,
    )

    return voice


async def _get_piper_voice() -> PiperVoice:

    global _PIPER_VOICE
    global _PIPER_LOAD_LOCK

    if _PIPER_VOICE is not None:
        return _PIPER_VOICE

    if _PIPER_LOAD_LOCK is None:
        _PIPER_LOAD_LOCK = asyncio.Lock()

    async with _PIPER_LOAD_LOCK:

        if _PIPER_VOICE is not None:
            return _PIPER_VOICE

        _PIPER_VOICE = await asyncio.to_thread(
            _load_piper_sync
        )

        return _PIPER_VOICE


def _synthesize_piper_sync(
    voice: PiperVoice,
    text: str,
    speed: float,
) -> bytes:

    speed = normalize_speech_speed(
        speed
    )

    synthesis_config = SynthesisConfig(
        length_scale=1.0 / speed,
        volume=1.0,
        noise_scale=0.667,
        noise_w_scale=0.8,
        normalize_audio=True,
    )

    chunks: list[bytes] = []

    first_chunk = True

    for chunk in voice.synthesize(
        text,
        synthesis_config,
    ):

        if first_chunk:

            logger.info(
                "Piper first audio chunk ready"
            )

            first_chunk = False

        audio_bytes = getattr(
            chunk,
            "audio_int16_bytes",
            None,
        )

        if audio_bytes:
            chunks.append(
                bytes(audio_bytes)
            )

    pcm = b"".join(
        chunks
    )

    if not pcm:
        raise RuntimeError(
            "Piper returned no audio."
        )

    source_rate = int(
        voice.config.sample_rate
    )

    if (
        source_rate
        != PIPER_OUTPUT_SAMPLE_RATE
    ):

        pcm, _ = audioop.ratecv(
            pcm,
            2,
            1,
            source_rate,
            PIPER_OUTPUT_SAMPLE_RATE,
            None,
        )

    return pcm


# ============================================================
# GEMINI ENGINE
# ============================================================

class GeminiEngine:

    def __init__(
        self,
        *,
        api_key: str | None = None,
        chat_model: str | None = None,
        transcribe_model: str | None = None,
        tts_model: str | None = None,
    ) -> None:

        self.api_key = (
            api_key
            or GEMINI_API_KEY
        )

        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is missing."
            )

        # FORCE Flash-Lite.
        # Ignore old config.py model values.
        self.chat_model = (
            chat_model
            or LIGHT_MODEL
        )

        # FORCE Flash-Lite for audio understanding.
        # This is intentional and prevents the bot from
        # using gemini-3.5-transcribe.
        self.transcribe_model = LIGHT_MODEL

        # Backwards compatible attribute.
        self.tts_model = LOCAL_TTS_NAME

        self.client = genai.Client(
            api_key=self.api_key
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

    # ========================================================
    # RETRY
    # ========================================================

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
                    operation(),
                    timeout=API_TIMEOUT_SECONDS,
                )

            except Exception as error:

                last_error = error

                if _is_quota_error(
                    error
                ):

                    self.quota_exhausted = True

                    logger.error(
                        "%s stopped: Gemini quota exhausted. "
                        "No retry will be performed.",
                        operation_name,
                    )

                    raise

                if (
                    attempt >= attempts
                    or not _is_retryable_error(
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
    # AUDIO TRANSCRIPTION
    # ========================================================

    async def transcribe(
        self,
        audio: bytes,
        *,
        mime_type: str = DEFAULT_AUDIO_MIME_TYPE,
    ) -> str:

        if not audio:
            return ""

        async def operation():

            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=LIGHT_MODEL,
                contents=[
                    types.Part.from_bytes(
                        data=audio,
                        mime_type=mime_type,
                    ),
                    (
                        "Listen to this audio carefully.\n"
                        "Transcribe exactly what the user said.\n"
                        "The user may speak Arabic or English.\n"
                        "Detect the spoken language automatically.\n"
                        "Return ONLY the transcript.\n"
                        "Do not translate.\n"
                        "Do not summarize.\n"
                        "Do not explain.\n"
                        "Do not answer the user."
                    ),
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=512,
                ),
            )

            return response

        response = await self._with_retry(
            operation,
            operation_name="Gemini STT",
        )

        transcript = _extract_response_text(
            response
        )

        transcript = _limit_text(
            transcript,
            MAX_TRANSCRIPT_LENGTH,
        )

        if transcript:

            logger.info(
                "STT successful | transcript=%s",
                transcript,
            )

        else:

            logger.warning(
                "STT returned no transcript"
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
            "model",
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
    # CHAT CONTENT
    # ========================================================

    def _build_chat_contents(
        self,
        text: str,
        username: str,
        memory: Iterable[
            dict[str, str]
        ] | None = None,
    ) -> list[types.Content]:

        contents: list[
            types.Content
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

            sdk_role = (
                "model"
                if role in {
                    "assistant",
                    "model",
                }
                else "user"
            )

            contents.append(
                types.Content(
                    role=sdk_role,
                    parts=[
                        types.Part.from_text(
                            text=_limit_text(
                                content,
                                800,
                            )
                        )
                    ],
                )
            )

        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=(
                            f"{_safe_username(username)} said:\n"
                            f"{_limit_text(text, MAX_TRANSCRIPT_LENGTH)}"
                        )
                    )
                ],
            )
        )

        return contents

    # ========================================================
    # AI RESPONSE
    # ========================================================

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
            "Keep replies concise and natural.\n"
            "Usually answer in 1-3 short sentences.\n"
            "Do not use markdown.\n"
            "Do not use code blocks.\n"
            "Do not write long explanations unless explicitly asked."
        )

        contents = (
            self._build_chat_contents(
                text,
                username,
                memory,
            )
        )

        async def operation():

            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=LIGHT_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=final_system_prompt,
                    temperature=min(
                        float(
                            GEMINI_TEMPERATURE
                        ),
                        0.8,
                    ),
                    max_output_tokens=min(
                        int(
                            GEMINI_MAX_OUTPUT_TOKENS
                        ),
                        512,
                    ),
                ),
            )

            return response

        response = await self._with_retry(
            operation,
            operation_name="Gemini AI",
        )

        answer = _extract_response_text(
            response
        )

        answer = _limit_text(
            answer,
            MAX_RESPONSE_LENGTH,
        )

        answer = _strip_markdown_for_voice(
            answer
        )

        if answer:

            logger.info(
                "AI response generated | response=%s",
                answer,
            )

        return answer

    # ========================================================
    # LOCAL PIPER TTS
    # ========================================================

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

        selected_speed = normalize_speech_speed(
            speed
        )

        logger.info(
            "Local Piper TTS | model=%s | mapped_voice=%s | speed=%.2f",
            PIPER_VOICE_MODEL,
            selected_voice,
            selected_speed,
        )

        voice_model = await _get_piper_voice()

        started = (
            asyncio.get_running_loop().time()
        )

        try:

            audio = await asyncio.wait_for(
                asyncio.to_thread(
                    _synthesize_piper_sync,
                    voice_model,
                    text,
                    selected_speed,
                ),
                timeout=max(
                    20.0,
                    float(
                        API_TIMEOUT_SECONDS
                    ),
                ),
            )

        except asyncio.TimeoutError:

            raise RuntimeError(
                "Piper TTS timed out."
            )

        elapsed = (
            asyncio.get_running_loop().time()
            - started
        )

        logger.info(
            "Piper TTS generated | model=%s | "
            "voice=%s | speed=%.2f | bytes=%s | %.2fs",
            PIPER_VOICE_MODEL,
            selected_voice,
            selected_speed,
            len(audio),
            elapsed,
        )

        return audio

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
            "speed": normalize_speech_speed(
                speed
            ),
            "character": _character_value(
                character,
                "name",
                None,
            ),
            "tts_model": LOCAL_TTS_NAME,
            "error": None,
        }

        if not audio:

            result["error"] = (
                "No audio received."
            )

            return result

        try:

            # ==================================================
            # STT
            # ==================================================

            transcript = await self.transcribe(
                audio,
                mime_type=mime_type,
            )

            if not transcript:

                result["error"] = (
                    "No speech detected."
                )

                return result

            result["transcript"] = (
                transcript
            )

            # ==================================================
            # MEMORY
            # ==================================================

            self.add_user_message(
                f"{_safe_username(username)}: "
                f"{transcript}"
            )

            # ==================================================
            # AI
            # ==================================================

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
                    "AI returned an empty response."
                )

                return result

            result["response"] = (
                response
            )

            self.add_assistant_message(
                response
            )

            # ==================================================
            # PIPER TTS
            # ==================================================

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
                    "Piper returned no audio."
                )

                return result

            result["audio"] = (
                speech
            )

            result["success"] = True

            self.processed_requests += 1

            logger.info(
                "Voice pipeline completed | "
                "user=%s | stt=%s | chat=%s | "
                "tts=%s | voice=%s | speed=%.2f",
                username,
                LIGHT_MODEL,
                LIGHT_MODEL,
                LOCAL_TTS_NAME,
                selected_voice,
                normalize_speech_speed(
                    speed
                ),
            )

            return result

        except Exception as error:

            self.failed_requests += 1

            result["error"] = (
                str(error)
            )

            if _is_quota_error(
                error
            ):

                self.quota_exhausted = True

                logger.error(
                    "Gemini quota exhausted."
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
            "chat_model": LIGHT_MODEL,
            "transcribe_model": LIGHT_MODEL,
            "tts_model": LOCAL_TTS_NAME,
            "tts_local": True,
            "piper_voice": PIPER_VOICE_MODEL,
        }

    # ========================================================
    # CLOSE
    # ========================================================

    async def close(
        self,
    ) -> None:

        try:

            close_method = getattr(
                self.client,
                "close",
                None,
            )

            if close_method:

                result = close_method()

                if asyncio.iscoroutine(
                    result
                ):
                    await result

        except Exception:

            logger.exception(
                "Failed to close Gemini client"
            )


# ============================================================
# FACTORY
# ============================================================

def create_gemini_engine() -> GeminiEngine:
    return GeminiEngine()


__all__ = [
    "GeminiEngine",
    "create_gemini_engine",
]
