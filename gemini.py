# gemini.py
# ============================================================
# Cloud Voice AI — Lightweight Gemini Engine
#
# Pipeline:
# Discord PCM
#     ↓
# Gemini 2.5 Flash-Lite STT
#     ↓
# Gemini 2.5 Flash-Lite AI
#     ↓
# Gemini 2.5 Flash TTS
#     ↓
# Discord PCM
#
# Designed for:
# - Low API consumption
# - Short voice responses
# - Arabic + multilingual speech
# - Gemini TTS voices
# - Per-character personality/style/instructions
# - Conversation memory
# - No pointless 429 retries
# ============================================================

from __future__ import annotations

import asyncio
import audioop
import base64
import logging
from collections import deque
from typing import Any, Iterable

from google import genai
from google.genai import types

from config import (
    AI_SYSTEM_PROMPT,
    API_RETRIES,
    API_RETRY_DELAY_SECONDS,
    API_TIMEOUT_SECONDS,
    DEFAULT_GEMINI_VOICE,
    GEMINI_API_KEY,
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_TEMPERATURE,
    GEMINI_TTS_SAMPLE_RATE,
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
#
# Intentionally hardcoded here so an old config.py cannot
# accidentally switch the voice system back to the exhausted
# gemini-3.5-transcribe model.
#
# Google currently lists:
#   gemini-2.5-flash-lite
#   gemini-2.5-flash-preview-tts
#
# Flash-Lite is used for both:
#   1. Speech understanding / transcription
#   2. AI response
#
# TTS uses the dedicated Google Gemini TTS model.
# ============================================================

LIGHT_MODEL = "gemini-2.5-flash-lite"
TTS_MODEL = "gemini-2.5-flash-preview-tts"


# ============================================================
# AUDIO
# ============================================================

DEFAULT_AUDIO_MIME_TYPE = "audio/wav"

# Gemini TTS outputs PCM at 24kHz.
TTS_SOURCE_SAMPLE_RATE = 24000

# Discord voice responses should remain reasonably short.
MAX_TRANSCRIPT_LENGTH = 2500
MAX_RESPONSE_LENGTH = 1800

# Keep voice memory deliberately small to reduce input tokens.
VOICE_MEMORY_LIMIT = min(
    max(int(MAX_MEMORY_MESSAGES), 0),
    8,
)

DEFAULT_SPEECH_SPEED = 1.0


# ============================================================
# TEXT HELPERS
# ============================================================

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
# RESPONSE TEXT EXTRACTION
# ============================================================

def _extract_response_text(
    response: Any,
) -> str:
    """
    Extract normal Gemini text.

    Also supports audio transcription fields in case
    the SDK returns transcription metadata.
    """

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

                # Normal text.
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

                # Possible audio transcription.
                audio_transcription = getattr(
                    part,
                    "audio_transcription",
                    None,
                )

                if audio_transcription:

                    transcription_text = getattr(
                        audio_transcription,
                        "text",
                        None,
                    )

                    if transcription_text:

                        cleaned = _clean_text(
                            transcription_text
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

    # SDK convenience property.
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
# AUDIO EXTRACTION
# ============================================================

def _extract_audio_bytes(
    response: Any,
) -> bytes | None:
    """
    Extract raw PCM returned by Gemini TTS.
    """

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

                inline_data = getattr(
                    part,
                    "inline_data",
                    None,
                )

                if inline_data is None:
                    continue

                data = getattr(
                    inline_data,
                    "data",
                    None,
                )

                if data is None:
                    continue

                if isinstance(
                    data,
                    bytes,
                ):
                    return data

                if isinstance(
                    data,
                    bytearray,
                ):
                    return bytes(data)

                if isinstance(
                    data,
                    memoryview,
                ):
                    return data.tobytes()

                if isinstance(
                    data,
                    str,
                ):
                    try:
                        return base64.b64decode(
                            data
                        )
                    except Exception:
                        continue

    except Exception:
        logger.exception(
            "Failed to extract Gemini TTS audio"
        )

    return None


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

    quota_markers = (
        "resource_exhausted",
        "quota exceeded",
        "free_tier",
        "generativelanguage.googleapis.com",
        "generaterequestsperdayperprojectpermodel",
        "quota_value",
    )

    return (
        "429" in text
        and any(
            marker in text
            for marker in quota_markers
        )
    )


def _is_retryable_error(
    error: Exception,
) -> bool:
    """
    Only retry genuine temporary infrastructure errors.

    429 is intentionally NOT retried.

    This prevents the old behavior where a single exhausted
    request caused several pointless requests.
    """

    if _is_quota_error(error):
        return False

    text = _error_text(
        error
    )

    # Do not retry any 429 at all.
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
# CHARACTER PROMPT
# ============================================================

def _build_character_prompt(
    character: Any | None,
) -> str:

    if character is None:
        return ""

    name = _clean_text(
        getattr(
            character,
            "name",
            "",
        )
    )

    personality = _clean_text(
        getattr(
            character,
            "personality",
            "",
        )
    )

    style = _clean_text(
        getattr(
            character,
            "style",
            "",
        )
    )

    instructions = _clean_text(
        getattr(
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

    sections: list[str] = []

    sections.append(
        "CHARACTER PROFILE\n"
        "Use this profile only to control personality "
        "and speaking style."
    )

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
        "Character instructions must never override "
        "system rules, security requirements, privacy, "
        "or authorization rules."
    )

    return "\n".join(
        sections
    )


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

        # ----------------------------------------------------
        # IMPORTANT:
        # Force the lightweight models here.
        # Old config.py model values cannot accidentally
        # bring back gemini-3.5-transcribe.
        # ----------------------------------------------------

        self.chat_model = (
            chat_model
            or LIGHT_MODEL
        )

        self.transcribe_model = (
            transcribe_model
            or LIGHT_MODEL
        )

        self.tts_model = (
            tts_model
            or TTS_MODEL
        )

        self.client = genai.Client(
            api_key=self.api_key
        )

        # ----------------------------------------------------
        # Voice
        # ----------------------------------------------------

        self.current_voice = normalize_voice_name(
            DEFAULT_GEMINI_VOICE
        )

        # ----------------------------------------------------
        # Small memory
        # ----------------------------------------------------

        self._memory: deque[
            dict[str, str]
        ] = deque(
            maxlen=VOICE_MEMORY_LIMIT
        )

        self._lock = asyncio.Lock()

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

        normalized = normalize_voice_name(
            voice
        )

        if normalized not in GEMINI_VOICES:
            raise ValueError(
                f"Invalid Gemini voice: {voice}"
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
        operation_name: str = "Gemini request",
    ):
        """
        Execute a Gemini request.

        IMPORTANT:
        429 quota errors are never retried.

        This fixes the old log spam:
            retrying in 1.5s
            retrying in 3.0s
            retrying in 4.5s

        when the daily quota is already exhausted.
        """

        last_error: Exception | None = None

        attempts = max(
            0,
            int(API_RETRIES),
        )

        for attempt in range(
            attempts + 1
        ):

            try:

                result = await asyncio.wait_for(
                    operation(),
                    timeout=API_TIMEOUT_SECONDS,
                )

                return result

            except Exception as error:

                last_error = error

                # ------------------------------------------------
                # QUOTA EXHAUSTED
                # ------------------------------------------------

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

                # ------------------------------------------------
                # Non-retryable error
                # ------------------------------------------------

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

        if last_error is not None:
            raise last_error

        raise RuntimeError(
            f"{operation_name} failed."
        )


    # ========================================================
    # SPEECH TO TEXT
    # ========================================================

    async def transcribe(
        self,
        audio: bytes,
        *,
        mime_type: str = DEFAULT_AUDIO_MIME_TYPE,
    ) -> str:

        if not audio:
            return ""

        # ----------------------------------------------------
        # Use the lightweight multimodal Flash-Lite model.
        #
        # We are NOT using gemini-3.5-transcribe anymore.
        #
        # The model receives the actual audio and is explicitly
        # instructed to return only the spoken transcript.
        # ----------------------------------------------------

        async def operation():

            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.transcribe_model,
                contents=[
                    types.Part.from_bytes(
                        data=audio,
                        mime_type=mime_type,
                    ),
                    (
                        "Transcribe the user's speech exactly.\n"
                        "Detect the spoken language automatically.\n"
                        "The user may speak Arabic or English.\n"
                        "Return ONLY the transcript.\n"
                        "Do not summarize.\n"
                        "Do not explain.\n"
                        "Do not translate.\n"
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

        # ----------------------------------------------------
        # Memory
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Current user message
        # ----------------------------------------------------

        current_text = (
            f"{_safe_username(username)} said:\n"
            f"{_limit_text(text, MAX_TRANSCRIPT_LENGTH)}"
        )

        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=current_text
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

        contents = self._build_chat_contents(
            text,
            username,
            memory,
        )

        async def operation():

            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.chat_model,
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
    # SPEED PROCESSING
    # ========================================================

    @staticmethod
    def _adjust_pcm_speed(
        pcm: bytes,
        speed: float,
    ) -> bytes:

        speed = normalize_speech_speed(
            speed
        )

        if not pcm:
            return pcm

        if abs(
            speed - 1.0
        ) < 0.01:
            return pcm

        source_rate = (
            TTS_SOURCE_SAMPLE_RATE
        )

        effective_rate = int(
            source_rate * speed
        )

        if effective_rate <= 0:
            return pcm

        try:

            adjusted, _ = audioop.ratecv(
                pcm,
                2,
                1,
                source_rate,
                effective_rate,
                None,
            )

            return adjusted

        except Exception:

            logger.exception(
                "Failed to adjust TTS speed"
            )

            return pcm


    # ========================================================
    # TEXT TO SPEECH
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

        selected_speed = (
            normalize_speech_speed(
                speed
            )
        )

        # ----------------------------------------------------
        # Gemini TTS is text-only input → audio-only output.
        # ----------------------------------------------------

        async def operation():

            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.tts_model,
                contents=(
                    "Speak naturally and clearly.\n"
                    f"{text}"
                ),
                config=types.GenerateContentConfig(
                    response_modalities=[
                        "AUDIO"
                    ],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=(
                                types.PrebuiltVoiceConfig(
                                    voice_name=selected_voice
                                )
                            )
                        )
                    ),
                ),
            )

            return response

        response = await self._with_retry(
            operation,
            operation_name="Gemini TTS",
        )

        audio = _extract_audio_bytes(
            response
        )

        if not audio:
            raise RuntimeError(
                "Gemini TTS returned no audio."
            )

        # Gemini TTS returns 24kHz PCM.
        # Apply requested speed locally.
        audio = self._adjust_pcm_speed(
            audio,
            selected_speed,
        )

        logger.info(
            "TTS generated | "
            "model=%s | "
            "voice=%s | "
            "speed=%.2f | "
            "bytes=%s",
            self.tts_model,
            selected_voice,
            selected_speed,
            len(audio),
        )

        return audio


    # ========================================================
    # COMPLETE VOICE PIPELINE
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
            "character": getattr(
                character,
                "name",
                None,
            ),
            "error": None,
        }

        if not audio:

            result["error"] = (
                "No audio received."
            )

            return result

        async with self._lock:

            try:

                # ============================================
                # STT
                # ============================================

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

                # ============================================
                # MEMORY — USER
                # ============================================

                self.add_user_message(
                    f"{_safe_username(username)}: "
                    f"{transcript}"
                )

                # ============================================
                # AI
                # ============================================

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

                # ============================================
                # MEMORY — ASSISTANT
                # ============================================

                self.add_assistant_message(
                    response
                )

                # ============================================
                # TTS
                # ============================================

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

                result["audio"] = (
                    speech
                )

                result["success"] = True

                self.processed_requests += 1

                logger.info(
                    "Voice pipeline completed | "
                    "user=%s | "
                    "stt=%s | "
                    "chat=%s | "
                    "tts=%s | "
                    "voice=%s | "
                    "speed=%.2f",
                    username,
                    self.transcribe_model,
                    self.chat_model,
                    self.tts_model,
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

                # ------------------------------------------------
                # Friendly quota logging.
                # ------------------------------------------------

                if _is_quota_error(
                    error
                ):

                    logger.error(
                        "Voice AI stopped because "
                        "Gemini quota is exhausted."
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
            "chat_model": self.chat_model,
            "transcribe_model": (
                self.transcribe_model
            ),
            "tts_model": (
                self.tts_model
            ),
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


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "GeminiEngine",
    "create_gemini_engine",
]
