# gemini.py
# ============================================================
# Cloud Voice AI — Gemini Engine
# STT + Chat + TTS + Memory + Retry Handling
# ============================================================

from __future__ import annotations

import asyncio
import base64
import logging
import re
from collections import deque
from typing import Any, Iterable

from google import genai
from google.genai import types

from config import (
    AI_SYSTEM_PROMPT,
    API_MAX_RETRIES,
    API_TIMEOUT_SECONDS,
    DEFAULT_GEMINI_VOICE,
    GEMINI_API_KEY,
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_TEMPERATURE,
    GEMINI_VOICES,
    MAX_MEMORY_MESSAGES,
    MEMORY_ENABLED,
    RETRY_DELAY_SECONDS,
    CHAT_MODEL,
    TRANSCRIBE_MODEL,
    TTS_MODEL,
    normalize_voice_name,
)


# ============================================================
# LOGGER
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# CONSTANTS
# ============================================================

MAX_TRANSCRIPT_LENGTH = 4000
MAX_RESPONSE_LENGTH = 4000
MAX_TTS_TEXT_LENGTH = 3500

DEFAULT_AUDIO_MIME_TYPE = "audio/wav"

RETRYABLE_ERROR_NAMES = (
    "TimeoutError",
    "ConnectError",
    "ConnectionError",
    "ServiceUnavailable",
    "InternalServerError",
    "TooManyRequests",
    "ResourceExhausted",
)


# ============================================================
# HELPERS
# ============================================================

def _clean_text(
    value: Any,
) -> str:
    if value is None:
        return ""

    text = str(value)

    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    return text.strip()


def _limit_text(
    text: str,
    maximum: int,
) -> str:
    text = _clean_text(text)

    if len(text) <= maximum:
        return text

    return text[: maximum - 1].rstrip() + "…"


def _safe_username(
    username: str | None,
) -> str:
    username = _clean_text(username)

    if not username:
        return "User"

    username = username.replace("@", "")

    return _limit_text(
        username,
        80,
    )


def _extract_response_text(
    response: Any,
) -> str:
    """
    Extract text from Gemini responses.

    Supports both:

        part.text

    and Gemini transcription responses:

        part.audio_transcription.text

    This is especially important for Gemini Transcribe models,
    where the transcription may be returned as an
    AudioTranscription object rather than a normal text part.
    """

    collected: list[str] = []

    try:
        candidates = getattr(
            response,
            "candidates",
            None,
        )

        for candidate in candidates or []:
            content = getattr(
                candidate,
                "content",
                None,
            )

            if content is None:
                continue

            parts = getattr(
                content,
                "parts",
                None,
            )

            for part in parts or []:

                # ------------------------------------------------
                # Normal text part
                # ------------------------------------------------

                text = getattr(
                    part,
                    "text",
                    None,
                )

                if text:
                    text = _clean_text(text)

                    if text:
                        collected.append(text)

                # ------------------------------------------------
                # Gemini audio transcription part
                # ------------------------------------------------

                audio_transcription = getattr(
                    part,
                    "audio_transcription",
                    None,
                )

                if audio_transcription is not None:
                    transcription_text = getattr(
                        audio_transcription,
                        "text",
                        None,
                    )

                    if transcription_text:
                        transcription_text = _clean_text(
                            transcription_text
                        )

                        if transcription_text:
                            collected.append(
                                transcription_text
                            )

    except Exception:
        logger.exception(
            "Failed to extract Gemini response text."
        )

    # ------------------------------------------------------------
    # Remove duplicates while preserving order.
    # ------------------------------------------------------------

    if collected:
        unique: list[str] = []

        for item in collected:
            if item not in unique:
                unique.append(item)

        return "\n".join(unique).strip()

    # ------------------------------------------------------------
    # Final fallback.
    #
    # We intentionally use response.text only AFTER inspecting
    # structured parts, because Gemini Transcribe can return
    # audio_transcription instead of normal text.
    # ------------------------------------------------------------

    try:
        text = getattr(
            response,
            "text",
            None,
        )

        if text:
            return _clean_text(text)

    except Exception:
        pass

    return ""


def _extract_audio_bytes(
    response: Any,
) -> bytes:
    """
    Extract raw audio bytes from a Gemini TTS response.
    """

    try:
        candidates = getattr(
            response,
            "candidates",
            None,
        )

        if not candidates:
            return b""

        for candidate in candidates:
            content = getattr(
                candidate,
                "content",
                None,
            )

            if content is None:
                continue

            parts = getattr(
                content,
                "parts",
                None,
            )

            if not parts:
                continue

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

                if isinstance(data, bytes):
                    return data

                if isinstance(data, bytearray):
                    return bytes(data)

                if isinstance(data, memoryview):
                    return data.tobytes()

                if isinstance(data, str):
                    try:
                        return base64.b64decode(
                            data,
                            validate=True,
                        )
                    except Exception:
                        return data.encode(
                            "latin-1",
                            errors="ignore",
                        )

    except Exception:
        logger.exception(
            "Failed to extract TTS audio."
        )

    return b""


def _is_retryable_error(
    error: Exception,
) -> bool:
    """
    Determines whether an API exception is worth retrying.
    """

    error_name = type(error).__name__

    if error_name in RETRYABLE_ERROR_NAMES:
        return True

    text = str(error).lower()

    retry_markers = (
        "429",
        "rate limit",
        "resource exhausted",
        "temporarily unavailable",
        "service unavailable",
        "internal server error",
        "timeout",
        "timed out",
        "connection reset",
        "connection aborted",
        "503",
        "500",
        "502",
        "504",
    )

    return any(
        marker in text
        for marker in retry_markers
    )


def _strip_markdown_for_voice(
    text: str,
) -> str:
    """
    Removes common markdown formatting so TTS sounds natural.
    """

    text = _clean_text(text)

    if not text:
        return ""

    # Code blocks.
    text = re.sub(
        r"```[\s\S]*?```",
        "",
        text,
    )

    # Inline code.
    text = re.sub(
        r"`([^`]*)`",
        r"\1",
        text,
    )

    # Markdown links.
    text = re.sub(
        r"\[([^\]]+)\]\([^)]+\)",
        r"\1",
        text,
    )

    # Bold / italic / strike.
    text = text.replace("**", "")
    text = text.replace("__", "")
    text = text.replace("~~", "")
    text = text.replace("*", "")

    # Headings.
    text = re.sub(
        r"(?m)^\s*#{1,6}\s*",
        "",
        text,
    )

    # Bullets.
    text = re.sub(
        r"(?m)^\s*[-•]\s+",
        "",
        text,
    )

    # Excessive whitespace.
    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return _limit_text(
        text,
        MAX_TTS_TEXT_LENGTH,
    )


# ============================================================
# GEMINI ENGINE
# ============================================================

class GeminiEngine:
    """
    Main Gemini engine used by Cloud Voice AI.

    Responsibilities:
        - Speech-to-text
        - Text generation
        - Text-to-speech
        - Conversation memory
        - Retry handling
        - Voice selection
    """

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

        self.chat_model = (
            chat_model
            or CHAT_MODEL
        )

        self.transcribe_model = (
            transcribe_model
            or TRANSCRIBE_MODEL
        )

        self.tts_model = (
            tts_model
            or TTS_MODEL
        )

        self.client = genai.Client(
            api_key=self.api_key,
        )

        self._memory: deque[dict[str, str]] = deque(
            maxlen=MAX_MEMORY_MESSAGES
        )

        self._lock = asyncio.Lock()

        self.current_voice = normalize_voice_name(
            DEFAULT_GEMINI_VOICE
        )

        self.total_transcriptions = 0
        self.total_responses = 0
        self.total_speeches = 0
        self.total_errors = 0

        logger.info(
            "GeminiEngine initialized | chat=%s | stt=%s | tts=%s | voice=%s",
            self.chat_model,
            self.transcribe_model,
            self.tts_model,
            self.current_voice,
        )

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

        logger.info(
            "Gemini voice changed to %s",
            normalized,
        )

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
        Executes a Gemini operation with retry handling.
        """

        attempts = max(
            1,
            API_MAX_RETRIES + 1,
        )

        last_error: Exception | None = None

        for attempt in range(
            1,
            attempts + 1,
        ):
            try:
                return await asyncio.wait_for(
                    operation(),
                    timeout=API_TIMEOUT_SECONDS,
                )

            except asyncio.CancelledError:
                raise

            except Exception as error:
                last_error = error
                self.total_errors += 1

                retryable = _is_retryable_error(
                    error
                )

                logger.warning(
                    "%s failed | attempt=%s/%s | retryable=%s | error=%s",
                    operation_name,
                    attempt,
                    attempts,
                    retryable,
                    error,
                )

                if (
                    not retryable
                    or attempt >= attempts
                ):
                    raise

                delay = RETRY_DELAY_SECONDS * (
                    1.5 ** (attempt - 1)
                )

                await asyncio.sleep(
                    min(delay, 10.0)
                )

        if last_error is not None:
            raise last_error

        raise RuntimeError(
            f"{operation_name} failed."
        )

    # ========================================================
    # TRANSCRIPTION
    # ========================================================

    async def transcribe(
        self,
        audio: bytes,
        *,
        mime_type: str = DEFAULT_AUDIO_MIME_TYPE,
    ) -> str:
        """
        Converts WAV audio into text using Gemini Transcribe.
        """

        if not audio:
            return ""

        if not isinstance(audio, bytes):
            audio = bytes(audio)

        async def request():
            response = await self.client.aio.models.generate_content(
                model=self.transcribe_model,
                contents=[
                    types.Part.from_bytes(
                        data=audio,
                        mime_type=mime_type,
                    ),
                    (
                        "Transcribe exactly what the user said. "
                        "Automatically detect the spoken language. "
                        "Return only the transcript, without explanations."
                    ),
                ],
            )

            return response

        response = await self._with_retry(
            request,
            operation_name="Speech transcription",
        )

        text = _extract_response_text(
            response
        )

        text = _limit_text(
            text,
            MAX_TRANSCRIPT_LENGTH,
        )

        if text:
            self.total_transcriptions += 1

            logger.info(
                "STT successful | transcript=%s",
                text,
            )
        else:
            logger.warning(
                "STT returned no transcript."
            )

        return text

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

        content = _clean_text(
            content
        )

        if not content:
            return

        role = _clean_text(
            role
        ).lower()

        if role not in {
            "user",
            "assistant",
            "system",
        }:
            role = "user"

        self._memory.append(
            {
                "role": role,
                "content": _limit_text(
                    content,
                    MAX_RESPONSE_LENGTH,
                ),
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
        return len(self._memory)

    # ========================================================
    # PROMPT BUILDING
    # ========================================================

    def _build_chat_contents(
        self,
        *,
        username: str,
        user_text: str,
        memory: Iterable[dict[str, str]] | None = None,
    ) -> list[Any]:

        contents: list[Any] = []

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
                if role == "assistant"
                else "user"
            )

            contents.append(
                types.Content(
                    role=sdk_role,
                    parts=[
                        types.Part.from_text(
                            text=content
                        )
                    ],
                )
            )

        current_username = _safe_username(
            username
        )

        current_text = _limit_text(
            user_text,
            MAX_TRANSCRIPT_LENGTH,
        )

        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=(
                            f"The user speaking in Discord is "
                            f"{current_username}.\n"
                            f"User message:\n{current_text}"
                        )
                    )
                ],
            )
        )

        return contents

    # ========================================================
    # CHAT RESPONSE
    # ========================================================

    async def generate_response(
        self,
        text: str,
        *,
        username: str = "User",
        memory: Iterable[dict[str, str]] | None = None,
    ) -> str:

        text = _clean_text(
            text
        )

        if not text:
            return ""

        contents = self._build_chat_contents(
            username=username,
            user_text=text,
            memory=memory,
        )

        async def request():
            config = types.GenerateContentConfig(
                system_instruction=AI_SYSTEM_PROMPT,
                temperature=GEMINI_TEMPERATURE,
                max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
            )

            return await self.client.aio.models.generate_content(
                model=self.chat_model,
                contents=contents,
                config=config,
            )

        response = await self._with_retry(
            request,
            operation_name="Chat generation",
        )

        answer = _extract_response_text(
            response
        )

        answer = _limit_text(
            answer,
            MAX_RESPONSE_LENGTH,
        )

        if not answer:
            answer = (
                "ما قدرت أطلع رد حاليًا، جرّب مرة ثانية."
            )

        if MEMORY_ENABLED:
            self.add_user_message(
                text
            )

            self.add_assistant_message(
                answer
            )

        self.total_responses += 1

        logger.info(
            "AI response generated | response=%s",
            answer,
        )

        return answer

    # ========================================================
    # TEXT TO SPEECH
    # ========================================================

    async def generate_speech(
        self,
        text: str,
        *,
        voice: str | None = None,
    ) -> bytes:

        text = _strip_markdown_for_voice(
            text
        )

        if not text:
            return b""

        selected_voice = normalize_voice_name(
            voice or self.current_voice
        )

        if selected_voice not in GEMINI_VOICES:
            selected_voice = self.current_voice

        async def request():
            speech_config = types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=selected_voice,
                    )
                )
            )

            config = types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=speech_config,
            )

            return await self.client.aio.models.generate_content(
                model=self.tts_model,
                contents=text,
                config=config,
            )

        response = await self._with_retry(
            request,
            operation_name="Text to speech",
        )

        audio = _extract_audio_bytes(
            response
        )

        if not audio:
            raise RuntimeError(
                "Gemini TTS returned no audio data."
            )

        self.total_speeches += 1

        logger.info(
            "TTS generated | voice=%s | bytes=%s",
            selected_voice,
            len(audio),
        )

        return audio

    # ========================================================
    # FULL VOICE PIPELINE
    # ========================================================

    async def process_voice(
        self,
        audio: bytes,
        *,
        username: str = "User",
        voice: str | None = None,
        mime_type: str = DEFAULT_AUDIO_MIME_TYPE,
        memory: Iterable[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """
        Full pipeline:

            Discord audio
                ↓
            Gemini STT
                ↓
            Gemini Chat
                ↓
            Gemini TTS
                ↓
            Audio bytes
        """

        selected_voice = normalize_voice_name(
            voice or self.current_voice
        )

        if not audio:
            return {
                "success": False,
                "transcript": "",
                "response": "",
                "audio": b"",
                "voice": selected_voice,
                "error": "No audio received.",
            }

        async with self._lock:
            try:

                # ------------------------------------------------
                # STT
                # ------------------------------------------------

                transcript = await self.transcribe(
                    audio,
                    mime_type=mime_type,
                )

                if not transcript:
                    return {
                        "success": False,
                        "transcript": "",
                        "response": "",
                        "audio": b"",
                        "voice": selected_voice,
                        "error": "No speech detected.",
                    }

                logger.info(
                    "STT | user=%s | text=%s",
                    username,
                    transcript,
                )

                # ------------------------------------------------
                # AI
                # ------------------------------------------------

                response_text = await self.generate_response(
                    transcript,
                    username=username,
                    memory=memory,
                )

                if not response_text:
                    return {
                        "success": False,
                        "transcript": transcript,
                        "response": "",
                        "audio": b"",
                        "voice": selected_voice,
                        "error": "Empty AI response.",
                    }

                logger.info(
                    "AI | user=%s | response=%s",
                    username,
                    response_text,
                )

                # ------------------------------------------------
                # TTS
                # ------------------------------------------------

                speech = await self.generate_speech(
                    response_text,
                    voice=voice,
                )

                if not speech:
                    return {
                        "success": False,
                        "transcript": transcript,
                        "response": response_text,
                        "audio": b"",
                        "voice": selected_voice,
                        "error": "TTS returned no audio.",
                    }

                logger.info(
                    "Voice pipeline completed | user=%s | transcript=%s | response=%s | audio_bytes=%s",
                    username,
                    transcript,
                    response_text,
                    len(speech),
                )

                return {
                    "success": True,
                    "transcript": transcript,
                    "response": response_text,
                    "audio": speech,
                    "voice": selected_voice,
                    "error": None,
                }

            except asyncio.CancelledError:
                raise

            except Exception as error:
                self.total_errors += 1

                logger.exception(
                    "Voice pipeline failed."
                )

                return {
                    "success": False,
                    "transcript": "",
                    "response": "",
                    "audio": b"",
                    "voice": selected_voice,
                    "error": str(error),
                }

    # ========================================================
    # STATS
    # ========================================================

    def get_stats(
        self,
    ) -> dict[str, Any]:

        return {
            "chat_model": self.chat_model,
            "transcribe_model": self.transcribe_model,
            "tts_model": self.tts_model,
            "voice": self.current_voice,
            "memory_enabled": MEMORY_ENABLED,
            "memory_messages": self.memory_size(),
            "max_memory_messages": MAX_MEMORY_MESSAGES,
            "transcriptions": self.total_transcriptions,
            "responses": self.total_responses,
            "speeches": self.total_speeches,
            "errors": self.total_errors,
        }

    # ========================================================
    # CLOSE
    # ========================================================

    async def close(
        self,
    ) -> None:
        """
        Closes the underlying Gemini client if the installed
        SDK exposes an async close method.
        """

        try:
            aio_client = getattr(
                self.client,
                "aio",
                None,
            )

            if aio_client is not None:
                close_method = getattr(
                    aio_client,
                    "close",
                    None,
                )

                if close_method is not None:
                    result = close_method()

                    if asyncio.iscoroutine(result):
                        await result

        except Exception:
            logger.debug(
                "Gemini client close failed.",
                exc_info=True,
            )


# ============================================================
# FACTORY
# ============================================================

def create_gemini_engine() -> GeminiEngine:
    return GeminiEngine()


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "GeminiEngine",
    "create_gemini_engine",
]
