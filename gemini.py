# ============================================================
# AI VOICE BOT
# gemini.py
# ============================================================
#
# Gemini AI Engine
#
# المسؤوليات:
# - الاتصال بـ Gemini
# - تحويل الصوت إلى نص STT
# - إرسال النص إلى Gemini Chat
# - إنشاء رد AI
# - تحويل الرد إلى صوت TTS
# - إدارة الذاكرة
# - إدارة الأخطاء وإعادة المحاولة
#
# ============================================================

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Optional

from google import genai
from google.genai import types

from config import (
    GEMINI_API_KEY,
    CHAT_MODEL,
    TRANSCRIBE_MODEL,
    TTS_MODEL,
    DEFAULT_GEMINI_VOICE,
    GEMINI_VOICES,
    AI_SYSTEM_PROMPT,
    MAX_MEMORY_MESSAGES,
    GEMINI_TEMPERATURE,
    GEMINI_MAX_OUTPUT_TOKENS,
)


# ============================================================
# LOGGER
# ============================================================

logger = logging.getLogger(
    "ai_voice_bot.gemini"
)


# ============================================================
# CONSTANTS
# ============================================================

MAX_RETRIES = 3

RETRY_BASE_DELAY = 1.5

REQUEST_TIMEOUT = 45.0

MAX_AUDIO_BYTES = 15 * 1024 * 1024

MAX_TEXT_LENGTH = 12000


# ============================================================
# GEMINI ENGINE
# ============================================================

class GeminiEngine:
    """
    محرك Gemini الكامل للبوت الصوتي.

    Pipeline:

        Discord Audio
              ↓
        STT / Transcription
              ↓
        Gemini Chat
              ↓
        Text Response
              ↓
        Gemini TTS
              ↓
        Raw PCM Audio
    """

    def __init__(
        self,
    ):

        self.api_key = (
            GEMINI_API_KEY
            or os.getenv(
                "GEMINI_API_KEY"
            )
        )

        if not self.api_key:

            raise RuntimeError(
                "GEMINI_API_KEY is missing."
            )

        # ----------------------------------------------------
        # Gemini Client
        # ----------------------------------------------------

        self.client = genai.Client(
            api_key=self.api_key
        )

        # ----------------------------------------------------
        # Models
        # ----------------------------------------------------

        self.chat_model = (
            CHAT_MODEL
        )

        self.transcribe_model = (
            TRANSCRIBE_MODEL
        )

        self.tts_model = (
            TTS_MODEL
        )

        # ----------------------------------------------------
        # Memory
        # ----------------------------------------------------

        self.memory: list[
            dict[str, Any]
        ] = []

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        self.requests = 0

        self.transcriptions = 0

        self.chat_requests = 0

        self.tts_requests = 0

        self.errors = 0

        self.last_request_at = 0.0

        # ----------------------------------------------------
        # Closed
        # ----------------------------------------------------

        self.closed = False

        logger.info(
            "GeminiEngine initialized | chat=%s | stt=%s | tts=%s",
            self.chat_model,
            self.transcribe_model,
            self.tts_model,
        )

    # ========================================================
    # PROCESS VOICE
    # ========================================================

    async def process_voice(
        self,
        audio: bytes,
        memory: Optional[list[dict[str, Any]]] = None,
        username: Optional[str] = None,
        voice: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """
        يعالج مقطع صوت كامل.

        Returns:

        {
            "text": "نص المستخدم",
            "response": "رد Gemini",
            "audio": b"..."
        }
        """

        if self.closed:

            logger.warning(
                "GeminiEngine is closed."
            )

            return None

        if not audio:

            return None

        if len(audio) > MAX_AUDIO_BYTES:

            logger.warning(
                "Audio exceeds maximum size."
            )

            return None

        try:

            # ------------------------------------------------
            # STT
            # ------------------------------------------------

            text = await self.transcribe(
                audio
            )

            if not text:

                return None

            text = text.strip()

            if not text:

                return None

            logger.info(
                "Transcription: %s",
                text,
            )

            # ------------------------------------------------
            # Memory
            # ------------------------------------------------

            active_memory = (
                memory
                if memory is not None
                else self.memory
            )

            # ------------------------------------------------
            # Chat
            # ------------------------------------------------

            response = await self.generate_response(
                text=text,
                memory=active_memory,
                username=username,
            )

            if not response:

                return {
                    "text": text,
                    "response": "",
                    "audio": None,
                }

            # ------------------------------------------------
            # Voice
            # ------------------------------------------------

            selected_voice = (
                voice
                or DEFAULT_GEMINI_VOICE
            )

            if (
                selected_voice
                not in GEMINI_VOICES
            ):

                logger.warning(
                    "Invalid voice %s, using %s",
                    selected_voice,
                    DEFAULT_GEMINI_VOICE,
                )

                selected_voice = (
                    DEFAULT_GEMINI_VOICE
                )

            audio_response = (
                await self.generate_speech(
                    text=response,
                    voice=selected_voice,
                )
            )

            return {
                "text": text,
                "response": response,
                "audio": audio_response,
            }

        except asyncio.CancelledError:

            raise

        except Exception:

            self.errors += 1

            logger.exception(
                "process_voice failed"
            )

            return None

    # ========================================================
    # TRANSCRIBE
    # ========================================================

    async def transcribe(
        self,
        audio: bytes,
    ) -> Optional[str]:
        """
        يحول WAV إلى نص باستخدام Gemini.
        """

        if not audio:

            return None

        if len(audio) > MAX_AUDIO_BYTES:

            raise ValueError(
                "Audio file is too large."
            )

        self.transcriptions += 1

        self.requests += 1

        self.last_request_at = (
            time.monotonic()
        )

        # ----------------------------------------------------
        # Gemini API
        # ----------------------------------------------------

        async def request():

            response = (
                await self.client.aio.models.generate_content(
                    model=self.transcribe_model,
                    contents=[
                        types.Part.from_bytes(
                            data=audio,
                            mime_type="audio/wav",
                        ),
                        (
                            "Transcribe exactly what the user says. "
                            "Return only the spoken text. "
                            "Do not add explanations, labels, "
                            "descriptions, or commentary."
                        ),
                    ],
                )
            )

            return response

        response = await self._with_retry(
            request
        )

        if response is None:

            return None

        text = getattr(
            response,
            "text",
            None,
        )

        if not text:

            return None

        return text.strip()

    # ========================================================
    # GENERATE RESPONSE
    # ========================================================

    async def generate_response(
        self,
        text: str,
        memory: Optional[list[dict[str, Any]]] = None,
        username: Optional[str] = None,
    ) -> Optional[str]:
        """
        يرسل النص إلى Gemini ويولد الرد.
        """

        if not text:

            return None

        text = text.strip()

        if not text:

            return None

        if len(text) > MAX_TEXT_LENGTH:

            text = text[
                :MAX_TEXT_LENGTH
            ]

        self.chat_requests += 1

        self.requests += 1

        self.last_request_at = (
            time.monotonic()
        )

        # ----------------------------------------------------
        # Memory
        # ----------------------------------------------------

        active_memory = (
            memory
            if memory is not None
            else self.memory
        )

        memory_text = (
            self._format_memory(
                active_memory
            )
        )

        # ----------------------------------------------------
        # Username
        # ----------------------------------------------------

        user_label = (
            username
            if username
            else "User"
        )

        # ----------------------------------------------------
        # Prompt
        # ----------------------------------------------------

        prompt_parts = []

        if AI_SYSTEM_PROMPT:

            prompt_parts.append(
                AI_SYSTEM_PROMPT
            )

        prompt_parts.append(
            "\nYou are currently talking "
            "inside a Discord voice channel."
        )

        prompt_parts.append(
            "Keep your answer natural and "
            "comfortable for spoken conversation."
        )

        prompt_parts.append(
            "Do not use markdown unless necessary."
        )

        prompt_parts.append(
            "Do not describe your internal reasoning."
        )

        prompt_parts.append(
            f"\nCurrent speaker: {user_label}"
        )

        if memory_text:

            prompt_parts.append(
                "\nConversation memory:"
            )

            prompt_parts.append(
                memory_text
            )

        prompt_parts.append(
            "\nCurrent user message:"
        )

        prompt_parts.append(
            text
        )

        prompt = "\n".join(
            prompt_parts
        )

        # ----------------------------------------------------
        # Request
        # ----------------------------------------------------

        async def request():

            config_kwargs = {
                "temperature": GEMINI_TEMPERATURE,
                "max_output_tokens": GEMINI_MAX_OUTPUT_TOKENS,
            }

            response = (
                await self.client.aio.models.generate_content(
                    model=self.chat_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        **config_kwargs
                    ),
                )
            )

            return response

        response = await self._with_retry(
            request
        )

        if response is None:

            return None

        result = getattr(
            response,
            "text",
            None,
        )

        if not result:

            return None

        result = result.strip()

        if not result:

            return None

        # ----------------------------------------------------
        # تحديث Memory الداخلية
        # ----------------------------------------------------

        self._add_internal_memory(
            role="user",
            content=text,
            username=username,
        )

        self._add_internal_memory(
            role="assistant",
            content=result,
        )

        return result

    # ========================================================
    # GENERATE SPEECH
    # ========================================================

    async def generate_speech(
        self,
        text: str,
        voice: Optional[str] = None,
    ) -> Optional[bytes]:
        """
        يحول نص Gemini إلى raw PCM.

        Gemini TTS:

            24000Hz
            Mono
            16-bit PCM
        """

        if not text:

            return None

        text = text.strip()

        if not text:

            return None

        if len(text) > MAX_TEXT_LENGTH:

            text = text[
                :MAX_TEXT_LENGTH
            ]

        selected_voice = (
            voice
            or DEFAULT_GEMINI_VOICE
        )

        if (
            selected_voice
            not in GEMINI_VOICES
        ):

            selected_voice = (
                DEFAULT_GEMINI_VOICE
            )

        self.tts_requests += 1

        self.requests += 1

        self.last_request_at = (
            time.monotonic()
        )

        # ----------------------------------------------------
        # TTS request
        # ----------------------------------------------------

        async def request():

            response = (
                await self.client.aio.models.generate_content(
                    model=self.tts_model,
                    contents=text,
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
            )

            return response

        response = await self._with_retry(
            request
        )

        if response is None:

            return None

        # ----------------------------------------------------
        # استخراج الصوت
        # ----------------------------------------------------

        audio = self._extract_audio(
            response
        )

        if not audio:

            logger.warning(
                "Gemini TTS returned no audio."
            )

            return None

        return audio

    # ========================================================
    # EXTRACT AUDIO
    # ========================================================

    def _extract_audio(
        self,
        response: Any,
    ) -> Optional[bytes]:
        """
        استخراج PCM من استجابة Gemini TTS.
        """

        try:

            candidates = getattr(
                response,
                "candidates",
                None,
            )

            if not candidates:

                return None

            for candidate in candidates:

                content = getattr(
                    candidate,
                    "content",
                    None,
                )

                if not content:

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

                    if not inline_data:

                        continue

                    data = getattr(
                        inline_data,
                        "data",
                        None,
                    )

                    if isinstance(
                        data,
                        bytes,
                    ):

                        return data

                    if isinstance(
                        data,
                        bytearray,
                    ):

                        return bytes(
                            data
                        )

        except Exception:

            logger.exception(
                "Could not extract TTS audio."
            )

        return None

    # ========================================================
    # FORMAT MEMORY
    # ========================================================

    def _format_memory(
        self,
        memory: Optional[
            list[dict[str, Any]]
        ],
    ) -> str:
        """
        يحول الذاكرة إلى نص يفهمه Gemini.
        """

        if not memory:

            return ""

        recent = list(
            memory[
                -MAX_MEMORY_MESSAGES:
            ]
        )

        lines = []

        for item in recent:

            if not isinstance(
                item,
                dict,
            ):

                continue

            role = item.get(
                "role",
                "user",
            )

            content = item.get(
                "content",
                "",
            )

            username = item.get(
                "username",
            )

            if not content:

                continue

            if username:

                lines.append(
                    f"{username} ({role}): {content}"
                )

            else:

                lines.append(
                    f"{role}: {content}"
                )

        return "\n".join(
            lines
        )

    # ========================================================
    # INTERNAL MEMORY
    # ========================================================

    def _add_internal_memory(
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

        logger.info(
            "Gemini internal memory cleared."
        )

    # ========================================================
    # RESET
    # ========================================================

    async def reset(
        self,
    ) -> None:

        self.clear_memory()

        logger.info(
            "GeminiEngine reset."
        )

    # ========================================================
    # RETRY
    # ========================================================

    async def _with_retry(
        self,
        operation,
    ):
        """
        إعادة المحاولة تلقائيًا.

        المحاولات:
            1
            2
            3
        """

        last_error = None

        for attempt in range(
            1,
            MAX_RETRIES + 1,
        ):

            try:

                result = await asyncio.wait_for(
                    operation(),
                    timeout=REQUEST_TIMEOUT,
                )

                return result

            except asyncio.CancelledError:

                raise

            except Exception as error:

                last_error = error

                error_text = str(
                    error
                ).lower()

                # --------------------------------------------
                # أخطاء لا تستحق إعادة المحاولة
                # --------------------------------------------

                permanent_errors = (
                    "api key",
                    "permission denied",
                    "unauthorized",
                    "invalid argument",
                    "invalid api key",
                )

                if any(
                    item in error_text
                    for item in permanent_errors
                ):

                    logger.error(
                        "Permanent Gemini error: %s",
                        error,
                    )

                    break

                logger.warning(
                    "Gemini request failed "
                    "(attempt %s/%s): %s",
                    attempt,
                    MAX_RETRIES,
                    error,
                )

                if (
                    attempt
                    >= MAX_RETRIES
                ):

                    break

                delay = (
                    RETRY_BASE_DELAY
                    * attempt
                )

                await asyncio.sleep(
                    delay
                )

        self.errors += 1

        logger.error(
            "Gemini request failed after %s attempts: %s",
            MAX_RETRIES,
            last_error,
        )

        return None

    # ========================================================
    # CLOSE
    # ========================================================

    async def close(
        self,
    ) -> None:

        if self.closed:
            return

        self.closed = True

        self.memory.clear()

        # ----------------------------------------------------
        # google-genai client الحالي
        # لا يحتاج إغلاقًا إجباريًا
        # ----------------------------------------------------

        self.client = None

        logger.info(
            "GeminiEngine closed."
        )
