# gemini.py
# ============================================================
# Cloud Voice AI — Groq STT + Groq Chat + Groq Saudi TTS
#
# Pipeline:
#
# Discord PCM
#     ↓
# WAV 16kHz mono
#     ↓
# Groq Whisper Large V3 Turbo (Arabic forced)
#     ↓
# Groq GPT-OSS 20B
#     ↓
# Groq Orpheus Arabic Saudi TTS
#     ↓
# WAV -> PCM 24kHz mono
#     ↓
# Discord
#
# Gemini is no longer used by this file.
# Piper is completely removed.
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

GROQ_API_KEY = (
    os.getenv("GROQ_API_KEY", "")
    or ""
).strip()


# ============================================================
# MODELS
# ============================================================

GROQ_STT_MODEL = os.getenv(
    "GROQ_STT_MODEL",
    "whisper-large-v3-turbo",
).strip() or "whisper-large-v3-turbo"


GROQ_CHAT_MODEL = os.getenv(
    "GROQ_CHAT_MODEL",
    "openai/gpt-oss-20b",
).strip() or "openai/gpt-oss-20b"


GROQ_TTS_MODEL = os.getenv(
    "GROQ_TTS_MODEL",
    "canopylabs/orpheus-arabic-saudi",
).strip() or "canopylabs/orpheus-arabic-saudi"


# ============================================================
# GROQ ARABIC TTS VOICES
# ============================================================

# Your old Gemini-style voice names are kept for compatibility.
# They are mapped to Groq's Saudi Arabic voices.

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


DEFAULT_GROQ_TTS_VOICE = os.getenv(
    "GROQ_TTS_VOICE",
    "fahad",
).strip().lower() or "fahad"


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

# voice.py expects mono PCM at 24 kHz.
TTS_OUTPUT_SAMPLE_RATE = 24000
TTS_OUTPUT_CHANNELS = 1
TTS_OUTPUT_SAMPLE_WIDTH = 2

MAX_TRANSCRIPT_LENGTH = 2500
MAX_RESPONSE_LENGTH = 1800

# Orpheus Arabic Saudi currently accepts max 200 characters
# per TTS request.
MAX_TTS_CHARS = 200

VOICE_MEMORY_LIMIT = min(
    max(int(MAX_MEMORY_MESSAGES), 0),
    8,
)

DEFAULT_SPEECH_SPEED = 1.0


# ============================================================
# BASIC HELPERS
# ============================================================

def _clean_text(
    value: Any,
) -> str:

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
# TEXT CLEANUP FOR VOICE
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
# TTS TEXT SPLITTER
# ============================================================

def _split_tts_text(
    text: str,
    maximum: int = MAX_TTS_CHARS,
) -> list[str]:
    """
    Split text into chunks that fit Groq Orpheus' input limit.

    Prefers punctuation and spaces so sentences sound natural.
    """

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

        window = remaining[:maximum]

        split_at = -1

        for marker in punctuation:
            position = window.rfind(marker)

            if position > split_at:
                split_at = position

        # Avoid tiny chunks.
        if split_at < 80:
            split_at = window.rfind(" ")

        if split_at < 40:
            split_at = maximum

        chunk = remaining[:split_at].strip()

        if chunk:
            chunks.append(
                chunk
            )

        remaining = remaining[split_at:].strip()

    if remaining:
        chunks.append(
            remaining
        )

    return chunks


# ============================================================
# GROQ ENGINE
# ============================================================

class GeminiEngine:
    """
    Kept under the old class name so main.py and voice.py
    do not need to be rewritten.

    STT:
        Groq Whisper Large V3 Turbo

    Chat:
        Groq GPT-OSS 20B

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

        self.api_key = (
            api_key
            or GROQ_API_KEY
        ).strip()

        if not self.api_key:
            raise RuntimeError(
                "GROQ_API_KEY is missing."
            )

        self.chat_model = (
            chat_model
            or GROQ_CHAT_MODEL
        )

        self.transcribe_model = (
            transcribe_model
            or GROQ_STT_MODEL
        )

        # Actual TTS is Groq Orpheus.
        self.tts_model = (
            tts_model
            or GROQ_TTS_MODEL
        )

        self.client = Groq(
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

    def _get_tts_voice(self) -> str:

        mapped = GROQ_TTS_VOICE_MAP.get(
            self.current_voice
        )

        if mapped in VALID_GROQ_TTS_VOICES:
            return mapped

        if (
            DEFAULT_GROQ_TTS_VOICE
            in VALID_GROQ_TTS_VOICES
        ):
            return DEFAULT_GROQ_TTS_VOICE

        return "fahad"

    # ========================================================
    # ERROR / RETRY
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

        text = cls._error_text(
            error
        )

        markers = (
            "rate limit",
            "rate_limit",
            "quota",
            "too many requests",
            "429",
        )

        return any(
            marker in text
            for marker in markers
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

        markers = (
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
            for marker in markers
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

        if last_error is not None:
            raise last_error

        raise RuntimeError(
            f"{operation_name} failed."
        )

    # ========================================================
    # STT
    # ========================================================

    def _transcribe_sync(
        self,
        audio: bytes,
    ):

        return self.client.audio.transcriptions.create(
            file=(
                "discord_audio.wav",
                audio,
            ),
            model=self.transcribe_model,
            response_format="json",
            temperature=0.0,

            # Force Arabic.
            language="ar",
        )

    async def transcribe(
        self,
        audio: bytes,
        *,
        mime_type: str = DEFAULT_AUDIO_MIME_TYPE,
    ) -> str:

        if not audio:
            return ""

        response = await self._with_retry(
            lambda: self._transcribe_sync(
                audio
            ),
            operation_name="Groq STT",
        )

        transcript = _clean_text(
            getattr(
                response,
                "text",
                "",
            )
        )

        transcript = _limit_text(
            transcript,
            MAX_TRANSCRIPT_LENGTH,
        )

        if transcript:

            logger.info(
                "STT successful | model=%s | "
                "language=ar | transcript=%s",
                self.transcribe_model,
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
    # CHAT MESSAGES
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

            if role in {
                "assistant",
                "model",
            }:

                sdk_role = "assistant"

            else:

                sdk_role = "user"

            messages.append(
                {
                    "role": sdk_role,
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

    # ========================================================
    # CHAT
    # ========================================================

    def _generate_response_sync(
        self,
        messages: list[dict[str, str]],
        system_prompt: str,
    ):

        final_messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            *messages,
        ]

        return self.client.chat.completions.create(
            model=self.chat_model,
            messages=final_messages,

            # GPT-OSS reasoning is disabled for
            # fast voice responses and to ensure
            # the final answer is returned in content.
            include_reasoning=False,

            temperature=min(
                float(
                    os.getenv(
                        "GEMINI_TEMPERATURE",
                        "0.75",
                    )
                ),
                0.8,
            ),

            # Use completion tokens for GPT-OSS.
            max_completion_tokens=min(
                int(
                    os.getenv(
                        "GEMINI_MAX_OUTPUT_TOKENS",
                        "700",
                    )
                ),
                512,
            ),
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
            "Keep replies concise and natural.\n"
            "Usually answer in 1-3 short sentences.\n"
            "Match the user's language.\n"
            "If the user speaks Arabic, respond in Arabic.\n"
            "Do not use markdown.\n"
            "Do not use code blocks.\n"
            "Do not write long explanations unless explicitly asked."
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
            operation_name="Groq AI",
        )

        answer = ""

        try:

            choices = (
                getattr(
                    response,
                    "choices",
                    None,
                )
                or []
            )

            if choices:

                message = getattr(
                    choices[0],
                    "message",
                    None,
                )

                answer = _clean_text(
                    getattr(
                        message,
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

        if answer:

            logger.info(
                "AI response generated | model=%s | response=%s",
                self.chat_model,
                answer,
            )

        else:

            logger.warning(
                "AI returned an empty response | model=%s",
                self.chat_model,
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
        """
        Calls Groq Orpheus Arabic Saudi and returns
        the response as 24kHz mono PCM16.
        """

        text = _clean_text(
            text
        )

        if not text:
            return b""

        # Orpheus supports text input up to 200 chars.
        text = text[:MAX_TTS_CHARS]

        response = self.client.audio.speech.create(
            model=self.tts_model,
            voice=voice,
            input=text,
            response_format="wav",
        )

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

            # Convert to 16-bit PCM if needed.
            if sample_width != 2:

                pcm = audioop.lin2lin(
                    pcm,
                    sample_width,
                    2,
                )

                sample_width = 2

            # Convert stereo/multi-channel to mono.
            if channels > 1:

                if channels == 2:

                    pcm = audioop.tomono(
                        pcm,
                        2,
                        0.5,
                        0.5,
                    )

                else:

                    # For uncommon channel counts,
                    # keep the first channel.
                    pcm = audioop.tomono(
                        pcm,
                        2,
                        1.0,
                        0.0,
                    )

                channels = 1

            # Resample to the format voice.py expects.
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
                    Path(
                        temp_path
                    ).unlink(
                        missing_ok=True
                    )

                except Exception:

                    logger.warning(
                        "Failed to remove temporary TTS file: %s",
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

        selected_speed = normalize_speech_speed(
            speed
        )

        groq_voice = self._get_tts_voice()

        logger.info(
            "Groq TTS | model=%s | mapped_voice=%s | "
            "voice=%s | speed=%.2f",
            self.tts_model,
            selected_voice,
            groq_voice,
            selected_speed,
        )

        # Speed support varies by Groq TTS model/API version.
        # We keep the public speed setting for compatibility,
        # but the Saudi Orpheus model currently does not expose
        # a speed parameter in the documented endpoint.
        if selected_speed != 1.0:

            logger.info(
                "TTS speed=%.2f requested; "
                "Groq Orpheus Saudi endpoint will use native speed.",
                selected_speed,
            )

        chunks = _split_tts_text(
            text,
            MAX_TTS_CHARS,
        )

        if not chunks:
            return b""

        output_parts: list[bytes] = []

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
                lambda chunk=chunk: self._generate_tts_chunk_sync(
                    chunk,
                    groq_voice,
                    selected_speed,
                ),
                operation_name=(
                    f"Groq TTS chunk {index}/{len(chunks)}"
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
            "Groq TTS generated | model=%s | "
            "voice=%s | chunks=%d | bytes=%d | %.2fs",
            self.tts_model,
            groq_voice,
            len(chunks),
            len(final_audio),
            elapsed,
        )

        return final_audio

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
            "character": _character_value(
                character,
                "name",
                None,
            ),
            "tts_model": self.tts_model,
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
            # MEMORY - USER
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

            # ==================================================
            # MEMORY - ASSISTANT
            # ==================================================

            self.add_assistant_message(
                response
            )

            # ==================================================
            # TTS
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
                    "Groq TTS returned no audio."
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
                self.transcribe_model,
                self.chat_model,
                self.tts_model,
                self._get_tts_voice(),
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

            if self._is_quota_error(
                error
            ):

                self.quota_exhausted = True

                logger.error(
                    "Groq rate limit/quota reached."
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
            "stt_language": "ar",
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

        self.client = None


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
