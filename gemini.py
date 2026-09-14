# gemini.py
# ============================================================
# Cloud Voice AI — Groq + Local Piper
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
# Piper TTS in a separate worker process
#     ↓
# Discord PCM
#
# Gemini is no longer used by this file.
# ============================================================

from __future__ import annotations

import asyncio
import audioop
import logging
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

# Limit CPU threading used by ONNX/Piper.
# These are defaults and can still be overridden by environment variables.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("ORT_NUM_THREADS", "1")

from groq import Groq

from piper import PiperVoice
from piper.config import SynthesisConfig
from piper.download_voices import download_voice

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

# Speech-to-text
GROQ_STT_MODEL = os.getenv(
    "GROQ_STT_MODEL",
    "whisper-large-v3-turbo",
).strip() or "whisper-large-v3-turbo"

# Chat / AI
GROQ_CHAT_MODEL = os.getenv(
    "GROQ_CHAT_MODEL",
    "openai/gpt-oss-20b",
).strip() or "openai/gpt-oss-20b"

# Local TTS
PIPER_VOICE_MODEL = os.getenv(
    "PIPER_MODEL",
    "ar_JO-kareem-low",
).strip() or "ar_JO-kareem-low"

LOCAL_TTS_NAME = (
    f"Piper/{PIPER_VOICE_MODEL}"
)


# ============================================================
# AUDIO
# ============================================================

DEFAULT_AUDIO_MIME_TYPE = "audio/wav"

# Piper output exposed to voice.py
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
# PIPER STORAGE
# ============================================================

PIPER_MODEL_DIR = Path(
    "piper_models"
)


# ============================================================
# MAIN-PROCESS PIPER STATE
# ============================================================

_PIPER_EXECUTOR: ProcessPoolExecutor | None = None


# ============================================================
# PIPER WORKER PROCESS STATE
# ============================================================

_WORKER_PIPER_VOICE: PiperVoice | None = None


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
# MARKDOWN CLEANUP
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
# PIPER MODEL PATHS
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


# ============================================================
# PIPER WORKER PROCESS
# ============================================================

def _piper_worker_init(
    model_dir: str,
    model_name: str,
) -> None:
    """
    Runs once when the dedicated Piper worker process starts.
    """

    global _WORKER_PIPER_VOICE

    # Re-apply CPU limits inside the child process.
    os.environ.setdefault(
        "OMP_NUM_THREADS",
        "1",
    )

    os.environ.setdefault(
        "ORT_NUM_THREADS",
        "1",
    )

    worker_model_dir = Path(
        model_dir
    )

    worker_model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        worker_model_dir
        / f"{model_name}.onnx"
    )

    config_path = (
        worker_model_dir
        / f"{model_name}.onnx.json"
    )

    if (
        not model_path.exists()
        or not config_path.exists()
    ):
        download_voice(
            model_name,
            worker_model_dir,
        )

    if not model_path.exists():
        raise RuntimeError(
            f"Piper model missing in worker: {model_path}"
        )

    _WORKER_PIPER_VOICE = PiperVoice.load(
        str(model_path)
    )


def _piper_worker_synthesize(
    text: str,
    speed: float,
) -> bytes:

    global _WORKER_PIPER_VOICE

    if _WORKER_PIPER_VOICE is None:
        raise RuntimeError(
            "Piper worker voice is not initialized."
        )

    text = _clean_text(
        text
    )

    if not text:
        return b""

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
    first_chunk_logged = True

    for chunk in _WORKER_PIPER_VOICE.synthesize(
        text,
        synthesis_config,
    ):

        if first_chunk_logged:

            logger.info(
                "Piper first audio chunk ready"
            )

            first_chunk_logged = False

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
        _WORKER_PIPER_VOICE.config.sample_rate
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
# PIPER EXECUTOR
# ============================================================

def _get_piper_executor() -> ProcessPoolExecutor:

    global _PIPER_EXECUTOR

    if _PIPER_EXECUTOR is None:

        _PIPER_EXECUTOR = ProcessPoolExecutor(
            max_workers=1,
            initializer=_piper_worker_init,
            initargs=(
                str(PIPER_MODEL_DIR),
                PIPER_VOICE_MODEL,
            ),
        )

        logger.info(
            "Piper worker process created | model=%s",
            PIPER_VOICE_MODEL,
        )

    return _PIPER_EXECUTOR


async def _synthesize_piper_process(
    text: str,
    speed: float,
) -> bytes:

    loop = asyncio.get_running_loop()

    executor = _get_piper_executor()

    future = loop.run_in_executor(
        executor,
        _piper_worker_synthesize,
        text,
        speed,
    )

    try:

        audio = await asyncio.wait_for(
            future,
            timeout=max(
                60.0,
                float(
                    API_TIMEOUT_SECONDS
                ),
            ),
        )

    except asyncio.TimeoutError as error:

        logger.error(
            "Piper TTS timed out after %.1fs",
            max(
                60.0,
                float(
                    API_TIMEOUT_SECONDS
                ),
            ),
        )

        raise RuntimeError(
            "Piper TTS timed out."
        ) from error

    return audio


# ============================================================
# GROQ ENGINE
# ============================================================

class GeminiEngine:
    """
    Kept under the old class name so the existing main.py and
    voice.py do not need to be rewritten.

    STT  -> Groq Whisper Large V3 Turbo
    Chat -> Groq GPT-OSS 20B
    TTS  -> Local Piper in a separate worker process
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

        # Actual TTS remains local Piper.
        self.tts_model = LOCAL_TTS_NAME

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

        filename = (
            "discord_audio.wav"
        )

        kwargs: dict[str, Any] = {
            "file": (
                filename,
                audio,
            ),
            "model": self.transcribe_model,
            "response_format": "json",
            "temperature": 0.0,

            # ==================================================
            # FORCE ARABIC
            # ==================================================
            "language": "ar",
        }

        return self.client.audio.transcriptions.create(
            **kwargs
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
            temperature=min(
                float(
                    os.getenv(
                        "GEMINI_TEMPERATURE",
                        "0.75",
                    )
                ),
                0.8,
            ),
            max_tokens=min(
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

        selected_speed = (
            normalize_speech_speed(
                speed
            )
        )

        logger.info(
            "Local Piper TTS | model=%s | "
            "mapped_voice=%s | speed=%.2f | process=separate",
            PIPER_VOICE_MODEL,
            selected_voice,
            selected_speed,
        )

        started = (
            asyncio.get_running_loop().time()
        )

        try:

            audio = await _synthesize_piper_process(
                text,
                selected_speed,
            )

        except Exception:

            logger.exception(
                "Piper TTS failed"
            )

            raise

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
            # LOCAL TTS
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
                self.transcribe_model,
                self.chat_model,
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
            "tts_model": LOCAL_TTS_NAME,
            "tts_local": True,
            "piper_voice": PIPER_VOICE_MODEL,
            "piper_worker": True,
        }

    # ========================================================
    # CLOSE
    # ========================================================

    async def close(
        self,
    ) -> None:

        global _PIPER_EXECUTOR

        self.client = None

        if _PIPER_EXECUTOR is not None:

            try:

                _PIPER_EXECUTOR.shutdown(
                    wait=False,
                    cancel_futures=True,
                )

            except Exception:

                logger.exception(
                    "Failed to shutdown Piper worker"
                )

            finally:

                _PIPER_EXECUTOR = None


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
