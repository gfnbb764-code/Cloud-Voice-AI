# gemini.py
# ============================================================
# Cloud Voice AI — Lightweight Gemini + Local Piper TTS Engine
#
# Pipeline:
# Discord PCM
#     ↓
# Gemini STT
#     ↓
# Gemini Flash-Lite AI
#     ↓
# Piper TTS (LOCAL)
#     ↓
# Discord PCM
#
# Designed for:
# - Low Gemini API consumption
# - NO Gemini TTS quota usage
# - Local/offline speech synthesis after voice download
# - Arabic speech
# - Short voice responses
# - Per-character personality/style/instructions
# - Conversation memory
# - No pointless 429 retries
# ============================================================

from __future__ import annotations

import asyncio
import audioop
import io
import logging
import wave
from collections import deque
from pathlib import Path
from typing import Any, Iterable

from google import genai
from google.genai import types

try:
    from piper import PiperVoice
    from piper.download_voices import download_voice
    from piper.config import SynthesisConfig
except ImportError as exc:
    raise RuntimeError(
        "Piper TTS is not installed. "
        "Install piper-tts==1.8.0 from requirements.txt."
    ) from exc


from config import (
    AI_SYSTEM_PROMPT,
    API_RETRIES,
    API_RETRY_DELAY_SECONDS,
    API_TIMEOUT_SECONDS,
    CHAT_MODEL,
    DEFAULT_GEMINI_VOICE,
    GEMINI_API_KEY,
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_TEMPERATURE,
    GEMINI_VOICES,
    MEMORY_ENABLED,
    MAX_MEMORY_MESSAGES,
    TRANSCRIBE_MODEL,
    normalize_speech_speed,
    normalize_voice_name,
)

logger = logging.getLogger(__name__)


# ============================================================
# LOCAL PIPER SETTINGS
# ============================================================

# Piper Arabic voice.
#
# Piper provides Arabic ar_JO voices. The medium Kareem voice
# is used here as the default local Arabic TTS model.
#
# Piper voice files are downloaded once and then loaded locally.
PIPER_VOICE_MODEL = "ar_JO-kareem-medium"

# Keep models in a predictable project-local cache directory.
PIPER_MODEL_DIR = (
    Path(__file__).resolve().parent
    / "piper_models"
)

PIPER_MODEL_PATH = (
    PIPER_MODEL_DIR
    / f"{PIPER_VOICE_MODEL}.onnx"
)

# Prevent multiple concurrent startup/download operations.
_PIPER_LOAD_LOCK = asyncio.Lock()

# Shared loaded Piper model.
_PIPER_VOICE: PiperVoice | None = None

# Piper generates at the model's own sample rate.
# voice.py expects Gemini-like TTS audio at 24kHz, so we
# resample Piper output to 24kHz before returning it.
TTS_OUTPUT_SAMPLE_RATE = 24000
TTS_OUTPUT_CHANNELS = 1
TTS_OUTPUT_SAMPLE_WIDTH = 2

# Maximum useful voice text.
MAX_TRANSCRIPT_LENGTH = 2500
MAX_RESPONSE_LENGTH = 1800

# Keep memory deliberately small.
VOICE_MEMORY_LIMIT = min(
    max(int(MAX_MEMORY_MESSAGES), 0),
    8,
)

DEFAULT_SPEECH_SPEED = 1.0

# Local Piper should be represented clearly in logs/stats.
LOCAL_TTS_NAME = f"piper/{PIPER_VOICE_MODEL}"


# ============================================================
# TEXT HELPERS
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
    character: Any,
    key: str,
    default: Any = None,
) -> Any:
    """
    Supports both Character objects and dictionaries.
    """

    if character is None:
        return default

    if isinstance(character, dict):
        try:
            return character.get(
                key,
                default,
            )
        except Exception:
            return default

    try:
        value = getattr(
            character,
            key,
            default,
        )
    except Exception:
        return default

    return (
        default
        if value is None
        else value
    )


# ============================================================
# RESPONSE TEXT EXTRACTION
# ============================================================

def _extract_response_text(
    response: Any,
) -> str:
    """
    Extract normal Gemini text.

    Also supports audio transcription metadata.
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
    Retry only temporary infrastructure problems.

    429 quota/rate-limit errors are NEVER retried.
    """

    if _is_quota_error(error):
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
# CHARACTER PROMPT
# ============================================================

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
# PIPER MODEL MANAGEMENT
# ============================================================

def _ensure_piper_model_sync() -> PiperVoice:
    """
    Ensure the Arabic Piper voice exists, downloading it once
    if necessary, then load it.

    This function is intentionally synchronous and is always
    executed inside asyncio.to_thread().
    """

    global _PIPER_VOICE

    if _PIPER_VOICE is not None:
        return _PIPER_VOICE

    PIPER_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = PIPER_MODEL_PATH

    if not model_path.exists():

        logger.info(
            "Downloading Piper voice model: %s",
            PIPER_VOICE_MODEL,
        )

        download_voice(
            PIPER_VOICE_MODEL,
            PIPER_MODEL_DIR,
        )

    if not model_path.exists():
        raise RuntimeError(
            "Piper voice model was not found after download: "
            f"{model_path}"
        )

    logger.info(
        "Loading local Piper voice: %s",
        PIPER_VOICE_MODEL,
    )

    _PIPER_VOICE = PiperVoice.load(
        str(model_path)
    )

    logger.info(
        "Piper voice loaded | model=%s | sample_rate=%s",
        PIPER_VOICE_MODEL,
        _PIPER_VOICE.config.sample_rate,
    )

    return _PIPER_VOICE


async def _get_piper_voice() -> PiperVoice:
    global _PIPER_VOICE

    if _PIPER_VOICE is not None:
        return _PIPER_VOICE

    async with _PIPER_LOAD_LOCK:

        if _PIPER_VOICE is not None:
            return _PIPER_VOICE

        return await asyncio.to_thread(
            _ensure_piper_model_sync
        )


# ============================================================
# PIPER AUDIO
# ============================================================

def _synthesize_piper_sync(
    voice_model: PiperVoice,
    text: str,
    speed: float,
) -> tuple[bytes, int]:
    """
    Generate a WAV in memory and return:
        raw PCM bytes
        source sample rate
    """

    output = io.BytesIO()

    # Piper's length_scale:
    #   > 1.0 = slower
    #   < 1.0 = faster
    #
    # Speech speed:
    #   1.0 = normal
    #
    length_scale = 1.0 / max(
        0.5,
        min(2.0, float(speed)),
    )

    synthesis_config = SynthesisConfig(
        length_scale=length_scale,
        volume=1.0,
        noise_scale=0.667,
        noise_w_scale=0.8,
        normalize_audio=True,
    )

    with wave.open(
        output,
        "wb",
    ) as wav_file:

        voice_model.synthesize_wav(
            text,
            wav_file,
            syn_config=synthesis_config,
        )

    output.seek(0)

    with wave.open(
        output,
        "rb",
    ) as wav_file:

        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        pcm = wav_file.readframes(
            wav_file.getnframes()
        )

    if channels != 1:
        raise RuntimeError(
            f"Unexpected Piper channel count: {channels}"
        )

    if sample_width != 2:
        raise RuntimeError(
            f"Unexpected Piper sample width: {sample_width}"
        )

    if not pcm:
        raise RuntimeError(
            "Piper generated empty audio."
        )

    return pcm, sample_rate


def _resample_piper_audio(
    pcm: bytes,
    source_rate: int,
    target_rate: int,
) -> bytes:

    if not pcm:
        return b""

    if source_rate == target_rate:
        return pcm

    try:

        converted, _ = audioop.ratecv(
            pcm,
            2,
            1,
            source_rate,
            target_rate,
            None,
        )

        return converted

    except Exception as exc:

        raise RuntimeError(
            "Failed to resample Piper audio."
        ) from exc


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
        # Gemini models:
        #
        # STT  -> TRANSCRIBE_MODEL
        # CHAT -> CHAT_MODEL
        #
        # TTS is LOCAL Piper.
        #
        # tts_model is accepted for compatibility with older
        # code, but it is intentionally not used for synthesis.
        # ----------------------------------------------------

        self.chat_model = (
            chat_model
            or CHAT_MODEL
        )

        self.transcribe_model = (
            transcribe_model
            or TRANSCRIBE_MODEL
        )

        self.tts_model = LOCAL_TTS_NAME

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
        operation_name: str = "Gemini request",
    ):
        """
        Execute a Gemini request.

        429 quota errors are never retried.
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
        mime_type: str = "audio/wav",
    ) -> str:

        if not audio:
            return ""

        async def operation():

            return await asyncio.to_thread(
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

            return await asyncio.to_thread(
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
    # TEXT TO SPEECH — LOCAL PIPER
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

        # The Gemini voice name remains useful to the rest of
        # the bot/UI, but synthesis itself uses local Piper.
        logger.info(
            "Local Piper TTS | "
            "model=%s | "
            "mapped_voice=%s | "
            "speed=%.2f",
            PIPER_VOICE_MODEL,
            selected_voice,
            selected_speed,
        )

        try:

            piper_voice = await _get_piper_voice()

            pcm, source_rate = await asyncio.to_thread(
                _synthesize_piper_sync,
                piper_voice,
                text,
                selected_speed,
            )

            output_pcm = _resample_piper_audio(
                pcm,
                source_rate,
                TTS_OUTPUT_SAMPLE_RATE,
            )

            if not output_pcm:
                raise RuntimeError(
                    "Piper returned empty audio."
                )

            logger.info(
                "Piper TTS generated | "
                "source_rate=%s | "
                "output_rate=%s | "
                "voice=%s | "
                "speed=%.2f | "
                "bytes=%s",
                source_rate,
                TTS_OUTPUT_SAMPLE_RATE,
                selected_voice,
                selected_speed,
                len(output_pcm),
            )

            return output_pcm

        except asyncio.CancelledError:
            raise

        except Exception:

            logger.exception(
                "Local Piper TTS failed"
            )

            raise


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
        mime_type: str = "audio/wav",
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

                result["transcript"] = transcript

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

                response = await self.generate_response(
                    transcript,
                    username=username,
                    memory=memory,
                    character=character,
                    system_prompt=system_prompt,
                )

                if not response:

                    result["error"] = (
                        "AI returned an empty response."
                    )

                    return result

                result["response"] = response

                # ============================================
                # MEMORY — ASSISTANT
                # ============================================

                self.add_assistant_message(
                    response
                )

                # ============================================
                # LOCAL PIPER TTS
                # ============================================

                selected_voice = (
                    self.set_voice(
                        voice
                    )
                    if voice
                    else self.voice
                )

                result["voice"] = selected_voice

                speech = await self.generate_speech(
                    response,
                    voice=selected_voice,
                    speed=speed,
                )

                result["audio"] = speech

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
                    LOCAL_TTS_NAME,
                    selected_voice,
                    normalize_speech_speed(
                        speed
                    ),
                )

                return result

            except asyncio.CancelledError:
                raise

            except Exception as error:

                self.failed_requests += 1

                result["error"] = str(
                    error
                )

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
            "tts_model": LOCAL_TTS_NAME,
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
