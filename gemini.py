# ============================================================
# AI VOICE BOT
# gemini.py
# ============================================================
#
# Google Gemini AI Engine
#
# الوظائف:
#
# 1. Speech To Text
# 2. Gemini 3.5 Flash-Lite
# 3. Gemini TTS
# 4. تغيير الصوت
# 5. ذاكرة المحادثة
# 6. Retry
# 7. Timeout
# 8. معالجة أخطاء API
# 9. حماية من الردود الفارغة
# 10. تنظيف الاتصالات
#
# ============================================================

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import random
import time

from typing import Optional

from google import genai
from google.genai import types


# ============================================================
# LOGGER
# ============================================================

logger = logging.getLogger(
    "ai_voice_bot.gemini"
)


# ============================================================
# ENVIRONMENT
# ============================================================

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

if not GEMINI_API_KEY:

    raise RuntimeError(
        "GEMINI_API_KEY غير موجود في Environment Variables."
    )


# ============================================================
# MODELS
# ============================================================

# العقل الأساسي للبوت
CHAT_MODEL = os.getenv(
    "GEMINI_CHAT_MODEL",
    "gemini-3.5-flash-lite"
)


# تحويل الصوت إلى نص
TRANSCRIBE_MODEL = os.getenv(
    "GEMINI_TRANSCRIBE_MODEL",
    "gemini-3.5-transcribe"
)


# تحويل النص إلى صوت
TTS_MODEL = os.getenv(
    "GEMINI_TTS_MODEL",
    "gemini-3.1-flash-tts-preview"
)


# ============================================================
# DEFAULT VOICE
# ============================================================

DEFAULT_VOICE = os.getenv(
    "GEMINI_DEFAULT_VOICE",
    "Kore"
)


# ============================================================
# AVAILABLE VOICES
# ============================================================
#
# هذه أسماء أصوات Gemini TTS الرسمية.
#
# ============================================================

GEMINI_VOICES = [
    "Zephyr",
    "Puck",
    "Charon",
    "Kore",
    "Fenrir",
    "Leda",
    "Orus",
    "Aoede",
    "Callirrhoe",
    "Autonoe",
    "Enceladus",
    "Iapetus",
    "Umbriel",
    "Algieba",
    "Despina",
    "Erinome",
    "Algenib",
    "Rasalgethi",
    "Laomedeia",
    "Achernar",
    "Alnilam",
    "Gacrux",
    "Pulcherrima",
    "Achird",
    "Zubenelgenubi",
    "Vindemiatrix",
    "Sadachbia",
    "Sadaltager",
    "Sulafat",
]


# ============================================================
# VOICE DESCRIPTIONS
# ============================================================

VOICE_DESCRIPTIONS = {

    "Zephyr":
        "Bright",

    "Puck":
        "Upbeat",

    "Charon":
        "Informative",

    "Kore":
        "Firm",

    "Fenrir":
        "Excitable",

    "Leda":
        "Youthful",

    "Orus":
        "Firm",

    "Aoede":
        "Breezy",

    "Callirrhoe":
        "Easy-going",

    "Autonoe":
        "Bright",

    "Enceladus":
        "Breathy",

    "Iapetus":
        "Clear",

    "Umbriel":
        "Easy-going",

    "Algieba":
        "Smooth",

    "Despina":
        "Smooth",

    "Erinome":
        "Clear",

    "Algenib":
        "Gravelly",

    "Rasalgethi":
        "Informative",

    "Laomedeia":
        "Upbeat",

    "Achernar":
        "Soft",

    "Alnilam":
        "Firm",

    "Gacrux":
        "Mature",

    "Pulcherrima":
        "Forward",

    "Achird":
        "Friendly",

    "Zubenelgenubi":
        "Casual",

    "Vindemiatrix":
        "Gentle",

    "Sadachbia":
        "Lively",

    "Sadaltager":
        "Knowledgeable",

    "Sulafat":
        "Warm",
}


# ============================================================
# AI PERSONALITY
# ============================================================

SYSTEM_PROMPT = os.getenv(
    "GEMINI_SYSTEM_PROMPT",
    """
أنت مساعد ذكاء اصطناعي صوتي داخل Discord.

شخصيتك:
- طبيعي جدًا.
- عفوي.
- ذكي.
- سريع في الرد.
- تتكلم باللهجة السعودية بشكل طبيعي.
- لا تستخدم أسلوبًا رسميًا بشكل مبالغ فيه.
- لا تكرر السؤال.
- لا تكرر نفس الجملة.
- لا تبدأ كل رد بكلمة "بالتأكيد".
- لا تستخدم Markdown في الرد الصوتي.
- اجعل ردودك مناسبة للمحادثة الصوتية.
- إذا كان السؤال بسيطًا، أجب باختصار.
- إذا كان الموضوع يحتاج شرحًا، اشرح بوضوح.
- إذا لم تفهم الكلام، قل للشخص يعيده.
- لا تدّعي أنك إنسان.
"""
)


# ============================================================
# LIMITS
# ============================================================

MAX_MEMORY_ITEMS = int(
    os.getenv(
        "GEMINI_MAX_MEMORY",
        "12"
    )
)


MAX_RESPONSE_CHARS = int(
    os.getenv(
        "GEMINI_MAX_RESPONSE_CHARS",
        "700"
    )
)


MAX_RETRIES = int(
    os.getenv(
        "GEMINI_MAX_RETRIES",
        "3"
    )
)


REQUEST_TIMEOUT = float(
    os.getenv(
        "GEMINI_REQUEST_TIMEOUT",
        "35"
    )
)


# ============================================================
# CLIENT
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# GEMINI ENGINE
# ============================================================

class GeminiEngine:
    """
    محرك Gemini الكامل.

    لا يحتفظ بملفات صوتية على القرص.
    """

    def __init__(self):

        self.closed = False

        self.current_voice = (
            DEFAULT_VOICE
        )

        self.total_requests = 0

        self.total_transcriptions = 0

        self.total_tts = 0

        self.total_errors = 0

        self.last_request_time = 0.0

        self.request_lock = (
            asyncio.Lock()
        )

        logger.info(
            "GeminiEngine initialized."
        )

        logger.info(
            "Chat model: %s",
            CHAT_MODEL
        )

        logger.info(
            "Transcribe model: %s",
            TRANSCRIBE_MODEL
        )

        logger.info(
            "TTS model: %s",
            TTS_MODEL
        )

        logger.info(
            "Default voice: %s",
            DEFAULT_VOICE
        )

    # ========================================================
    # SET VOICE
    # ========================================================

    def set_voice(
        self,
        voice_name: str
    ) -> bool:

        if self.closed:
            return False

        voice_name = (
            voice_name
            .strip()
        )

        if (
            voice_name
            not in GEMINI_VOICES
        ):

            return False

        self.current_voice = (
            voice_name
        )

        return True

    # ========================================================
    # GET VOICE
    # ========================================================

    def get_voice(
        self
    ) -> str:

        return self.current_voice

    # ========================================================
    # VOICE EXISTS
    # ========================================================

    @staticmethod
    def voice_exists(
        voice_name: str
    ) -> bool:

        return (
            voice_name
            in GEMINI_VOICES
        )

    # ========================================================
    # LIST VOICES
    # ========================================================

    @staticmethod
    def list_voices():

        return list(
            GEMINI_VOICES
        )

    # ========================================================
    # TRANSCRIBE
    # ========================================================

    async def transcribe(
        self,
        audio: bytes
    ) -> str:

        if self.closed:

            raise RuntimeError(
                "GeminiEngine مغلق."
            )

        if not audio:

            return ""

        self.total_transcriptions += 1

        loop = (
            asyncio.get_running_loop()
        )

        last_error = None

        for attempt in range(
            MAX_RETRIES
        ):

            try:

                result = await asyncio.wait_for(

                    loop.run_in_executor(
                        None,
                        self._transcribe_sync,
                        audio
                    ),

                    timeout=REQUEST_TIMEOUT
                )

                result = (
                    result
                    or ""
                ).strip()

                return result

            except Exception as error:

                last_error = error

                self.total_errors += 1

                logger.warning(
                    "Transcription attempt %s failed: %s",
                    attempt + 1,
                    error
                )

                if (
                    attempt
                    >= MAX_RETRIES - 1
                ):

                    break

                await asyncio.sleep(
                    self._backoff(
                        attempt
                    )
                )

        raise RuntimeError(
            f"Gemini STT failed: {last_error}"
        )

    # ========================================================
    # TRANSCRIBE SYNC
    # ========================================================

    def _transcribe_sync(
        self,
        audio: bytes
    ) -> str:

        response = client.models.generate_content(

            model=TRANSCRIBE_MODEL,

            contents=[
                types.Part.from_bytes(
                    data=audio,
                    mime_type="audio/wav"
                ),

                (
                    "حوّل الكلام الموجود في "
                    "هذا التسجيل إلى نص عربي. "
                    "لا تضف شرحًا ولا تعليقًا. "
                    "اكتب الكلام المنطوق فقط."
                ),
            ],
        )

        text = getattr(
            response,
            "text",
            None
        )

        return (
            text
            or ""
        )

    # ========================================================
    # GENERATE RESPONSE
    # ========================================================

    async def generate_response(
        self,
        text: str,
        memory: Optional[list] = None,
        username: Optional[str] = None
    ) -> str:

        if self.closed:

            raise RuntimeError(
                "GeminiEngine مغلق."
            )

        text = (
            text
            or ""
        ).strip()

        if not text:

            return ""

        # ----------------------------------------------------
        # بناء السياق
        # ----------------------------------------------------

        prompt = self.build_prompt(
            text=text,
            memory=memory or [],
            username=username
        )

        loop = (
            asyncio.get_running_loop()
        )

        last_error = None

        for attempt in range(
            MAX_RETRIES
        ):

            try:

                async with (
                    self.request_lock
                ):

                    self.total_requests += 1

                    self.last_request_time = (
                        time.monotonic()
                    )

                    response = await asyncio.wait_for(

                        loop.run_in_executor(
                            None,
                            self._generate_sync,
                            prompt
                        ),

                        timeout=REQUEST_TIMEOUT
                    )

                response = (
                    response
                    or ""
                ).strip()

                response = (
                    self.clean_response(
                        response
                    )
                )

                return response

            except Exception as error:

                last_error = error

                self.total_errors += 1

                logger.warning(
                    "Gemini response attempt %s failed: %s",
                    attempt + 1,
                    error
                )

                if (
                    attempt
                    >= MAX_RETRIES - 1
                ):

                    break

                await asyncio.sleep(
                    self._backoff(
                        attempt
                    )
                )

        raise RuntimeError(
            f"Gemini response failed: {last_error}"
        )

    # ========================================================
    # GENERATE SYNC
    # ========================================================

    def _generate_sync(
        self,
        prompt: str
    ) -> str:

        response = client.models.generate_content(

            model=CHAT_MODEL,

            contents=prompt,

            config=types.GenerateContentConfig(

                system_instruction=(
                    SYSTEM_PROMPT
                ),

                temperature=0.8,

                max_output_tokens=250,
            ),
        )

        text = getattr(
            response,
            "text",
            None
        )

        return (
            text
            or ""
        )

    # ========================================================
    # BUILD PROMPT
    # ========================================================

    def build_prompt(
        self,
        text: str,
        memory: list,
        username: Optional[str]
    ) -> str:

        lines = []

        lines.append(
            "السياق السابق:"
        )

        # ----------------------------------------------------
        # Memory
        # ----------------------------------------------------

        recent_memory = (
            memory[
                -MAX_MEMORY_ITEMS:
            ]
        )

        for item in recent_memory:

            role = item.get(
                "role",
                "user"
            )

            content = item.get(
                "content",
                ""
            )

            item_username = item.get(
                "username"
            )

            if item_username:

                lines.append(
                    f"{item_username} "
                    f"({role}): {content}"
                )

            else:

                lines.append(
                    f"{role}: {content}"
                )

        # ----------------------------------------------------
        # Current speaker
        # ----------------------------------------------------

        if username:

            lines.append(
                ""
            )

            lines.append(
                f"المتحدث الحالي: {username}"
            )

        lines.append(
            ""
        )

        lines.append(
            "الرسالة الحالية:"
        )

        lines.append(
            text
        )

        lines.append(
            ""
        )

        lines.append(
            "أجب على الرسالة الحالية."
        )

        return "\n".join(
            lines
        )

    # ========================================================
    # TEXT TO SPEECH
    # ========================================================

    async def text_to_speech(
        self,
        text: str,
        voice: Optional[str] = None
    ) -> bytes:

        if self.closed:

            raise RuntimeError(
                "GeminiEngine مغلق."
            )

        text = (
            text
            or ""
        ).strip()

        if not text:

            return b""

        selected_voice = (
            voice
            or self.current_voice
        )

        if not self.voice_exists(
            selected_voice
        ):

            raise ValueError(
                f"Voice غير موجود: {selected_voice}"
            )

        self.total_tts += 1

        loop = (
            asyncio.get_running_loop()
        )

        last_error = None

        for attempt in range(
            MAX_RETRIES
        ):

            try:

                audio = await asyncio.wait_for(

                    loop.run_in_executor(
                        None,
                        self._tts_sync,
                        text,
                        selected_voice
                    ),

                    timeout=REQUEST_TIMEOUT
                )

                if audio:

                    return audio

                raise RuntimeError(
                    "Gemini TTS returned empty audio."
                )

            except Exception as error:

                last_error = error

                self.total_errors += 1

                logger.warning(
                    "TTS attempt %s failed: %s",
                    attempt + 1,
                    error
                )

                if (
                    attempt
                    >= MAX_RETRIES - 1
                ):

                    break

                await asyncio.sleep(
                    self._backoff(
                        attempt
                    )
                )

        raise RuntimeError(
            f"Gemini TTS failed: {last_error}"
        )

    # ========================================================
    # TTS SYNC
    # ========================================================

    def _tts_sync(
        self,
        text: str,
        voice: str
    ) -> bytes:

        # ----------------------------------------------------
        # توجيه الصوت
        # ----------------------------------------------------

        prompt = (
            "تحدث باللغة العربية السعودية "
            "بصوت طبيعي وواضح.\n"
            "لا تقرأ التعليمات التالية بصوت عالٍ.\n"
            "اقرأ النص فقط.\n\n"
            f"النص:\n{text}"
        )

        response = (
            client.models.generate_content(

                model=TTS_MODEL,

                contents=prompt,

                config=types.GenerateContentConfig(

                    response_modalities=[
                        "AUDIO"
                    ],

                    speech_config=types.SpeechConfig(

                        voice_config=(
                            types.VoiceConfig(

                                prebuilt_voice_config=(
                                    types.PrebuiltVoiceConfig(
                                        voice_name=voice
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )

        # ----------------------------------------------------
        # البحث عن audio part
        # ----------------------------------------------------

        try:

            candidates = (
                response.candidates
            )

            if not candidates:

                return b""

            content = (
                candidates[0].content
            )

            if not content:

                return b""

            parts = (
                content.parts
            )

            for part in parts:

                inline_data = getattr(
                    part,
                    "inline_data",
                    None
                )

                if not inline_data:
                    continue

                data = getattr(
                    inline_data,
                    "data",
                    None
                )

                if not data:
                    continue

                # --------------------------------------------
                # SDK قد يرجع bytes مباشرة
                # أو بيانات قابلة للتحويل
                # --------------------------------------------

                if isinstance(
                    data,
                    bytes
                ):

                    return data

                if isinstance(
                    data,
                    str
                ):

                    try:

                        return base64.b64decode(
                            data
                        )

                    except Exception:

                        return data.encode()

        except Exception:

            logger.exception(
                "Could not extract TTS audio."
            )

        return b""

    # ========================================================
    # PROCESS VOICE
    # ========================================================

    async def process_voice(
        self,
        audio: bytes,
        memory: Optional[list] = None,
        username: Optional[str] = None,
        voice: Optional[str] = None
    ) -> Optional[dict]:

        if self.closed:

            return None

        if not audio:

            return None

        # ----------------------------------------------------
        # STT
        # ----------------------------------------------------

        text = await self.transcribe(
            audio
        )

        text = (
            text
            or ""
        ).strip()

        # ----------------------------------------------------
        # تجاهل التسجيلات الفارغة
        # ----------------------------------------------------

        if not text:

            return None

        # ----------------------------------------------------
        # AI
        # ----------------------------------------------------

        response = (
            await self.generate_response(
                text=text,
                memory=memory,
                username=username
            )
        )

        response = (
            response
            or ""
        ).strip()

        # ----------------------------------------------------
        # لا يوجد رد
        # ----------------------------------------------------

        if not response:

            return {
                "text": text,
                "response": "",
                "audio": b"",
            }

        # ----------------------------------------------------
        # TTS
        # ----------------------------------------------------

        audio_output = (
            await self.text_to_speech(
                text=response,
                voice=voice
            )
        )

        return {
            "text": text,
            "response": response,
            "audio": audio_output,
        }

    # ========================================================
    # CLEAN RESPONSE
    # ========================================================

    @staticmethod
    def clean_response(
        text: str
    ) -> str:

        if not text:

            return ""

        # ----------------------------------------------------
        # إزالة Markdown البسيط
        # ----------------------------------------------------

        replacements = {

            "**": "",

            "__": "",

            "```": "",

            "`": "",

            "#": "",

        }

        for old, new in (
            replacements.items()
        ):

            text = text.replace(
                old,
                new
            )

        # ----------------------------------------------------
        # إزالة المسافات الزائدة
        # ----------------------------------------------------

        lines = [
            line.strip()
            for line
            in text.splitlines()
            if line.strip()
        ]

        text = " ".join(
            lines
        )

        # ----------------------------------------------------
        # حماية الطول
        # ----------------------------------------------------

        if (
            len(text)
            > MAX_RESPONSE_CHARS
        ):

            text = (
                text[
                    :MAX_RESPONSE_CHARS
                ]
                .rsplit(
                    " ",
                    1
                )[0]
                + "..."
            )

        return text.strip()

    # ========================================================
    # BACKOFF
    # ========================================================

    @staticmethod
    def _backoff(
        attempt: int
    ) -> float:

        base = (
            1.0
            * (
                2 ** attempt
            )
        )

        jitter = random.uniform(
            0.1,
            0.5
        )

        return min(
            base + jitter,
            8.0
        )

    # ========================================================
    # CLEAR MEMORY
    # ========================================================

    def clear_memory(
        self
    ) -> None:

        # Gemini نفسه ما عنده Memory محلية
        # هنا فقط Hook للمستقبل.

        logger.debug(
            "Gemini memory cleared."
        )

    # ========================================================
    # RESET
    # ========================================================

    async def reset(
        self
    ) -> None:

        self.total_requests = 0

        self.total_transcriptions = 0

        self.total_tts = 0

        self.total_errors = 0

        self.last_request_time = 0.0

        logger.info(
            "Gemini engine statistics reset."
        )

    # ========================================================
    # STATUS
    # ========================================================

    def status(
        self
    ) -> dict:

        return {

            "closed":
                self.closed,

            "chat_model":
                CHAT_MODEL,

            "transcribe_model":
                TRANSCRIBE_MODEL,

            "tts_model":
                TTS_MODEL,

            "voice":
                self.current_voice,

            "requests":
                self.total_requests,

            "transcriptions":
                self.total_transcriptions,

            "tts":
                self.total_tts,

            "errors":
                self.total_errors,
        }

    # ========================================================
    # CLOSE
    # ========================================================

    async def close(
        self
    ) -> None:

        if self.closed:
            return

        self.closed = True

        logger.info(
            "Closing GeminiEngine..."
        )

        # google-genai client لا يحتاج
        # جلسة HTTP يدوية هنا في هذا المسار.

        logger.info(
            "GeminiEngine closed."
        )


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def get_voice_description(
    voice: str
) -> str:

    return VOICE_DESCRIPTIONS.get(
        voice,
        "Unknown"
    )


def get_all_voice_descriptions():

    return {
        voice: VOICE_DESCRIPTIONS.get(
            voice,
            "Unknown"
        )
        for voice in GEMINI_VOICES
    }


def normalize_voice_name(
    voice: str
) -> Optional[str]:

    if not voice:

        return None

    voice = (
        voice
        .strip()
    )

    # --------------------------------------------------------
    # تطابق مباشر
    # --------------------------------------------------------

    if voice in GEMINI_VOICES:

        return voice

    # --------------------------------------------------------
    # تطابق بدون حساسية حالة الأحرف
    # --------------------------------------------------------

    lowered = (
        voice.lower()
    )

    for available in GEMINI_VOICES:

        if (
            available.lower()
            == lowered
        ):

            return available

    return None


# ============================================================
# TEST FUNCTION
# ============================================================

async def test_gemini():

    engine = GeminiEngine()

    try:

        response = (
            await engine.generate_response(
                text="هلا وش أخبارك؟",
                memory=[],
                username="TestUser"
            )
        )

        logger.info(
            "Test response: %s",
            response
        )

        audio = (
            await engine.text_to_speech(
                response,
                "Kore"
            )
        )

        logger.info(
            "Generated audio: %s bytes",
            len(audio)
        )

    finally:

        await engine.close()


# ============================================================
# END
# ============================================================
