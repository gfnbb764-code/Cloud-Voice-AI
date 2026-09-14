# ============================================================
# config.py
# Gemini Discord AI Voice Bot
# ============================================================
#
# ملف الإعدادات الرئيسي للمشروع.
#
# الملفات:
#   main.py
#   voice.py
#   gemini.py
#   config.py
#
# ============================================================

from __future__ import annotations

import os
from pathlib import Path
from typing import Final


# ============================================================
# PROJECT
# ============================================================

PROJECT_NAME: Final[str] = "Gemini Discord AI Voice Bot"
PROJECT_VERSION: Final[str] = "1.0.0"

BASE_DIR: Final[Path] = Path(__file__).resolve().parent

DATA_DIR: Final[Path] = BASE_DIR / "data"
CACHE_DIR: Final[Path] = BASE_DIR / "cache"
LOG_DIR: Final[Path] = BASE_DIR / "logs"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# ENVIRONMENT
# ============================================================

def env(name: str, default: str = "") -> str:
    """
    قراءة متغير من البيئة مع تنظيف المسافات.
    """
    return os.getenv(name, default).strip()


def env_bool(name: str, default: bool = False) -> bool:
    """
    قراءة True/False من البيئة.
    """
    value = env(name)

    if not value:
        return default

    return value.lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
        "enabled",
    }


def env_int(name: str, default: int) -> int:
    """
    قراءة رقم صحيح من البيئة.
    """
    value = env(name)

    if not value:
        return default

    try:
        return int(value)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    """
    قراءة رقم عشري من البيئة.
    """
    value = env(name)

    if not value:
        return default

    try:
        return float(value)
    except ValueError:
        return default


# ============================================================
# DISCORD
# ============================================================

DISCORD_TOKEN: Final[str] = env("DISCORD_TOKEN")

DISCORD_PREFIX: Final[str] = env(
    "DISCORD_PREFIX",
    "!"
)

DISCORD_OWNER_ID: Final[int] = env_int(
    "DISCORD_OWNER_ID",
    0
)

DISCORD_STATUS: Final[str] = env(
    "DISCORD_STATUS",
    "Gemini AI • Voice"
)

DISCORD_ACTIVITY_TYPE: Final[str] = env(
    "DISCORD_ACTIVITY_TYPE",
    "watching"
)

DISCORD_SHARD_COUNT: Final[int] = env_int(
    "DISCORD_SHARD_COUNT",
    1
)


# ============================================================
# GEMINI
# ============================================================

GEMINI_API_KEY: Final[str] = env(
    "GEMINI_API_KEY"
)

# النموذج الأساسي للمحادثة.
CHAT_MODEL: Final[str] = env(
    "CHAT_MODEL",
    "gemini-3.5-flash-lite"
)

# نموذج تحويل الصوت إلى نص.
TRANSCRIBE_MODEL: Final[str] = env(
    "TRANSCRIBE_MODEL",
    "gemini-3.5-transcribe"
)

# نموذج تحويل النص إلى صوت.
TTS_MODEL: Final[str] = env(
    "TTS_MODEL",
    "gemini-3.1-flash-tts-preview"
)


# ============================================================
# GEMINI GENERATION
# ============================================================

GEMINI_TEMPERATURE: Final[float] = env_float(
    "GEMINI_TEMPERATURE",
    0.75
)

GEMINI_TOP_P: Final[float] = env_float(
    "GEMINI_TOP_P",
    0.95
)

GEMINI_TOP_K: Final[int] = env_int(
    "GEMINI_TOP_K",
    40
)

GEMINI_MAX_OUTPUT_TOKENS: Final[int] = env_int(
    "GEMINI_MAX_OUTPUT_TOKENS",
    350
)


# ============================================================
# AI PERSONALITY
# ============================================================

AI_NAME: Final[str] = env(
    "AI_NAME",
    "Gemini"
)

AI_LANGUAGE: Final[str] = env(
    "AI_LANGUAGE",
    "ar"
)

AI_SYSTEM_PROMPT: Final[str] = env(
    "AI_SYSTEM_PROMPT",
    """
أنت مساعد ذكاء اصطناعي داخل Discord.

تحدث بطريقة طبيعية وودية ومختصرة.
افهم اللهجة العربية العامية قدر الإمكان.
يمكنك استخدام العربية والإنجليزية حسب لغة المستخدم.

لا تتحدث بطريقة آلية أو رسمية زيادة عن اللزوم.
لا تكرر كلام المستخدم بدون سبب.
لا تكتب مقدمات طويلة.
إذا كان السؤال بسيطاً، أعط إجابة بسيطة.
إذا كان المستخدم يمزح، تفاعل معه بشكل طبيعي.

أنت تعمل داخل مكالمة صوتية في Discord،
لذلك يجب أن تكون إجاباتك مناسبة للاستماع وليست طويلة جداً.
""".strip()
)


# ============================================================
# MEMORY
# ============================================================

MAX_MEMORY_MESSAGES: Final[int] = env_int(
    "MAX_MEMORY_MESSAGES",
    12
)

MAX_MEMORY_CHARS: Final[int] = env_int(
    "MAX_MEMORY_CHARS",
    12000
)

MEMORY_ENABLED: Final[bool] = env_bool(
    "MEMORY_ENABLED",
    True
)

MEMORY_FILE: Final[Path] = DATA_DIR / "memory.json"


# ============================================================
# GUILD SESSIONS
# ============================================================

MAX_GUILD_SESSIONS: Final[int] = env_int(
    "MAX_GUILD_SESSIONS",
    10
)

SESSION_TIMEOUT_SECONDS: Final[int] = env_int(
    "SESSION_TIMEOUT_SECONDS",
    900
)


# ============================================================
# VOICE
# ============================================================

# صوت Gemini الافتراضي.
DEFAULT_VOICE: Final[str] = env(
    "DEFAULT_VOICE",
    "Kore"
)

# لغة الصوت.
DEFAULT_TTS_LANGUAGE: Final[str] = env(
    "DEFAULT_TTS_LANGUAGE",
    "ar"
)

# هل يسمح بتغيير الصوت من Discord؟
ALLOW_VOICE_CHANGE: Final[bool] = env_bool(
    "ALLOW_VOICE_CHANGE",
    True
)

# هل نحفظ اختيار الصوت بعد إعادة تشغيل البوت؟
PERSIST_VOICE_SETTINGS: Final[bool] = env_bool(
    "PERSIST_VOICE_SETTINGS",
    True
)

VOICE_SETTINGS_FILE: Final[Path] = (
    DATA_DIR / "voice_settings.json"
)


# ============================================================
# GEMINI TTS VOICES
# ============================================================

GEMINI_VOICES: Final[dict[str, str]] = {

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
# VOICE ALIASES
# ============================================================

VOICE_ALIASES: Final[dict[str, str]] = {

    "zephyr": "Zephyr",
    "زفير": "Zephyr",

    "puck": "Puck",
    "بوك": "Puck",

    "charon": "Charon",
    "شارون": "Charon",

    "kore": "Kore",
    "كوري": "Kore",

    "fenrir": "Fenrir",
    "فنرير": "Fenrir",

    "leda": "Leda",
    "ليدا": "Leda",

    "orus": "Orus",
    "أوروس": "Orus",

    "aoede": "Aoede",
    "أويدي": "Aoede",

    "callirrhoe": "Callirrhoe",
    "كاليرو": "Callirrhoe",

    "autonoe": "Autonoe",
    "أوتونو": "Autonoe",

    "achird": "Achird",
    "أشيرد": "Achird",

    "sulafat": "Sulafat",
    "سلافات": "Sulafat",

    "gacrux": "Gacrux",
    "جاكروكس": "Gacrux",
}


# ============================================================
# AUDIO
# ============================================================

# Discord يحتاج PCM بمعدل 48kHz.
DISCORD_SAMPLE_RATE: Final[int] = 48000

DISCORD_CHANNELS: Final[int] = 2

DISCORD_SAMPLE_WIDTH: Final[int] = 2

DISCORD_FRAME_SIZE: Final[int] = 960

DISCORD_AUDIO_BYTES_PER_FRAME: Final[int] = (
    DISCORD_FRAME_SIZE
    * DISCORD_CHANNELS
    * DISCORD_SAMPLE_WIDTH
)


# Gemini Live / transcription input.
GEMINI_INPUT_SAMPLE_RATE: Final[int] = 16000

GEMINI_INPUT_CHANNELS: Final[int] = 1

GEMINI_INPUT_SAMPLE_WIDTH: Final[int] = 2


# Gemini TTS output.
GEMINI_TTS_SAMPLE_RATE: Final[int] = 24000

GEMINI_TTS_CHANNELS: Final[int] = 1

GEMINI_TTS_SAMPLE_WIDTH: Final[int] = 2


# ============================================================
# AUDIO LIMITS
# ============================================================

MAX_RECORDING_SECONDS: Final[int] = env_int(
    "MAX_RECORDING_SECONDS",
    15
)

MIN_RECORDING_SECONDS: Final[float] = env_float(
    "MIN_RECORDING_SECONDS",
    0.25
)

MAX_AUDIO_BYTES: Final[int] = (
    GEMINI_INPUT_SAMPLE_RATE
    * GEMINI_INPUT_SAMPLE_WIDTH
    * GEMINI_INPUT_CHANNELS
    * MAX_RECORDING_SECONDS
)


# ============================================================
# VOICE ACTIVITY
# ============================================================

VOICE_ACTIVITY_ENABLED: Final[bool] = env_bool(
    "VOICE_ACTIVITY_ENABLED",
    True
)

VOICE_SILENCE_TIMEOUT: Final[float] = env_float(
    "VOICE_SILENCE_TIMEOUT",
    1.1
)

VOICE_MIN_ENERGY: Final[int] = env_int(
    "VOICE_MIN_ENERGY",
    450
)

VOICE_MAX_SILENCE_SECONDS: Final[float] = env_float(
    "VOICE_MAX_SILENCE_SECONDS",
    2.5
)


# ============================================================
# CONCURRENCY
# ============================================================

MAX_AI_REQUESTS: Final[int] = env_int(
    "MAX_AI_REQUESTS",
    2
)

MAX_TTS_REQUESTS: Final[int] = env_int(
    "MAX_TTS_REQUESTS",
    1
)

MAX_TRANSCRIPTION_REQUESTS: Final[int] = env_int(
    "MAX_TRANSCRIPTION_REQUESTS",
    2
)


# ============================================================
# TIMEOUTS
# ============================================================

AI_TIMEOUT: Final[float] = env_float(
    "AI_TIMEOUT",
    30.0
)

TRANSCRIPTION_TIMEOUT: Final[float] = env_float(
    "TRANSCRIPTION_TIMEOUT",
    20.0
)

TTS_TIMEOUT: Final[float] = env_float(
    "TTS_TIMEOUT",
    30.0
)

DISCORD_CONNECT_TIMEOUT: Final[float] = env_float(
    "DISCORD_CONNECT_TIMEOUT",
    20.0
)


# ============================================================
# RETRIES
# ============================================================

MAX_RETRIES: Final[int] = env_int(
    "MAX_RETRIES",
    3
)

RETRY_DELAY: Final[float] = env_float(
    "RETRY_DELAY",
    1.5
)

RETRY_BACKOFF: Final[float] = env_float(
    "RETRY_BACKOFF",
    2.0
)


# ============================================================
# RATE LIMITING
# ============================================================

USER_COOLDOWN_SECONDS: Final[float] = env_float(
    "USER_COOLDOWN_SECONDS",
    1.0
)

MAX_REQUESTS_PER_MINUTE: Final[int] = env_int(
    "MAX_REQUESTS_PER_MINUTE",
    20
)


# ============================================================
# TEXT LIMITS
# ============================================================

MAX_USER_MESSAGE_LENGTH: Final[int] = env_int(
    "MAX_USER_MESSAGE_LENGTH",
    2000
)

MAX_AI_RESPONSE_LENGTH: Final[int] = env_int(
    "MAX_AI_RESPONSE_LENGTH",
    1500
)

MAX_TTS_TEXT_LENGTH: Final[int] = env_int(
    "MAX_TTS_TEXT_LENGTH",
    1200
)


# ============================================================
# CACHE
# ============================================================

CACHE_ENABLED: Final[bool] = env_bool(
    "CACHE_ENABLED",
    True
)

MAX_CACHE_ITEMS: Final[int] = env_int(
    "MAX_CACHE_ITEMS",
    100
)


# ============================================================
# LOGGING
# ============================================================

LOG_LEVEL: Final[str] = env(
    "LOG_LEVEL",
    "INFO"
).upper()

LOG_TO_FILE: Final[bool] = env_bool(
    "LOG_TO_FILE",
    True
)

LOG_FILE: Final[Path] = (
    LOG_DIR / "bot.log"
)

MAX_LOG_SIZE_MB: Final[int] = env_int(
    "MAX_LOG_SIZE_MB",
    5
)

MAX_LOG_BACKUPS: Final[int] = env_int(
    "MAX_LOG_BACKUPS",
    2
)


# ============================================================
# DEBUG
# ============================================================

DEBUG: Final[bool] = env_bool(
    "DEBUG",
    False
)

SHOW_AI_ERRORS: Final[bool] = env_bool(
    "SHOW_AI_ERRORS",
    False
)

SHOW_AUDIO_DEBUG: Final[bool] = env_bool(
    "SHOW_AUDIO_DEBUG",
    False
)


# ============================================================
# SECURITY
# ============================================================

# لا تسجل مفاتيح API أو Tokens.
HIDE_SECRETS_IN_LOGS: Final[bool] = True

# لا نحفظ تسجيلات المستخدمين على القرص.
SAVE_USER_AUDIO: Final[bool] = env_bool(
    "SAVE_USER_AUDIO",
    False
)

# حذف ملفات الصوت المؤقتة مباشرة.
DELETE_TEMP_AUDIO: Final[bool] = True


# ============================================================
# COMMANDS
# ============================================================

COMMANDS: Final[dict[str, str]] = {

    "ping":
        "فحص سرعة البوت",

    "botinfo":
        "معلومات البوت",

    "help_ai":
        "عرض أوامر الذكاء الاصطناعي",

    "join":
        "دخول البوت إلى الروم الصوتي",

    "leave":
        "خروج البوت من الروم الصوتي",

    "voice":
        "عرض الصوت الحالي",

    "voices":
        "عرض أصوات Gemini",

    "setvoice":
        "تغيير صوت البوت",

    "voiceinfo":
        "معلومات صوت معين",

    "clear":
        "مسح ذاكرة المحادثة",

    "memory":
        "عرض حالة الذاكرة",

    "reset":
        "إعادة ضبط الجلسة",

    "stats":
        "إحصائيات البوت",
}


# ============================================================
# VOICE COMMAND HELP
# ============================================================

VOICE_COMMAND_EXAMPLES: Final[list[str]] = [

    f"{DISCORD_PREFIX}voices",

    f"{DISCORD_PREFIX}voice",

    f"{DISCORD_PREFIX}setvoice Kore",

    f"{DISCORD_PREFIX}setvoice Achird",

    f"{DISCORD_PREFIX}voiceinfo Kore",
]


# ============================================================
# DEFAULT AI VOICE PROMPT
# ============================================================

TTS_STYLE_PROMPT: Final[str] = env(
    "TTS_STYLE_PROMPT",
    """
تحدث بطريقة طبيعية وودية.
استخدم نبرة مناسبة للمحادثة الصوتية.
لا تقرأ علامات Markdown بصوت عالٍ.
لا تقل أسماء الأوامر البرمجية إلا إذا كانت جزءاً من الإجابة.
اجعل الكلام واضحاً ومناسباً للاستماع.
""".strip()
)


# ============================================================
# VALIDATION
# ============================================================

def validate_config() -> tuple[bool, list[str]]:
    """
    فحص الإعدادات الأساسية قبل تشغيل البوت.
    """

    errors: list[str] = []

    if not DISCORD_TOKEN:
        errors.append(
            "DISCORD_TOKEN غير موجود."
        )

    if not GEMINI_API_KEY:
        errors.append(
            "GEMINI_API_KEY غير موجود."
        )

    if not DISCORD_PREFIX:
        errors.append(
            "DISCORD_PREFIX لا يمكن أن يكون فارغاً."
        )

    if GEMINI_TEMPERATURE < 0:
        errors.append(
            "GEMINI_TEMPERATURE يجب أن يكون >= 0."
        )

    if GEMINI_TOP_P < 0 or GEMINI_TOP_P > 1:
        errors.append(
            "GEMINI_TOP_P يجب أن يكون بين 0 و 1."
        )

    if GEMINI_TOP_K < 1:
        errors.append(
            "GEMINI_TOP_K يجب أن يكون >= 1."
        )

    if MAX_MEMORY_MESSAGES < 1:
        errors.append(
            "MAX_MEMORY_MESSAGES يجب أن يكون >= 1."
        )

    if MAX_RECORDING_SECONDS < 1:
        errors.append(
            "MAX_RECORDING_SECONDS يجب أن يكون >= 1."
        )

    if DEFAULT_VOICE not in GEMINI_VOICES:
        errors.append(
            f"الصوت الافتراضي غير صحيح: {DEFAULT_VOICE}"
        )

    return (
        len(errors) == 0,
        errors,
    )


# ============================================================
# VOICE HELPERS
# ============================================================

def normalize_voice_name(
    voice_name: str,
) -> str | None:
    """
    تحويل اسم الصوت إلى الاسم الرسمي.

    أمثلة:

        kore
        KORE
        كوري

    كلها تصبح:

        Kore
    """

    if not voice_name:
        return None

    cleaned = voice_name.strip()

    # الاسم الرسمي
    if cleaned in GEMINI_VOICES:
        return cleaned

    # بدون حساسية لحالة الأحرف
    lowered = cleaned.lower()

    for official_name in GEMINI_VOICES:

        if official_name.lower() == lowered:
            return official_name

    # aliases
    alias = VOICE_ALIASES.get(lowered)

    if alias:
        return alias

    return None


def is_valid_voice(
    voice_name: str,
) -> bool:
    """
    التحقق من وجود الصوت.
    """

    return (
        normalize_voice_name(voice_name)
        is not None
    )


def get_voice_description(
    voice_name: str,
) -> str | None:
    """
    الحصول على وصف الصوت.
    """

    official = normalize_voice_name(
        voice_name
    )

    if not official:
        return None

    return GEMINI_VOICES.get(
        official
    )


def get_voice_list() -> list[str]:
    """
    إرجاع قائمة الأصوات.
    """

    return list(
        GEMINI_VOICES.keys()
    )


# ============================================================
# DISPLAY HELPERS
# ============================================================

def mask_secret(
    value: str,
    visible: int = 4,
) -> str:
    """
    إخفاء مفتاح أو Token عند عرضه.
    """

    if not value:
        return "غير موجود"

    if len(value) <= visible:
        return "*" * len(value)

    return (
        value[:visible]
        + "..."
        + "*" * 8
    )


def get_public_config() -> dict:
    """
    إعدادات آمنة للعرض في !botinfo.
    """

    return {

        "project":
            PROJECT_NAME,

        "version":
            PROJECT_VERSION,

        "chat_model":
            CHAT_MODEL,

        "transcribe_model":
            TRANSCRIBE_MODEL,

        "tts_model":
            TTS_MODEL,

        "default_voice":
            DEFAULT_VOICE,

        "memory":
            MEMORY_ENABLED,

        "max_memory":
            MAX_MEMORY_MESSAGES,

        "max_recording":
            MAX_RECORDING_SECONDS,

        "debug":
            DEBUG,

        "gemini_key":
            mask_secret(GEMINI_API_KEY),

        "discord_token":
            mask_secret(DISCORD_TOKEN),
    }


# ============================================================
# STARTUP VALIDATION
# ============================================================

CONFIG_OK, CONFIG_ERRORS = validate_config()


# ============================================================
# OPTIONAL STARTUP MESSAGE
# ============================================================

if DEBUG:

    print("=" * 60)
    print(PROJECT_NAME)
    print("=" * 60)

    print(
        f"Version: {PROJECT_VERSION}"
    )

    print(
        f"Chat model: {CHAT_MODEL}"
    )

    print(
        f"Transcribe model: {TRANSCRIBE_MODEL}"
    )

    print(
        f"TTS model: {TTS_MODEL}"
    )

    print(
        f"Default voice: {DEFAULT_VOICE}"
    )

    print(
        f"Memory enabled: {MEMORY_ENABLED}"
    )

    print(
        f"Configuration valid: {CONFIG_OK}"
    )

    if CONFIG_ERRORS:

        print("\nConfiguration errors:")

        for error in CONFIG_ERRORS:
            print(
                f"  - {error}"
            )

    print("=" * 60)


# ============================================================
# END OF CONFIG.PY
# ============================================================
