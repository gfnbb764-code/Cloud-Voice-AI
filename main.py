# ============================================================
# main.py
# Gemini Discord AI Voice Bot
# FINAL SLASH COMMAND VERSION
# ============================================================

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Optional

import discord
from discord import app_commands
from discord.ext import commands

from config import (
    PROJECT_NAME,
    PROJECT_VERSION,

    DISCORD_TOKEN,
    DISCORD_STATUS,

    DEFAULT_VOICE,
    GEMINI_VOICES,

    MAX_GUILD_SESSIONS,
    MAX_MEMORY_MESSAGES,

    MEMORY_ENABLED,
    ALLOW_VOICE_CHANGE,

    DEBUG,
    LOG_LEVEL,

    validate_config,
    normalize_voice_name,
)

from voice import VoiceSession


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=getattr(
        logging,
        LOG_LEVEL,
        logging.INFO,
    ),
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger("GeminiBot")


# ============================================================
# INTENTS
# ============================================================

intents = discord.Intents.default()

intents.guilds = True
intents.voice_states = True
intents.messages = True
intents.message_content = False


# ============================================================
# BOT
# ============================================================

class GeminiBot(commands.Bot):

    def __init__(self):

        super().__init__(
            command_prefix=commands.when_mentioned,
            intents=intents,
            help_command=None,
        )

        # ----------------------------------------------------
        # Voice sessions
        # ----------------------------------------------------

        self.voice_sessions: dict[int, VoiceSession] = {}

        # ----------------------------------------------------
        # Per-user cooldown
        # ----------------------------------------------------

        self.cooldowns: dict[int, deque[float]] = defaultdict(
            lambda: deque(maxlen=20)
        )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        self.started_at = time.monotonic()

        self.stats = {
            "commands": 0,
            "voice_joins": 0,
            "voice_leaves": 0,
            "voice_messages": 0,
            "ai_requests": 0,
            "errors": 0,
        }

        # ----------------------------------------------------
        # Lock
        # ----------------------------------------------------

        self.session_lock = asyncio.Lock()

    # ========================================================
    # SETUP HOOK
    # ========================================================

    async def setup_hook(self):

        logger.info(
            "Loading Gemini Discord AI Bot..."
        )

        # ----------------------------------------------------
        # Sync slash commands
        # ----------------------------------------------------

        try:

            synced = await self.tree.sync()

            logger.info(
                "Synced %s global slash commands.",
                len(synced),
            )

        except Exception:

            logger.exception(
                "Failed to sync slash commands."
            )

            raise

    # ========================================================
    # READY
    # ========================================================

    async def on_ready(self):

        logger.info(
            "=========================================="
        )

        logger.info(
            "%s v%s",
            PROJECT_NAME,
            PROJECT_VERSION,
        )

        logger.info(
            "Logged in as %s (%s)",
            self.user,
            self.user.id if self.user else "unknown",
        )

        logger.info(
            "Guilds: %s",
            len(self.guilds),
        )

        logger.info(
            "=========================================="
        )

        activity = discord.Activity(
            type=discord.ActivityType.watching,
            name=DISCORD_STATUS,
        )

        try:

            await self.change_presence(
                status=discord.Status.online,
                activity=activity,
            )

        except Exception:

            logger.exception(
                "Failed to update presence."
            )

    # ========================================================
    # ERROR HANDLER
    # ========================================================

    async def on_error(
        self,
        event_method: str,
        *args,
        **kwargs,
    ):

        self.stats["errors"] += 1

        logger.exception(
            "Unhandled Discord event error: %s",
            event_method,
        )

    # ========================================================
    # SESSION
    # ========================================================

    async def get_or_create_session(
        self,
        guild: discord.Guild,
    ) -> VoiceSession:

        guild_id = guild.id

        async with self.session_lock:

            existing = self.voice_sessions.get(
                guild_id
            )

            if existing:

                return existing

            if (
                len(self.voice_sessions)
                >= MAX_GUILD_SESSIONS
            ):

                raise RuntimeError(
                    "Maximum guild sessions reached."
                )

            session = VoiceSession(
                bot=self,
                guild=guild,
                default_voice=DEFAULT_VOICE,
            )

            self.voice_sessions[guild_id] = session

            return session

    # ========================================================
    # GET SESSION
    # ========================================================

    def get_session(
        self,
        guild_id: int,
    ) -> Optional[VoiceSession]:

        return self.voice_sessions.get(
            guild_id
        )

    # ========================================================
    # REMOVE SESSION
    # ========================================================

    async def remove_session(
        self,
        guild_id: int,
    ):

        async with self.session_lock:

            session = self.voice_sessions.pop(
                guild_id,
                None,
            )

        if session:

            try:

                await session.close()

            except Exception:

                logger.exception(
                    "Failed to close voice session."
                )

    # ========================================================
    # COOLDOWN
    # ========================================================

    def check_cooldown(
        self,
        user_id: int,
        cooldown: float = 1.0,
    ) -> float:

        now = time.monotonic()

        history = self.cooldowns[user_id]

        while history and (
            now - history[0] > 60
        ):

            history.popleft()

        if history:

            remaining = cooldown - (
                now - history[-1]
            )

            if remaining > 0:

                return remaining

        history.append(now)

        return 0.0

    # ========================================================
    # UPTIME
    # ========================================================

    def uptime_seconds(self) -> int:

        return int(
            time.monotonic()
            - self.started_at
        )

    # ========================================================
    # FORMAT UPTIME
    # ========================================================

    @staticmethod
    def format_uptime(
        seconds: int,
    ) -> str:

        days, remainder = divmod(
            seconds,
            86400,
        )

        hours, remainder = divmod(
            remainder,
            3600,
        )

        minutes, seconds = divmod(
            remainder,
            60,
        )

        parts = []

        if days:
            parts.append(
                f"{days}d"
            )

        if hours:
            parts.append(
                f"{hours}h"
            )

        if minutes:
            parts.append(
                f"{minutes}m"
            )

        parts.append(
            f"{seconds}s"
        )

        return " ".join(parts)


# ============================================================
# CREATE BOT
# ============================================================

bot = GeminiBot()


# ============================================================
# EMBED HELPERS
# ============================================================

def success_embed(
    title: str,
    description: str,
) -> discord.Embed:

    return discord.Embed(
        title=f"✅ {title}",
        description=description,
        color=discord.Color.green(),
    )


def error_embed(
    title: str,
    description: str,
) -> discord.Embed:

    return discord.Embed(
        title=f"❌ {title}",
        description=description,
        color=discord.Color.red(),
    )


def info_embed(
    title: str,
    description: str,
) -> discord.Embed:

    return discord.Embed(
        title=f"ℹ️ {title}",
        description=description,
        color=discord.Color.blurple(),
    )


# ============================================================
# /PING
# ============================================================

@bot.tree.command(
    name="ping",
    description="فحص سرعة واستجابة البوت",
)
async def ping(
    interaction: discord.Interaction,
):

    bot.stats["commands"] += 1

    latency = round(
        bot.latency * 1000
    )

    await interaction.response.send_message(
        embed=info_embed(
            "Pong!",
            f"🏓 Discord Latency: **{latency}ms**",
        ),
        ephemeral=True,
    )


# ============================================================
# /BOTINFO
# ============================================================

@bot.tree.command(
    name="botinfo",
    description="عرض معلومات البوت والنظام",
)
async def botinfo(
    interaction: discord.Interaction,
):

    bot.stats["commands"] += 1

    uptime = bot.format_uptime(
        bot.uptime_seconds()
    )

    embed = discord.Embed(
        title="🤖 Gemini AI Bot",
        description=(
            "بوت ذكاء اصطناعي صوتي يعمل داخل Discord."
        ),
        color=discord.Color.blurple(),
    )

    embed.add_field(
        name="📦 Version",
        value=f"`{PROJECT_VERSION}`",
        inline=True,
    )

    embed.add_field(
        name="🏠 Servers",
        value=f"`{len(bot.guilds)}`",
        inline=True,
    )

    embed.add_field(
        name="🎙️ Voice Sessions",
        value=f"`{len(bot.voice_sessions)}`",
        inline=True,
    )

    embed.add_field(
        name="⏱️ Uptime",
        value=f"`{uptime}`",
        inline=True,
    )

    embed.add_field(
        name="🧠 Memory",
        value=(
            f"`{MAX_MEMORY_MESSAGES} messages`"
            if MEMORY_ENABLED
            else "`Disabled`"
        ),
        inline=True,
    )

    embed.add_field(
        name="🎤 Default Voice",
        value=f"`{DEFAULT_VOICE}`",
        inline=True,
    )

    await interaction.response.send_message(
        embed=embed,
        ephemeral=True,
    )


# ============================================================
# /HELP
# ============================================================

@bot.tree.command(
    name="help",
    description="عرض جميع أوامر البوت",
)
async def help_command(
    interaction: discord.Interaction,
):

    bot.stats["commands"] += 1

    embed = discord.Embed(
        title="📚 أوامر Gemini AI",
        description=(
            "هذه جميع أوامر البوت المتاحة:"
        ),
        color=discord.Color.blurple(),
    )

    embed.add_field(
        name="🎙️ الصوت",
        value=(
            "`/join` — دخول الروم الصوتي\n"
            "`/leave` — الخروج من الروم\n"
            "`/voices` — عرض الأصوات\n"
            "`/setvoice` — تغيير الصوت\n"
            "`/voice` — الصوت الحالي"
        ),
        inline=False,
    )

    embed.add_field(
        name="🧠 الذكاء الاصطناعي",
        value=(
            "`/memory` — حالة الذاكرة\n"
            "`/clear` — مسح الذاكرة\n"
            "`/reset` — إعادة ضبط الجلسة"
        ),
        inline=False,
    )

    embed.add_field(
        name="⚙️ النظام",
        value=(
            "`/ping` — فحص السرعة\n"
            "`/botinfo` — معلومات البوت\n"
            "`/stats` — الإحصائيات"
        ),
        inline=False,
    )

    await interaction.response.send_message(
        embed=embed,
        ephemeral=True,
    )


# ============================================================
# /JOIN
# ============================================================

@bot.tree.command(
    name="join",
    description="إدخال البوت إلى الروم الصوتي",
)
async def join(
    interaction: discord.Interaction,
):

    bot.stats["commands"] += 1

    if not interaction.guild:

        await interaction.response.send_message(
            embed=error_embed(
                "غير متاح",
                "هذا الأمر يعمل داخل السيرفرات فقط.",
            ),
            ephemeral=True,
        )

        return

    member = interaction.user

    if not isinstance(
        member,
        discord.Member,
    ):

        await interaction.response.send_message(
            embed=error_embed(
                "خطأ",
                "تعذر تحديد عضويتك في السيرفر.",
            ),
            ephemeral=True,
        )

        return

    voice_state = member.voice

    if not voice_state or not voice_state.channel:

        await interaction.response.send_message(
            embed=error_embed(
                "أنت لست في روم صوتي",
                "ادخل روم صوتي أولاً ثم استخدم `/join`.",
            ),
            ephemeral=True,
        )

        return

    channel = voice_state.channel

    await interaction.response.defer(
        ephemeral=True
    )

    try:

        session = await bot.get_or_create_session(
            interaction.guild
        )

        await session.join(
            channel
        )

        bot.stats["voice_joins"] += 1

        await interaction.followup.send(
            embed=success_embed(
                "تم الدخول 🎙️",
                (
                    f"دخلت إلى **{channel.name}**.\n\n"
                    f"🎤 الصوت الحالي: **{session.get_voice()}**\n"
                    "🧠 الذكاء الاصطناعي جاهز للاستماع والرد."
                ),
            ),
            ephemeral=True,
        )

    except Exception as exc:

        logger.exception(
            "Join failed."
        )

        bot.stats["errors"] += 1

        await interaction.followup.send(
            embed=error_embed(
                "تعذر الدخول",
                f"حدث خطأ أثناء دخول الروم الصوتي.\n`{type(exc).__name__}`",
            ),
            ephemeral=True,
        )


# ============================================================
# /LEAVE
# ============================================================

@bot.tree.command(
    name="leave",
    description="إخراج البوت من الروم الصوتي",
)
async def leave(
    interaction: discord.Interaction,
):

    bot.stats["commands"] += 1

    if not interaction.guild:

        await interaction.response.send_message(
            embed=error_embed(
                "غير متاح",
                "هذا الأمر يعمل داخل السيرفرات فقط.",
            ),
            ephemeral=True,
        )

        return

    await interaction.response.defer(
        ephemeral=True
    )

    session = bot.get_session(
        interaction.guild.id
    )

    if not session:

        await interaction.followup.send(
            embed=info_embed(
                "البوت غير موجود",
                "البوت ليس لديه جلسة صوتية في هذا السيرفر.",
            ),
            ephemeral=True,
        )

        return

    try:

        await bot.remove_session(
            interaction.guild.id
        )

        bot.stats["voice_leaves"] += 1

        await interaction.followup.send(
            embed=success_embed(
                "تم الخروج 👋",
                "خرجت من الروم الصوتي.",
            ),
            ephemeral=True,
        )

    except Exception as exc:

        logger.exception(
            "Leave failed."
        )

        bot.stats["errors"] += 1

        await interaction.followup.send(
            embed=error_embed(
                "تعذر الخروج",
                f"`{type(exc).__name__}`",
            ),
            ephemeral=True,
        )


# ============================================================
# /VOICE
# ============================================================

@bot.tree.command(
    name="voice",
    description="عرض الصوت المستخدم حالياً",
)
async def voice(
    interaction: discord.Interaction,
):

    bot.stats["commands"] += 1

    if not interaction.guild:

        await interaction.response.send_message(
            embed=error_embed(
                "غير متاح",
                "هذا الأمر يعمل داخل السيرفرات فقط.",
            ),
            ephemeral=True,
        )

        return

    session = bot.get_session(
        interaction.guild.id
    )

    current_voice = DEFAULT_VOICE

    if session:

        try:
            current_voice = session.get_voice()
        except Exception:
            pass

    description = (
        f"🎙️ الصوت الحالي: **{current_voice}**\n"
        f"✨ الشخصية الصوتية: **{GEMINI_VOICES.get(current_voice, 'Unknown')}**"
    )

    await interaction.response.send_message(
        embed=info_embed(
            "الصوت الحالي",
            description,
        ),
        ephemeral=True,
    )


# ============================================================
# VOICE AUTOCOMPLETE
# ============================================================

async def voice_autocomplete(
    interaction: discord.Interaction,
    current: str,
):

    current = current.lower().strip()

    choices = []

    for name, style in GEMINI_VOICES.items():

        if (
            not current
            or current in name.lower()
            or current in style.lower()
        ):

            choices.append(
                app_commands.Choice(
                    name=f"{name} — {style}",
                    value=name,
                )
            )

    return choices[:25]


# ============================================================
# /SETVOICE
# ============================================================

@bot.tree.command(
    name="setvoice",
    description="تغيير صوت الذكاء الاصطناعي",
)
@app_commands.describe(
    voice_name="اختر صوت Gemini الذي تريد استخدامه"
)
@app_commands.autocomplete(
    voice_name=voice_autocomplete
)
async def setvoice(
    interaction: discord.Interaction,
    voice_name: str,
):

    bot.stats["commands"] += 1

    if not ALLOW_VOICE_CHANGE:

        await interaction.response.send_message(
            embed=error_embed(
                "تغيير الصوت معطل",
                "تم تعطيل تغيير الأصوات من إعدادات البوت.",
            ),
            ephemeral=True,
        )

        return

    if not interaction.guild:

        await interaction.response.send_message(
            embed=error_embed(
                "غير متاح",
                "هذا الأمر يعمل داخل السيرفرات فقط.",
            ),
            ephemeral=True,
        )

        return

    normalized = normalize_voice_name(
        voice_name
    )

    if not normalized:

        await interaction.response.send_message(
            embed=error_embed(
                "صوت غير موجود",
                (
                    "لم أجد هذا الصوت.\n"
                    "استخدم `/voices` لرؤية الأصوات المتاحة."
                ),
            ),
            ephemeral=True,
        )

        return

    await interaction.response.defer(
        ephemeral=True
    )

    try:

        session = await bot.get_or_create_session(
            interaction.guild
        )

        result = session.set_voice(
            normalized
        )

        # بعض النسخ قد ترجع True/False،
        # وبعضها قد لا ترجع شيئاً.
        if result is False:

            raise RuntimeError(
                "Voice session rejected the voice."
            )

        await interaction.followup.send(
            embed=success_embed(
                "تم تغيير الصوت 🎙️",
                (
                    f"الصوت الجديد: **{normalized}**\n"
                    f"الطابع: **{GEMINI_VOICES[normalized]}**"
                ),
            ),
            ephemeral=True,
        )

    except Exception as exc:

        logger.exception(
            "Failed to set voice."
        )

        bot.stats["errors"] += 1

        await interaction.followup.send(
            embed=error_embed(
                "تعذر تغيير الصوت",
                f"`{type(exc).__name__}`",
            ),
            ephemeral=True,
        )


# ============================================================
# /VOICES
# ============================================================

@bot.tree.command(
    name="voices",
    description="عرض جميع أصوات Gemini المتاحة",
)
async def voices(
    interaction: discord.Interaction,
):

    bot.stats["commands"] += 1

    lines = []

    for name, style in GEMINI_VOICES.items():

        lines.append(
            f"🎙️ **{name}** — {style}"
        )

    # Discord embed field/description limits
    # لذلك نقسم القائمة إلى أكثر من جزء.

    embed = discord.Embed(
        title="🎙️ أصوات Gemini",
        description=(
            "يمكنك تغيير الصوت باستخدام `/setvoice`.\n\n"
        ),
        color=discord.Color.blurple(),
    )

    first_half = lines[:15]
    second_half = lines[15:]

    embed.add_field(
        name="الأصوات 1–15",
        value="\n".join(first_half),
        inline=True,
    )

    embed.add_field(
        name="الأصوات 16–29",
        value="\n".join(second_half),
        inline=True,
    )

    await interaction.response.send_message(
        embed=embed,
        ephemeral=True,
    )


# ============================================================
# /VOICEINFO
# ============================================================

@bot.tree.command(
    name="voiceinfo",
    description="عرض معلومات صوت Gemini",
)
@app_commands.describe(
    voice_name="اسم الصوت"
)
@app_commands.autocomplete(
    voice_name=voice_autocomplete
)
async def voiceinfo(
    interaction: discord.Interaction,
    voice_name: str,
):

    bot.stats["commands"] += 1

    normalized = normalize_voice_name(
        voice_name
    )

    if not normalized:

        await interaction.response.send_message(
            embed=error_embed(
                "الصوت غير موجود",
                "استخدم `/voices` لرؤية القائمة.",
            ),
            ephemeral=True,
        )

        return

    style = GEMINI_VOICES[
        normalized
    ]

    embed = discord.Embed(
        title=f"🎙️ {normalized}",
        description=(
            f"الطابع الصوتي: **{style}**"
        ),
        color=discord.Color.blurple(),
    )

    embed.add_field(
        name="Voice ID",
        value=f"`{normalized}`",
        inline=True,
    )

    embed.add_field(
        name="Style",
        value=f"`{style}`",
        inline=True,
    )

    await interaction.response.send_message(
        embed=embed,
        ephemeral=True,
    )


# ============================================================
# /MEMORY
# ============================================================

@bot.tree.command(
    name="memory",
    description="عرض حالة ذاكرة المحادثة",
)
async def memory(
    interaction: discord.Interaction,
):

    bot.stats["commands"] += 1

    if not interaction.guild:

        await interaction.response.send_message(
            embed=error_embed(
                "غير متاح",
                "هذا الأمر يعمل داخل السيرفرات فقط.",
            ),
            ephemeral=True,
        )

        return

    session = bot.get_session(
        interaction.guild.id
    )

    if not MEMORY_ENABLED:

        await interaction.response.send_message(
            embed=info_embed(
                "الذاكرة",
                "🧠 ذاكرة المحادثة معطلة حالياً.",
            ),
            ephemeral=True,
        )

        return

    message_count = 0

    if session:

        try:

            if hasattr(
                session,
                "memory",
            ):

                memory_obj = session.memory

                if hasattr(
                    memory_obj,
                    "__len__",
                ):

                    message_count = len(
                        memory_obj
                    )

        except Exception:

            message_count = 0

    embed = discord.Embed(
        title="🧠 ذاكرة المحادثة",
        color=discord.Color.blurple(),
    )

    embed.add_field(
        name="الحالة",
        value="🟢 مفعلة",
        inline=True,
    )

    embed.add_field(
        name="الحد الأقصى",
        value=f"`{MAX_MEMORY_MESSAGES}`",
        inline=True,
    )

    embed.add_field(
        name="الرسائل الحالية",
        value=f"`{message_count}`",
        inline=True,
    )

    await interaction.response.send_message(
        embed=embed,
        ephemeral=True,
    )


# ============================================================
# /CLEAR
# ============================================================

@bot.tree.command(
    name="clear",
    description="مسح ذاكرة المحادثة الحالية",
)
async def clear(
    interaction: discord.Interaction,
):

    bot.stats["commands"] += 1

    if not interaction.guild:

        await interaction.response.send_message(
            embed=error_embed(
                "غير متاح",
                "هذا الأمر يعمل داخل السيرفرات فقط.",
            ),
            ephemeral=True,
        )

        return

    session = bot.get_session(
        interaction.guild.id
    )

    if not session:

        await interaction.response.send_message(
            embed=info_embed(
                "لا توجد جلسة",
                "لا توجد جلسة AI نشطة حالياً.",
            ),
            ephemeral=True,
        )

        return

    try:

        cleared = False

        if hasattr(
            session,
            "clear_memory",
        ):

            result = session.clear_memory()

            if asyncio.iscoroutine(result):
                await result

            cleared = True

        elif hasattr(
            session,
            "memory",
        ):

            memory_obj = session.memory

            if hasattr(
                memory_obj,
                "clear",
            ):

                memory_obj.clear()

                cleared = True

        if cleared:

            await interaction.response.send_message(
                embed=success_embed(
                    "تم مسح الذاكرة 🧹",
                    "تم حذف سياق المحادثة الحالية.",
                ),
                ephemeral=True,
            )

        else:

            await interaction.response.send_message(
                embed=error_embed(
                    "غير مدعوم",
                    "نسخة جلسة الصوت الحالية لا تدعم مسح الذاكرة.",
                ),
                ephemeral=True,
            )

    except Exception as exc:

        logger.exception(
            "Clear memory failed."
        )

        bot.stats["errors"] += 1

        await interaction.response.send_message(
            embed=error_embed(
                "تعذر المسح",
                f"`{type(exc).__name__}`",
            ),
            ephemeral=True,
        )


# ============================================================
# /RESET
# ============================================================

@bot.tree.command(
    name="reset",
    description="إعادة ضبط جلسة الذكاء الاصطناعي",
)
async def reset(
    interaction: discord.Interaction,
):

    bot.stats["commands"] += 1

    if not interaction.guild:

        await interaction.response.send_message(
            embed=error_embed(
                "غير متاح",
                "هذا الأمر يعمل داخل السيرفرات فقط.",
            ),
            ephemeral=True,
        )

        return

    await interaction.response.defer(
        ephemeral=True
    )

    try:

        old_session = bot.get_session(
            interaction.guild.id
        )

        current_channel = None

        if old_session:

            try:

                if hasattr(
                    old_session,
                    "voice_client",
                ):

                    vc = old_session.voice_client

                    if vc and vc.channel:
                        current_channel = vc.channel

            except Exception:
                pass

        await bot.remove_session(
            interaction.guild.id
        )

        if current_channel:

            new_session = (
                await bot.get_or_create_session(
                    interaction.guild
                )
            )

            await new_session.join(
                current_channel
            )

        await interaction.followup.send(
            embed=success_embed(
                "تمت إعادة الضبط 🔄",
                (
                    "تمت إعادة إنشاء جلسة الذكاء الاصطناعي."
                ),
            ),
            ephemeral=True,
        )

    except Exception as exc:

        logger.exception(
            "Reset failed."
        )

        bot.stats["errors"] += 1

        await interaction.followup.send(
            embed=error_embed(
                "تعذر إعادة الضبط",
                f"`{type(exc).__name__}`",
            ),
            ephemeral=True,
        )


# ============================================================
# /STATS
# ============================================================

@bot.tree.command(
    name="stats",
    description="عرض إحصائيات البوت",
)
async def stats(
    interaction: discord.Interaction,
):

    bot.stats["commands"] += 1

    uptime = bot.format_uptime(
        bot.uptime_seconds()
    )

    embed = discord.Embed(
        title="📊 إحصائيات البوت",
        color=discord.Color.blurple(),
    )

    embed.add_field(
        name="⏱️ Uptime",
        value=f"`{uptime}`",
        inline=True,
    )

    embed.add_field(
        name="🏠 Servers",
        value=f"`{len(bot.guilds)}`",
        inline=True,
    )

    embed.add_field(
        name="🎙️ Sessions",
        value=f"`{len(bot.voice_sessions)}`",
        inline=True,
    )

    embed.add_field(
        name="⚡ Commands",
        value=f"`{bot.stats['commands']}`",
        inline=True,
    )

    embed.add_field(
        name="🧠 AI Requests",
        value=f"`{bot.stats['ai_requests']}`",
        inline=True,
    )

    embed.add_field(
        name="🎤 Voice Joins",
        value=f"`{bot.stats['voice_joins']}`",
        inline=True,
    )

    embed.add_field(
        name="👋 Voice Leaves",
        value=f"`{bot.stats['voice_leaves']}`",
        inline=True,
    )

    embed.add_field(
        name="❌ Errors",
        value=f"`{bot.stats['errors']}`",
        inline=True,
    )

    await interaction.response.send_message(
        embed=embed,
        ephemeral=True,
    )


# ============================================================
# APP COMMAND ERROR HANDLER
# ============================================================

@bot.tree.error
async def on_app_command_error(
    interaction: discord.Interaction,
    error: app_commands.AppCommandError,
):

    bot.stats["errors"] += 1

    logger.error(
        "Slash command error: %s",
        error,
        exc_info=True,
    )

    if isinstance(
        error,
        app_commands.CommandOnCooldown,
    ):

        message = (
            "⏳ انتظر قليلاً ثم حاول مرة أخرى."
        )

    elif isinstance(
        error,
        app_commands.CheckFailure,
    ):

        message = (
            "🚫 لا تملك صلاحية استخدام هذا الأمر."
        )

    else:

        message = (
            "حدث خطأ غير متوقع أثناء تنفيذ الأمر."
        )

    try:

        if interaction.response.is_done():

            await interaction.followup.send(
                embed=error_embed(
                    "حدث خطأ",
                    message,
                ),
                ephemeral=True,
            )

        else:

            await interaction.response.send_message(
                embed=error_embed(
                    "حدث خطأ",
                    message,
                ),
                ephemeral=True,
            )

    except Exception:

        logger.exception(
            "Failed to send command error."
        )


# ============================================================
# STARTUP
# ============================================================

def main():

    ok, errors = validate_config()

    if not ok:

        print()
        print("=" * 60)
        print("CONFIGURATION ERROR")
        print("=" * 60)

        for error in errors:

            print(
                f"❌ {error}"
            )

        print("=" * 60)
        print()

        raise SystemExit(1)

    logger.info(
        "Configuration validated successfully."
    )

    logger.info(
        "Starting Discord bot..."
    )

    try:

        bot.run(
            DISCORD_TOKEN
        )

    except KeyboardInterrupt:

        logger.info(
            "Bot stopped by user."
        )

    except Exception:

        logger.exception(
            "Fatal bot error."
        )

        raise


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
