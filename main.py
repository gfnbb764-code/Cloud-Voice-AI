# ============================================================
# AI VOICE BOT
# main.py
# ============================================================
#
# Discord AI Voice Bot
#
# الوظائف الرئيسية:
# - تشغيل البوت
# - أوامر Discord
# - الدخول والخروج من Voice
# - إدارة جلسات السيرفر
# - إدارة إعدادات السيرفر
# - إدارة الذاكرة
# - التحكم في حالة الصوت
# - معالجة الأخطاء
# - مراقبة Voice State
# - Health / Ping
#
# يعتمد على:
#   discord.py
#   discord-ext-voice-recv
#   Gemini API
#
# ============================================================

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time

from dataclasses import dataclass, field
from typing import Optional

import discord
from discord.ext import commands, voice_recv

from config import (
    DISCORD_TOKEN,
    BOT_PREFIX,
    BOT_STATUS,
    BOT_ACTIVITY,
    MAX_MEMORY_MESSAGES,
    MAX_GUILD_SESSIONS,
)

from voice import VoiceSession


# ============================================================
# LOGGING
# ============================================================

LOG_LEVEL = os.getenv(
    "LOG_LEVEL",
    "INFO"
).upper()

logging.basicConfig(
    level=getattr(
        logging,
        LOG_LEVEL,
        logging.INFO
    ),
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(
    "ai_voice_bot"
)


# ============================================================
# INTENTS
# ============================================================

intents = discord.Intents.default()

intents.guilds = True
intents.voice_states = True
intents.messages = True
intents.message_content = True


# ============================================================
# BOT CLASS
# ============================================================

class AIVoiceBot(commands.Bot):
    """
    الكلاس الرئيسي للبوت.

    يحتوي على:
    - جلسات Voice
    - إعدادات السيرفر
    - ذاكرة المحادثة
    - Locks لمنع العمليات المتزامنة
    """

    def __init__(self):

        super().__init__(
            command_prefix=BOT_PREFIX,
            intents=intents,
            case_insensitive=True,
            help_command=None,
        )

        # ----------------------------------------------------
        # جلسات Voice
        # guild_id -> VoiceSession
        # ----------------------------------------------------

        self.voice_sessions: dict[
            int,
            VoiceSession
        ] = {}

        # ----------------------------------------------------
        # Locks
        # ----------------------------------------------------

        self.guild_locks: dict[
            int,
            asyncio.Lock
        ] = {}

        # ----------------------------------------------------
        # إعدادات السيرفر
        # ----------------------------------------------------

        self.guild_settings: dict[
            int,
            dict
        ] = {}

        # ----------------------------------------------------
        # وقت التشغيل
        # ----------------------------------------------------

        self.started_at = time.monotonic()

        # ----------------------------------------------------
        # حالة الإغلاق
        # ----------------------------------------------------

        self.shutting_down = False

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        self.stats = {
            "voice_joins": 0,
            "voice_leaves": 0,
            "messages_processed": 0,
            "ai_requests": 0,
            "errors": 0,
        }

    # ========================================================
    # LOCK
    # ========================================================

    def get_guild_lock(
        self,
        guild_id: int
    ) -> asyncio.Lock:

        lock = self.guild_locks.get(
            guild_id
        )

        if lock is None:

            lock = asyncio.Lock()

            self.guild_locks[
                guild_id
            ] = lock

        return lock

    # ========================================================
    # READY
    # ========================================================

    async def setup_hook(self):

        logger.info(
            "Running setup_hook..."
        )

        # ----------------------------------------------------
        # تحميل الإضافات
        # ----------------------------------------------------

        await self.load_commands()

        logger.info(
            "setup_hook completed."
        )

    # ========================================================
    # COMMAND LOADER
    # ========================================================

    async def load_commands(self):

        """
        الأوامر موجودة في هذا الملف حاليًا.

        هذا النظام موجود عشان نقدر نفصلها لاحقًا
        إلى Cogs بدون تغيير البنية الأساسية.
        """

        logger.info(
            "Command system initialized."
        )

    # ========================================================
    # BOT READY
    # ========================================================

    async def on_ready(self):

        logger.info(
            "======================================"
        )

        logger.info(
            "AI Voice Bot is ONLINE"
        )

        logger.info(
            "Bot: %s",
            self.user
        )

        logger.info(
            "Bot ID: %s",
            self.user.id
        )

        logger.info(
            "Guilds: %s",
            len(self.guilds)
        )

        logger.info(
            "Latency: %.0fms",
            self.latency * 1000
        )

        logger.info(
            "======================================"
        )

        # ----------------------------------------------------
        # Presence
        # ----------------------------------------------------

        activity = None

        if BOT_ACTIVITY:

            activity = discord.Game(
                name=BOT_ACTIVITY
            )

        await self.change_presence(
            status=discord.Status.online,
            activity=activity
        )

    # ========================================================
    # MESSAGE
    # ========================================================

    async def on_message(
        self,
        message: discord.Message
    ):

        # ----------------------------------------------------
        # تجاهل البوتات
        # ----------------------------------------------------

        if message.author.bot:
            return

        # ----------------------------------------------------
        # معالجة الأوامر
        # ----------------------------------------------------

        await self.process_commands(
            message
        )

    # ========================================================
    # ERROR
    # ========================================================

    async def on_command_error(
        self,
        ctx: commands.Context,
        error: Exception
    ):

        # ----------------------------------------------------
        # تجاهل CommandNotFound
        # ----------------------------------------------------

        if isinstance(
            error,
            commands.CommandNotFound
        ):
            return

        # ----------------------------------------------------
        # Missing arguments
        # ----------------------------------------------------

        if isinstance(
            error,
            commands.MissingRequiredArgument
        ):

            await safe_send(
                ctx,
                "❌ ناقصك شيء في الأمر."
            )

            return

        # ----------------------------------------------------
        # Missing permissions
        # ----------------------------------------------------

        if isinstance(
            error,
            commands.MissingPermissions
        ):

            await safe_send(
                ctx,
                "❌ ما عندك الصلاحية لهذا الأمر."
            )

            return

        # ----------------------------------------------------
        # Bot permissions
        # ----------------------------------------------------

        if isinstance(
            error,
            commands.BotMissingPermissions
        ):

            await safe_send(
                ctx,
                "❌ البوت ما عنده الصلاحيات المطلوبة."
            )

            return

        # ----------------------------------------------------
        # Cooldown
        # ----------------------------------------------------

        if isinstance(
            error,
            commands.CommandOnCooldown
        ):

            await safe_send(
                ctx,
                (
                    "⏳ انتظر "
                    f"{error.retry_after:.1f} ثانية."
                )
            )

            return

        # ----------------------------------------------------
        # Unexpected error
        # ----------------------------------------------------

        self.stats["errors"] += 1

        logger.exception(
            "Command error:",
            exc_info=error
        )

        await safe_send(
            ctx,
            "❌ صار خطأ غير متوقع."
        )


# ============================================================
# SAFE SEND
# ============================================================

async def safe_send(
    ctx: commands.Context,
    content: str
):

    try:

        await ctx.send(
            content
        )

    except Exception as error:

        logger.warning(
            "Could not send message: %s",
            error
        )


# ============================================================
# BOT INSTANCE
# ============================================================

bot = AIVoiceBot()


# ============================================================
# BASIC COMMANDS
# ============================================================

@bot.command(
    name="ping"
)
async def ping_command(
    ctx: commands.Context
):

    start = time.perf_counter()

    message = await ctx.send(
        "🏓 جاري القياس..."
    )

    elapsed = (
        time.perf_counter()
        - start
    ) * 1000

    websocket_latency = (
        bot.latency * 1000
    )

    await message.edit(
        content=(
            "🏓 **Pong!**\n"
            f"📡 Discord: `{websocket_latency:.0f}ms`\n"
            f"⚡ Response: `{elapsed:.0f}ms`"
        )
    )


# ============================================================
# BOT INFO
# ============================================================

@bot.command(
    name="botinfo"
)
async def botinfo_command(
    ctx: commands.Context
):

    uptime = (
        time.monotonic()
        - bot.started_at
    )

    hours = int(
        uptime // 3600
    )

    minutes = int(
        (uptime % 3600)
        // 60
    )

    seconds = int(
        uptime % 60
    )

    embed = discord.Embed(
        title="🤖 AI Voice Bot",
        description=(
            "بوت ذكاء اصطناعي صوتي "
            "يعمل داخل Discord."
        ),
        color=discord.Color.blurple()
    )

    embed.add_field(
        name="📡 Latency",
        value=(
            f"`{bot.latency * 1000:.0f}ms`"
        ),
        inline=True
    )

    embed.add_field(
        name="🏠 Servers",
        value=str(
            len(bot.guilds)
        ),
        inline=True
    )

    embed.add_field(
        name="⏱️ Uptime",
        value=(
            f"`{hours:02d}:"
            f"{minutes:02d}:"
            f"{seconds:02d}`"
        ),
        inline=True
    )

    embed.add_field(
        name="🎙️ Voice Sessions",
        value=str(
            len(
                bot.voice_sessions
            )
        ),
        inline=True
    )

    embed.add_field(
        name="🧠 AI Requests",
        value=str(
            bot.stats[
                "ai_requests"
            ]
        ),
        inline=True
    )

    embed.add_field(
        name="❌ Errors",
        value=str(
            bot.stats[
                "errors"
            ]
        ),
        inline=True
    )

    await ctx.send(
        embed=embed
    )


# ============================================================
# HELP
# ============================================================

@bot.command(
    name="help_ai",
    aliases=["commands", "cmds"]
)
async def help_command(
    ctx: commands.Context
):

    embed = discord.Embed(
        title="🤖 أوامر AI Voice Bot",
        description=(
            "هذه أوامر البوت الأساسية."
        ),
        color=discord.Color.blurple()
    )

    embed.add_field(
        name="🎙️ الصوت",
        value=(
            "`!join` — دخول الروم الصوتي\n"
            "`!leave` — الخروج من الروم\n"
            "`!voice` — حالة الاتصال"
        ),
        inline=False
    )

    embed.add_field(
        name="🧠 الذكاء",
        value=(
            "`!clear` — مسح ذاكرة المحادثة\n"
            "`!memory` — عرض حالة الذاكرة"
        ),
        inline=False
    )

    embed.add_field(
        name="⚙️ النظام",
        value=(
            "`!ping` — سرعة البوت\n"
            "`!botinfo` — معلومات البوت\n"
            "`!help_ai` — هذه القائمة"
        ),
        inline=False
    )

    await ctx.send(
        embed=embed
    )


# ============================================================
# JOIN
# ============================================================

@bot.command(
    name="join"
)
@commands.guild_only()
async def join_command(
    ctx: commands.Context
):

    guild = ctx.guild

    # --------------------------------------------------------
    # لازم المستخدم يكون داخل Voice
    # --------------------------------------------------------

    if ctx.author.voice is None:

        await safe_send(
            ctx,
            (
                "🎙️ ادخل روم صوتي أول، "
                "وبعدين استخدم `!join`."
            )
        )

        return

    channel = (
        ctx.author.voice.channel
    )

    # --------------------------------------------------------
    # تحقق من الحد
    # --------------------------------------------------------

    if (
        guild.id
        not in bot.voice_sessions
        and len(bot.voice_sessions)
        >= MAX_GUILD_SESSIONS
    ):

        await safe_send(
            ctx,
            (
                "⚠️ البوت وصل للحد الأقصى "
                "من جلسات Voice."
            )
        )

        return

    lock = bot.get_guild_lock(
        guild.id
    )

    async with lock:

        # ----------------------------------------------------
        # إذا فيه اتصال سابق
        # ----------------------------------------------------

        existing = (
            guild.voice_client
        )

        if existing:

            if (
                existing.channel
                == channel
            ):

                await safe_send(
                    ctx,
                    (
                        "🎙️ أنا أصلًا داخل "
                        "هذا الروم."
                    )
                )

                return

            try:

                await existing.disconnect(
                    force=True
                )

            except Exception as error:

                logger.warning(
                    "Failed to disconnect old VC: %s",
                    error
                )

        # ----------------------------------------------------
        # الاتصال
        # ----------------------------------------------------

        status_message = await ctx.send(
            (
                f"🔌 جاري الدخول إلى "
                f"**{channel.name}**..."
            )
        )

        try:

            voice_client = (
                await channel.connect(
                    cls=voice_recv.VoiceRecvClient
                )
            )

        except Exception as error:

            logger.exception(
                "Voice connection failed"
            )

            await status_message.edit(
                content=(
                    "❌ ما قدرت أدخل الروم.\n"
                    f"```text\n{error}\n```"
                )
            )

            return

        # ----------------------------------------------------
        # إنشاء جلسة AI
        # ----------------------------------------------------

        try:

            session = VoiceSession(
                bot=bot,
                guild=guild,
                voice_client=voice_client,
                text_channel=ctx.channel,
            )

            await session.start()

            bot.voice_sessions[
                guild.id
            ] = session

            bot.stats[
                "voice_joins"
            ] += 1

        except Exception as error:

            logger.exception(
                "Failed to create voice session"
            )

            try:

                await voice_client.disconnect(
                    force=True
                )

            except Exception:
                pass

            await status_message.edit(
                content=(
                    "❌ فشل تشغيل نظام الصوت.\n"
                    f"```text\n{error}\n```"
                )
            )

            return

        # ----------------------------------------------------
        # نجاح
        # ----------------------------------------------------

        await status_message.edit(
            content=(
                f"🎙️ **دخلت الروم!**\n"
                f"📍 `{channel.name}`\n\n"
                "🧠 نظام الذكاء جاهز.\n"
                "🎤 تكلم بشكل طبيعي."
            )
        )


# ============================================================
# LEAVE
# ============================================================

@bot.command(
    name="leave",
    aliases=["disconnect", "dc"]
)
@commands.guild_only()
async def leave_command(
    ctx: commands.Context
):

    guild = ctx.guild

    lock = bot.get_guild_lock(
        guild.id
    )

    async with lock:

        session = (
            bot.voice_sessions.pop(
                guild.id,
                None
            )
        )

        voice_client = (
            guild.voice_client
        )

        if session:

            try:

                await session.stop()

            except Exception as error:

                logger.warning(
                    "Session stop error: %s",
                    error
                )

        if voice_client:

            try:

                await voice_client.disconnect(
                    force=True
                )

            except Exception as error:

                logger.warning(
                    "Disconnect error: %s",
                    error
                )

            bot.stats[
                "voice_leaves"
            ] += 1

            await safe_send(
                ctx,
                "👋 طلعت من الروم."
            )

            return

        await safe_send(
            ctx,
            "ℹ️ أنا مو داخل أي روم صوتي."
        )


# ============================================================
# VOICE STATUS
# ============================================================

@bot.command(
    name="voice",
    aliases=["vc", "voicestatus"]
)
@commands.guild_only()
async def voice_status_command(
    ctx: commands.Context
):

    session = (
        bot.voice_sessions.get(
            ctx.guild.id
        )
    )

    voice_client = (
        ctx.guild.voice_client
    )

    if not voice_client:

        await safe_send(
            ctx,
            "🔴 مو متصل بـ Voice."
        )

        return

    channel = (
        voice_client.channel
    )

    listening = False

    if session:

        listening = (
            session.listening
        )

    embed = discord.Embed(
        title="🎙️ Voice Status",
        color=(
            discord.Color.green()
            if listening
            else discord.Color.orange()
        )
    )

    embed.add_field(
        name="📍 Channel",
        value=(
            channel.mention
            if channel
            else "Unknown"
        ),
        inline=False
    )

    embed.add_field(
        name="👂 Listening",
        value=(
            "🟢 نعم"
            if listening
            else "🔴 لا"
        ),
        inline=True
    )

    embed.add_field(
        name="🧠 Session",
        value=(
            "🟢 Active"
            if session
            else "🔴 None"
        ),
        inline=True
    )

    await ctx.send(
        embed=embed
    )


# ============================================================
# CLEAR MEMORY
# ============================================================

@bot.command(
    name="clear",
    aliases=[
        "clearmemory",
        "forget"
    ]
)
@commands.guild_only()
async def clear_memory_command(
    ctx: commands.Context
):

    session = (
        bot.voice_sessions.get(
            ctx.guild.id
        )
    )

    if not session:

        await safe_send(
            ctx,
            (
                "ℹ️ ما فيه جلسة AI "
                "فعالة حاليًا."
            )
        )

        return

    try:

        session.clear_memory()

    except Exception as error:

        logger.exception(
            "Memory clear failed"
        )

        await safe_send(
            ctx,
            "❌ ما قدرت أمسح الذاكرة."
        )

        return

    await safe_send(
        ctx,
        "🧠 تم مسح ذاكرة المحادثة."
    )


# ============================================================
# MEMORY STATUS
# ============================================================

@bot.command(
    name="memory"
)
@commands.guild_only()
async def memory_command(
    ctx: commands.Context
):

    session = (
        bot.voice_sessions.get(
            ctx.guild.id
        )
    )

    if not session:

        await safe_send(
            ctx,
            "ℹ️ ما فيه جلسة Voice."
        )

        return

    try:

        count = (
            session.memory_size()
        )

    except Exception:

        count = 0

    embed = discord.Embed(
        title="🧠 Memory",
        color=discord.Color.blurple()
    )

    embed.add_field(
        name="Messages",
        value=(
            f"`{count}/{MAX_MEMORY_MESSAGES}`"
        ),
        inline=True
    )

    embed.add_field(
        name="Limit",
        value=(
            f"`{MAX_MEMORY_MESSAGES}`"
        ),
        inline=True
    )

    await ctx.send(
        embed=embed
    )


# ============================================================
# RESET SESSION
# ============================================================

@bot.command(
    name="reset"
)
@commands.guild_only()
async def reset_command(
    ctx: commands.Context
):

    session = (
        bot.voice_sessions.get(
            ctx.guild.id
        )
    )

    if not session:

        await safe_send(
            ctx,
            "ℹ️ ما فيه جلسة Voice."
        )

        return

    try:

        await session.reset()

    except Exception as error:

        logger.exception(
            "Session reset failed"
        )

        await safe_send(
            ctx,
            "❌ فشل إعادة تشغيل الجلسة."
        )

        return

    await safe_send(
        ctx,
        "🔄 تم إعادة تشغيل جلسة AI."
    )


# ============================================================
# STATS
# ============================================================

@bot.command(
    name="stats"
)
@commands.guild_only()
async def stats_command(
    ctx: commands.Context
):

    embed = discord.Embed(
        title="📊 Bot Statistics",
        color=discord.Color.blurple()
    )

    embed.add_field(
        name="🎙️ Voice Joins",
        value=str(
            bot.stats[
                "voice_joins"
            ]
        ),
        inline=True
    )

    embed.add_field(
        name="🚪 Voice Leaves",
        value=str(
            bot.stats[
                "voice_leaves"
            ]
        ),
        inline=True
    )

    embed.add_field(
        name="🧠 AI Requests",
        value=str(
            bot.stats[
                "ai_requests"
            ]
        ),
        inline=True
    )

    embed.add_field(
        name="💬 Processed",
        value=str(
            bot.stats[
                "messages_processed"
            ]
        ),
        inline=True
    )

    embed.add_field(
        name="❌ Errors",
        value=str(
            bot.stats[
                "errors"
            ]
        ),
        inline=True
    )

    embed.add_field(
        name="🏠 Servers",
        value=str(
            len(bot.guilds)
        ),
        inline=True
    )

    await ctx.send(
        embed=embed
    )


# ============================================================
# SHUTDOWN
# ============================================================

async def shutdown():

    if bot.shutting_down:
        return

    bot.shutting_down = True

    logger.info(
        "Starting graceful shutdown..."
    )

    # --------------------------------------------------------
    # إيقاف كل جلسات Voice
    # --------------------------------------------------------

    sessions = list(
        bot.voice_sessions.values()
    )

    bot.voice_sessions.clear()

    for session in sessions:

        try:

            await session.stop()

        except Exception as error:

            logger.warning(
                "Session shutdown error: %s",
                error
            )

    # --------------------------------------------------------
    # Disconnect all voice clients
    # --------------------------------------------------------

    for guild in bot.guilds:

        voice_client = (
            guild.voice_client
        )

        if voice_client:

            try:

                await voice_client.disconnect(
                    force=True
                )

            except Exception as error:

                logger.warning(
                    "VC shutdown error: %s",
                    error
                )

    # --------------------------------------------------------
    # Close bot
    # --------------------------------------------------------

    try:

        await bot.close()

    except Exception as error:

        logger.warning(
            "Bot close error: %s",
            error
        )

    logger.info(
        "Shutdown complete."
    )


# ============================================================
# SIGNAL HANDLERS
# ============================================================

def handle_shutdown_signal(
    signum,
    frame
):

    logger.info(
        "Received signal: %s",
        signum
    )

    try:

        loop = asyncio.get_running_loop()

    except RuntimeError:

        return

    loop.create_task(
        shutdown()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    logger.info(
        "======================================"
    )

    logger.info(
        "Starting AI Voice Bot..."
    )

    logger.info(
        "======================================"
    )

    # --------------------------------------------------------
    # Signals
    # --------------------------------------------------------

    if sys.platform != "win32":

        try:

            signal.signal(
                signal.SIGTERM,
                handle_shutdown_signal
            )

            signal.signal(
                signal.SIGINT,
                handle_shutdown_signal
            )

        except Exception as error:

            logger.warning(
                "Could not install signal handlers: %s",
                error
            )

    # --------------------------------------------------------
    # Start
    # --------------------------------------------------------

    try:

        bot.run(
            DISCORD_TOKEN,
            log_handler=None
        )

    except KeyboardInterrupt:

        logger.info(
            "Keyboard interrupt."
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
