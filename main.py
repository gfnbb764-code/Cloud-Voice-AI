# main.py
# ============================================================
# Cloud Voice AI — Discord Bot
# Main Application
# ============================================================

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import discord
from discord import app_commands
from discord.ext import commands

from config import (
    PROJECT_NAME,
    PROJECT_VERSION,
    PROJECT_DESCRIPTION,
    PROJECT_AUTHOR,
    DISCORD_TOKEN,
    DISCORD_PREFIX,
    DISCORD_OWNER_ID,
    DISCORD_GUILD_ID,
    DISCORD_STATUS,
    DISCORD_ACTIVITY_TYPE,
    DISCORD_SHARD_COUNT,
    DEFAULT_GEMINI_VOICE,
    GEMINI_VOICES,
    MEMORY_ENABLED,
    MAX_MEMORY_MESSAGES,
    COMMAND_DESCRIPTIONS,
    DEBUG,
    LOG_LEVEL,
    normalize_voice_name,
    is_valid_voice,
)

from voice import (
    VoiceSession,
    VoiceSessionManager,
    send_voice_list,
)


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

logger = logging.getLogger("cloud_voice_ai")


if DEBUG:
    logger.setLevel(logging.DEBUG)


# ============================================================
# CONSTANTS
# ============================================================

BOT_START_TIME = time.time()

MAX_DISCORD_MESSAGE_LENGTH = 1900

VOICE_ACTIVITY_TYPES = {
    "playing": discord.ActivityType.playing,
    "streaming": discord.ActivityType.streaming,
    "listening": discord.ActivityType.listening,
    "watching": discord.ActivityType.watching,
    "custom": discord.ActivityType.custom,
    "competing": discord.ActivityType.competing,
}


# ============================================================
# HELPERS
# ============================================================

def format_uptime(seconds: float) -> str:
    seconds = max(0, int(seconds))

    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    parts: list[str] = []

    if days:
        parts.append(f"{days}d")

    if hours:
        parts.append(f"{hours}h")

    if minutes:
        parts.append(f"{minutes}m")

    parts.append(f"{seconds}s")

    return " ".join(parts)


def truncate_text(
    text: str,
    limit: int = MAX_DISCORD_MESSAGE_LENGTH,
) -> str:

    if len(text) <= limit:
        return text

    return text[: limit - 3] + "..."


def get_user_voice_channel(
    interaction: discord.Interaction,
) -> Optional[discord.VoiceChannel]:

    if not interaction.guild:
        return None

    member = interaction.guild.get_member(
        interaction.user.id
    )

    if member is None:
        return None

    channel = member.voice.channel if member.voice else None

    if isinstance(channel, discord.VoiceChannel):
        return channel

    return None


def make_activity() -> discord.Activity:
    activity_type = DISCORD_ACTIVITY_TYPE.lower().strip()

    if activity_type == "streaming":
        return discord.Streaming(
            name=DISCORD_STATUS,
            url="https://www.twitch.tv/",
        )

    activity_enum = VOICE_ACTIVITY_TYPES.get(
        activity_type,
        discord.ActivityType.listening,
    )

    return discord.Activity(
        type=activity_enum,
        name=DISCORD_STATUS,
    )


def is_owner(
    interaction: discord.Interaction,
) -> bool:

    if not DISCORD_OWNER_ID:
        return False

    return interaction.user.id == DISCORD_OWNER_ID


def is_guild_command(
    interaction: discord.Interaction,
) -> bool:

    return interaction.guild is not None


# ============================================================
# BOT
# ============================================================

class CloudVoiceBot(commands.Bot):
    """Main Discord bot for Cloud Voice AI."""

    def __init__(self) -> None:

        intents = discord.Intents.none()

        intents.guilds = True
        intents.voice_states = True
        intents.messages = True
        intents.message_content = False

        super().__init__(
            command_prefix=DISCORD_PREFIX,
            intents=intents,
            shard_count=DISCORD_SHARD_COUNT,
            help_command=None,
        )

        self.started_at: float = BOT_START_TIME

        self.voice_manager = VoiceSessionManager()

        self.ready_once = False
        self.sync_completed = False

        self.total_commands = 0
        self.total_errors = 0
        self.total_joins = 0
        self.total_leaves = 0

        self._shutdown_lock = asyncio.Lock()

    # ========================================================
    # SETUP
    # ========================================================

    async def setup_hook(self) -> None:

        logger.info(
            "Starting %s v%s...",
            PROJECT_NAME,
            PROJECT_VERSION,
        )

        logger.info(
            "Loading Discord application commands..."
        )

        await self._sync_commands()

        logger.info(
            "Application commands loaded."
        )

    # ========================================================
    # COMMAND SYNC
    # ========================================================

    async def _sync_commands(self) -> None:

        try:

            if DISCORD_GUILD_ID:

                guild = discord.Object(
                    id=DISCORD_GUILD_ID
                )

                self.tree.copy_global_to(
                    guild=guild
                )

                synced = await self.tree.sync(
                    guild=guild
                )

                logger.info(
                    "Synced %d commands to guild %s.",
                    len(synced),
                    DISCORD_GUILD_ID,
                )

            else:

                synced = await self.tree.sync()

                logger.info(
                    "Synced %d global commands.",
                    len(synced),
                )

            self.sync_completed = True

        except Exception:

            logger.exception(
                "Failed to synchronize application commands."
            )

            raise

    # ========================================================
    # READY
    # ========================================================

    async def on_ready(self) -> None:

        if self.user is None:
            return

        logger.info(
            "Logged in as %s (%s)",
            self.user,
            self.user.id,
        )

        logger.info(
            "Connected to %d guild(s).",
            len(self.guilds),
        )

        if not self.ready_once:

            self.ready_once = True

            logger.info(
                "Cloud Voice AI is ready."
            )

        try:

            await self.change_presence(
                status=discord.Status.online,
                activity=make_activity(),
            )

        except Exception:

            logger.exception(
                "Failed to update bot presence."
            )

    # ========================================================
    # COMMAND TRACKING
    # ========================================================

    async def on_app_command_completion(
        self,
        interaction: discord.Interaction,
        command: app_commands.Command,
    ) -> None:

        self.total_commands += 1

        logger.debug(
            "Command completed: /%s by %s (%s)",
            command.name,
            interaction.user,
            interaction.user.id,
        )

    # ========================================================
    # VOICE STATE
    # ========================================================

    async def on_voice_state_update(
        self,
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ) -> None:

        if self.user is None:
            return

        if member.id != self.user.id:
            return

        if before.channel is None and after.channel is not None:

            logger.info(
                "Joined voice channel: %s",
                after.channel,
            )

        elif before.channel is not None and after.channel is None:

            logger.info(
                "Left voice channel: %s",
                before.channel,
            )

    # ========================================================
    # ERROR HANDLING
    # ========================================================

    async def on_error(
        self,
        event_method: str,
        *args,
        **kwargs,
    ) -> None:

        self.total_errors += 1

        logger.exception(
            "Unhandled Discord event error: %s",
            event_method,
        )

    async def on_command_error(
        self,
        ctx: commands.Context,
        error: Exception,
    ) -> None:

        if isinstance(
            error,
            commands.CommandNotFound,
        ):
            return

        self.total_errors += 1

        logger.exception(
            "Command error: %s",
            error,
        )

    # ========================================================
    # SHUTDOWN
    # ========================================================

    async def shutdown(self) -> None:

        async with self._shutdown_lock:

            logger.info(
                "Shutting down Cloud Voice AI..."
            )

            try:
                await self.voice_manager.disconnect_all()
            except Exception:
                logger.exception(
                    "Failed to disconnect voice sessions."
                )

            try:
                await self.close()
            except Exception:
                logger.exception(
                    "Failed to close Discord client."
                )


# ============================================================
# BOT INSTANCE
# ============================================================

bot = CloudVoiceBot()


# ============================================================
# /PING
# ============================================================

@bot.tree.command(
    name="ping",
    description=COMMAND_DESCRIPTIONS["ping"],
)
async def ping(
    interaction: discord.Interaction,
) -> None:

    latency = round(bot.latency * 1000)

    await interaction.response.send_message(
        f"🏓 Pong!\n"
        f"⚡ Latency: `{latency}ms`",
        ephemeral=True,
    )


# ============================================================
# /BOTINFO
# ============================================================

@bot.tree.command(
    name="botinfo",
    description=COMMAND_DESCRIPTIONS["botinfo"],
)
async def botinfo(
    interaction: discord.Interaction,
) -> None:

    embed = discord.Embed(
        title=f"🎙️ {PROJECT_NAME}",
        description=PROJECT_DESCRIPTION,
        color=discord.Color.blurple(),
    )

    embed.add_field(
        name="📦 Version",
        value=f"`{PROJECT_VERSION}`",
        inline=True,
    )

    embed.add_field(
        name="👨‍💻 Author",
        value=PROJECT_AUTHOR,
        inline=True,
    )

    embed.add_field(
        name="🏠 Servers",
        value=f"`{len(bot.guilds)}`",
        inline=True,
    )

    embed.add_field(
        name="🎙️ Voice Sessions",
        value=f"`{bot.voice_manager.count}`",
        inline=True,
    )

    embed.add_field(
        name="🧠 Memory",
        value=(
            "Enabled"
            if MEMORY_ENABLED
            else "Disabled"
        ),
        inline=True,
    )

    embed.add_field(
        name="🎤 Default Voice",
        value=f"`{DEFAULT_GEMINI_VOICE}`",
        inline=True,
    )

    embed.add_field(
        name="⏱️ Uptime",
        value=f"`{format_uptime(time.time() - bot.started_at)}`",
        inline=False,
    )

    await interaction.response.send_message(
        embed=embed
    )


# ============================================================
# /HELP
# ============================================================

@bot.tree.command(
    name="help",
    description=COMMAND_DESCRIPTIONS["help"],
)
async def help_command(
    interaction: discord.Interaction,
) -> None:

    embed = discord.Embed(
        title="🎙️ Cloud Voice AI — Help",
        description=(
            "أوامر البوت الأساسية:"
        ),
        color=discord.Color.blurple(),
    )

    commands_list = [
        ("🏓 `/ping`", "اختبار الاستجابة."),
        ("ℹ️ `/botinfo`", "معلومات البوت."),
        ("🎙️ `/join`", "دخول الروم الصوتي."),
        ("👋 `/leave`", "الخروج من الروم الصوتي."),
        ("🎤 `/voice`", "عرض الصوت الحالي."),
        ("🔊 `/setvoice`", "تغيير الصوت."),
        ("🎚️ `/voices`", "عرض جميع الأصوات."),
        ("🔎 `/voiceinfo`", "معلومات صوت معين."),
        ("🧠 `/memory`", "عرض حالة الذاكرة."),
        ("🧹 `/clear`", "مسح الذاكرة."),
        ("🔄 `/reset`", "إعادة ضبط الجلسة."),
        ("📊 `/stats`", "إحصائيات جلسة الصوت."),
    ]

    for name, description in commands_list:

        embed.add_field(
            name=name,
            value=description,
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
    description=COMMAND_DESCRIPTIONS["join"],
)
async def join(
    interaction: discord.Interaction,
) -> None:

    if not interaction.guild:

        await interaction.response.send_message(
            "❌ هذا الأمر يعمل داخل السيرفر فقط.",
            ephemeral=True,
        )

        return

    channel = get_user_voice_channel(
        interaction
    )

    if channel is None:

        await interaction.response.send_message(
            "🎙️ ادخل روم صوتي أولًا ثم استخدم `/join`.",
            ephemeral=True,
        )

        return

    await interaction.response.defer()

    try:

        session = await bot.voice_manager.join(
            guild=interaction.guild,
            channel=channel,
            voice=DEFAULT_GEMINI_VOICE,
        )

        bot.total_joins += 1

        await interaction.followup.send(
            f"🎙️ دخلت **{channel.name}**!\n"
            f"🗣️ الصوت: `{session.voice}`\n"
            f"🧠 الاستماع والرد الصوتي جاهز.",
        )

    except Exception as exc:

        bot.total_errors += 1

        logger.exception(
            "Failed to join voice channel."
        )

        await interaction.followup.send(
            f"❌ ما قدرت أدخل الروم الصوتي.\n"
            f"```{truncate_text(str(exc), 1200)}```"
        )


# ============================================================
# /LEAVE
# ============================================================

@bot.tree.command(
    name="leave",
    description=COMMAND_DESCRIPTIONS["leave"],
)
async def leave(
    interaction: discord.Interaction,
) -> None:

    if not interaction.guild:

        await interaction.response.send_message(
            "❌ هذا الأمر يعمل داخل السيرفر فقط.",
            ephemeral=True,
        )

        return

    await interaction.response.defer(
        ephemeral=True
    )

    try:

        disconnected = await bot.voice_manager.leave(
            interaction.guild.id
        )

        if disconnected:

            bot.total_leaves += 1

            await interaction.followup.send(
                "👋 طلعت من الروم الصوتي."
            )

        else:

            await interaction.followup.send(
                "ℹ️ أنا مو داخل روم صوتي حاليًا."
            )

    except Exception as exc:

        bot.total_errors += 1

        logger.exception(
            "Failed to leave voice channel."
        )

        await interaction.followup.send(
            f"❌ حصل خطأ أثناء الخروج.\n"
            f"```{truncate_text(str(exc), 1200)}```"
        )


# ============================================================
# /VOICE
# ============================================================

@bot.tree.command(
    name="voice",
    description=COMMAND_DESCRIPTIONS["voice"],
)
async def voice(
    interaction: discord.Interaction,
) -> None:

    if not interaction.guild:

        await interaction.response.send_message(
            "❌ هذا الأمر يعمل داخل السيرفر فقط.",
            ephemeral=True,
        )

        return

    session = bot.voice_manager.get(
        interaction.guild.id
    )

    if session is None:

        await interaction.response.send_message(
            f"🎤 الصوت الافتراضي: `{DEFAULT_GEMINI_VOICE}`\n"
            "ℹ️ لا توجد جلسة صوتية نشطة حاليًا.",
            ephemeral=True,
        )

        return

    await interaction.response.send_message(
        f"🎤 الصوت الحالي: `{session.voice}`\n"
        f"🎙️ القناة: `{session.channel.name}`",
        ephemeral=True,
    )


# ============================================================
# VOICE AUTOCOMPLETE
# ============================================================

async def voice_autocomplete(
    interaction: discord.Interaction,
    current: str,
) -> list[app_commands.Choice[str]]:

    current = current.lower().strip()

    matches = [
        voice
        for voice in GEMINI_VOICES
        if current in voice.lower()
    ]

    return [
        app_commands.Choice(
            name=voice,
            value=voice,
        )
        for voice in matches[:25]
    ]


# ============================================================
# /SETVOICE
# ============================================================

@bot.tree.command(
    name="setvoice",
    description=COMMAND_DESCRIPTIONS["setvoice"],
)
@app_commands.describe(
    voice="اختر صوت Gemini."
)
@app_commands.autocomplete(
    voice=voice_autocomplete
)
async def setvoice(
    interaction: discord.Interaction,
    voice: str,
) -> None:

    if not interaction.guild:

        await interaction.response.send_message(
            "❌ هذا الأمر يعمل داخل السيرفر فقط.",
            ephemeral=True,
        )

        return

    if not is_valid_voice(voice):

        await interaction.response.send_message(
            "❌ هذا الصوت غير موجود.\n"
            "استخدم `/voices` لرؤية الأصوات المتاحة.",
            ephemeral=True,
        )

        return

    selected_voice = normalize_voice_name(
        voice
    )

    session = bot.voice_manager.get(
        interaction.guild.id
    )

    if session is None:

        await interaction.response.send_message(
            f"🎤 الصوت الافتراضي الحالي: `{DEFAULT_GEMINI_VOICE}`\n\n"
            f"💡 ادخل البوت أولًا باستخدام `/join` "
            f"ثم غيّر الصوت إلى `{selected_voice}`.",
            ephemeral=True,
        )

        return

    try:

        session.set_voice(
            selected_voice
        )

        await interaction.response.send_message(
            f"✅ تم تغيير صوت الذكاء الاصطناعي إلى:\n"
            f"🎙️ **{selected_voice}**",
        )

    except Exception as exc:

        bot.total_errors += 1

        logger.exception(
            "Failed to change voice."
        )

        await interaction.response.send_message(
            f"❌ تعذر تغيير الصوت.\n"
            f"```{truncate_text(str(exc), 1000)}```",
            ephemeral=True,
        )


# ============================================================
# /VOICES
# ============================================================

@bot.tree.command(
    name="voices",
    description=COMMAND_DESCRIPTIONS["voices"],
)
async def voices(
    interaction: discord.Interaction,
) -> None:

    await interaction.response.defer(
        ephemeral=True
    )

    try:

        await send_voice_list(
            interaction
        )

    except Exception:

        logger.exception(
            "Failed to send voice list."
        )

        lines = [
            f"`{index + 1:02d}` — **{voice}**"
            for index, voice in enumerate(
                GEMINI_VOICES
            )
        ]

        embed = discord.Embed(
            title="🎙️ Gemini Voices",
            description="\n".join(lines),
            color=discord.Color.blurple(),
        )

        await interaction.followup.send(
            embed=embed
        )


# ============================================================
# /VOICEINFO
# ============================================================

@bot.tree.command(
    name="voiceinfo",
    description=COMMAND_DESCRIPTIONS["voiceinfo"],
)
@app_commands.describe(
    voice="الصوت الذي تريد معلومات عنه."
)
@app_commands.autocomplete(
    voice=voice_autocomplete
)
async def voiceinfo(
    interaction: discord.Interaction,
    voice: str,
) -> None:

    if not is_valid_voice(voice):

        await interaction.response.send_message(
            "❌ الصوت غير موجود.",
            ephemeral=True,
        )

        return

    selected = normalize_voice_name(
        voice
    )

    is_default = (
        selected == normalize_voice_name(
            DEFAULT_GEMINI_VOICE
        )
    )

    embed = discord.Embed(
        title=f"🎙️ {selected}",
        description=(
            "صوت متاح لنظام Gemini TTS في Cloud Voice AI."
        ),
        color=discord.Color.blurple(),
    )

    embed.add_field(
        name="🔊 Voice ID",
        value=f"`{selected}`",
        inline=True,
    )

    embed.add_field(
        name="⭐ Default",
        value="نعم" if is_default else "لا",
        inline=True,
    )

    embed.add_field(
        name="📋 Available",
        value="نعم",
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
    description=COMMAND_DESCRIPTIONS["memory"],
)
async def memory(
    interaction: discord.Interaction,
) -> None:

    if not interaction.guild:

        await interaction.response.send_message(
            "❌ هذا الأمر يعمل داخل السيرفر فقط.",
            ephemeral=True,
        )

        return

    session = bot.voice_manager.get(
        interaction.guild.id
    )

    if session is None:

        await interaction.response.send_message(
            "ℹ️ لا توجد جلسة صوتية نشطة.",
            ephemeral=True,
        )

        return

    try:

        count = session.memory_count

    except AttributeError:

        count = 0

    status = (
        "مفعلة"
        if MEMORY_ENABLED
        else "معطلة"
    )

    await interaction.response.send_message(
        f"🧠 **Memory**\n\n"
        f"الحالة: `{status}`\n"
        f"الرسائل: `{count}/{MAX_MEMORY_MESSAGES}`",
        ephemeral=True,
    )


# ============================================================
# /CLEAR
# ============================================================

@bot.tree.command(
    name="clear",
    description=COMMAND_DESCRIPTIONS["clear"],
)
async def clear(
    interaction: discord.Interaction,
) -> None:

    if not interaction.guild:

        await interaction.response.send_message(
            "❌ هذا الأمر يعمل داخل السيرفر فقط.",
            ephemeral=True,
        )

        return

    session = bot.voice_manager.get(
        interaction.guild.id
    )

    if session is None:

        await interaction.response.send_message(
            "ℹ️ لا توجد جلسة صوتية نشطة.",
            ephemeral=True,
        )

        return

    try:

        session.clear_memory()

        await interaction.response.send_message(
            "🧹 تم مسح ذاكرة المحادثة.",
            ephemeral=True,
        )

    except Exception as exc:

        bot.total_errors += 1

        logger.exception(
            "Failed to clear memory."
        )

        await interaction.response.send_message(
            f"❌ تعذر مسح الذاكرة.\n"
            f"```{truncate_text(str(exc), 1000)}```",
            ephemeral=True,
        )


# ============================================================
# /RESET
# ============================================================

@bot.tree.command(
    name="reset",
    description=COMMAND_DESCRIPTIONS["reset"],
)
async def reset(
    interaction: discord.Interaction,
) -> None:

    if not interaction.guild:

        await interaction.response.send_message(
            "❌ هذا الأمر يعمل داخل السيرفر فقط.",
            ephemeral=True,
        )

        return

    session = bot.voice_manager.get(
        interaction.guild.id
    )

    if session is None:

        await interaction.response.send_message(
            "ℹ️ لا توجد جلسة صوتية نشطة.",
            ephemeral=True,
        )

        return

    try:

        session.reset()

        await interaction.response.send_message(
            "🔄 تم إعادة ضبط جلسة الذكاء الاصطناعي.",
            ephemeral=True,
        )

    except Exception as exc:

        bot.total_errors += 1

        logger.exception(
            "Failed to reset session."
        )

        await interaction.response.send_message(
            f"❌ تعذر إعادة ضبط الجلسة.\n"
            f"```{truncate_text(str(exc), 1000)}```",
            ephemeral=True,
        )


# ============================================================
# /STATS
# ============================================================

@bot.tree.command(
    name="stats",
    description=COMMAND_DESCRIPTIONS["stats"],
)
async def stats(
    interaction: discord.Interaction,
) -> None:

    if not interaction.guild:

        await interaction.response.send_message(
            "❌ هذا الأمر يعمل داخل السيرفر فقط.",
            ephemeral=True,
        )

        return

    session = bot.voice_manager.get(
        interaction.guild.id
    )

    if session is None:

        await interaction.response.send_message(
            "ℹ️ لا توجد جلسة صوتية نشطة.",
            ephemeral=True,
        )

        return

    embed = discord.Embed(
        title="📊 Voice AI Statistics",
        color=discord.Color.blurple(),
    )

    embed.add_field(
        name="🎙️ Voice",
        value=f"`{session.voice}`",
        inline=True,
    )

    embed.add_field(
        name="👥 Channel",
        value=f"`{session.channel.name}`",
        inline=True,
    )

    try:
        processed = session.processed_requests
    except AttributeError:
        processed = 0

    try:
        failed = session.failed_requests
    except AttributeError:
        failed = 0

    try:
        memory_count = session.memory_count
    except AttributeError:
        memory_count = 0

    embed.add_field(
        name="🧠 Memory",
        value=f"`{memory_count}/{MAX_MEMORY_MESSAGES}`",
        inline=True,
    )

    embed.add_field(
        name="💬 Requests",
        value=f"`{processed}`",
        inline=True,
    )

    embed.add_field(
        name="❌ Failed",
        value=f"`{failed}`",
        inline=True,
    )

    embed.add_field(
        name="🎤 Default",
        value=f"`{DEFAULT_GEMINI_VOICE}`",
        inline=True,
    )

    await interaction.response.send_message(
        embed=embed,
        ephemeral=True,
    )


# ============================================================
# AUTOCOMPLETE ERROR HANDLER
# ============================================================

@bot.tree.error
async def on_tree_error(
    interaction: discord.Interaction,
    error: app_commands.AppCommandError,
) -> None:

    bot.total_errors += 1

    logger.error(
        "Application command error: %s",
        error,
        exc_info=(
            type(error),
            error,
            error.__traceback__,
        ),
    )

    if interaction.response.is_done():

        try:

            await interaction.followup.send(
                "❌ حدث خطأ أثناء تنفيذ الأمر.",
                ephemeral=True,
            )

        except Exception:
            pass

    else:

        try:

            await interaction.response.send_message(
                "❌ حدث خطأ أثناء تنفيذ الأمر.",
                ephemeral=True,
            )

        except Exception:
            pass


# ============================================================
# GLOBAL EXCEPTION HANDLER
# ============================================================

def handle_unhandled_exception(
    exception_type,
    exception,
    traceback_object,
) -> None:

    logger.critical(
        "Unhandled Python exception.",
        exc_info=(
            exception_type,
            exception,
            traceback_object,
        ),
    )


# ============================================================
# MAIN
# ============================================================

async def main() -> None:

    if not DISCORD_TOKEN:
        raise RuntimeError(
            "DISCORD_TOKEN is missing."
        )

    logger.info(
        "============================================================"
    )

    logger.info(
        "%s v%s",
        PROJECT_NAME,
        PROJECT_VERSION,
    )

    logger.info(
        "%s",
        PROJECT_DESCRIPTION,
    )

    logger.info(
        "Starting Discord client..."
    )

    logger.info(
        "Default Gemini voice: %s",
        DEFAULT_GEMINI_VOICE,
    )

    logger.info(
        "Memory: %s",
        "enabled" if MEMORY_ENABLED else "disabled",
    )

    logger.info(
        "============================================================"
    )

    try:

        async with bot:

            await bot.start(
                DISCORD_TOKEN
            )

    except KeyboardInterrupt:

        logger.info(
            "Shutdown requested by keyboard."
        )

    except asyncio.CancelledError:

        logger.info(
            "Main task cancelled."
        )

        raise

    except Exception:

        logger.exception(
            "Fatal bot error."
        )

        raise

    finally:

        if not bot.is_closed():

            await bot.shutdown()

        logger.info(
            "Cloud Voice AI stopped."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        asyncio.run(
            main()
        )

    except KeyboardInterrupt:

        logger.info(
            "Cloud Voice AI terminated."
        )

    except Exception:

        logger.exception(
            "Cloud Voice AI terminated with an error."
        )

        raise
