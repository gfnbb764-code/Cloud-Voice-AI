# main.py
# ============================================================
# Cloud Voice AI — Main Discord Bot
# ============================================================

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

import discord
from discord import app_commands
from discord.ext import commands

from config import (
    ALLOW_VOICE_CHANGE,
    AUTO_LEAVE_EMPTY_CHANNEL,
    AUTO_LEAVE_DELAY_SECONDS,
    CHARACTER_STORAGE_FILE,
    CHARACTERS_ENABLED,
    COMMAND_DESCRIPTIONS,
    DEFAULT_CHARACTER_INSTRUCTIONS,
    DEFAULT_CHARACTER_NAME,
    DEFAULT_CHARACTER_PERSONALITY,
    DEFAULT_CHARACTER_STYLE,
    DEFAULT_GEMINI_VOICE,
    DEFAULT_SPEECH_SPEED,
    DISCORD_ACTIVITY_TYPE,
    DISCORD_GUILD_ID,
    DISCORD_STATUS,
    DISCORD_TOKEN,
    GEMINI_VOICES,
    LOG_LEVEL,
    MAX_CHARACTERS_PER_GUILD,
    MAX_SPEECH_SPEED,
    MIN_SPEECH_SPEED,
    MODERATION_AI_ENABLED,
    MODERATION_MAX_REASON_LENGTH,
    MODERATION_OWNER_ONLY,
    get_safe_config,
    is_valid_voice,
    normalize_speech_speed,
    normalize_voice_name,
)

from voice import (
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

logger = logging.getLogger("CloudVoiceAI")


# ============================================================
# CHARACTER DATA
# ============================================================

class Character:

    def __init__(
        self,
        *,
        name: str,
        voice: str = DEFAULT_GEMINI_VOICE,
        speed: float = DEFAULT_SPEECH_SPEED,
        personality: str = DEFAULT_CHARACTER_PERSONALITY,
        style: str = DEFAULT_CHARACTER_STYLE,
        instructions: str = DEFAULT_CHARACTER_INSTRUCTIONS,
    ) -> None:

        self.name = name.strip()

        self.voice = normalize_voice_name(
            voice
        )

        self.speed = normalize_speech_speed(
            speed
        )

        self.personality = (
            personality.strip()
            or DEFAULT_CHARACTER_PERSONALITY
        )

        self.style = (
            style.strip()
            or DEFAULT_CHARACTER_STYLE
        )

        self.instructions = (
            instructions.strip()
            or DEFAULT_CHARACTER_INSTRUCTIONS
        )

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "name": self.name,
            "voice": self.voice,
            "speed": self.speed,
            "personality": self.personality,
            "style": self.style,
            "instructions": self.instructions,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "Character":

        return cls(
            name=str(
                data.get(
                    "name",
                    DEFAULT_CHARACTER_NAME,
                )
            ),
            voice=str(
                data.get(
                    "voice",
                    DEFAULT_GEMINI_VOICE,
                )
            ),
            speed=float(
                data.get(
                    "speed",
                    DEFAULT_SPEECH_SPEED,
                )
            ),
            personality=str(
                data.get(
                    "personality",
                    DEFAULT_CHARACTER_PERSONALITY,
                )
            ),
            style=str(
                data.get(
                    "style",
                    DEFAULT_CHARACTER_STYLE,
                )
            ),
            instructions=str(
                data.get(
                    "instructions",
                    DEFAULT_CHARACTER_INSTRUCTIONS,
                )
            ),
        )


# ============================================================
# CHARACTER MANAGER
# ============================================================

class CharacterManager:

    def __init__(
        self,
        path: str,
    ) -> None:

        self.path = Path(path)

        self.characters: dict[
            str,
            dict[str, Character],
        ] = {}

        self.selected: dict[
            str,
            str | None,
        ] = {}

        self._lock = asyncio.Lock()

        self._load()

    # ========================================================
    # STORAGE
    # ========================================================

    def _load(
        self,
    ) -> None:

        if not self.path.exists():
            return

        try:

            with self.path.open(
                "r",
                encoding="utf-8",
            ) as file:

                data = json.load(file)

            if not isinstance(data, dict):
                return

            raw_characters = data.get(
                "characters",
                {},
            )

            raw_selected = data.get(
                "selected",
                {},
            )

            if not isinstance(raw_characters, dict):
                raw_characters = {}

            if not isinstance(raw_selected, dict):
                raw_selected = {}

            for guild_id, items in raw_characters.items():

                if not isinstance(items, dict):
                    continue

                guild_characters: dict[
                    str,
                    Character,
                ] = {}

                for name, character_data in items.items():

                    if not isinstance(
                        character_data,
                        dict,
                    ):
                        continue

                    try:

                        character = Character.from_dict(
                            character_data
                        )

                        guild_characters[
                            str(name).lower()
                        ] = character

                    except Exception:

                        logger.exception(
                            "Failed to load character: %s",
                            name,
                        )

                self.characters[
                    str(guild_id)
                ] = guild_characters

            self.selected = {
                str(guild_id): (
                    str(value)
                    if value is not None
                    else None
                )
                for guild_id, value
                in raw_selected.items()
            }

            logger.info(
                "Character storage loaded"
            )

        except Exception:

            logger.exception(
                "Failed to load character storage"
            )

    async def _save(
        self,
    ) -> None:

        async with self._lock:

            data = {
                "characters": {
                    guild_id: {
                        key: character.to_dict()
                        for key, character in items.items()
                    }
                    for guild_id, items in self.characters.items()
                },
                "selected": self.selected,
            }

            self.path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            temporary = self.path.with_suffix(
                self.path.suffix + ".tmp"
            )

            def write_file() -> None:

                with temporary.open(
                    "w",
                    encoding="utf-8",
                ) as file:

                    json.dump(
                        data,
                        file,
                        ensure_ascii=False,
                        indent=2,
                    )

                os.replace(
                    temporary,
                    self.path,
                )

            await asyncio.to_thread(
                write_file
            )

    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def _key(
        name: str,
    ) -> str:

        return name.strip().lower()

    def _guild(
        self,
        guild_id: int,
    ) -> dict[str, Character]:

        key = str(guild_id)

        if key not in self.characters:
            self.characters[key] = {}

        return self.characters[key]

    # ========================================================
    # CREATE
    # ========================================================

    async def create(
        self,
        guild_id: int,
        character: Character,
    ) -> None:

        items = self._guild(
            guild_id
        )

        key = self._key(
            character.name
        )

        if not key:
            raise ValueError(
                "Character name cannot be empty."
            )

        if key in items:
            raise ValueError(
                "A character with this name already exists."
            )

        if len(items) >= MAX_CHARACTERS_PER_GUILD:
            raise ValueError(
                "Maximum number of characters reached."
            )

        items[key] = character

        await self._save()

    # ========================================================
    # GET
    # ========================================================

    def get(
        self,
        guild_id: int,
        name: str,
    ) -> Character | None:

        return self._guild(
            guild_id
        ).get(
            self._key(name)
        )

    # ========================================================
    # LIST
    # ========================================================

    def list(
        self,
        guild_id: int,
    ) -> list[Character]:

        return list(
            self._guild(
                guild_id
            ).values()
        )

    # ========================================================
    # SELECTED
    # ========================================================

    def get_selected(
        self,
        guild_id: int,
    ) -> Character | None:

        selected = self.selected.get(
            str(guild_id)
        )

        if not selected:
            return None

        return self.get(
            guild_id,
            selected,
        )

    async def select(
        self,
        guild_id: int,
        name: str,
    ) -> Character:

        character = self.get(
            guild_id,
            name,
        )

        if character is None:
            raise ValueError(
                "Character not found."
            )

        self.selected[
            str(guild_id)
        ] = character.name

        await self._save()

        return character

    async def clear_selected(
        self,
        guild_id: int,
    ) -> None:

        self.selected[
            str(guild_id)
        ] = None

        await self._save()

    # ========================================================
    # DELETE
    # ========================================================

    async def delete(
        self,
        guild_id: int,
        name: str,
    ) -> Character:

        items = self._guild(
            guild_id
        )

        key = self._key(name)

        character = items.pop(
            key,
            None,
        )

        if character is None:
            raise ValueError(
                "Character not found."
            )

        selected = self.selected.get(
            str(guild_id)
        )

        if (
            selected
            and selected.lower()
            == character.name.lower()
        ):

            self.selected[
                str(guild_id)
            ] = None

        await self._save()

        return character

    # ========================================================
    # UPDATE
    # ========================================================

    async def update(
        self,
        guild_id: int,
        old_name: str,
        *,
        name: str | None = None,
        voice: str | None = None,
        speed: float | None = None,
        personality: str | None = None,
        style: str | None = None,
        instructions: str | None = None,
    ) -> Character:

        items = self._guild(
            guild_id
        )

        old_key = self._key(
            old_name
        )

        character = items.get(
            old_key
        )

        if character is None:
            raise ValueError(
                "Character not found."
            )

        if name is not None:

            new_name = name.strip()

            if not new_name:
                raise ValueError(
                    "Character name cannot be empty."
                )

            new_key = self._key(
                new_name
            )

            if (
                new_key != old_key
                and new_key in items
            ):
                raise ValueError(
                    "Another character already uses that name."
                )

            character.name = new_name

            if new_key != old_key:

                items.pop(old_key)

                items[new_key] = character

                selected = self.selected.get(
                    str(guild_id)
                )

                if (
                    selected
                    and selected.lower()
                    == old_name.lower()
                ):

                    self.selected[
                        str(guild_id)
                    ] = character.name

        if voice is not None:

            normalized = normalize_voice_name(
                voice
            )

            if not is_valid_voice(
                normalized
            ):
                raise ValueError(
                    f"Invalid voice: {voice}"
                )

            character.voice = normalized

        if speed is not None:

            character.speed = normalize_speech_speed(
                speed
            )

        if personality is not None:

            character.personality = (
                personality.strip()
                or DEFAULT_CHARACTER_PERSONALITY
            )

        if style is not None:

            character.style = (
                style.strip()
                or DEFAULT_CHARACTER_STYLE
            )

        if instructions is not None:

            character.instructions = (
                instructions.strip()
                or DEFAULT_CHARACTER_INSTRUCTIONS
            )

        await self._save()

        return character


# ============================================================
# BOT
# ============================================================

class CloudVoiceBot(commands.Bot):

    def __init__(self) -> None:

        intents = discord.Intents.default()

        intents.guilds = True
        intents.voice_states = True
        intents.members = True
        intents.message_content = False

        super().__init__(
            command_prefix="!",
            intents=intents,
        )

        self.voice_manager = VoiceSessionManager(
            self
        )

        self.character_manager = CharacterManager(
            CHARACTER_STORAGE_FILE
        )

        self.ready_once = False
        self.synced = False

    # ========================================================
    # SETUP
    # ========================================================

    async def setup_hook(
        self,
    ) -> None:

        await self._sync_commands()

    async def _sync_commands(
        self,
    ) -> None:

        if self.synced:
            return

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

            else:

                synced = await self.tree.sync()

            self.synced = True

            logger.info(
                "Synced %s application commands",
                len(synced),
            )

        except Exception:

            logger.exception(
                "Failed to sync commands"
            )

    # ========================================================
    # READY
    # ========================================================

    async def on_ready(
        self,
    ) -> None:

        if self.user is None:
            return

        activity_type = (
            DISCORD_ACTIVITY_TYPE.lower()
        )

        if activity_type == "watching":

            activity = discord.Activity(
                type=discord.ActivityType.watching,
                name=DISCORD_STATUS,
            )

        elif activity_type == "playing":

            activity = discord.Game(
                name=DISCORD_STATUS
            )

        else:

            activity = discord.Activity(
                type=discord.ActivityType.listening,
                name=DISCORD_STATUS,
            )

        await self.change_presence(
            activity=activity
        )

        if not self.ready_once:

            logger.info(
                "Cloud Voice AI is online as %s",
                self.user,
            )

            self.ready_once = True

    # ========================================================
    # CLOSE
    # ========================================================

    async def close(
        self,
    ) -> None:

        logger.info(
            "Shutting down Cloud Voice AI..."
        )

        await self.voice_manager.close_all()

        await super().close()


bot = CloudVoiceBot()


# ============================================================
# VOICE AUTOCOMPLETE
# ============================================================

async def voice_autocomplete(
    interaction: discord.Interaction,
    current: str,
) -> list[app_commands.Choice[str]]:

    current = current.strip().lower()

    matches = [
        voice
        for voice in GEMINI_VOICES
        if (
            not current
            or current in voice.lower()
        )
    ]

    return [
        app_commands.Choice(
            name=voice,
            value=voice,
        )
        for voice in matches[:25]
    ]


# ============================================================
# CHARACTER AUTOCOMPLETE
# ============================================================

async def character_autocomplete(
    interaction: discord.Interaction,
    current: str,
) -> list[app_commands.Choice[str]]:

    if interaction.guild is None:
        return []

    current = current.strip().lower()

    characters = bot.character_manager.list(
        interaction.guild.id
    )

    selected = bot.character_manager.get_selected(
        interaction.guild.id
    )

    matches: list[app_commands.Choice[str]] = []

    for character in characters:

        if (
            current
            and current not in character.name.lower()
        ):
            continue

        marker = (
            "🟢 "
            if (
                selected
                and selected.name.lower()
                == character.name.lower()
            )
            else ""
        )

        matches.append(
            app_commands.Choice(
                name=(
                    f"{marker}{character.name}"
                )[:100],
                value=character.name,
            )
        )

    return matches[:25]


# ============================================================
# BASIC COMMANDS
# ============================================================

@bot.tree.command(
    name="ping",
    description=COMMAND_DESCRIPTIONS["ping"],
)
async def ping(
    interaction: discord.Interaction,
) -> None:

    latency = round(
        bot.latency * 1000
    )

    await interaction.response.send_message(
        f"🏓 Pong! `{latency}ms`"
    )


@bot.tree.command(
    name="botinfo",
    description=COMMAND_DESCRIPTIONS["botinfo"],
)
async def botinfo(
    interaction: discord.Interaction,
) -> None:

    config = get_safe_config()

    embed = discord.Embed(
        title="🤖 Cloud Voice AI",
        color=discord.Color.blurple(),
    )

    embed.add_field(
        name="Version",
        value=str(config["version"]),
        inline=True,
    )

    embed.add_field(
        name="Chat",
        value=str(config["chat_model"]),
        inline=True,
    )

    embed.add_field(
        name="TTS",
        value=str(config["tts_model"]),
        inline=True,
    )

    embed.add_field(
        name="Voices",
        value=str(config["voice_count"]),
        inline=True,
    )

    embed.add_field(
        name="Characters",
        value=(
            "Enabled"
            if config["characters_enabled"]
            else "Disabled"
        ),
        inline=True,
    )

    embed.add_field(
        name="Moderation",
        value=(
            "Owner only"
            if config["moderation_owner_only"]
            else "Enabled"
        ),
        inline=True,
    )

    await interaction.response.send_message(
        embed=embed
    )


@bot.tree.command(
    name="help",
    description=COMMAND_DESCRIPTIONS["help"],
)
async def help_command(
    interaction: discord.Interaction,
) -> None:

    embed = discord.Embed(
        title="📚 Cloud Voice AI",
        description=(
            "أوامر البوت الأساسية والشخصيات والإشراف."
        ),
        color=discord.Color.blurple(),
    )

    embed.add_field(
        name="🎙️ Voice",
        value=(
            "`/join`\n"
            "`/leave`\n"
            "`/voice`\n"
            "`/setvoice`\n"
            "`/voices`\n"
            "`/speed`"
        ),
        inline=True,
    )

    embed.add_field(
        name="🤖 Characters",
        value=(
            "`/character create`\n"
            "`/character edit`\n"
            "`/character select`\n"
            "`/character list`\n"
            "`/character view`\n"
            "`/character delete`"
        ),
        inline=True,
    )

    embed.add_field(
        name="🧠 Memory",
        value=(
            "`/memory`\n"
            "`/clear`\n"
            "`/reset`\n"
            "`/stats`"
        ),
        inline=True,
    )

    embed.add_field(
        name="🛡️ Moderation",
        value=(
            "`/mod status`\n"
            "`/mod enable`\n"
            "`/mod disable`\n"
            "`/mod request`"
        ),
        inline=True,
    )

    await interaction.response.send_message(
        embed=embed
    )


# ============================================================
# VOICE COMMANDS
# ============================================================

@bot.tree.command(
    name="join",
    description=COMMAND_DESCRIPTIONS["join"],
)
async def join(
    interaction: discord.Interaction,
) -> None:

    if interaction.guild is None:

        await interaction.response.send_message(
            "❌ هذا الأمر يعمل داخل السيرفر فقط.",
            ephemeral=True,
        )
        return

    member = interaction.user

    if not isinstance(
        member,
        discord.Member,
    ):

        await interaction.response.send_message(
            "❌ تعذر معرفة الروم الصوتي.",
            ephemeral=True,
        )
        return

    if member.voice is None:

        await interaction.response.send_message(
            "❌ ادخل روم صوتي أولًا.",
            ephemeral=True,
        )
        return

    channel = member.voice.channel

    if channel is None:

        await interaction.response.send_message(
            "❌ ادخل روم صوتي أولًا.",
            ephemeral=True,
        )
        return

    await interaction.response.defer()

    selected = (
        bot.character_manager.get_selected(
            interaction.guild.id
        )
        if CHARACTERS_ENABLED
        else None
    )

    try:

        session = await bot.voice_manager.join(
            channel=channel,
            voice=(
                selected.voice
                if selected
                else DEFAULT_GEMINI_VOICE
            ),
            speed=(
                selected.speed
                if selected
                else DEFAULT_SPEECH_SPEED
            ),
            character=selected,
        )

        await interaction.followup.send(
            (
                f"🎙️ دخلت **{channel.name}**.\n"
                f"🔊 الصوت: **{session.voice}**\n"
                f"⚡ السرعة: **{session.speed:.2f}x**"
                + (
                    f"\n🤖 الشخصية: **{selected.name}**"
                    if selected
                    else ""
                )
            )
        )

    except Exception as error:

        logger.exception(
            "Join failed"
        )

        await interaction.followup.send(
            f"❌ فشل الدخول: `{error}`"
        )


@bot.tree.command(
    name="leave",
    description=COMMAND_DESCRIPTIONS["leave"],
)
async def leave(
    interaction: discord.Interaction,
) -> None:

    if interaction.guild is None:

        await interaction.response.send_message(
            "❌ هذا الأمر يعمل داخل السيرفر فقط.",
            ephemeral=True,
        )
        return

    success = await bot.voice_manager.leave(
        interaction.guild.id
    )

    await interaction.response.send_message(
        (
            "👋 طلعت من الروم الصوتي."
            if success
            else "ℹ️ البوت غير موجود في روم صوتي."
        )
    )


@bot.tree.command(
    name="voice",
    description=COMMAND_DESCRIPTIONS["voice"],
)
async def voice(
    interaction: discord.Interaction,
) -> None:

    if interaction.guild is None:

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
            "🔇 البوت غير موجود في روم صوتي."
        )
        return

    character = session.character

    await interaction.response.send_message(
        (
            f"🎙️ الصوت: **{session.voice}**\n"
            f"⚡ السرعة: **{session.speed:.2f}x**\n"
            f"📡 الروم: **{session.channel.name}**\n"
            f"🤖 الشخصية: **{character.name if character else 'بدون شخصية'}**"
        )
    )


@bot.tree.command(
    name="setvoice",
    description=COMMAND_DESCRIPTIONS["setvoice"],
)
@app_commands.describe(
    voice="اسم صوت Gemini",
)
@app_commands.autocomplete(
    voice=voice_autocomplete
)
async def setvoice(
    interaction: discord.Interaction,
    voice: str,
) -> None:

    if interaction.guild is None:

        await interaction.response.send_message(
            "❌ هذا الأمر يعمل داخل السيرفر فقط.",
            ephemeral=True,
        )
        return

    if not ALLOW_VOICE_CHANGE:

        await interaction.response.send_message(
            "❌ تغيير الأصوات معطل.",
            ephemeral=True,
        )
        return

    session = bot.voice_manager.get(
        interaction.guild.id
    )

    if session is None:

        await interaction.response.send_message(
            "❌ استخدم `/join` أولًا.",
            ephemeral=True,
        )
        return

    try:

        selected_voice = session.set_voice(
            voice
        )

        selected_character = session.character

        if selected_character:

            await bot.character_manager.update(
                interaction.guild.id,
                selected_character.name,
                voice=selected_voice,
            )

        await interaction.response.send_message(
            f"🔊 تم تغيير الصوت إلى **{selected_voice}**."
        )

    except Exception as error:

        await interaction.response.send_message(
            f"❌ {error}",
            ephemeral=True,
        )


@bot.tree.command(
    name="voices",
    description=COMMAND_DESCRIPTIONS["voices"],
)
async def voices(
    interaction: discord.Interaction,
) -> None:

    await send_voice_list(
        interaction
    )


@bot.tree.command(
    name="speed",
    description=COMMAND_DESCRIPTIONS["speed"],
)
@app_commands.describe(
    speed="سرعة الكلام",
)
async def speed(
    interaction: discord.Interaction,
    speed: app_commands.Range[
        float,
        MIN_SPEECH_SPEED,
        MAX_SPEECH_SPEED,
    ],
) -> None:

    if interaction.guild is None:

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
            "❌ استخدم `/join` أولًا.",
            ephemeral=True,
        )
        return

    new_speed = session.set_speed(
        float(speed)
    )

    character = session.character

    if character:

        await bot.character_manager.update(
            interaction.guild.id,
            character.name,
            speed=new_speed,
        )

        session.set_character(
            bot.character_manager.get(
                interaction.guild.id,
                character.name,
            )
        )

    await interaction.response.send_message(
        (
            f"⚡ سرعة الكلام الآن **{new_speed:.2f}x**."
            + (
                f"\n🤖 تم حفظها للشخصية **{character.name}**."
                if character
                else ""
            )
        )
    )


# ============================================================
# CHARACTER GROUP
# ============================================================

character_group = app_commands.Group(
    name="character",
    description=COMMAND_DESCRIPTIONS["character"],
)


@character_group.command(
    name="create",
    description=COMMAND_DESCRIPTIONS["character_create"],
)
@app_commands.describe(
    name="اسم الشخصية",
    voice="صوت الشخصية",
    speed="سرعة الكلام",
    personality="شخصية وطبع الذكاء الاصطناعي",
    style="أسلوب الكلام",
    instructions="تعليمات إضافية للشخصية",
)
@app_commands.autocomplete(
    voice=voice_autocomplete
)
async def character_create(
    interaction: discord.Interaction,
    name: app_commands.Range[str, 1, 50],
    voice: str,
    speed: app_commands.Range[float, 0.5, 2.0],
    personality: app_commands.Range[str, 1, 500],
    style: app_commands.Range[str, 1, 500],
    instructions: app_commands.Range[str, 1, 1500],
) -> None:

    if not CHARACTERS_ENABLED:

        await interaction.response.send_message(
            "❌ نظام الشخصيات معطل.",
            ephemeral=True,
        )
        return

    if interaction.guild is None:

        await interaction.response.send_message(
            "❌ هذا الأمر يعمل داخل السيرفر فقط.",
            ephemeral=True,
        )
        return

    normalized_voice = normalize_voice_name(
        voice
    )

    if not is_valid_voice(
        normalized_voice
    ):

        await interaction.response.send_message(
            "❌ الصوت غير صالح.",
            ephemeral=True,
        )
        return

    character = Character(
        name=name,
        voice=normalized_voice,
        speed=float(speed),
        personality=personality,
        style=style,
        instructions=instructions,
    )

    try:

        await bot.character_manager.create(
            interaction.guild.id,
            character,
        )

        await interaction.response.send_message(
            (
                f"✅ تم إنشاء الشخصية **{character.name}**.\n"
                f"🔊 الصوت: **{character.voice}**\n"
                f"⚡ السرعة: **{character.speed:.2f}x**\n"
                f"🎭 الشخصية: **{character.personality}**\n"
                f"🗣️ الأسلوب: **{character.style}**"
            )
        )

    except Exception as error:

        await interaction.response.send_message(
            f"❌ {error}",
            ephemeral=True,
        )


@character_group.command(
    name="edit",
    description=COMMAND_DESCRIPTIONS["character_edit"],
)
@app_commands.describe(
    character="اسم الشخصية الحالية",
    new_name="الاسم الجديد",
    voice="الصوت الجديد",
    speed="السرعة الجديدة",
    personality="الشخصية الجديدة",
    style="أسلوب الكلام الجديد",
    instructions="التعليمات الجديدة",
)
@app_commands.autocomplete(
    voice=voice_autocomplete,
    character=character_autocomplete,
)
async def character_edit(
    interaction: discord.Interaction,
    character: str,
    new_name: str | None = None,
    voice: str | None = None,
    speed: app_commands.Range[
        float,
        0.5,
        2.0,
    ] | None = None,
    personality: str | None = None,
    style: str | None = None,
    instructions: str | None = None,
) -> None:

    if not CHARACTERS_ENABLED:

        await interaction.response.send_message(
            "❌ نظام الشخصيات معطل.",
            ephemeral=True,
        )
        return

    if interaction.guild is None:

        await interaction.response.send_message(
            "❌ هذا الأمر يعمل داخل السيرفر فقط.",
            ephemeral=True,
        )
        return

    if all(
        value is None
        for value in (
            new_name,
            voice,
            speed,
            personality,
            style,
            instructions,
        )
    ):

        await interaction.response.send_message(
            "❌ حدد شيئًا واحدًا على الأقل لتعديله.",
            ephemeral=True,
        )
        return

    try:

        updated = await bot.character_manager.update(
            interaction.guild.id,
            character,
            name=new_name,
            voice=voice,
            speed=(
                float(speed)
                if speed is not None
                else None
            ),
            personality=personality,
            style=style,
            instructions=instructions,
        )

        session = bot.voice_manager.get(
            interaction.guild.id
        )

        if session:

            live_character = (
                bot.character_manager.get(
                    interaction.guild.id,
                    updated.name,
                )
            )

            session.set_character(
                live_character
            )

            if live_character:

                session.voice_name = (
                    live_character.voice
                )

                session.speech_speed = (
                    live_character.speed
                )

        await interaction.response.send_message(
            f"✅ تم تعديل الشخصية **{updated.name}**."
        )

    except Exception as error:

        await interaction.response.send_message(
            f"❌ {error}",
            ephemeral=True,
        )


@character_group.command(
    name="select",
    description=COMMAND_DESCRIPTIONS["character_select"],
)
@app_commands.describe(
    character="اختر الشخصية",
)
@app_commands.autocomplete(
    character=character_autocomplete,
)
async def character_select(
    interaction: discord.Interaction,
    character: str,
) -> None:

    if not CHARACTERS_ENABLED:

        await interaction.response.send_message(
            "❌ نظام الشخصيات معطل.",
            ephemeral=True,
        )
        return

    if interaction.guild is None:

        await interaction.response.send_message(
            "❌ هذا الأمر يعمل داخل السيرفر فقط.",
            ephemeral=True,
        )
        return

    # ========================================================
    # Verify the character belongs to this guild.
    # ========================================================

    selected_character = bot.character_manager.get(
        interaction.guild.id,
        character,
    )

    if selected_character is None:

        await interaction.response.send_message(
            "❌ اختر شخصية موجودة في هذا السيرفر من القائمة.",
            ephemeral=True,
        )
        return

    try:

        selected = await bot.character_manager.select(
            interaction.guild.id,
            selected_character.name,
        )

        session = bot.voice_manager.get(
            interaction.guild.id
        )

        if session:

            session.set_character(
                selected
            )

            session.voice_name = (
                selected.voice
            )

            session.speech_speed = (
                selected.speed
            )

        await interaction.response.send_message(
            (
                f"🤖 تم اختيار الشخصية **{selected.name}**!\n"
                f"🔊 الصوت: **{selected.voice}**\n"
                f"⚡ السرعة: **{selected.speed:.2f}x**"
            )
        )

    except Exception as error:

        logger.exception(
            "Character selection failed"
        )

        await interaction.response.send_message(
            f"❌ فشل اختيار الشخصية: `{error}`",
            ephemeral=True,
        )


@character_group.command(
    name="list",
    description=COMMAND_DESCRIPTIONS["character_list"],
)
async def character_list(
    interaction: discord.Interaction,
) -> None:

    if not CHARACTERS_ENABLED:

        await interaction.response.send_message(
            "❌ نظام الشخصيات معطل.",
            ephemeral=True,
        )
        return

    if interaction.guild is None:

        await interaction.response.send_message(
            "❌ هذا الأمر يعمل داخل السيرفر فقط.",
            ephemeral=True,
        )
        return

    characters = bot.character_manager.list(
        interaction.guild.id
    )

    selected = bot.character_manager.get_selected(
        interaction.guild.id
    )

    if not characters:

        await interaction.response.send_message(
            "📭 لا توجد شخصيات في هذا السيرفر."
        )
        return

    lines: list[str] = []

    for character in characters:

        marker = (
            "🟢"
            if (
                selected
                and selected.name.lower()
                == character.name.lower()
            )
            else "⚪"
        )

        lines.append(
            f"{marker} **{character.name}** "
            f"— `{character.voice}` "
            f"— `{character.speed:.2f}x`"
        )

    embed = discord.Embed(
        title="🤖 الشخصيات",
        description="\n".join(lines),
        color=discord.Color.blurple(),
    )

    await interaction.response.send_message(
        embed=embed
    )


@character_group.command(
    name="view",
    description=COMMAND_DESCRIPTIONS["character_view"],
)
@app_commands.describe(
    character="اسم الشخصية",
)
@app_commands.autocomplete(
    character=character_autocomplete,
)
async def character_view(
    interaction: discord.Interaction,
    character: str,
) -> None:

    if not CHARACTERS_ENABLED:

        await interaction.response.send_message(
            "❌ نظام الشخصيات معطل.",
            ephemeral=True,
        )
        return

    if interaction.guild is None:

        await interaction.response.send_message(
            "❌ هذا الأمر يعمل داخل السيرفر فقط.",
            ephemeral=True,
        )
        return

    selected = bot.character_manager.get(
        interaction.guild.id,
        character,
    )

    if selected is None:

        await interaction.response.send_message(
            "❌ الشخصية غير موجودة.",
            ephemeral=True,
        )
        return

    embed = discord.Embed(
        title=f"🤖 {selected.name}",
        color=discord.Color.blurple(),
    )

    embed.add_field(
        name="🔊 Voice",
        value=selected.voice,
        inline=True,
    )

    embed.add_field(
        name="⚡ Speed",
        value=f"{selected.speed:.2f}x",
        inline=True,
    )

    embed.add_field(
        name="🎭 Personality",
        value=selected.personality[:1024],
        inline=False,
    )

    embed.add_field(
        name="🗣️ Style",
        value=selected.style[:1024],
        inline=False,
    )

    embed.add_field(
        name="📜 Instructions",
        value=selected.instructions[:1024],
        inline=False,
    )

    await interaction.response.send_message(
        embed=embed
    )


@character_group.command(
    name="delete",
    description=COMMAND_DESCRIPTIONS["character_delete"],
)
@app_commands.describe(
    character="اسم الشخصية",
)
@app_commands.autocomplete(
    character=character_autocomplete,
)
async def character_delete(
    interaction: discord.Interaction,
    character: str,
) -> None:

    if not CHARACTERS_ENABLED:

        await interaction.response.send_message(
            "❌ نظام الشخصيات معطل.",
            ephemeral=True,
        )
        return

    if interaction.guild is None:

        await interaction.response.send_message(
            "❌ هذا الأمر يعمل داخل السيرفر فقط.",
            ephemeral=True,
        )
        return

    try:

        deleted = await bot.character_manager.delete(
            interaction.guild.id,
            character,
        )

        session = bot.voice_manager.get(
            interaction.guild.id
        )

        if (
            session
            and session.character
            and session.character.name.lower()
            == deleted.name.lower()
        ):

            session.set_character(
                None
            )

            session.voice_name = (
                DEFAULT_GEMINI_VOICE
            )

            session.speech_speed = (
                DEFAULT_SPEECH_SPEED
            )

        await interaction.response.send_message(
            f"🗑️ تم حذف الشخصية **{deleted.name}**."
        )

    except Exception as error:

        await interaction.response.send_message(
            f"❌ {error}",
            ephemeral=True,
        )


bot.tree.add_command(
    character_group
)


# ============================================================
# MEMORY COMMANDS
# ============================================================

@bot.tree.command(
    name="memory",
    description=COMMAND_DESCRIPTIONS["memory"],
)
async def memory(
    interaction: discord.Interaction,
) -> None:

    if interaction.guild is None:

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
            "🔇 لا توجد جلسة صوتية."
        )
        return

    stats = session.engine.get_stats()

    await interaction.response.send_message(
        (
            f"🧠 الذاكرة: `{stats['memory_size']}` رسالة\n"
            f"📦 الحالة: "
            f"`{'Enabled' if stats['memory_enabled'] else 'Disabled'}`"
        )
    )


@bot.tree.command(
    name="clear",
    description=COMMAND_DESCRIPTIONS["clear"],
)
async def clear(
    interaction: discord.Interaction,
) -> None:

    if interaction.guild is None:

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
            "🔇 لا توجد جلسة صوتية."
        )
        return

    session.engine.clear_memory()

    await interaction.response.send_message(
        "🧹 تم مسح ذاكرة المحادثة."
    )


@bot.tree.command(
    name="reset",
    description=COMMAND_DESCRIPTIONS["reset"],
)
async def reset(
    interaction: discord.Interaction,
) -> None:

    if interaction.guild is None:

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
            "🔇 لا توجد جلسة صوتية."
        )
        return

    session.engine.reset()

    await bot.character_manager.clear_selected(
        interaction.guild.id
    )

    session.set_character(
        None
    )

    session.voice_name = DEFAULT_GEMINI_VOICE
    session.speech_speed = DEFAULT_SPEECH_SPEED

    await interaction.response.send_message(
        "♻️ تم إعادة ضبط جلسة الذكاء الاصطناعي."
    )


@bot.tree.command(
    name="stats",
    description=COMMAND_DESCRIPTIONS["stats"],
)
async def stats(
    interaction: discord.Interaction,
) -> None:

    if interaction.guild is None:

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
            "🔇 لا توجد جلسة صوتية."
        )
        return

    data = session.engine.get_stats()

    embed = discord.Embed(
        title="📊 Voice AI Stats",
        color=discord.Color.blurple(),
    )

    embed.add_field(
        name="📡 Channel",
        value=session.channel.name,
        inline=True,
    )

    embed.add_field(
        name="🔊 Voice",
        value=session.voice,
        inline=True,
    )

    embed.add_field(
        name="⚡ Speed",
        value=f"{session.speed:.2f}x",
        inline=True,
    )

    embed.add_field(
        name="🤖 Character",
        value=(
            session.character.name
            if session.character
            else "None"
        ),
        inline=True,
    )

    embed.add_field(
        name="🧠 Memory",
        value=str(data["memory_size"]),
        inline=True,
    )

    embed.add_field(
        name="✅ Processed",
        value=str(data["processed_requests"]),
        inline=True,
    )

    embed.add_field(
        name="❌ Failed",
        value=str(data["failed_requests"]),
        inline=True,
    )

    await interaction.response.send_message(
        embed=embed
    )


# ============================================================
# MODERATION HELPERS
# ============================================================

def is_server_owner(
    interaction: discord.Interaction,
) -> bool:

    if interaction.guild is None:
        return False

    return (
        interaction.user.id
        == interaction.guild.owner_id
    )


def bot_has_permission(
    guild: discord.Guild,
    permission: str,
) -> bool:

    me = guild.me

    if me is None:
        return False

    return bool(
        getattr(
            me.guild_permissions,
            permission,
            False,
        )
    )


async def require_moderation_owner(
    interaction: discord.Interaction,
) -> bool:

    if interaction.guild is None:

        await interaction.response.send_message(
            "❌ هذا الأمر يعمل داخل السيرفر فقط.",
            ephemeral=True,
        )
        return False

    if not MODERATION_AI_ENABLED:

        await interaction.response.send_message(
            "❌ نظام الإشراف معطل.",
            ephemeral=True,
        )
        return False

    if (
        MODERATION_OWNER_ONLY
        and not is_server_owner(interaction)
    ):

        await interaction.response.send_message(
            "🛡️ هذا النظام متاح **لصاحب السيرفر فقط**.",
            ephemeral=True,
        )
        return False

    return True


# ============================================================
# MODERATION GROUP
# ============================================================

mod_group = app_commands.Group(
    name="mod",
    description=COMMAND_DESCRIPTIONS["mod"],
)


@mod_group.command(
    name="status",
    description=COMMAND_DESCRIPTIONS["mod_status"],
)
async def mod_status(
    interaction: discord.Interaction,
) -> None:

    if interaction.guild is None:

        await interaction.response.send_message(
            "❌ هذا الأمر يعمل داخل السيرفر فقط.",
            ephemeral=True,
        )
        return

    owner = is_server_owner(
        interaction
    )

    embed = discord.Embed(
        title="🛡️ AI Moderation",
        color=discord.Color.blurple(),
    )

    embed.add_field(
        name="Status",
        value=(
            "🟢 Enabled"
            if MODERATION_AI_ENABLED
            else "🔴 Disabled"
        ),
        inline=True,
    )

    embed.add_field(
        name="Access",
        value=(
            "👑 Server Owner Only"
            if MODERATION_OWNER_ONLY
            else "⚠️ Owner restriction disabled"
        ),
        inline=True,
    )

    embed.add_field(
        name="Your Access",
        value=(
            "✅ Owner"
            if owner
            else "❌ Not Owner"
        ),
        inline=True,
    )

    await interaction.response.send_message(
        embed=embed,
        ephemeral=True,
    )


@mod_group.command(
    name="enable",
    description=COMMAND_DESCRIPTIONS["mod_enable"],
)
async def mod_enable(
    interaction: discord.Interaction,
) -> None:

    if not await require_moderation_owner(
        interaction
    ):
        return

    await interaction.response.send_message(
        (
            "🛡️ نظام الإشراف متاح بالفعل.\n"
            "يمكن تنفيذ إجراءات الإشراف من خلال أوامر "
            "الإدارة المسموحة فقط، وبصلاحيات Discord الفعلية."
        ),
        ephemeral=True,
    )


@mod_group.command(
    name="disable",
    description=COMMAND_DESCRIPTIONS["mod_disable"],
)
async def mod_disable(
    interaction: discord.Interaction,
) -> None:

    if not await require_moderation_owner(
        interaction
    ):
        return

    await interaction.response.send_message(
        (
            "ℹ️ نظام الإشراف مضبوط حاليًا على أنه "
            "متاح فقط لصاحب السيرفر.\n"
            "لا يوجد وضع يمنح الذكاء الاصطناعي صلاحيات "
            "إدارية مستقلة."
        ),
        ephemeral=True,
    )


@mod_group.command(
    name="request",
    description="طلب إجراء إشرافي من البوت.",
)
@app_commands.describe(
    request="اكتب الإجراء المطلوب بوضوح",
)
async def mod_request(
    interaction: discord.Interaction,
    request: str,
) -> None:

    if not await require_moderation_owner(
        interaction
    ):
        return

    request = request.strip()

    if not request:

        await interaction.response.send_message(
            "❌ اكتب الطلب أولًا.",
            ephemeral=True,
        )
        return

    supported = (
        "clear messages\n"
        "kick member\n"
        "ban member\n"
        "unban member\n"
        "timeout member\n"
        "remove timeout\n"
        "create role\n"
        "delete role\n"
        "create channel\n"
        "delete channel\n"
        "rename channel\n"
        "add role\n"
        "remove role"
    )

    await interaction.response.send_message(
        (
            "🛡️ **طلب الإشراف استُلم.**\n\n"
            f"📝 الطلب: `{request[:500]}`\n\n"
            "⚠️ تنفيذ الطلبات الإدارية الحرة من النص "
            "غير مفعل في هذه النسخة حتى لا يستطيع نموذج "
            "الذكاء الاصطناعي تنفيذ إجراء غير مقصود.\n\n"
            "**الإجراءات المدعومة:**\n"
            f"{supported}\n\n"
            "استخدم أوامر Discord المحددة للإجراء المطلوب."
        ),
        ephemeral=True,
    )


bot.tree.add_command(
    mod_group
)


# ============================================================
# DISCORD MODERATION COMMANDS
# ============================================================

@bot.tree.command(
    name="clear_messages",
    description="حذف عدد من الرسائل — صاحب السيرفر فقط.",
)
@app_commands.describe(
    amount="عدد الرسائل",
)
async def clear_messages(
    interaction: discord.Interaction,
    amount: app_commands.Range[
        int,
        1,
        100,
    ],
) -> None:

    if not await require_moderation_owner(
        interaction
    ):
        return

    channel = interaction.channel

    if not isinstance(
        channel,
        discord.TextChannel,
    ):

        await interaction.response.send_message(
            "❌ هذا الأمر يعمل في القنوات النصية.",
            ephemeral=True,
        )
        return

    if not bot_has_permission(
        interaction.guild,
        "manage_messages",
    ):

        await interaction.response.send_message(
            "❌ البوت لا يملك Manage Messages.",
            ephemeral=True,
        )
        return

    await interaction.response.defer(
        ephemeral=True
    )

    try:

        deleted = await channel.purge(
            limit=int(amount),
        )

        await interaction.followup.send(
            f"🧹 تم حذف **{len(deleted)}** رسالة.",
            ephemeral=True,
        )

    except Exception as error:

        logger.exception(
            "Message deletion failed"
        )

        await interaction.followup.send(
            f"❌ فشل الحذف: `{error}`",
            ephemeral=True,
        )


@bot.tree.command(
    name="kick",
    description="طرد عضو — صاحب السيرفر فقط.",
)
@app_commands.describe(
    member="العضو",
    reason="السبب",
)
async def kick(
    interaction: discord.Interaction,
    member: discord.Member,
    reason: str | None = None,
) -> None:

    if not await require_moderation_owner(
        interaction
    ):
        return

    if not bot_has_permission(
        interaction.guild,
        "kick_members",
    ):

        await interaction.response.send_message(
            "❌ البوت لا يملك Kick Members.",
            ephemeral=True,
        )
        return

    me = interaction.guild.me

    if (
        me
        and member.top_role >= me.top_role
    ):

        await interaction.response.send_message(
            "❌ لا أستطيع طرد عضو رتبته أعلى من رتبتي أو مساوية لها.",
            ephemeral=True,
        )
        return

    reason = (
        reason[:MODERATION_MAX_REASON_LENGTH]
        if reason
        else "Cloud Voice AI moderation"
    )

    try:

        await member.kick(
            reason=reason
        )

        await interaction.response.send_message(
            f"👢 تم طرد **{member}**."
        )

    except Exception as error:

        await interaction.response.send_message(
            f"❌ فشل الطرد: `{error}`",
            ephemeral=True,
        )


@bot.tree.command(
    name="ban",
    description="حظر عضو — صاحب السيرفر فقط.",
)
@app_commands.describe(
    member="العضو",
    reason="السبب",
)
async def ban(
    interaction: discord.Interaction,
    member: discord.Member,
    reason: str | None = None,
) -> None:

    if not await require_moderation_owner(
        interaction
    ):
        return

    if not bot_has_permission(
        interaction.guild,
        "ban_members",
    ):

        await interaction.response.send_message(
            "❌ البوت لا يملك Ban Members.",
            ephemeral=True,
        )
        return

    me = interaction.guild.me

    if (
        me
        and member.top_role >= me.top_role
    ):

        await interaction.response.send_message(
            "❌ لا أستطيع حظر عضو رتبته أعلى من رتبتي أو مساوية لها.",
            ephemeral=True,
        )
        return

    reason = (
        reason[:MODERATION_MAX_REASON_LENGTH]
        if reason
        else "Cloud Voice AI moderation"
    )

    try:

        await member.ban(
            reason=reason,
            delete_message_seconds=0,
        )

        await interaction.response.send_message(
            f"🔨 تم حظر **{member}**."
        )

    except Exception as error:

        await interaction.response.send_message(
            f"❌ فشل الحظر: `{error}`",
            ephemeral=True,
        )


# ============================================================
# VOICE STATE EVENTS
# ============================================================

@bot.event
async def on_voice_state_update(
    member: discord.Member,
    before: discord.VoiceState,
    after: discord.VoiceState,
) -> None:

    if not AUTO_LEAVE_EMPTY_CHANNEL:
        return

    session = bot.voice_manager.get(
        member.guild.id
    )

    if session is None:
        return

    channel = session.channel

    if channel is None:
        return

    human_members = [
        m
        for m in channel.members
        if not m.bot
    ]

    if human_members:
        return

    async def delayed_leave() -> None:

        try:

            await asyncio.sleep(
                AUTO_LEAVE_DELAY_SECONDS
            )

            current = bot.voice_manager.get(
                member.guild.id
            )

            if current is None:
                return

            humans = [
                m
                for m in current.channel.members
                if not m.bot
            ]

            if not humans:

                await bot.voice_manager.leave(
                    member.guild.id
                )

                logger.info(
                    "Left empty voice channel | guild=%s",
                    member.guild.id,
                )

        except asyncio.CancelledError:

            return

        except Exception:

            logger.exception(
                "Auto leave failed"
            )

    asyncio.create_task(
        delayed_leave()
    )


# ============================================================
# APPLICATION COMMAND ERROR
# ============================================================

@bot.tree.error
async def on_app_command_error(
    interaction: discord.Interaction,
    error: app_commands.AppCommandError,
) -> None:

    logger.error(
        "Application command error: %r",
        error,
        exc_info=(
            type(error),
            error,
            error.__traceback__,
        ),
    )

    message = (
        "❌ حدث خطأ أثناء تنفيذ الأمر."
    )

    if isinstance(
        error,
        app_commands.CommandOnCooldown,
    ):

        message = (
            "⏳ الأمر عليه انتظار، حاول بعد قليل."
        )

    try:

        if interaction.response.is_done():

            await interaction.followup.send(
                message,
                ephemeral=True,
            )

        else:

            await interaction.response.send_message(
                message,
                ephemeral=True,
            )

    except Exception:
        pass


# ============================================================
# GLOBAL ERROR
# ============================================================

@bot.event
async def on_error(
    event: str,
    *args: Any,
    **kwargs: Any,
) -> None:

    logger.exception(
        "Discord event error: %s",
        event,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    if not DISCORD_TOKEN:

        raise RuntimeError(
            "DISCORD_TOKEN is missing."
        )

    logger.info(
        "Starting Cloud Voice AI..."
    )

    bot.run(
        DISCORD_TOKEN
    )


if __name__ == "__main__":
    main()
