"""Telegram bot for generating short abstract videos.

Usage:
    export TELEGRAM_BOT_TOKEN=<your token>
    python -m src.bot

The bot lets a user pick a video duration up to one minute and
then responds with a freshly rendered MP4 clip generated entirely
with NumPy and MoviePy.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Final, Iterable

import numpy as np
from moviepy.editor import VideoClip
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import (Application, CallbackQueryHandler, CommandHandler,
                          ContextTypes)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DurationOption:
    label: str
    seconds: int


MAX_DURATION_SECONDS: Final[int] = 60

DURATION_OPTIONS: Final[tuple[DurationOption, ...]] = (
    DurationOption(label="15 секунд", seconds=15),
    DurationOption(label="30 секунд", seconds=30),
    DurationOption(label="45 секунд", seconds=45),
    DurationOption(label="60 секунд", seconds=60),
)


def _build_duration_keyboard(options: Iterable[DurationOption]) -> InlineKeyboardMarkup:
    buttons = [[InlineKeyboardButton(option.label, callback_data=str(option.seconds))]
               for option in options]
    return InlineKeyboardMarkup(buttons)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send greeting and duration selector."""
    message = update.effective_message
    if message is None:
        LOGGER.warning("Received /start update without a message object")
        return

    keyboard = _build_duration_keyboard(DURATION_OPTIONS)
    welcome_message = (
        "Привет! Я бот, который генерирует абстрактные видео до 1 минуты.\n"
        "Выбери длительность ролика ниже, и я создам его специально для тебя."
    )
    await message.reply_text(welcome_message, reply_markup=keyboard)


async def handle_duration_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Generate and send a video clip for the chosen duration."""
    query = update.callback_query
    if query is None:
        LOGGER.warning("Duration callback triggered without a query object")
        return

    await query.answer()

    try:
        duration_seconds = int(query.data or "0")
    except ValueError:
        await query.edit_message_text("Не удалось распознать выбранную длительность.")
        return

    if duration_seconds <= 0 or duration_seconds > MAX_DURATION_SECONDS:
        await query.edit_message_text(
            "Можно выбрать длительность только от 1 до 60 секунд. Попробуй снова командой /start."
        )
        return

    await query.edit_message_text(
        f"Генерирую видео длительностью {duration_seconds} секунд. Это может занять до минуты..."
    )

    if query.message is not None:
        await context.bot.send_chat_action(chat_id=query.message.chat_id, action=ChatAction.UPLOAD_VIDEO)

    video_path = await asyncio.to_thread(generate_abstract_video, duration_seconds)

    try:
        with open(video_path, "rb") as video_file:
            await query.message.reply_video(
                video=video_file,
                caption=f"Готово! Вот твоё видео на {duration_seconds} секунд.",
            )
    finally:
        try:
            os.remove(video_path)
        except OSError:
            LOGGER.exception("Failed to remove temporary video file %s", video_path)


def generate_abstract_video(
    duration_seconds: int,
    *,
    size: tuple[int, int] = (640, 640),
    fps: int = 24,
) -> str:
    """Render an abstract colorful animation and return the path to the created MP4 file."""
    if duration_seconds <= 0:
        raise ValueError("Duration must be a positive integer")
    if duration_seconds > MAX_DURATION_SECONDS:
        raise ValueError(f"Duration must not exceed {MAX_DURATION_SECONDS} seconds")

    width, height = size

    def make_frame(t: float) -> np.ndarray:
        x = np.linspace(0, np.pi * 2, width)
        y = np.linspace(0, np.pi * 2, height)
        xv, yv = np.meshgrid(x, y)

        # Create a smooth, looping color pattern based on sine waves.
        phase_shift = t * 2
        r = (np.sin(xv * 2 + phase_shift) + 1) / 2
        g = (np.sin(yv * 2 + phase_shift * 0.7) + 1) / 2
        b = (np.sin((xv + yv) * 1.3 + phase_shift * 1.3) + 1) / 2

        frame = np.stack((r, g, b), axis=-1)
        return (frame * 255).astype(np.uint8)

    clip = VideoClip(make_frame, duration=duration_seconds)
    clip = clip.set_fps(fps)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.close()

    clip.write_videofile(temp_file.name, codec="libx264", audio=False, fps=fps, verbose=False, logger=None)

    return temp_file.name


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("Environment variable TELEGRAM_BOT_TOKEN is not set")

    application = Application.builder().token(token).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CallbackQueryHandler(handle_duration_choice))

    LOGGER.info("Bot is starting...")
    application.run_polling()


if __name__ == "__main__":
    main()
