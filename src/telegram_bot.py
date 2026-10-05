"""
Telegram Bot for trade alerts and daily summaries.
Uses python-telegram-bot for async communication.
"""
import asyncio
import logging

from config.settings import settings
from src.logger import _console

_pending_tasks: set[asyncio.Task] = set()


class TelegramNotifier:
    def __init__(self):
        self.bot = None
        self.chat_id = settings.TELEGRAM_CHAT_ID
        self.enabled = bool(settings.TELEGRAM_BOT_TOKEN and settings.TELEGRAM_CHAT_ID)
        if self.enabled:
            try:
                from telegram import Bot
                self.bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)
            except ImportError:
                _console("python-telegram-bot not installed; Telegram notifications disabled", level=logging.WARNING)
                self.enabled = False

    def send_message(self, text: str):
        """Send text message to configured chat (or background task with strong reference)."""
        if not self.enabled:
            _console(f"[TELEGRAM] {text}")
            return

        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(self._send(text))
            _pending_tasks.add(task)
            task.add_done_callback(_pending_tasks.discard)
        except RuntimeError:
            _console(f"[TELEGRAM] {text}")

    async def _send(self, text: str):
        if not self.enabled or self.bot is None:
            _console(f"[TELEGRAM] {text}")
            return
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode="HTML",
            )
        except Exception as e:
            _console(f"Telegram error: {e}", level=logging.ERROR)

    async def send_daily_summary_async(self, summary_text: str):
        """Async wrapper for daily summary."""
        await self.send_message_async(f"<b>📊 Daily Summary</b>\n\n{summary_text}")

    async def send_message_async(self, text: str):
        """Send a message and wait for delivery when Telegram is enabled."""
        if not self.enabled:
            _console(f"[TELEGRAM] {text}")
            return
        await self._send(text)


telegram = TelegramNotifier()