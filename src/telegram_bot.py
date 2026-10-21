"""
Telegram Bot for trade alerts and daily summaries.
Uses python-telegram-bot for async communication.
"""
import asyncio
import logging

from config.settings import settings
from src.logger import _console

_pending_tasks: set[asyncio.Task] = set()
logger = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(self):
        self.bot = None
        self.chat_id = settings.TELEGRAM_CHAT_ID
        self.enabled = bool(settings.TELEGRAM_BOT_TOKEN and settings.TELEGRAM_CHAT_ID)
        self._initialized = False
        self._initialize_lock = asyncio.Lock()
        if self.enabled:
            try:
                from telegram import Bot
                self.bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)
            except ImportError:
                _console(
                    "python-telegram-bot not installed; Telegram notifications disabled",
                    level=logging.WARNING,
                )
                self.enabled = False
        else:
            _console(
                "Telegram notifications disabled: TELEGRAM_BOT_TOKEN or "
                "TELEGRAM_CHAT_ID is missing",
                level=logging.WARNING,
            )

    async def initialize(self) -> bool:
        """Initialize the Telegram client before sending messages."""
        if not self.enabled or self.bot is None:
            return False
        async with self._initialize_lock:
            if self._initialized:
                return True
            try:
                await self.bot.initialize()
                self._initialized = True
                _console("Telegram bot initialized", level=logging.INFO)
            except Exception as exc:
                logger.exception("Telegram initialization failed")
                _console(f"Telegram initialization failed: {exc!r}", level=logging.ERROR)
            return self._initialized

    async def shutdown(self) -> None:
        """Release resources opened by python-telegram-bot."""
        if self.bot is not None and self._initialized:
            try:
                await self.bot.shutdown()
            except Exception as exc:
                logger.exception("Telegram shutdown failed")
                _console(f"Telegram shutdown failed: {exc!r}", level=logging.ERROR)
            finally:
                self._initialized = False

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

    async def _send(self, text: str) -> bool:
        if not self.enabled or self.bot is None:
            logger.error("Telegram message was not sent because notifications are disabled")
            return False
        for attempt in range(1, 4):
            if not self._initialized and not await self.initialize():
                if attempt < 3:
                    await asyncio.sleep(attempt)
                    continue
                return False
            try:
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=text,
                )
                return True
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Telegram send failed (attempt %s/3)", attempt)
                _console(
                    f"Telegram send failed (attempt {attempt}/3): {exc!r}",
                    level=logging.ERROR,
                )
                if attempt < 3:
                    await asyncio.sleep(attempt)
        return False

    async def send_daily_summary_async(self, summary_text: str) -> bool:
        """Async wrapper for daily summary."""
        return await self.send_message_async(
            f"📊 Daily Summary\n\n{summary_text}"
        )

    async def send_message_async(self, text: str) -> bool:
        """Send a message and report whether Telegram accepted it."""
        return await self._send(text)


telegram = TelegramNotifier()