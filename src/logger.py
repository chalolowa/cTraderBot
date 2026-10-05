"""Centralized logging for tradebot with rotating file and console handlers."""
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = ROOT_DIR / "logs"
LOG_FILE = LOGS_DIR / "tradebot.log"


def setup_logger() -> logging.Logger:
    log = logging.getLogger("tradebot")
    if not log.handlers:
        log.setLevel(logging.DEBUG)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

        # File formatter with timestamp and log level
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] [tradebot] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        try:
            file_handler = RotatingFileHandler(
                str(LOG_FILE),
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(file_formatter)
            log.addHandler(file_handler)
        except Exception:
            # Fallback if log file cannot be opened (e.g. read-only env in tests)
            pass

        # Console handler prints [tradebot] %(message)s
        console_formatter = logging.Formatter("[tradebot] %(message)s")
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        log.addHandler(console_handler)

    return log


logger = setup_logger()


def _console(message: str, level: int = logging.INFO) -> None:
    """Log a message at the given level to both console and rotating log file."""
    logger.log(level, message)
