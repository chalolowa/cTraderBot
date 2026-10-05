"""
Database utilities for trade persistence.
Provides robust SQLite connection management with WAL mode and busy timeouts.
"""

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from config.settings import settings


def configure_sqlite_connection(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Configure WAL journal mode and busy timeout on a SQLite connection."""
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
    return conn


def create_connection(db_path: str | None = None, timeout: float = 30.0) -> sqlite3.Connection:
    """Create and configure a new SQLite connection."""
    path_str = db_path or settings.DB_PATH
    path = Path(path_str)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=timeout)
    configure_sqlite_connection(conn)
    return conn


@contextmanager
def get_db_connection(db_path: str | None = None) -> Generator[sqlite3.Connection]:
    """Context manager for database connections ensuring WAL mode, busy timeout, and clean close."""
    conn = create_connection(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_database(db_path: str | None = None) -> None:
    """Ensure all tables exist. Called on startup."""
    from src.risk_manager import risk_manager
    # RiskManager init creates tables with WAL mode
    _ = risk_manager