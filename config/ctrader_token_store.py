"""SQLite-backed TokenStore for ctrader-api-client.

cTrader rotates both the access token and refresh token on every refresh and
invalidates the old pair immediately. Without persistence, losing the new
pair (e.g. a crash right after a rotation) would strand the account until
someone re-runs scripts/ctrader_oauth_setup.py. This store writes every
rotated pair to the same SQLite database used for trades/events, so the
bot can resume after a restart without any manual step.
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import closing

from ctrader_api_client import AccountCredentials, TokenStore


class SQLiteTokenStore(TokenStore):
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        parent = os.path.dirname(self.db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._init_db()

    def _connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _init_db(self) -> None:
        with closing(self._connection()) as conn, conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ctrader_tokens (
                    account_id INTEGER PRIMARY KEY,
                    access_token TEXT NOT NULL,
                    refresh_token TEXT NOT NULL,
                    expires_at INTEGER NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    async def save(self, credentials: AccountCredentials) -> None:
        """Called by the client every time a token pair is rotated (and once
        manually by scripts/ctrader_oauth_setup.py to seed the first pair)."""
        with closing(self._connection()) as conn, conn:
            conn.execute(
                """
                INSERT INTO ctrader_tokens (account_id, access_token, refresh_token, expires_at, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(account_id) DO UPDATE SET
                    access_token = excluded.access_token,
                    refresh_token = excluded.refresh_token,
                    expires_at = excluded.expires_at,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (credentials.account_id, credentials.access_token,
                 credentials.refresh_token, credentials.expires_at),
            )

    async def load(self, account_id: int) -> AccountCredentials | None:
        """Not part of the TokenStore protocol -- called by our own
        ctrader_client.connect() to read back whatever was last persisted."""
        with closing(self._connection()) as conn:
            row = conn.execute(
                "SELECT access_token, refresh_token, expires_at FROM ctrader_tokens WHERE account_id = ?",
                (account_id,),
            ).fetchone()
        if row is None:
            return None
        access_token, refresh_token, expires_at = row
        return AccountCredentials(
            account_id=account_id,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
        )
