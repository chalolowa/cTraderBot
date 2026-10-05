"""Durable local event journal and best-effort MongoDB synchronizer."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import threading
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from config.settings import settings
from src.database import create_connection
from src.logger import _console

MAX_SYNC_ATTEMPTS = 10


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class EventOutbox:
    """SQLite-backed outbox. Local append is independent of MongoDB health."""

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or settings.DB_PATH
        parent = os.path.dirname(self.db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection]:
        conn = create_connection(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._lock, self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS event_outbox (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    aggregate_id TEXT,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    synced_at TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_event_outbox_pending "
                "ON event_outbox (synced_at, created_at)"
            )

    def append(
        self,
        event_type: str,
        payload: dict[str, Any],
        aggregate_id: int | str | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> str:
        """Append an event to the outbox. If connection is passed, uses it (for transactions)."""
        event_id = str(uuid.uuid4())
        sql = """
            INSERT INTO event_outbox
                (event_id, event_type, aggregate_id, payload, created_at)
            VALUES (?, ?, ?, ?, ?)
        """
        params = (
            event_id,
            event_type,
            str(aggregate_id) if aggregate_id is not None else None,
            json.dumps(payload, default=str, sort_keys=True),
            _utc_now(),
        )
        if connection is not None:
            connection.execute(sql, params)
            _console(
                f"SQLite outbox write succeeded ({event_type}, event={event_id})",
                level=logging.DEBUG,
            )
            return event_id

        try:
            with self._lock, self._connection() as conn:
                conn.execute(sql, params)
        except sqlite3.Error as exc:
            _console(
                f"SQLite outbox write failed ({event_type}): {exc}", level=logging.ERROR
            )
            raise
        _console(
            f"SQLite outbox write succeeded ({event_type}, event={event_id})",
            level=logging.DEBUG,
        )
        return event_id

    def pending(
        self, limit: int, max_attempts: int = MAX_SYNC_ATTEMPTS
    ) -> list[sqlite3.Row]:
        with self._lock, self._connection() as connection:
            return connection.execute(
                """
                SELECT * FROM event_outbox
                WHERE synced_at IS NULL AND attempts < ?
                ORDER BY created_at
                LIMIT ?
                """,
                (max_attempts, limit),
            ).fetchall()

    def mark_synced(self, event_id: str) -> None:
        try:
            with self._lock, self._connection() as connection:
                connection.execute(
                    "UPDATE event_outbox SET synced_at = ?, last_error = NULL "
                    "WHERE event_id = ?",
                    (_utc_now(), event_id),
                )
        except sqlite3.Error as exc:
            _console(
                f"SQLite sync-state update failed (event={event_id}): {exc}",
                level=logging.ERROR,
            )
            raise
        _console(
            f"SQLite sync-state update succeeded (event={event_id})",
            level=logging.DEBUG,
        )

    def mark_failed(self, event_id: str, error: str) -> None:
        try:
            with self._lock, self._connection() as connection:
                connection.execute(
                    "UPDATE event_outbox SET attempts = attempts + 1, last_error = ? "
                    "WHERE event_id = ?",
                    (error[:1000], event_id),
                )
        except sqlite3.Error as exc:
            _console(
                f"SQLite failure-state update failed (event={event_id}): {exc}",
                level=logging.ERROR,
            )
            raise
        _console(
            f"SQLite failure-state update recorded (event={event_id})",
            level=logging.DEBUG,
        )

    def pending_count(self, max_attempts: int = MAX_SYNC_ATTEMPTS) -> int:
        with self._lock, self._connection() as connection:
            return connection.execute(
                "SELECT COUNT(*) FROM event_outbox WHERE synced_at IS NULL AND attempts < ?",
                (max_attempts,),
            ).fetchone()[0]


class MongoEventSynchronizer:
    """Uploads acknowledged outbox events without ever blocking trading."""

    def __init__(self, outbox: EventOutbox) -> None:
        self.outbox = outbox
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._client: Any = None

    def _get_client(self) -> Any:
        if not settings.MONGO_URI:
            return None
        if self._client is None:
            try:
                from pymongo import MongoClient

                self._client = MongoClient(
                    settings.MONGO_URI,
                    serverSelectionTimeoutMS=5000,
                    connectTimeoutMS=5000,
                    socketTimeoutMS=5000,
                )
            except Exception as exc:
                _console(
                    f"MongoDB client initialization failed: {exc}", level=logging.ERROR
                )
                raise
        return self._client

    def close(self) -> None:
        """Close long-lived MongoDB client."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    async def start(self) -> None:
        if not settings.MONGO_URI:
            _console(
                "MongoDB synchronization disabled: MONGO_URI is not set",
                level=logging.INFO,
            )
            return
        if self._task and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="mongo-event-sync")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            await self._task
            self._task = None
        self.close()

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.to_thread(self.sync_once)
            except Exception as exc:
                _console(f"MongoDB synchronization failed: {exc}", level=logging.ERROR)
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=settings.OUTBOX_SYNC_INTERVAL_SECONDS
                )
            except TimeoutError:
                pass

    def sync_once(self) -> int:
        """Synchronize one batch; failures remain pending for a later retry."""
        if not settings.MONGO_URI:
            return 0
        try:
            from pymongo.errors import (
                ConnectionFailure,
                NetworkTimeout,
                PyMongoError,
                ServerSelectionTimeoutError,
            )
        except ImportError:
            _console(
                "MongoDB synchronization unavailable: pymongo not installed",
                level=logging.WARNING,
            )
            return 0

        try:
            client = self._get_client()
            if client is None:
                return 0
            collection = client[settings.MONGO_DATABASE][settings.MONGO_COLLECTION]
        except Exception as exc:
            _console(f"MongoDB connection failed: {exc}", level=logging.ERROR)
            self.close()
            return 0

        uploaded = 0
        pending_events = self.outbox.pending(settings.OUTBOX_BATCH_SIZE)
        for row in pending_events:
            document = {
                "event_id": row["event_id"],
                "event_type": row["event_type"],
                "aggregate_id": row["aggregate_id"],
                "payload": json.loads(row["payload"]),
                "created_at": row["created_at"],
                "synced_at": _utc_now(),
            }
            try:
                result = collection.replace_one(
                    {"_id": row["event_id"]},
                    {"_id": row["event_id"], **document},
                    upsert=True,
                )
                if getattr(result, "acknowledged", False):
                    self.outbox.mark_synced(row["event_id"])
                    uploaded += 1
                    _console(
                        f"MongoDB synchronization succeeded (event={row['event_id']})",
                        level=logging.DEBUG,
                    )
                else:
                    _console(
                        f"MongoDB synchronization failed (event={row['event_id']}): write was not acknowledged",
                        level=logging.ERROR,
                    )
                    self.outbox.mark_failed(row["event_id"], "write not acknowledged")
            except (
                ServerSelectionTimeoutError,
                ConnectionFailure,
                NetworkTimeout,
                PyMongoError,
            ) as exc:
                _console(
                    f"MongoDB connection error during sync (aborting batch): {exc}",
                    level=logging.ERROR,
                )
                self.outbox.mark_failed(row["event_id"], str(exc))
                self.close()
                break
            except Exception as exc:
                _console(
                    f"MongoDB synchronization item failed (event={row['event_id']}): {exc}",
                    level=logging.ERROR,
                )
                self.outbox.mark_failed(row["event_id"], str(exc))

        return uploaded


event_outbox = EventOutbox()
mongo_synchronizer = MongoEventSynchronizer(event_outbox)
