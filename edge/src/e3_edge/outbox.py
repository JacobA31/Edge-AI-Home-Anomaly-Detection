"""Offline event queue (SQLite). Events wait here until the backend confirms them.

Covers the proposal's outage-handling requirement: detection keeps running without
network, and queued events are retried when the connection comes back.
"""

import json
import sqlite3
from datetime import UTC, datetime


class Outbox:
    def __init__(self, path: str):
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS outbox (event_id TEXT PRIMARY KEY, body TEXT NOT NULL)"
        )
        self.db.commit()

    def put(self, event: dict) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO outbox (event_id, body) VALUES (?, ?)",
            (event["event_id"], json.dumps(event)),
        )
        self.db.commit()

    def pending(self, limit: int = 50) -> list[dict]:
        rows = self.db.execute("SELECT body FROM outbox ORDER BY rowid LIMIT ?", (limit,)).fetchall()
        return [json.loads(r[0]) for r in rows]

    def mark_sent(self, event_id: str) -> None:
        self.db.execute("DELETE FROM outbox WHERE event_id = ?", (event_id,))
        self.db.commit()

    def __len__(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]


def with_elapsed(event: dict, now: datetime | None = None) -> dict:
    """Fill elapsed_seconds at send time, as the proposal's event record requires."""
    now = now or datetime.now(UTC)
    detected = datetime.fromisoformat(event["detected_at"].replace("Z", "+00:00"))
    return event | {"elapsed_seconds": round(max(0.0, (now - detected).total_seconds()), 1)}
