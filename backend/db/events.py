"""Store Pi events and retrieve recent event history."""

from datetime import UTC, datetime

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from db.connection import get_connection


def save_event(event: dict) -> bool:
    """Return True for a new event, or False when its ID was already saved."""
    with get_connection() as conn:
        # A device must exist before its events can reference it. Both inserts
        # share a transaction, so a failed event does not leave a partial save.
        conn.execute(
            "INSERT INTO devices (device_id, room) VALUES (%s, %s) "
            "ON CONFLICT (device_id) DO NOTHING",
            (event["device_id"], event["room"]),
        )
        row = conn.execute(
            "INSERT INTO events "
            "(event_id, device_id, event_type, room, detected_at, elapsed_seconds, confidence, details) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (event_id) DO NOTHING RETURNING event_id",
            (
                event["event_id"],
                event["device_id"],
                event["event_type"],
                event["room"],
                event["detected_at"],
                event["elapsed_seconds"],
                event.get("confidence"),
                Jsonb(event["details"]) if "details" in event else None,
            ),
        ).fetchone()
    return row is not None


def get_events(limit: int, since: datetime | None = None, device_id: str | None = None) -> list[dict]:
    """Read events newest detection first; since is an exclusive detection-time cutoff."""
    conditions = []
    parameters: list = []
    if since is not None:
        conditions.append("detected_at > %s")
        parameters.append(since)
    if device_id is not None:
        conditions.append("device_id = %s")
        parameters.append(device_id)
    where = " WHERE " + " AND ".join(conditions) if conditions else ""
    parameters.append(limit)
    with get_connection() as conn, conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            "SELECT event_id, device_id, event_type, room, detected_at, elapsed_seconds, "
            "confidence, details, received_at FROM events"
            + where
            + " ORDER BY detected_at DESC, received_at DESC, event_id DESC LIMIT %s",
            parameters,
        )
        events = cursor.fetchall()
    for event in events:
        event["event_id"] = str(event["event_id"])
        for field in ("detected_at", "received_at"):
            event[field] = event[field].astimezone(UTC).isoformat().replace("+00:00", "Z")
        for field in ("confidence", "details"):
            if event[field] is None:
                del event[field]
    return events
