"""Store Pi events and retrieve recent event history."""

from datetime import UTC, datetime, timedelta
from uuid import UUID

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from alert_delivery import delivery_info
from db.connection import get_connection
from db.devices import DeviceError
from db.notifications import enqueue_alert
from device_status import MAX_REPORT_AGE_SECONDS, monitoring_report
from event_cursor import encode_cursor


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
        consent_state, consent_updated_at = conn.execute(
            "SELECT consent_state, consent_updated_at FROM devices WHERE device_id = %s FOR UPDATE",
            (event["device_id"],),
        ).fetchone()
        # Already saved events remain safe to retry after withdrawal. No new data
        # or notifications are created, and the device is not marked seen again.
        if conn.execute("SELECT 1 FROM events WHERE event_id = %s", (event["event_id"],)).fetchone():
            return False
        if event["event_type"] in {"fall", "anomaly", "inactivity"}:
            if consent_state != "granted":
                raise DeviceError("consent is required; stop collection and discard this alert", 403)
            age = event["elapsed_seconds"]
            if (age > MAX_REPORT_AGE_SECONDS
                    or datetime.now(UTC) - timedelta(seconds=age) < consent_updated_at):
                raise DeviceError("alert predates the current consent period; discard it", 403)
        elif consent_state != "granted":
            # Retain minimal status traffic for reconnect/stop confirmation, not
            # arbitrary monitoring details or scores while permission is absent.
            event = event.copy()
            event.pop("confidence", None)
            state = event.get("details", {}).get("monitoring_state")
            event["details"] = {"monitoring_state": state} if state in ("active", "muted", "unknown") else {}
        row = conn.execute(
            "INSERT INTO events "
            "(event_id, device_id, event_type, room, detected_at, elapsed_seconds, confidence, details) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (event_id) DO NOTHING RETURNING received_at",
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
        if row is not None:
            received_at = row[0]
            enqueue_alert(conn, event)
            conn.execute(
                "UPDATE devices SET last_seen_at = GREATEST(last_seen_at, %s) WHERE device_id = %s",
                (received_at, event["device_id"]),
            )
            report = monitoring_report(event, received_at)
            if report is not None:
                state, reported_at = report
                if consent_state != "granted" and state == "active":
                    # A Pi still reporting active has not maintained the stop.
                    conn.execute(
                        "UPDATE devices SET consent_confirmed_version = NULL WHERE device_id = %s "
                        "AND %s >= consent_updated_at "
                        "AND (monitoring_updated_at IS NULL OR %s >= monitoring_updated_at)",
                        (event["device_id"], reported_at, reported_at),
                    )
                    state = "unknown"
                conn.execute(
                    "UPDATE devices SET monitoring_state = %s, monitoring_updated_at = %s "
                    "WHERE device_id = %s AND (monitoring_updated_at IS NULL OR monitoring_updated_at < %s "
                    "OR (monitoring_updated_at = %s AND "
                    "CASE %s WHEN 'muted' THEN 2 WHEN 'unknown' THEN 1 ELSE 0 END > "
                    "CASE monitoring_state WHEN 'muted' THEN 2 WHEN 'unknown' THEN 1 ELSE 0 END))",
                    (state, reported_at, event["device_id"], reported_at, reported_at, state),
                )
    return row is not None


def get_events(account_id: str, limit: int, since: datetime | None = None,
               device_id: str | None = None, alerts_only: bool = False,
               cursor: tuple[datetime, datetime, UUID] | None = None) -> dict:
    """Read an authorized page; stable ordering prevents skips when newer events arrive."""
    conditions = ["EXISTS (SELECT 1 FROM device_memberships m JOIN devices d ON d.device_id = m.device_id "
                  "WHERE m.device_id = events.device_id AND m.account_id = %s "
                  "AND d.consent_first_granted_at IS NOT NULL)"]
    parameters: list = [account_id]
    if since is not None:
        conditions.append("detected_at > %s")
        parameters.append(since)
    if device_id is not None:
        conditions.append("device_id = %s")
        parameters.append(device_id)
    if alerts_only:
        # Filter before limiting: routine status messages must not hide older alerts.
        conditions.append("event_type IN ('fall', 'anomaly', 'inactivity')")
    if cursor is not None:
        conditions.append("(detected_at, received_at, event_id) < (%s, %s, %s)")
        parameters.extend(cursor)
    where = " WHERE " + " AND ".join(conditions) if conditions else ""
    parameters.append(limit + 1)
    with get_connection() as conn, conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            "SELECT event_id, device_id, event_type, room, detected_at, elapsed_seconds, "
            "confidence, details, received_at FROM events"
            + where
            + " ORDER BY detected_at DESC, received_at DESC, event_id DESC LIMIT %s",
            parameters,
        )
        rows = cursor.fetchall()
    more = len(rows) > limit
    events = rows[:limit]
    for event in events:
        event.update(delivery_info(event))
        event["event_id"] = str(event["event_id"])
        for field in ("detected_at", "received_at"):
            event[field] = event[field].astimezone(UTC).isoformat().replace("+00:00", "Z")
        for field in ("confidence", "details"):
            if event[field] is None:
                del event[field]
    return {"events": events, "next_cursor": encode_cursor(events[-1]) if more else None}
