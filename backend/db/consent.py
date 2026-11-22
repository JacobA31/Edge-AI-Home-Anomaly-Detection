"""Account-independent resident consent, auditable decisions and Pi policy sync."""

import secrets
from datetime import UTC, datetime, timedelta

from psycopg.rows import dict_row

from consent_disclosure import DISCLOSURE, DISCLOSURE_VERSION
from db.connection import get_connection
from db.devices import DeviceError
from security import token_hash

POLICY_FIELDS = (
    "consent_state", "consent_version", "consent_updated_at", "consent_text_version",
    "consent_first_granted_at", "consent_delivered_version", "consent_confirmed_version",
)


def policy(row: dict) -> dict:
    return {
        "state": row["consent_state"], "version": row["consent_version"],
        "updated_at": row["consent_updated_at"].astimezone(UTC).isoformat(),
        "disclosure_version": row["consent_text_version"],
        "first_granted_at": (row["consent_first_granted_at"].astimezone(UTC).isoformat()
                             if row["consent_first_granted_at"] else None),
        "monitoring_allowed": row["consent_state"] == "granted",
        "device_confirmed": row["consent_confirmed_version"] == row["consent_version"],
    }


def device_policy(conn, device_id: str) -> dict:
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute("SELECT * FROM devices WHERE device_id = %s", (device_id,))
        return policy(cursor.fetchone())


def require_consent(conn, device_id: str) -> None:
    row = conn.execute("SELECT consent_state FROM devices WHERE device_id = %s", (device_id,)).fetchone()
    if row is None or row[0] != "granted":
        raise DeviceError("monitoring consent has not been granted or has been withdrawn", 403)


def provision_code(device_id: str) -> str:
    """DB operator only: browser caregivers cannot issue or retrieve this code."""
    code = secrets.token_urlsafe(32)
    with get_connection() as conn:
        if not conn.execute(
            "SELECT 1 FROM devices WHERE device_id = %s FOR UPDATE", (device_id,),
        ).fetchone():
            raise DeviceError("provision or link this device first")
        conn.execute(
            "INSERT INTO device_consent_credentials (device_id, token_hash) VALUES (%s, %s) "
            "ON CONFLICT (device_id) DO UPDATE SET token_hash = EXCLUDED.token_hash, "
            "issued_at = clock_timestamp()", (device_id, token_hash(code)),
        )
    return code


def _resident_device(conn, code: str) -> str:
    query = "SELECT device_id FROM device_consent_credentials WHERE token_hash = %s"
    row = conn.execute(query, (token_hash(code),)).fetchone()
    if row is None:
        raise DeviceError("consent code is invalid or has been replaced", 403)
    conn.execute("SELECT 1 FROM devices WHERE device_id = %s FOR UPDATE", (row[0],))
    if conn.execute(query, (token_hash(code),)).fetchone() is None:
        raise DeviceError("consent code is invalid or has been replaced", 403)
    return row[0]


def _details(conn, device_id: str) -> dict:
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute("SELECT * FROM devices WHERE device_id = %s", (device_id,))
        device = cursor.fetchone()
        cursor.execute(
            "SELECT a.display_name, a.email, m.role FROM device_memberships m "
            "JOIN accounts a ON a.account_id = m.account_id WHERE m.device_id = %s "
            "ORDER BY m.role DESC, a.email", (device_id,),
        )
        caregivers = cursor.fetchall()
        cursor.execute(
            "SELECT decision, disclosure_version, recorded_at FROM device_consent_history "
            "WHERE device_id = %s ORDER BY version DESC LIMIT 10", (device_id,),
        )
        history = cursor.fetchall()
    for record in history:
        record["recorded_at"] = record["recorded_at"].astimezone(UTC).isoformat()
    return {"device_id": device_id, "room": device["room"], "consent": policy(device),
            "caregivers": caregivers, "history": history,
            "disclosure": {"version": DISCLOSURE_VERSION, "paragraphs": DISCLOSURE}}


def inspect_consent(code: str) -> dict:
    with get_connection() as conn:
        return _details(conn, _resident_device(conn, code))


def change_consent(code: str, decision: str, expected_version: int,
                   disclosure_version: str | None = None) -> dict:
    with get_connection() as conn:
        device_id = _resident_device(conn, code)
        current = device_policy(conn, device_id)
        if expected_version != current["version"]:
            raise DeviceError("consent changed; review the current information and try again", 409)
        if decision not in {"grant", "withdraw"}:
            raise DeviceError("choose grant or withdraw")
        caregiver = conn.execute(
            "SELECT account_id FROM device_memberships WHERE device_id = %s AND role = 'trusted_caregiver'",
            (device_id,),
        ).fetchone()
        if decision == "grant":
            if disclosure_version != DISCLOSURE_VERSION:
                raise DeviceError("review the current monitoring explanation before agreeing", 409)
            if caregiver is None:
                raise DeviceError("the initial caregiver must link their account before you can agree", 409)
            if current["state"] == "withdrawn" and not current["device_confirmed"]:
                raise DeviceError("wait for device stop confirmation before giving permission again", 409)
        state = "granted" if decision == "grant" else "withdrawn"
        if state == current["state"]:
            return _details(conn, device_id)
        # The same device lock serializes consent, command pickup/ack and event
        # ingestion. No stale resume remains executable on the server afterwards.
        conn.execute(
            "UPDATE device_commands SET status = 'cancelled', finished_at = clock_timestamp() "
            "WHERE device_id = %s AND status IN ('pending', 'delivered')", (device_id,),
        )
        conn.execute(
            "UPDATE devices SET consent_state = %s, consent_version = consent_version + 1, "
            "consent_updated_at = clock_timestamp(), consent_text_version = %s, "
            "consent_first_granted_at = CASE WHEN %s = 'granted' THEN "
            "COALESCE(consent_first_granted_at, clock_timestamp()) ELSE consent_first_granted_at END, "
            "consent_delivered_version = NULL, consent_confirmed_version = NULL WHERE device_id = %s",
            (state, DISCLOSURE_VERSION, state, device_id),
        )
        conn.execute(
            "INSERT INTO device_consent_history "
            "(device_id, version, decision, disclosure_version, initial_caregiver_id) "
            "VALUES (%s, %s, %s, %s, %s)",
            (device_id, current["version"] + 1, state, DISCLOSURE_VERSION,
             caregiver[0] if caregiver else None),
        )
        return _details(conn, device_id)


def acknowledge_policy(device_id: str, token: str, version: int,
                       monitoring_state: str, elapsed_seconds: float) -> dict:
    from db.device_commands import authenticate_device

    with get_connection() as conn:
        authenticate_device(conn, device_id, token)
        current = device_policy(conn, device_id)
        delivered = conn.execute(
            "SELECT consent_delivered_version FROM devices WHERE device_id = %s", (device_id,),
        ).fetchone()[0]
        if current["version"] != version or delivered != version:
            raise DeviceError("consent policy changed or was not collected; collect it again", 409)
        if not current["monitoring_allowed"] and monitoring_state != "muted":
            raise DeviceError("capture and monitoring must stop before confirming this policy", 409)
        if current["device_confirmed"]:
            return current  # Replayed policy ack must not overwrite newer reports.
        reported_at = max(datetime.fromisoformat(current["updated_at"]),
                          datetime.now(UTC) - timedelta(seconds=elapsed_seconds))
        state, updated_at = conn.execute(
            "SELECT monitoring_state, monitoring_updated_at FROM devices WHERE device_id = %s", (device_id,),
        ).fetchone()
        if (not current["monitoring_allowed"] and updated_at is not None
                and updated_at > reported_at and state != "muted"):
            raise DeviceError("a newer report does not confirm stopping; report the current state", 409)
        conn.execute(
            "UPDATE devices SET consent_confirmed_version = %s, "
            "monitoring_state = CASE WHEN monitoring_updated_at IS NULL OR monitoring_updated_at <= %s "
            "THEN %s ELSE monitoring_state END, "
            "monitoring_updated_at = GREATEST(monitoring_updated_at, %s) WHERE device_id = %s",
            (version, reported_at, monitoring_state, reported_at, device_id),
        )
        return device_policy(conn, device_id)
