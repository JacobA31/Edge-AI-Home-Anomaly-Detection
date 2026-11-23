"""Durable, ordered mute/resume requests and device-scoped control credentials."""

import secrets
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from psycopg.rows import dict_row

from db.connection import get_connection
from db.consent import device_policy, require_consent
from db.devices import DeviceError, require_manager
from security import token_hash


def public_command(row: dict | None) -> dict | None:
    if row is None:
        return None
    fields = ("command_id", "action", "status", "requested_at", "expires_at",
              "delivered_at", "finished_at", "consent_version")
    result = {key: row[key] for key in fields}
    if result["status"] == "pending" and result["expires_at"] <= datetime.now(UTC):
        result["status"] = "expired"
    result["command_id"] = str(result["command_id"])
    for key in fields[3:7]:
        if result[key] is not None:
            result[key] = result[key].astimezone(UTC).isoformat()
    return result


def latest_commands(conn, device_ids: list[str]) -> dict:
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            "SELECT DISTINCT ON (device_id) * FROM device_commands "
            "WHERE device_id = ANY(%s) ORDER BY device_id, sequence DESC", (device_ids,),
        )
        return {row["device_id"]: public_command(row) for row in cursor.fetchall()}


def _expire(conn, device_id: str) -> None:
    # Delivered commands stay outstanding until acknowledged: a disconnected Pi
    # may already have applied them. Never send an opposite request out of order.
    conn.execute(
        "UPDATE device_commands SET status = 'expired', finished_at = clock_timestamp() "
        "WHERE device_id = %s AND status = 'pending' AND expires_at <= clock_timestamp()",
        (device_id,),
    )


def request_command(device_id: str, account_id: str, action: str) -> tuple[dict, bool]:
    if action not in {"mute", "resume"}:
        raise DeviceError("choose mute or resume")
    with get_connection() as conn:
        require_manager(conn, device_id, account_id)
        if action == "resume":
            require_consent(conn, device_id)
        if not conn.execute(
            "SELECT 1 FROM device_control_credentials WHERE device_id = %s", (device_id,),
        ).fetchone():
            raise DeviceError("remote controls have not been configured for this device", 409)
        _expire(conn, device_id)
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                "SELECT * FROM device_commands WHERE device_id = %s "
                "AND status IN ('pending', 'delivered')", (device_id,),
            )
            existing = cursor.fetchone()
            if existing:
                if existing["action"] != action:
                    raise DeviceError("wait for the outstanding request to be confirmed", 409)
                return public_command(existing), False
            cursor.execute(
                "INSERT INTO device_commands (command_id, device_id, requested_by, action, consent_version) "
                "SELECT %s, device_id, %s, %s, consent_version FROM devices WHERE device_id = %s "
                "RETURNING *", (uuid4(), account_id, action, device_id),
            )
            return public_command(cursor.fetchone()), True


def provision_token(device_id: str) -> str:
    """DB operator only. Rotating the secret preserves outstanding commands."""
    token = secrets.token_urlsafe(32)
    with get_connection() as conn:
        if not conn.execute(
            "SELECT 1 FROM devices WHERE device_id = %s FOR UPDATE", (device_id,),
        ).fetchone():
            raise DeviceError("provision or link this device first")
        conn.execute(
            "INSERT INTO device_control_credentials (device_id, token_hash) VALUES (%s, %s) "
            "ON CONFLICT (device_id) DO UPDATE SET token_hash = EXCLUDED.token_hash, "
            "issued_at = clock_timestamp()", (device_id, token_hash(token)),
        )
    return token


def authenticate_device(conn, device_id: str, token: str) -> None:
    # Authenticate before locking so unauthenticated callers cannot hold the
    # device lock. Recheck after locking to serialize with credential rotation.
    query = "SELECT 1 FROM device_control_credentials WHERE device_id = %s AND token_hash = %s"
    params = (device_id, token_hash(token))
    if not conn.execute(query, params).fetchone():
        raise DeviceError("invalid device control credential", 401)
    conn.execute("SELECT 1 FROM devices WHERE device_id = %s FOR UPDATE", (device_id,))
    if not conn.execute(query, params).fetchone():
        raise DeviceError("invalid device control credential", 401)
    conn.execute("UPDATE devices SET last_seen_at = clock_timestamp() WHERE device_id = %s", (device_id,))


def pick_up(device_id: str, token: str) -> dict:
    with get_connection() as conn:
        authenticate_device(conn, device_id, token)
        _expire(conn, device_id)
        conn.execute(
            "UPDATE devices SET consent_delivered_version = consent_version WHERE device_id = %s",
            (device_id,),
        )
        current_policy = device_policy(conn, device_id)
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                "UPDATE device_commands SET status = 'delivered', "
                "delivered_at = COALESCE(delivered_at, clock_timestamp()) "
                "WHERE device_id = %s AND status IN ('pending', 'delivered') RETURNING *", (device_id,),
            )
            return {"command": public_command(cursor.fetchone()), "consent": current_policy}


def acknowledge(device_id: str, token: str, command_id: str,
                succeeded: bool, elapsed_seconds: float) -> dict:
    with get_connection() as conn:
        authenticate_device(conn, device_id, token)
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                "SELECT * FROM device_commands WHERE device_id = %s AND command_id = %s",
                (device_id, command_id),
            )
            row = cursor.fetchone()
            if row is None:
                raise DeviceError("command not found", 404)
            current_policy = device_policy(conn, device_id)
            if row["action"] == "resume" and succeeded and (
                not current_policy["monitoring_allowed"]
                or row["consent_version"] != current_policy["version"]
                or not current_policy["device_confirmed"]
            ):
                raise DeviceError("collect and confirm current consent before resuming monitoring", 409)
            status = "completed" if succeeded else "failed"
            if row["status"] == status:
                return public_command(row)  # A lost acknowledgement response is safe to retry.
            if row["status"] != "delivered":
                raise DeviceError("command cannot be acknowledged in its current state", 409)
            cursor.execute(
                "UPDATE device_commands SET status = %s, finished_at = clock_timestamp() "
                "WHERE command_id = %s RETURNING *", (status, command_id),
            )
            result = public_command(cursor.fetchone())
            # A delayed acknowledgement must not overwrite a newer heartbeat
            # or local mute/unmute report. Preserve execution age on retries.
            reported_at = max(row["delivered_at"], datetime.now(UTC) - timedelta(seconds=elapsed_seconds))
            if succeeded:
                state = "muted" if row["action"] == "mute" else "active"
                conn.execute(
                    "UPDATE devices SET monitoring_state = %s, monitoring_updated_at = %s "
                    "WHERE device_id = %s AND "
                    "(monitoring_updated_at IS NULL OR monitoring_updated_at <= %s)",
                    (state, reported_at, device_id, reported_at),
                )
            else:
                # A failed operation may have partially affected the microphone.
                # Keep a newer explicit Pi report if one has already arrived.
                conn.execute(
                    "UPDATE devices SET monitoring_state = 'unknown', monitoring_updated_at = %s "
                    "WHERE device_id = %s AND "
                    "(monitoring_updated_at IS NULL OR monitoring_updated_at <= %s)",
                    (reported_at, device_id, reported_at),
                )
            return result
