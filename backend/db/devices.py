"""Device membership, single-use setup codes, and read-only caregiver invitations."""

import secrets
from datetime import UTC, datetime
from uuid import uuid4

from psycopg.rows import dict_row

from db.connection import get_connection
from device_status import add_device_status, offline_after_seconds
from security import token_hash


class DeviceError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _device_lock(conn, device_id: str) -> None:
    # Serialize claims, invitations and revocations for this device.
    row = conn.execute(
        "SELECT device_id FROM devices WHERE device_id = %s FOR UPDATE", (device_id,),
    ).fetchone()
    if row is None:
        raise DeviceError("device is unavailable or you do not have access", 403)


def require_manager(conn, device_id: str, account_id: str) -> None:
    _device_lock(conn, device_id)
    row = conn.execute(
        "SELECT role FROM device_memberships WHERE device_id = %s AND account_id = %s",
        (device_id, account_id),
    ).fetchone()
    if row is None or row[0] != "trusted_caregiver":
        raise DeviceError("only the trusted caregiver can manage this device", 403)


def list_devices(account_id: str) -> list[dict]:
    # Import here: command authorization reuses the device membership helpers.
    from db.consent import POLICY_FIELDS, policy
    from db.device_commands import latest_commands

    timeout = offline_after_seconds()
    with get_connection() as conn, conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            "SELECT d.device_id, d.room, m.role, d.last_seen_at, "
            "d.monitoring_state, d.monitoring_updated_at, "
            + ", ".join("d." + field for field in POLICY_FIELDS) + " "
            "FROM devices d "
            "JOIN device_memberships m ON m.device_id = d.device_id "
            "WHERE m.account_id = %s ORDER BY d.device_id", (account_id,),
        )
        rows = cursor.fetchall()
        commands = latest_commands(conn, [row["device_id"] for row in rows])
        configured = {row[0] for row in conn.execute(
            "SELECT device_id FROM device_control_credentials WHERE device_id = ANY(%s)",
            ([row["device_id"] for row in rows],),
        ).fetchall()}
        for row in rows:
            row["consent"] = policy(row)
            for field in POLICY_FIELDS:
                del row[field]
            row["latest_command"] = commands.get(row["device_id"])
            row["controls_configured"] = row["device_id"] in configured
    now = datetime.now(UTC)
    return [add_device_status(row, now, timeout) for row in rows]


def provision_device(device_id: str, room: str) -> str:
    """Operator-only entry point used by the CLI, never exposed as a browser API."""
    code = secrets.token_urlsafe(32)
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO devices (device_id, room) VALUES (%s, %s) ON CONFLICT (device_id) DO NOTHING",
            (device_id, room),
        )
        _device_lock(conn, device_id)
        if conn.execute("SELECT 1 FROM device_memberships WHERE device_id = %s", (device_id,)).fetchone():
            raise DeviceError("this device is already linked; its trusted caregiver manages access", 409)
        conn.execute(
            "INSERT INTO device_claim_codes (device_id, token_hash, expires_at) "
            "VALUES (%s, %s, now() + interval '24 hours') "
            "ON CONFLICT (device_id) DO UPDATE SET token_hash = EXCLUDED.token_hash, "
            "expires_at = EXCLUDED.expires_at",
            (device_id, token_hash(code)),
        )
    return code


def claim_device(account_id: str, code: str) -> str:
    digest = token_hash(code)
    with get_connection() as conn:
        row = conn.execute(
            "SELECT device_id FROM device_claim_codes WHERE token_hash = %s AND expires_at > now()",
            (digest,),
        ).fetchone()
        if row is None:
            raise DeviceError("linking code is invalid, expired, or already used")
        device_id = row[0]
        _device_lock(conn, device_id)
        # Recheck under the device lock: simultaneous claims may have consumed the code.
        row = conn.execute(
            "DELETE FROM device_claim_codes WHERE device_id = %s AND token_hash = %s "
            "AND expires_at > now() RETURNING device_id", (device_id, digest),
        ).fetchone()
        if row is None:
            raise DeviceError("linking code is invalid, expired, or already used")
        if conn.execute("SELECT 1 FROM device_memberships WHERE device_id = %s", (device_id,)).fetchone():
            raise DeviceError("this device is already linked", 409)
        conn.execute(
            "INSERT INTO device_memberships (device_id, account_id, role) "
            "VALUES (%s, %s, 'trusted_caregiver')",
            (device_id, account_id),
        )
    return device_id


def get_access(device_id: str, account_id: str) -> dict:
    with get_connection() as conn:
        require_manager(conn, device_id, account_id)
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                "SELECT a.account_id, a.email, a.display_name, m.role FROM device_memberships m "
                "JOIN accounts a ON a.account_id = m.account_id WHERE m.device_id = %s "
                "ORDER BY m.role DESC, a.email", (device_id,),
            )
            members = cursor.fetchall()
            cursor.execute(
                "SELECT invitation_id, email, expires_at FROM device_invitations "
                "WHERE device_id = %s AND expires_at > now() ORDER BY email", (device_id,),
            )
            invitations = cursor.fetchall()
    for member in members:
        member["account_id"] = str(member["account_id"])
    for invitation in invitations:
        invitation["invitation_id"] = str(invitation["invitation_id"])
        invitation["expires_at"] = invitation["expires_at"].astimezone(UTC).isoformat()
    return {"members": members, "invitations": invitations}


def invite_caregiver(device_id: str, account_id: str, email: str) -> str:
    from db.consent import require_consent

    code = secrets.token_urlsafe(32)
    with get_connection() as conn:
        require_manager(conn, device_id, account_id)
        require_consent(conn, device_id)
        if conn.execute(
            "SELECT 1 FROM device_memberships m JOIN accounts a ON a.account_id = m.account_id "
            "WHERE m.device_id = %s AND a.email = %s", (device_id, email),
        ).fetchone():
            raise DeviceError("this caregiver already has access", 409)
        # Reissuing an invitation invalidates its previous code.
        conn.execute(
            "INSERT INTO device_invitations "
            "(invitation_id, device_id, email, token_hash, created_by, expires_at) "
            "VALUES (%s, %s, %s, %s, %s, now() + interval '3 days') "
            "ON CONFLICT (device_id, email) DO UPDATE SET token_hash = EXCLUDED.token_hash, "
            "created_by = EXCLUDED.created_by, created_at = now(), expires_at = EXCLUDED.expires_at",
            (uuid4(), device_id, email, token_hash(code), account_id),
        )
    return code


def accept_invitation(account_id: str, email: str, code: str) -> str:
    from db.consent import require_consent

    digest = token_hash(code)
    with get_connection() as conn:
        row = conn.execute(
            "SELECT device_id FROM device_invitations "
            "WHERE token_hash = %s AND email = %s AND expires_at > now()", (digest, email),
        ).fetchone()
        if row is None:
            raise DeviceError("invitation is invalid, expired, or does not match your account email")
        device_id = row[0]
        _device_lock(conn, device_id)
        require_consent(conn, device_id)
        row = conn.execute(
            "DELETE FROM device_invitations WHERE device_id = %s AND token_hash = %s "
            "AND email = %s AND expires_at > now() RETURNING invitation_id", (device_id, digest, email),
        ).fetchone()
        if row is None:
            raise DeviceError("invitation is invalid, expired, or already used")
        conn.execute(
            "INSERT INTO device_memberships (device_id, account_id, role) VALUES (%s, %s, 'caregiver') "
            "ON CONFLICT (device_id, account_id) DO NOTHING", (device_id, account_id),
        )
    return device_id


def remove_caregiver(device_id: str, manager_id: str, account_id: str) -> None:
    with get_connection() as conn:
        require_manager(conn, device_id, manager_id)
        if manager_id == account_id:
            raise DeviceError("the trusted caregiver cannot remove their own access", 409)
        row = conn.execute(
            "DELETE FROM device_memberships WHERE device_id = %s AND account_id = %s "
            "AND role = 'caregiver' RETURNING account_id", (device_id, account_id),
        ).fetchone()
        if row is None:
            raise DeviceError("caregiver membership was not found", 404)
        # Revocation must also cancel any outstanding invitations for this account.
        conn.execute(
            "DELETE FROM device_invitations WHERE device_id = %s "
            "AND email = (SELECT email FROM accounts WHERE account_id = %s)", (device_id, account_id),
        )


def revoke_invitation(device_id: str, account_id: str, invitation_id: str) -> None:
    with get_connection() as conn:
        require_manager(conn, device_id, account_id)
        row = conn.execute(
            "DELETE FROM device_invitations WHERE device_id = %s AND invitation_id = %s "
            "RETURNING invitation_id", (device_id, invitation_id),
        ).fetchone()
        if row is None:
            raise DeviceError("invitation was not found", 404)
