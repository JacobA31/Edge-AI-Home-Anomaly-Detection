"""Account records and revocable, expiring browser sessions."""

import secrets
from uuid import uuid4

from psycopg.rows import dict_row

from db.connection import get_connection
from security import SESSION_SECONDS, token_hash


def public_account(row: dict) -> dict:
    return {
        "account_id": str(row["account_id"]),
        "email": row["email"],
        "display_name": row["display_name"],
    }


def create_account(email: str, display_name: str, password_hash: str) -> dict:
    with get_connection() as conn, conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            "INSERT INTO accounts (account_id, email, display_name, password_hash) "
            "VALUES (%s, %s, %s, %s) RETURNING account_id, email, display_name",
            (uuid4(), email, display_name, password_hash),
        )
        return public_account(cursor.fetchone())


def find_account(email: str) -> dict | None:
    with get_connection() as conn, conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute("SELECT * FROM accounts WHERE email = %s", (email,))
        return cursor.fetchone()


def allow_attempt(email: str) -> bool:
    key = token_hash(email)
    with get_connection() as conn:
        conn.execute("DELETE FROM auth_attempts WHERE window_started_at < now() - interval '15 minutes'")
        row = conn.execute(
            "INSERT INTO auth_attempts (attempt_key, attempts) VALUES (%s, 1) "
            "ON CONFLICT (attempt_key) DO UPDATE SET attempts = auth_attempts.attempts + 1 "
            "RETURNING attempts",
            (key,),
        ).fetchone()
    return row[0] <= 5


def create_session(account_id: str, email: str, expected_password_hash: str) -> str | None:
    token = secrets.token_urlsafe(32)
    with get_connection() as conn:
        # Lock against concurrent password changes so an old password cannot create a new session.
        row = conn.execute(
            "SELECT password_hash FROM accounts WHERE account_id = %s FOR UPDATE", (account_id,)
        ).fetchone()
        if row is None or row[0] != expected_password_hash:
            return None
        conn.execute("DELETE FROM account_sessions WHERE expires_at <= now()")
        conn.execute("DELETE FROM auth_attempts WHERE attempt_key = %s", (token_hash(email),))
        conn.execute(
            "INSERT INTO account_sessions (token_hash, account_id, expires_at) "
            "VALUES (%s, %s, now() + %s * interval '1 second')",
            (token_hash(token), account_id, SESSION_SECONDS),
        )
    return token


def get_session_account(token: str | None) -> dict | None:
    if token is None:
        return None
    with get_connection() as conn, conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            "SELECT a.account_id, a.email, a.display_name FROM accounts a "
            "JOIN account_sessions s ON s.account_id = a.account_id "
            "WHERE s.token_hash = %s AND s.expires_at > now()",
            (token_hash(token),),
        )
        row = cursor.fetchone()
    return public_account(row) if row else None


def delete_session(token: str | None) -> None:
    if token is not None:
        with get_connection() as conn:
            conn.execute("DELETE FROM account_sessions WHERE token_hash = %s", (token_hash(token),))


def update_name(account_id: str, display_name: str, token: str) -> dict | None:
    with get_connection() as conn, conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            "UPDATE accounts SET display_name = %s WHERE account_id = %s "
            "AND EXISTS (SELECT 1 FROM account_sessions WHERE token_hash = %s "
            "AND account_id = accounts.account_id AND expires_at > now()) "
            "RETURNING account_id, email, display_name",
            (display_name, account_id, token_hash(token)),
        )
        row = cursor.fetchone()
    return public_account(row) if row else None


def change_password(account_id: str, old_hash: str, new_hash: str, token: str) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            "UPDATE accounts SET password_hash = %s WHERE account_id = %s AND password_hash = %s "
            "AND EXISTS (SELECT 1 FROM account_sessions WHERE token_hash = %s "
            "AND account_id = accounts.account_id AND expires_at > now()) RETURNING account_id",
            (new_hash, account_id, old_hash, token_hash(token)),
        ).fetchone()
        if row is None:
            return False
        conn.execute("DELETE FROM account_sessions WHERE account_id = %s", (account_id,))
    return True
