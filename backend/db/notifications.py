"""Durable, per-caregiver alert email queue. No email is sent while receiving an event."""

from uuid import uuid4

from psycopg.rows import dict_row

from db.connection import get_connection
from notification_email import DeliveryError, build_message, email_mode, submit_smtp

MAX_ATTEMPTS = 6


def enqueue_alert(conn, event: dict) -> None:
    if event["event_type"] not in {"fall", "anomaly", "inactivity"}:
        return
    mode = email_mode()
    recipients = conn.execute(
        "SELECT a.account_id, a.email FROM device_memberships m "
        "JOIN accounts a ON a.account_id = m.account_id WHERE m.device_id = %s "
        "AND m.role IN ('trusted_caregiver', 'caregiver')", (event["device_id"],),
    ).fetchall()
    for account_id, email in recipients:
        conn.execute(
            "INSERT INTO email_notifications "
            "(notification_id, event_id, account_id, recipient_email, delivery_mode, status) "
            "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (event_id, account_id) DO NOTHING",
            (uuid4(), event["event_id"], account_id, email, mode,
             "disabled" if mode == "disabled" else "pending"),
        )


def claim_next(mode: str) -> tuple | None:
    token = uuid4()
    with get_connection() as conn:
        # SKIP LOCKED also protects an active sender holding its delivery row lock.
        # SMTP crashes have an unknowable delivery outcome; preview crashes are safe to retry.
        conn.execute(
            "UPDATE email_notifications SET status = %s, claim_token = NULL, "
            "last_error = %s, finished_at = CASE WHEN delivery_mode = 'smtp' THEN now() ELSE NULL END "
            "WHERE notification_id IN (SELECT notification_id FROM email_notifications "
            "WHERE delivery_mode = %s AND status = 'processing' "
            "AND claimed_at < now() - interval '2 minutes' FOR UPDATE SKIP LOCKED)",
            ("uncertain" if mode == "smtp" else "pending",
             "worker stopped before delivery confirmation" if mode == "smtp" else "preview worker restarted",
             mode),
        )
        row = conn.execute(
            "UPDATE email_notifications SET status = 'processing', claim_token = %s, claimed_at = now(), "
            "attempt_count = attempt_count + 1 WHERE notification_id = ("
            "SELECT notification_id FROM email_notifications WHERE delivery_mode = %s "
            "AND status = 'pending' AND next_attempt_at <= now() "
            "ORDER BY queued_at, notification_id FOR UPDATE SKIP LOCKED LIMIT 1) "
            "RETURNING notification_id, claim_token", (token, mode),
        ).fetchone()
    return row


def _finish(conn, job: dict, status: str, error: str | None = None, message=None) -> None:
    delay = min(300, 10 * 2 ** (job["attempt_count"] - 1))
    conn.execute(
        "UPDATE email_notifications SET status = %s, claim_token = NULL, last_error = %s, "
        "next_attempt_at = clock_timestamp() + %s * interval '1 second', "
        "finished_at = CASE WHEN %s = 'pending' THEN NULL ELSE clock_timestamp() END, "
        "preview_subject = %s, preview_body = %s "
        "WHERE notification_id = %s AND claim_token = %s",
        (status, error, delay, status,
         str(message["Subject"]) if message is not None else None,
         message.get_content() if message is not None else None,
         job["notification_id"], job["claim_token"]),
    )


def process_claim(claim: tuple, mode: str, settings=None) -> str:
    notification_id, token = claim
    with get_connection() as conn, conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            "SELECT n.*, e.device_id, e.event_type, e.room, e.detected_at, e.received_at, e.elapsed_seconds "
            "FROM email_notifications n JOIN events e ON e.event_id = n.event_id "
            "WHERE n.notification_id = %s AND n.claim_token = %s AND n.status = 'processing' "
            "AND n.delivery_mode = %s FOR UPDATE OF n", (notification_id, token, mode),
        )
        job = cursor.fetchone()
        if job is None:
            return "skipped"
        # Legacy queued jobs must not bypass the new initial-consent requirement.
        if conn.execute(
            "SELECT consent_first_granted_at FROM devices WHERE device_id = %s", (job["device_id"],),
        ).fetchone()[0] is None:
            _finish(conn, job, "cancelled", "initial monitoring consent has not been recorded")
            return "cancelled"
        # Lock the membership during SMTP submission. Revocation either completes
        # first (cancel this delivery), or waits until the already-authorized send ends.
        permitted = conn.execute(
            "SELECT 1 FROM device_memberships m JOIN accounts a ON a.account_id = m.account_id "
            "WHERE m.device_id = %s AND m.account_id = %s AND a.email = %s "
            "AND m.role IN ('trusted_caregiver', 'caregiver') FOR SHARE OF m, a",
            (job["device_id"], job["account_id"], job["recipient_email"]),
        ).fetchone()
        if permitted is None:
            _finish(conn, job, "cancelled", "caregiver no longer has access or their email changed")
            return "cancelled"
        try:
            sender = settings.sender if settings is not None else "e3-preview@example.invalid"
            message = build_message(job, sender)
            if mode == "preview":
                _finish(conn, job, "previewed", message=message)
                return "previewed"
            if mode != "smtp" or settings is None:
                raise RuntimeError("SMTP delivery requires validated settings")
            submit_smtp(message, job["recipient_email"], settings)
        except DeliveryError as exc:
            status = "uncertain" if exc.uncertain else (
                "pending" if exc.retryable and job["attempt_count"] < MAX_ATTEMPTS else "failed"
            )
            _finish(conn, job, status, str(exc))
            return status
        _finish(conn, job, "sent")
        return "sent"


def list_notifications(limit: int = 10) -> list[dict]:
    with get_connection() as conn, conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            "SELECT notification_id, event_id, recipient_email, delivery_mode, status, attempt_count, "
            "manual_retry_count, queued_at, next_attempt_at, finished_at, last_error "
            "FROM email_notifications ORDER BY queued_at DESC, notification_id DESC LIMIT %s", (limit,),
        )
        return cursor.fetchall()


def get_preview(notification_id: str | None = None) -> dict | None:
    with get_connection() as conn, conn.cursor(row_factory=dict_row) as cursor:
        condition = " AND notification_id = %s" if notification_id else ""
        cursor.execute(
            "SELECT notification_id, recipient_email, preview_subject, preview_body FROM email_notifications "
            "WHERE delivery_mode = 'preview' AND status = 'previewed'" + condition
            + " ORDER BY queued_at DESC LIMIT 1", (notification_id,) if notification_id else (),
        )
        return cursor.fetchone()


def retry_notification(notification_id: str, confirm_not_sent: bool) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT status FROM email_notifications WHERE notification_id = %s FOR UPDATE",
            (notification_id,),
        ).fetchone()
        if row is None or row[0] not in {"failed", "uncertain"}:
            return False
        if row[0] == "uncertain" and not confirm_not_sent:
            raise ValueError("check the email service first, then use --confirm-not-sent "
                             "to retry an uncertain email")
        conn.execute(
            "UPDATE email_notifications SET status = 'pending', attempt_count = 0, "
            "manual_retry_count = manual_retry_count + 1, next_attempt_at = now(), "
            "finished_at = NULL, last_error = 'manually requeued' WHERE notification_id = %s",
            (notification_id,),
        )
    return True
