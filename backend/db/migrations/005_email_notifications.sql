CREATE TABLE email_notifications (
    notification_id UUID PRIMARY KEY,
    event_id UUID NOT NULL REFERENCES events (event_id) ON DELETE CASCADE,
    account_id UUID NOT NULL REFERENCES accounts (account_id) ON DELETE CASCADE,
    recipient_email TEXT NOT NULL,
    delivery_mode TEXT NOT NULL CHECK (delivery_mode IN ('preview', 'smtp', 'disabled')),
    status TEXT NOT NULL CHECK (status IN (
        'pending', 'processing', 'sent', 'previewed', 'failed', 'cancelled', 'uncertain', 'disabled'
    )),
    attempt_count INTEGER NOT NULL DEFAULT 0,
    manual_retry_count INTEGER NOT NULL DEFAULT 0,
    queued_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    claimed_at TIMESTAMPTZ,
    claim_token UUID,
    finished_at TIMESTAMPTZ,
    last_error TEXT,
    preview_subject TEXT,
    preview_body TEXT,
    UNIQUE (event_id, account_id),
    CHECK ((status = 'processing') = (claim_token IS NOT NULL))
);
CREATE INDEX email_notifications_pending_idx ON email_notifications (delivery_mode, next_attempt_at, queued_at)
    WHERE status = 'pending';
CREATE INDEX email_notifications_processing_idx ON email_notifications (claimed_at)
    WHERE status = 'processing';
-- Only new alerts create notifications. Historical/test alerts are not backfilled.
