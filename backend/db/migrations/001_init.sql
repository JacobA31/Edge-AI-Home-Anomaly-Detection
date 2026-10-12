-- 001: first tables. Never edit this file after it's merged; add 002_*.sql instead.

CREATE TABLE devices (
    device_id    TEXT PRIMARY KEY,
    room         TEXT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at TIMESTAMPTZ
);

CREATE TABLE events (
    event_id        UUID PRIMARY KEY,               -- from the Pi; makes retries safe
    device_id       TEXT NOT NULL REFERENCES devices (device_id),
    event_type      TEXT NOT NULL,
    room            TEXT NOT NULL,
    detected_at     TIMESTAMPTZ NOT NULL,
    elapsed_seconds DOUBLE PRECISION NOT NULL,
    confidence      DOUBLE PRECISION,
    details         JSONB,
    received_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    acknowledged_at TIMESTAMPTZ                     -- set when a caregiver dismisses the alert
);

CREATE INDEX events_detected_at_idx ON events (detected_at DESC);
