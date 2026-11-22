-- Never infer consent from an existing device, caregiver membership, or mute state.
ALTER TABLE devices
    ADD COLUMN consent_state TEXT NOT NULL DEFAULT 'not_recorded'
        CHECK (consent_state IN ('not_recorded', 'granted', 'withdrawn')),
    ADD COLUMN consent_version BIGINT NOT NULL DEFAULT 0 CHECK (consent_version >= 0),
    ADD COLUMN consent_updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    ADD COLUMN consent_text_version TEXT,
    ADD COLUMN consent_first_granted_at TIMESTAMPTZ,
    ADD COLUMN consent_delivered_version BIGINT,
    ADD COLUMN consent_confirmed_version BIGINT;

CREATE TABLE device_consent_credentials (
    device_id TEXT PRIMARY KEY REFERENCES devices(device_id) ON DELETE CASCADE,
    token_hash TEXT NOT NULL UNIQUE,
    issued_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE device_consent_history (
    device_id TEXT NOT NULL REFERENCES devices(device_id) ON DELETE CASCADE,
    version BIGINT NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN ('granted', 'withdrawn')),
    disclosure_version TEXT NOT NULL,
    initial_caregiver_id UUID REFERENCES accounts(account_id) ON DELETE SET NULL,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (device_id, version)
);

ALTER TABLE device_commands
    ADD COLUMN consent_version BIGINT NOT NULL DEFAULT 0,
    DROP CONSTRAINT device_commands_status_check,
    DROP CONSTRAINT device_commands_check,
    DROP CONSTRAINT device_commands_check1,
    ADD CONSTRAINT device_commands_status_check
        CHECK (status IN ('pending', 'delivered', 'completed', 'failed', 'expired', 'cancelled')),
    ADD CONSTRAINT device_commands_delivery_check CHECK (
        (status NOT IN ('delivered', 'completed', 'failed') OR delivered_at IS NOT NULL)
        AND (status NOT IN ('pending', 'expired') OR delivered_at IS NULL)),
    ADD CONSTRAINT device_commands_finished_check CHECK (
        (status IN ('completed', 'failed', 'expired', 'cancelled')) = (finished_at IS NOT NULL));

-- Old outstanding commands cannot bypass the new setup requirement.
UPDATE device_commands SET status = 'cancelled', finished_at = now()
    WHERE status IN ('pending', 'delivered');
