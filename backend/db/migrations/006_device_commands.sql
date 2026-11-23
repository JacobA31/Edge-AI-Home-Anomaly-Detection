-- Remote requests are separate from the Pi's reported monitoring state.
CREATE TABLE device_control_credentials (
    device_id TEXT PRIMARY KEY REFERENCES devices(device_id) ON DELETE CASCADE,
    token_hash TEXT NOT NULL UNIQUE,
    issued_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE device_commands (
    command_id UUID PRIMARY KEY,
    sequence BIGSERIAL NOT NULL UNIQUE,
    device_id TEXT NOT NULL REFERENCES devices(device_id) ON DELETE CASCADE,
    requested_by UUID REFERENCES accounts(account_id) ON DELETE SET NULL,
    action TEXT NOT NULL CHECK (action IN ('mute', 'resume')),
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'delivered', 'completed', 'failed', 'expired')),
    requested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL DEFAULT (now() + interval '10 minutes'),
    delivered_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    CHECK ((status IN ('delivered', 'completed', 'failed')) = (delivered_at IS NOT NULL)),
    CHECK ((status IN ('completed', 'failed', 'expired')) = (finished_at IS NOT NULL))
);

CREATE UNIQUE INDEX device_commands_one_outstanding ON device_commands(device_id)
    WHERE status IN ('pending', 'delivered');
CREATE INDEX device_commands_latest ON device_commands(device_id, sequence DESC);
