CREATE TABLE device_memberships (
    device_id TEXT NOT NULL REFERENCES devices (device_id) ON DELETE CASCADE,
    account_id UUID NOT NULL REFERENCES accounts (account_id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('trusted_caregiver', 'caregiver')),
    joined_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (device_id, account_id)
);
CREATE UNIQUE INDEX device_one_trusted_caregiver_idx ON device_memberships (device_id)
    WHERE role = 'trusted_caregiver';
CREATE INDEX device_memberships_account_idx ON device_memberships (account_id);

-- Only the setup operator with DB access can issue a first-caregiver linking code.
CREATE TABLE device_claim_codes (
    device_id TEXT PRIMARY KEY REFERENCES devices (device_id) ON DELETE CASCADE,
    token_hash TEXT NOT NULL UNIQUE,
    expires_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE device_invitations (
    invitation_id UUID PRIMARY KEY,
    device_id TEXT NOT NULL REFERENCES devices (device_id) ON DELETE CASCADE,
    email TEXT NOT NULL CHECK (email = lower(email)),
    token_hash TEXT NOT NULL UNIQUE,
    created_by UUID NOT NULL REFERENCES accounts (account_id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL,
    UNIQUE (device_id, email)
);
