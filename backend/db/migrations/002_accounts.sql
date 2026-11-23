-- Accounts establish identity; device membership and roles are added separately.
CREATE TABLE accounts (
    account_id UUID PRIMARY KEY,
    email TEXT NOT NULL UNIQUE CHECK (email = lower(email)),
    display_name TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE account_sessions (
    token_hash TEXT PRIMARY KEY,
    account_id UUID NOT NULL REFERENCES accounts (account_id) ON DELETE CASCADE,
    expires_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX account_sessions_account_idx ON account_sessions (account_id);
CREATE INDEX account_sessions_expiry_idx ON account_sessions (expires_at);

-- Shared across Function App instances; failed login limits survive restarts.
CREATE TABLE auth_attempts (
    attempt_key TEXT PRIMARY KEY,
    attempts INTEGER NOT NULL,
    window_started_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
