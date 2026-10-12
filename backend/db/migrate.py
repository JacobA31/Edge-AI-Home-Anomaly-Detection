"""Tiny migration runner: applies db/migrations/NNN_name.sql files in order, once each.

Run it locally with:  python -m db.migrate     (needs DATABASE_URL in your environment)
In Azure it runs through POST /api/admin/migrate after every deploy.
"""

from pathlib import Path

from db.connection import get_connection

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
_LOCK_ID = 4907  # any constant; stops two runs applying migrations at the same time


def pending_files(applied: set[str]) -> list[Path]:
    return [p for p in sorted(MIGRATIONS_DIR.glob("*.sql")) if p.stem not in applied]


def apply_migrations() -> list[str]:
    done: list[str] = []
    with get_connection(autocommit=True) as conn:
        conn.execute("SELECT pg_advisory_lock(%s)", (_LOCK_ID,))
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                " version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
            )
            applied = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
            for path in pending_files(applied):
                with conn.transaction():
                    conn.execute(path.read_text())
                    conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (path.stem,))
                done.append(path.stem)
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (_LOCK_ID,))
    return done


if __name__ == "__main__":
    print("Applied:", apply_migrations() or "nothing new")
