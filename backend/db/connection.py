"""Database connection helper. Reads DATABASE_URL from app settings / local.settings.json."""

import os

import psycopg


def get_connection(autocommit: bool = False) -> psycopg.Connection:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    return psycopg.connect(url, connect_timeout=10, autocommit=autocommit)
