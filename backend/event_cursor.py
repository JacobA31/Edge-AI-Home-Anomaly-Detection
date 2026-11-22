"""Opaque history markers retaining PostgreSQL's full timestamp precision."""

import base64
import json
from datetime import datetime
from uuid import UUID


def encode_cursor(event: dict) -> str:
    marker = {key: event[key] for key in ("detected_at", "received_at", "event_id")}
    payload = json.dumps(marker, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


def decode_cursor(value: str) -> tuple[datetime, datetime, UUID]:
    try:
        if not value or len(value) > 1024:
            raise ValueError("invalid cursor length")
        payload = base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)
        marker = json.loads(payload)
        if not isinstance(marker, dict) or set(marker) != {"detected_at", "received_at", "event_id"}:
            raise ValueError("invalid cursor fields")
        detected = datetime.fromisoformat(marker["detected_at"])
        received = datetime.fromisoformat(marker["received_at"])
        if detected.utcoffset() is None or received.utcoffset() is None:
            raise ValueError("cursor timestamps require timezones")
        return detected, received, UUID(marker["event_id"])
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise ValueError("cursor is invalid; reload the alert history") from exc
