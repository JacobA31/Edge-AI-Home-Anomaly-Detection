"""Builds event records matching shared/event.schema.json."""

import uuid
from datetime import UTC, datetime


def make_event(device_id: str, room: str, event_type: str, confidence: float | None = None,
               details: dict | None = None) -> dict:
    event = {
        "event_id": str(uuid.uuid4()),
        "device_id": device_id,
        "event_type": event_type,
        "room": room,
        "detected_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "elapsed_seconds": 0.0,
    }
    if confidence is not None:
        event["confidence"] = confidence
    if details:
        event["details"] = details
    return event
