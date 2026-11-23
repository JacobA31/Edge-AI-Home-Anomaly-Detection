"""Connection freshness and explicitly reported monitoring state."""

import os
from datetime import UTC, datetime, timedelta

# Three missed check-ins with the Pi's default 90-second heartbeat interval.
DEFAULT_OFFLINE_AFTER_SECONDS = 270
MAX_REPORT_AGE_SECONDS = 3155760000  # 100 years; older reports cannot establish current state.


def offline_after_seconds() -> int:
    try:
        seconds = int(os.environ.get("DEVICE_OFFLINE_AFTER_SECONDS", str(DEFAULT_OFFLINE_AFTER_SECONDS)))
    except ValueError as exc:
        raise RuntimeError("DEVICE_OFFLINE_AFTER_SECONDS must be a positive whole number") from exc
    if seconds < 1:
        raise RuntimeError("DEVICE_OFFLINE_AFTER_SECONDS must be a positive whole number")
    return seconds


def monitoring_report(event: dict, received_at: datetime) -> tuple[str, datetime] | None:
    """A bare heartbeat means connected, not necessarily monitoring audio.

    Matthew's Pi integration can include details.monitoring_state in heartbeats
    (active/muted/unknown), or send device_muted/device_unmuted transitions.
    """
    if event["event_type"] == "device_muted":
        state = "muted"
    elif event["event_type"] == "device_unmuted":
        state = "active"
    elif event["event_type"] == "heartbeat":
        state = event.get("details", {}).get("monitoring_state")
    else:
        return None
    if not isinstance(state, str) or state not in {"active", "muted", "unknown"}:
        return None
    age = event["elapsed_seconds"]
    if age > MAX_REPORT_AGE_SECONDS:
        return None
    detected = datetime.fromisoformat(event["detected_at"].replace("Z", "+00:00")).astimezone(UTC)
    # Use both clocks conservatively; future/skewed Pi timestamps cannot extend
    # a report beyond receipt time, and queued events keep their original age.
    reported_at = min(detected, received_at - timedelta(seconds=age))
    return state, reported_at


def add_device_status(device: dict, now: datetime, timeout: int) -> dict:
    last_seen = device["last_seen_at"]
    if last_seen is None:
        device["connection_status"] = "never_seen"
    else:
        device["connection_status"] = "online" if (now - last_seen).total_seconds() < timeout else "offline"
    device["offline_after_seconds"] = timeout
    for field in ("last_seen_at", "monitoring_updated_at"):
        if device[field] is not None:
            device[field] = device[field].astimezone(UTC).isoformat().replace("+00:00", "Z")
    return device
