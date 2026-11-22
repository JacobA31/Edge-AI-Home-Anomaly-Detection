"""Explain late-arriving alerts using the Pi's recorded detection-to-send age."""

DELAYED_ALERT_AFTER_SECONDS = 30
ALERT_TYPES = {"fall", "anomaly", "inactivity"}


def delay_label(seconds: float) -> str:
    if seconds >= 365 * 86400:
        return "more than a year"
    remaining = int(seconds)
    if remaining < 1:
        return "less than a second"
    parts = []
    for size, unit in ((86400, "day"), (3600, "hour"), (60, "minute"), (1, "second")):
        amount, remaining = divmod(remaining, size)
        if amount:
            parts.append(f"{amount} {unit}{'s' if amount != 1 else ''}")
        if len(parts) == 2:
            break
    return " ".join(parts)


def delivery_info(event: dict) -> dict:
    # The Pi fills elapsed_seconds when sending from its outbox. It is a
    # duration measured on one clock, unlike server_received - pi_detected.
    # Use the first accepted record: retries must not reclassify old alerts.
    seconds = max(0.0, float(event["elapsed_seconds"]))
    return {
        "is_delayed": event["event_type"] in ALERT_TYPES and seconds >= DELAYED_ALERT_AFTER_SECONDS,
        "delivery_delay_seconds": seconds,
        "delivery_delay_label": delay_label(seconds),
    }
