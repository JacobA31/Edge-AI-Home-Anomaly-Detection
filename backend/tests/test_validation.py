import json
from pathlib import Path

import pytest

from db.migrate import MIGRATIONS_DIR
from validation import EventValidationError, validate_event

SAMPLE = Path(__file__).resolve().parents[2] / "shared" / "sample-event.json"


def sample() -> dict:
    return json.loads(SAMPLE.read_text())


def test_sample_event_is_valid():
    validate_event(sample())


def test_rejects_unknown_event_type():
    event = sample() | {"event_type": "party"}
    with pytest.raises(EventValidationError, match="event_type"):
        validate_event(event)


def test_rejects_extra_fields_such_as_audio():
    event = sample() | {"audio": "UklGRiQAAABXQVZF..."}
    with pytest.raises(EventValidationError):
        validate_event(event)


def test_migrations_are_numbered():
    names = [p.stem for p in sorted(MIGRATIONS_DIR.glob("*.sql"))]
    assert names, "expected at least one migration"
    assert all(n[:3].isdigit() and n[3] == "_" for n in names), names
