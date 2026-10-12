"""Validates incoming events against shared/event.schema.json."""

import json
from functools import lru_cache
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

_HERE = Path(__file__).resolve().parent
# Deployed: CI copies shared/ into backend/shared/. Local dev: use the repo's shared/ folder.
_SCHEMA_CANDIDATES = [_HERE / "shared" / "event.schema.json", _HERE.parent / "shared" / "event.schema.json"]


class EventValidationError(ValueError):
    pass


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    for path in _SCHEMA_CANDIDATES:
        if path.exists():
            schema = json.loads(path.read_text())
            return Draft202012Validator(schema, format_checker=FormatChecker())
    raise FileNotFoundError("event.schema.json not found in backend/shared or ../shared")


def validate_event(event: object) -> None:
    errors = sorted(_validator().iter_errors(event), key=lambda e: list(e.path))
    if errors:
        first = errors[0]
        where = ".".join(str(p) for p in first.path) or "event"
        raise EventValidationError(f"{where}: {first.message}")
