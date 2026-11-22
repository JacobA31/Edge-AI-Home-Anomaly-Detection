"""Event endpoints.

POST /api/events  – called by the Raspberry Pi. Needs a function key (x-functions-key header).
GET  /api/events  – called by the dashboard to list recent events.
"""

import json
import logging
from datetime import datetime

import azure.functions as func
from psycopg import Error

from db.accounts import get_session_account
from db.events import get_events, save_event
from event_cursor import decode_cursor
from security import session_token
from validation import EventValidationError, validate_event

bp = func.Blueprint()


@bp.route(route="events", methods=["POST"], auth_level=func.AuthLevel.FUNCTION)
def post_event(req: func.HttpRequest) -> func.HttpResponse:
    try:
        event = req.get_json()
        # Python's JSON decoder also accepts NaN/Infinity, which are not valid JSON.
        json.dumps(event, allow_nan=False)
    except ValueError:
        return _json({"error": "body must be JSON"}, 400)

    try:
        validate_event(event)
    except EventValidationError as exc:
        return _json({"error": str(exc)}, 422)

    try:
        inserted = save_event(event)
    except (Error, RuntimeError):
        logging.exception("Could not save event %s", event["event_id"])
        return _json({"error": "event could not be saved; retry later"}, 503)

    logging.info("%s event %s", "Saved" if inserted else "Already saved", event["event_id"])
    return _json(
        {"status": "accepted" if inserted else "duplicate", "event_id": event["event_id"]},
        202 if inserted else 200,
    )


@bp.route(route="events", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def list_events(req: func.HttpRequest) -> func.HttpResponse:
    try:
        user = get_session_account(session_token(req))
    except (Error, RuntimeError):
        logging.exception("Could not check dashboard session")
        return _json({"error": "account service unavailable; try again later"}, 503)
    if user is None:
        return _json({"error": "please log in"}, 401)
    try:
        limit = int(req.params.get("limit", "50"))
    except ValueError:
        return _json({"error": "limit must be a whole number between 1 and 100"}, 400)
    if not 1 <= limit <= 100:
        return _json({"error": "limit must be a whole number between 1 and 100"}, 400)

    since = None
    if "since" in req.params:
        try:
            since = datetime.fromisoformat(req.params["since"])
            if since.utcoffset() is None:
                raise ValueError("timezone required")
        except ValueError:
            return _json({"error": "since must be an ISO 8601 timestamp with a timezone"}, 400)

    device_id = req.params.get("device_id")
    if device_id is not None and not 1 <= len(device_id) <= 64:
        return _json({"error": "device_id must contain between 1 and 64 characters"}, 400)

    alerts_only = req.params.get("alerts_only", "false")
    if alerts_only not in {"true", "false"}:
        return _json({"error": "alerts_only must be true or false"}, 400)
    cursor = None
    if "cursor" in req.params:
        try:
            cursor = decode_cursor(req.params["cursor"])
        except ValueError as exc:
            return _json({"error": str(exc)}, 400)

    try:
        page = get_events(
            account_id=user["account_id"], limit=limit, since=since, device_id=device_id,
            alerts_only=alerts_only == "true", cursor=cursor,
        )
    except (Error, RuntimeError):
        logging.exception("Could not retrieve events")
        return _json({"error": "events could not be loaded; retry later"}, 503)
    return _json(page)


def _json(body: dict, status: int = 200) -> func.HttpResponse:
    return func.HttpResponse(
        json.dumps(body), status_code=status, mimetype="application/json",
        headers={"Cache-Control": "no-store"},
    )
