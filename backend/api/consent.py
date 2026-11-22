"""Resident code access requires no caregiver account or dashboard session."""

import re
from functools import wraps

import azure.functions as func

from api.auth import InvalidInput, auth_endpoint, read_body, response
from api.device_commands import elapsed_value, pi_endpoint
from db import consent
from db.devices import DeviceError

bp = func.Blueprint()


def resident_endpoint(handler):
    @wraps(handler)
    def wrapped(req: func.HttpRequest) -> func.HttpResponse:
        try:
            return handler(req)
        except DeviceError as exc:
            return response({"error": str(exc)}, exc.status)
    return auth_endpoint(wrapped)


def code_value(body: dict) -> str:
    value = body.get("code")
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_-]{43}", value.strip()) is None:
        raise InvalidInput("enter the complete private consent code provided during setup")
    return value.strip()


def version_value(body: dict, key: str) -> int:
    value = body.get(key)
    if type(value) is not int or not 0 <= value <= 9007199254740991:
        raise InvalidInput("invalid consent version; refresh the page or collect the policy again")
    return value


@bp.route(route="consent/inspect", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS)
@resident_endpoint
def inspect_monitoring_consent(req: func.HttpRequest) -> func.HttpResponse:
    body = read_body(req)
    if set(body) != {"code"}:
        raise InvalidInput("provide only the private consent code")
    return response(consent.inspect_consent(code_value(body)))


@bp.route(route="consent/decision", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS)
@resident_endpoint
def decide_monitoring_consent(req: func.HttpRequest) -> func.HttpResponse:
    body = read_body(req)
    grant = body.get("decision") == "grant"
    required = {"code", "decision", "expected_version"}
    if grant:
        required |= {"disclosure_version", "confirmed"}
    if set(body) != required or body.get("decision") not in ("grant", "withdraw"):
        raise InvalidInput("provide a consent decision and the version you reviewed")
    if grant and (body.get("confirmed") is not True or not isinstance(body.get("disclosure_version"), str)):
        raise InvalidInput("confirm that you understand and agree to monitoring")
    return response(consent.change_consent(
        code_value(body), body["decision"], version_value(body, "expected_version"),
        body.get("disclosure_version"),
    ))


@bp.route(route="device-control/{device_id}/consent/ack", methods=["POST"],
          auth_level=func.AuthLevel.ANONYMOUS)
@pi_endpoint
def acknowledge_monitoring_consent(req: func.HttpRequest, token: str) -> func.HttpResponse:
    body = read_body(req)
    if (set(body) != {"version", "monitoring_state", "elapsed_seconds"}
            or body.get("monitoring_state") not in ("muted", "active")):
        raise InvalidInput("provide consent version, actual monitoring_state and elapsed_seconds")
    return response({"consent": consent.acknowledge_policy(
        req.route_params["device_id"], token, version_value(body, "version"),
        body["monitoring_state"], elapsed_value(body["elapsed_seconds"]),
    )})
