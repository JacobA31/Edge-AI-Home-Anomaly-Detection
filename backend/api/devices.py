"""Authenticated device linking and trusted-caregiver access management."""

import re
from uuid import UUID

import azure.functions as func

from api.auth import InvalidInput, auth_endpoint, email_address, read_body, response
from db import accounts, devices
from security import session_token

bp = func.Blueprint()


def device_endpoint(handler):
    def wrapped(req: func.HttpRequest) -> func.HttpResponse:
        user = accounts.get_session_account(session_token(req))
        if user is None:
            return response({"error": "please log in"}, 401)
        try:
            return handler(req, user)
        except devices.DeviceError as exc:
            return response({"error": str(exc)}, exc.status)
    # Azure binds only `req`. Preserve the name, without publishing the internal
    # handler's extra `user` parameter as an HTTP-trigger argument.
    wrapped.__name__ = handler.__name__
    wrapped.__doc__ = handler.__doc__
    return auth_endpoint(wrapped)


def code_value(req: func.HttpRequest, name: str) -> str:
    code = read_body(req).get(name)
    if not isinstance(code, str) or re.fullmatch(r"[A-Za-z0-9_-]{43}", code.strip()) is None:
        raise InvalidInput("enter the complete code you were given")
    return code.strip()


def code_limit(user: dict) -> func.HttpResponse | None:
    if accounts.allow_attempt("device-code:" + user["account_id"]):
        return None
    result = response({"error": "too many code attempts; try again in 15 minutes"}, 429)
    result.headers["retry-after"] = "900"
    return result


def uuid_parameter(req: func.HttpRequest, name: str) -> str:
    try:
        return str(UUID(req.route_params.get(name, "")))
    except ValueError as exc:
        raise InvalidInput("invalid account or invitation ID") from exc


@bp.route(route="devices", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
@device_endpoint
def list_devices(req: func.HttpRequest, user: dict) -> func.HttpResponse:
    return response({"devices": devices.list_devices(user["account_id"])})


@bp.route(route="devices/claim", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS)
@device_endpoint
def claim_device(req: func.HttpRequest, user: dict) -> func.HttpResponse:
    code = code_value(req, "linking_code")
    if (blocked := code_limit(user)) is not None:
        return blocked
    device_id = devices.claim_device(user["account_id"], code)
    return response({"device_id": device_id, "role": "trusted_caregiver"}, 201)


@bp.route(route="devices/invitations/accept", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS)
@device_endpoint
def accept_invitation(req: func.HttpRequest, user: dict) -> func.HttpResponse:
    code = code_value(req, "invitation_code")
    if (blocked := code_limit(user)) is not None:
        return blocked
    device_id = devices.accept_invitation(user["account_id"], user["email"], code)
    return response({"device_id": device_id, "role": "caregiver"})


@bp.route(route="devices/{device_id}/access", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
@device_endpoint
def device_access(req: func.HttpRequest, user: dict) -> func.HttpResponse:
    return response(devices.get_access(req.route_params["device_id"], user["account_id"]))


@bp.route(route="devices/{device_id}/invitations", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS)
@device_endpoint
def invite_caregiver(req: func.HttpRequest, user: dict) -> func.HttpResponse:
    body = read_body(req)
    if set(body) != {"email"}:
        raise InvalidInput("provide only the caregiver email; invitations always grant read-only access")
    code = devices.invite_caregiver(req.route_params["device_id"], user["account_id"], email_address(body))
    return response({"invitation_code": code, "expires_in_days": 3}, 201)


@bp.route(route="devices/{device_id}/members/{account_id}", methods=["DELETE"],
          auth_level=func.AuthLevel.ANONYMOUS)
@device_endpoint
def remove_caregiver(req: func.HttpRequest, user: dict) -> func.HttpResponse:
    devices.remove_caregiver(
        req.route_params["device_id"], user["account_id"], uuid_parameter(req, "account_id"),
    )
    return response({"status": "removed"})


@bp.route(route="devices/{device_id}/invitations/{invitation_id}", methods=["DELETE"],
          auth_level=func.AuthLevel.ANONYMOUS)
@device_endpoint
def revoke_invitation(req: func.HttpRequest, user: dict) -> func.HttpResponse:
    devices.revoke_invitation(
        req.route_params["device_id"], user["account_id"], uuid_parameter(req, "invitation_id"),
    )
    return response({"status": "revoked"})
