"""Browser requests use account sessions; Pi pickup/ack use per-device tokens."""

import logging
import math
import re
from uuid import UUID

import azure.functions as func
from psycopg import Error

from api.auth import InvalidInput, read_body, response
from api.devices import device_endpoint
from db import device_commands
from db.devices import DeviceError

bp = func.Blueprint()


def elapsed_value(value) -> float:
    if (type(value) not in (int, float) or not 0 <= value <= 3155760000
            or not math.isfinite(value)):
        raise InvalidInput("elapsed_seconds must be finite and between 0 and 3155760000")
    return value


@bp.route(route="devices/{device_id}/commands", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS)
@device_endpoint
def request_device_command(req: func.HttpRequest, user: dict) -> func.HttpResponse:
    body = read_body(req)
    if set(body) != {"action"} or body["action"] not in ("mute", "resume"):
        raise InvalidInput("provide an action of mute or resume")
    command, created = device_commands.request_command(
        req.route_params["device_id"], user["account_id"], body["action"],
    )
    return response({"command": command}, 202 if created else 200)


def pi_endpoint(handler):
    def wrapped(req: func.HttpRequest) -> func.HttpResponse:
        # Azure anonymous means no function key is required, NOT public access:
        # these routes always verify a hashed, device-scoped bearer credential.
        match = re.fullmatch(r"Bearer ([A-Za-z0-9_-]{43})", req.headers.get("authorization", ""))
        if match is None:
            return response({"error": "device control credential required"}, 401)
        try:
            return handler(req, match[1])
        except InvalidInput as exc:
            return response({"error": str(exc)}, 400)
        except DeviceError as exc:
            return response({"error": str(exc)}, exc.status)
        except (Error, RuntimeError):
            logging.exception("Device command request could not be completed")
            return response({"error": "device control service unavailable; try again later"}, 503)
    # Azure must bind only the request, not the internal credential argument.
    wrapped.__name__ = handler.__name__
    wrapped.__doc__ = handler.__doc__
    return wrapped


@bp.route(route="device-control/{device_id}/next", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS)
@pi_endpoint
def pick_up_device_command(req: func.HttpRequest, token: str) -> func.HttpResponse:
    return response(device_commands.pick_up(req.route_params["device_id"], token))


@bp.route(route="device-control/{device_id}/commands/{command_id}/ack", methods=["POST"],
          auth_level=func.AuthLevel.ANONYMOUS)
@pi_endpoint
def acknowledge_device_command(req: func.HttpRequest, token: str) -> func.HttpResponse:
    body = read_body(req)
    if set(body) != {"succeeded", "elapsed_seconds"} or type(body.get("succeeded")) is not bool:
        raise InvalidInput("provide succeeded (boolean) and elapsed_seconds since execution")
    elapsed = elapsed_value(body["elapsed_seconds"])
    try:
        command_id = str(UUID(req.route_params["command_id"]))
    except ValueError as exc:
        raise InvalidInput("invalid command ID") from exc
    return response({"command": device_commands.acknowledge(
        req.route_params["device_id"], token, command_id, body["succeeded"], elapsed,
    )})
