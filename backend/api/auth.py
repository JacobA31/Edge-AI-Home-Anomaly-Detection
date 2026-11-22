"""Email/password accounts, profile updates, and cookie-based browser sessions."""

import json
import logging
import re
from functools import wraps

import azure.functions as func
from psycopg import Error
from psycopg.errors import UniqueViolation

from db import accounts
from http_response import CookieResponse
from security import hash_password, session_cookie, session_token, trusted_origin, verify_password

bp = func.Blueprint()
# Unknown email addresses still perform the same expensive password check.
_DUMMY_HASH = hash_password("dummy-account-password-not-used")


class InvalidInput(ValueError):
    pass


def response(body: dict, status: int = 200) -> func.HttpResponse:
    return CookieResponse(
        json.dumps(body), status_code=status, mimetype="application/json",
        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
    )


def auth_endpoint(handler):
    @wraps(handler)
    def wrapped(req: func.HttpRequest) -> func.HttpResponse:
        try:
            if req.method != "GET" and not trusted_origin(req):
                return response({"error": "request origin is not allowed"}, 403)
            return handler(req)
        except InvalidInput as exc:
            return response({"error": str(exc)}, 400)
        except (Error, RuntimeError):
            logging.exception("Account request could not be completed")
            return response({"error": "account service unavailable; try again later"}, 503)
    return wrapped


def read_body(req: func.HttpRequest) -> dict:
    if req.headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
        raise InvalidInput("body must be JSON")
    if len(req.get_body()) > 8192:
        raise InvalidInput("request is too large")
    try:
        body = req.get_json()
    except ValueError as exc:
        raise InvalidInput("body must be JSON") from exc
    if not isinstance(body, dict):
        raise InvalidInput("body must be a JSON object")
    return body


def email_address(body: dict) -> str:
    email = body.get("email")
    if not isinstance(email, str):
        raise InvalidInput("enter a valid email address")
    email = email.strip().lower()
    if len(email) > 254 or re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email) is None:
        raise InvalidInput("enter a valid email address")
    return email


def password_value(body: dict, key: str = "password", minimum: int = 1) -> str:
    password = body.get(key)
    if not isinstance(password, str) or not minimum <= len(password) <= 128:
        raise InvalidInput(f"{key.replace('_', ' ')} must contain {minimum} to 128 characters")
    return password


def display_name(body: dict) -> str:
    name = body.get("display_name")
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 100:
        raise InvalidInput("name must contain 1 to 100 characters")
    return name.strip()


def limited(email: str) -> func.HttpResponse | None:
    if accounts.allow_attempt(email):
        return None
    result = response({"error": "too many attempts; try again in 15 minutes"}, 429)
    result.headers["retry-after"] = "900"
    return result


@bp.route(route="auth/register", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS)
@auth_endpoint
def register(req: func.HttpRequest) -> func.HttpResponse:
    body = read_body(req)
    email = email_address(body)
    name = display_name(body)
    password = password_value(body, minimum=15)
    if (blocked := limited(email)) is not None:
        return blocked
    try:
        accounts.create_account(email, name, hash_password(password))
    except UniqueViolation:
        return response({"error": "an account already uses this email; log in instead"}, 409)
    # Sign in separately: no session is created before account creation commits.
    return response({"status": "created"}, 201)


@bp.route(route="auth/login", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS)
@auth_endpoint
def login(req: func.HttpRequest) -> func.HttpResponse:
    body = read_body(req)
    email = email_address(body)
    password = password_value(body)
    if (blocked := limited(email)) is not None:
        return blocked
    account = accounts.find_account(email)
    stored = account["password_hash"] if account else _DUMMY_HASH
    if not verify_password(password, stored) or account is None:
        return response({"error": "email or password is incorrect"}, 401)
    token = accounts.create_session(str(account["account_id"]), email, stored)
    if token is None:
        return response({"error": "email or password is incorrect"}, 401)
    result = response({"user": accounts.public_account(account)})
    result.headers["set-cookie"] = session_cookie(req, token)
    return result


@bp.route(route="auth/me", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
@auth_endpoint
def me(req: func.HttpRequest) -> func.HttpResponse:
    user = accounts.get_session_account(session_token(req))
    return response({"user": user}) if user else response({"error": "please log in"}, 401)


@bp.route(route="auth/logout", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS)
@auth_endpoint
def logout(req: func.HttpRequest) -> func.HttpResponse:
    accounts.delete_session(session_token(req))
    result = response({"status": "logged_out"})
    result.headers["set-cookie"] = session_cookie(req)
    return result


@bp.route(route="auth/profile", methods=["PATCH"], auth_level=func.AuthLevel.ANONYMOUS)
@auth_endpoint
def profile(req: func.HttpRequest) -> func.HttpResponse:
    token = session_token(req)
    user = accounts.get_session_account(token)
    if user is None:
        return response({"error": "please log in"}, 401)
    updated = accounts.update_name(user["account_id"], display_name(read_body(req)), token)
    return response({"user": updated}) if updated else response({"error": "please log in"}, 401)


@bp.route(route="auth/password", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS)
@auth_endpoint
def password(req: func.HttpRequest) -> func.HttpResponse:
    token = session_token(req)
    user = accounts.get_session_account(token)
    if user is None:
        return response({"error": "please log in"}, 401)
    body = read_body(req)
    current = password_value(body, "current_password")
    new = password_value(body, "new_password", minimum=15)
    if (blocked := limited(user["email"])) is not None:
        return blocked
    account = accounts.find_account(user["email"])
    old_hash = account["password_hash"]
    if not verify_password(current, old_hash):
        return response({"error": "current password is incorrect"}, 400)
    if not accounts.change_password(user["account_id"], old_hash, hash_password(new), token):
        return response({"error": "account changed; please log in again"}, 401)
    result = response({"status": "password_changed"})
    result.headers["set-cookie"] = session_cookie(req)
    return result
