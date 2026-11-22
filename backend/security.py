"""Password hashing and browser-session safeguards. Never log passwords or tokens."""

import hashlib
import hmac
import os
import re
import secrets
from http.cookies import CookieError, SimpleCookie
from urllib.parse import urlsplit

import azure.functions as func

PASSWORD_ITERATIONS = 600_000
SESSION_SECONDS = 8 * 60 * 60
COOKIE_NAME = "e3_session"


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), PASSWORD_ITERATIONS)
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    algorithm, iterations, salt, expected = stored.split("$")
    if algorithm != "pbkdf2_sha256":
        return False
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), int(iterations))
    return hmac.compare_digest(digest.hex(), expected)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def session_token(req: func.HttpRequest) -> str | None:
    cookies = SimpleCookie()
    try:
        cookies.load(req.headers.get("cookie", ""))
    except CookieError:
        return None
    cookie = cookies.get(COOKIE_NAME)
    if cookie is None or re.fullmatch(r"[A-Za-z0-9_-]{43}", cookie.value) is None:
        return None
    return cookie.value


def _local_development(req: func.HttpRequest) -> bool:
    # Core Tools also sets WEBSITE_HOSTNAME; its presence does not identify Azure.
    # Use the documented environment setting and require a loopback request URL.
    development = os.environ.get("AZURE_FUNCTIONS_ENVIRONMENT", "").casefold() == "development"
    return development and urlsplit(req.url).hostname in {"localhost", "127.0.0.1", "::1"}


def session_cookie(req: func.HttpRequest, token: str = "") -> str:
    # HTTP cookies are allowed only for local development. Azure always uses HTTPS.
    url = urlsplit(req.url)
    local = _local_development(req) and url.scheme == "http"
    attributes = "SameSite=Lax" if local else "SameSite=None; Secure; Partitioned"
    age = SESSION_SECONDS if token else 0
    return f"{COOKIE_NAME}={token}; Path=/api; Max-Age={age}; HttpOnly; {attributes}"


def trusted_origin(req: func.HttpRequest) -> bool:
    # Cookie-authenticated writes require an exact allowed browser origin (CSRF protection).
    configured = os.environ.get("AUTH_ALLOWED_ORIGINS")
    if configured is None:
        if not _local_development(req):
            raise RuntimeError("AUTH_ALLOWED_ORIGINS must be configured in Azure")
        configured = "http://localhost:5173"
    allowed = {origin.strip().rstrip("/") for origin in configured.split(",") if origin.strip()}
    return req.headers.get("origin", "") in allowed
