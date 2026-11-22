"""Email configuration, message formatting, and encrypted SMTP submission."""

import os
import smtplib
import ssl
from dataclasses import dataclass, field
from datetime import UTC
from email.errors import HeaderParseError
from email.headerregistry import Address
from email.message import EmailMessage
from email.policy import SMTP
from email.utils import format_datetime
from urllib.parse import urlencode, urlsplit

from alert_delivery import delivery_info


class DeliveryError(Exception):
    def __init__(self, reason: str, *, retryable: bool = False, uncertain: bool = False):
        super().__init__(reason)
        self.retryable = retryable
        self.uncertain = uncertain


def email_mode() -> str:
    mode = os.environ.get("EMAIL_MODE")
    if mode is None:
        host = urlsplit("//" + os.environ.get("WEBSITE_HOSTNAME", "")).hostname
        local = os.environ.get("AZURE_FUNCTIONS_ENVIRONMENT", "").casefold() == "development"
        mode = "preview" if local and host in {"localhost", "127.0.0.1", "::1"} else "disabled"
    if mode not in {"preview", "smtp", "disabled"}:
        raise RuntimeError("EMAIL_MODE must be preview, smtp, or disabled")
    return mode


def address(value: str) -> Address:
    try:
        result = Address(addr_spec=value)
        if not result.username or not result.domain:
            raise ValueError("incomplete address")
        result.addr_spec.encode("ascii")
        return result
    except (ValueError, IndexError, HeaderParseError) as exc:
        raise DeliveryError("email address is invalid or needs an ASCII mailbox/domain") from exc


@dataclass(frozen=True)
class SMTPSettings:
    host: str
    port: int
    security: str
    username: str = field(repr=False)
    password: str = field(repr=False)
    sender: str

    @classmethod
    def load(cls):
        required = ("SMTP_HOST", "SMTP_USERNAME", "SMTP_PASSWORD", "SMTP_FROM_EMAIL", "DASHBOARD_URL")
        if any(not os.environ.get(name) for name in required):
            raise RuntimeError("SMTP mode requires SMTP_HOST, SMTP_USERNAME, SMTP_PASSWORD, "
                               "SMTP_FROM_EMAIL and DASHBOARD_URL")
        security = os.environ.get("SMTP_SECURITY", "starttls")
        if security not in {"starttls", "ssl"}:
            raise RuntimeError("SMTP_SECURITY must be starttls or ssl; unencrypted SMTP is not supported")
        try:
            port = int(os.environ.get("SMTP_PORT", "465" if security == "ssl" else "587"))
            if not 1 <= port <= 65535:
                raise ValueError("invalid port")
            sender = address(os.environ["SMTP_FROM_EMAIL"]).addr_spec
        except (ValueError, DeliveryError) as exc:
            raise RuntimeError("SMTP_PORT or SMTP_FROM_EMAIL is invalid") from exc
        dashboard_url()
        return cls(os.environ["SMTP_HOST"], port, security,
                   os.environ["SMTP_USERNAME"], os.environ["SMTP_PASSWORD"], sender)


def dashboard_url() -> str:
    value = os.environ.get("DASHBOARD_URL", "http://localhost:5173").rstrip("/")
    parsed = urlsplit(value)
    local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if (not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in {"", "/"} or parsed.scheme not in {"http", "https"}
            or (parsed.scheme == "http" and not local)):
        raise RuntimeError("DASHBOARD_URL must be an HTTPS site origin (HTTP is allowed for localhost)")
    return value


def build_message(job: dict, sender: str) -> EmailMessage:
    recipient = address(job["recipient_email"])
    titles = {"fall": "Possible fall", "anomaly": "Unusual sound activity", "inactivity": "Inactivity"}
    title = titles[job["event_type"]]
    room = " ".join(job["room"].split())
    delivery = delivery_info(job)
    message = EmailMessage(policy=SMTP)
    message["From"] = str(Address(display_name="E3 Caregiver Alerts", addr_spec=address(sender).addr_spec))
    message["To"] = str(recipient)
    prefix = "E3 delayed alert" if delivery["is_delayed"] else "E3 alert"
    message["Subject"] = f"{prefix}: {title} — {room}"
    message["Message-ID"] = f"<{job['notification_id']}@{address(sender).domain}>"
    message["Date"] = format_datetime(job["queued_at"])
    link = f"{dashboard_url()}/history?{urlencode({'device_id': job['device_id']})}"
    delayed_notice = (
        "DELAYED ALERT — original detection time is shown below.\n"
        f"The device sent this alert about {delivery['delivery_delay_label']} after detection.\n\n"
        if delivery["is_delayed"] else ""
    )
    message.set_content(
        delayed_notice + f"{title} reported in {room}.\n\n"
        f"Device: {job['device_id']}\n"
        f"Detected (UTC): {job['detected_at'].astimezone(UTC).isoformat()}\n"
        f"Received by server (UTC): {job['received_at'].astimezone(UTC).isoformat()}\n\n"
        f"View alerts: {link}\n\n"
        "You received this notification because your account is linked to this device.\n",
        cte="quoted-printable",
    )
    return message


def submit_smtp(message: EmailMessage, recipient: str, settings: SMTPSettings) -> None:
    """Return only after SMTP accepts DATA; never retry a failed QUIT after acceptance."""
    client = None
    try:
        context = ssl.create_default_context()
        if settings.security == "ssl":
            client = smtplib.SMTP_SSL(settings.host, settings.port, timeout=5, context=context)
        else:
            client = smtplib.SMTP(settings.host, settings.port, timeout=5)
            client.ehlo()
            client.starttls(context=context)
        client.ehlo()
        client.login(settings.username, settings.password)
        code, _ = client.mail(settings.sender)
        if code != 250:
            raise smtplib.SMTPResponseException(code, b"sender rejected")
        code, _ = client.rcpt(address(recipient).addr_spec)
        if code not in {250, 251}:
            raise smtplib.SMTPResponseException(code, b"recipient rejected")
        try:
            code, _ = client.data(message.as_bytes())
        except smtplib.SMTPResponseException:
            raise
        except OSError as exc:
            # Losing the reply after DATA may mean the server already accepted it.
            raise DeliveryError("SMTP acceptance was not confirmed; review before retrying",
                                uncertain=True) from exc
        if code != 250:
            raise smtplib.SMTPResponseException(code, b"message rejected")
    except DeliveryError:
        raise
    except smtplib.SMTPResponseException as exc:
        raise DeliveryError(f"SMTP rejected submission ({exc.smtp_code})",
                            retryable=400 <= exc.smtp_code < 500) from exc
    except (smtplib.SMTPNotSupportedError, ssl.SSLCertVerificationError) as exc:
        raise DeliveryError("SMTP TLS or authentication configuration is not supported") from exc
    except OSError as exc:
        raise DeliveryError("SMTP connection failed before message submission", retryable=True) from exc
    finally:
        if client is not None:
            # close() performs no further protocol exchange, so an accepted message
            # cannot be retried merely because the SMTP server drops the connection.
            try:
                client.close()
            except OSError:
                pass
