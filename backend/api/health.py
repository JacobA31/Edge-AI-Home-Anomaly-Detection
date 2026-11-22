"""GET /api/health – lets us (and the dashboard) check the backend is up."""

import json

import azure.functions as func

from notification_email import email_mode
from security import _local_development

bp = func.Blueprint()


@bp.route(route="health", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def health(req: func.HttpRequest) -> func.HttpResponse:
    body = {"status": "ok"}
    if req.params.get("timing") == "1":
        if not _local_development(req):
            return func.HttpResponse(status_code=404)
        # Read-only safety preflight for the local timing helper. No credentials
        # or connection details are returned, and this is unavailable in Azure.
        body["email_mode"] = email_mode()
    return func.HttpResponse(json.dumps(body), mimetype="application/json",
                             headers={"Cache-Control": "no-store"})
