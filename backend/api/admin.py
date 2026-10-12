"""POST /api/admin/migrate – applies new SQL files in db/migrations/.

Called by the backend GitHub workflow right after each deploy. It runs inside Azure,
so the PostgreSQL firewall only needs "Allow access from Azure services".
Protected by a dedicated function key stored in the AZURE_FUNCTIONAPP_MIGRATE_KEY secret.
"""

import json
import logging

import azure.functions as func

from db.migrate import apply_migrations

bp = func.Blueprint()


@bp.route(route="admin/migrate", methods=["POST"], auth_level=func.AuthLevel.FUNCTION)
def migrate(req: func.HttpRequest) -> func.HttpResponse:
    try:
        applied = apply_migrations()
    except Exception:  # report any failure back to the workflow
        logging.exception("Migration failed")
        return func.HttpResponse(json.dumps({"error": "migration failed, see logs"}), status_code=500)
    return func.HttpResponse(json.dumps({"applied": applied}), mimetype="application/json")
