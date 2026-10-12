"""Entry point for the Azure Functions app (Python v2 programming model).

Each file in api/ is a Blueprint. Add new endpoints there and register them below.
"""

import azure.functions as func

from api.admin import bp as admin_bp
from api.events import bp as events_bp
from api.health import bp as health_bp

app = func.FunctionApp()

app.register_functions(health_bp)
app.register_functions(events_bp)
app.register_functions(admin_bp)
