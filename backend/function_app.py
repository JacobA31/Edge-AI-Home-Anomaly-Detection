"""Entry point for the Azure Functions app (Python v2 programming model).

Each file in api/ is a Blueprint. Add new endpoints there and register them below.
"""

import azure.functions as func

from api.admin import bp as admin_bp
from api.auth import bp as auth_bp
from api.consent import bp as consent_bp
from api.device_commands import bp as device_commands_bp
from api.devices import bp as devices_bp
from api.events import bp as events_bp
from api.health import bp as health_bp
from notification_worker import bp as notifications_bp

app = func.FunctionApp()

app.register_functions(health_bp)
app.register_functions(events_bp)
app.register_functions(admin_bp)
app.register_functions(auth_bp)
app.register_functions(devices_bp)
app.register_functions(device_commands_bp)
app.register_functions(consent_bp)
app.register_functions(notifications_bp)
