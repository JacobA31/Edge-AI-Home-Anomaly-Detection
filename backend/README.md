# backend/

Azure Functions app (Python, v2 programming model) on the **Flex Consumption** plan.
Deployed by `.github/workflows/backend.yml` on every merge to `main` that touches `backend/` or `shared/`.

```
backend/
├── function_app.py          registers the blueprints below
├── api/
│   ├── health.py            GET  /api/health          (public)
│   ├── events.py            POST /api/events          (Pi, needs function key)
│   │                        GET  /api/events          (dashboard)
│   └── admin.py             POST /api/admin/migrate   (CI only, needs function key)
├── validation.py            checks events against shared/event.schema.json
├── db/
│   ├── connection.py        reads DATABASE_URL
│   ├── migrate.py           applies migrations/NNN_*.sql once each
│   └── migrations/          001_init.sql, 002_..., never edit merged files
├── tests/
├── host.json
├── requirements.txt         what Azure installs
└── requirements-dev.txt     + pytest, ruff for local/CI
```

## Run locally

Needs Python 3.11, [Azure Functions Core Tools v4](https://learn.microsoft.com/azure/azure-functions/functions-run-local),
and (for anything touching the DB) a local PostgreSQL, e.g. `docker run -p 5432:5432 -e POSTGRES_PASSWORD=dev postgres:16`.

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp local.settings.example.json local.settings.json   # then edit DATABASE_URL
python -m db.migrate                                  # create tables locally
func start                                            # http://localhost:7071/api/health
pytest                                                # tests
```

Send a test event:

```bash
curl -X POST http://localhost:7071/api/events -H "Content-Type: application/json" -d @../shared/sample-event.json
```
