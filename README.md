# E3 – Edge-AI Home Anomaly Detection for Elder Care

SYSC 4907 capstone (Fall 2026 – Winter 2027), supervised by Dr. Rose Gomar.

One repository holds every part of the system. Each top-level folder has its own
GitHub Actions workflow and only runs when files in that folder change.

```
.
├── backend/     Azure Functions (Python). Receives events from the Pi, stores them in
│                PostgreSQL, serves the dashboard API. DB migrations live in backend/db/.
│                → deploys to Azure Functions on merge to main
├── frontend/    React + TypeScript (Vite) caregiver dashboard
│                → deploys to Azure Static Web Apps on merge to main
├── edge/        Code that runs on the Raspberry Pi 5 (capture, inference, baseline,
│                event queue). → tested in CI; installed on the Pi with git pull
├── ml/          Training notebooks and scripts (YAMNet + classifier, baseline experiments)
│                → lint only, never deployed
├── shared/      Things more than one part needs, e.g. the event JSON schema
│                → changing it re-runs backend, frontend and edge checks
├── docs/        Architecture notes, setup guide, decision records
└── scripts/     Helper scripts for the whole repo
```

## What runs when

| You change…           | On a pull request           | On merge to `main`                      |
|-----------------------|-----------------------------|-----------------------------------------|
| `backend/**`          | tests                       | tests → deploy Functions → run DB migrations |
| `frontend/**`         | build                       | build → deploy Static Web App           |
| `edge/**`             | tests                       | tests (Pi pulls manually)               |
| `ml/**`               | lint                        | lint                                    |
| `shared/**`           | backend + frontend + edge   | backend + frontend deploy, edge tests   |
| `docs/**`             | nothing                     | nothing                                 |

Every workflow can also be started by hand from the **Actions** tab (`workflow_dispatch`).

## Getting started

First-time GitHub and Azure setup: **[docs/SETUP.md](docs/SETUP.md)**.
How we work (branches, PRs, commits): **[CONTRIBUTING.md](CONTRIBUTING.md)**.

Run each part locally:

| Part      | Commands |
|-----------|----------|
| backend   | `cd backend && python -m venv .venv && source .venv/bin/activate && pip install -r requirements-dev.txt && func start` |
| frontend  | `cd frontend && npm install && npm run dev` |
| edge      | `cd edge && python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]" && python -m e3_edge` |
| ml        | `cd ml && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && jupyter lab` |

## Privacy rule for this repo

Raw audio never goes into Git (the proposal says raw audio is never stored or
transmitted). `.gitignore` blocks audio files and `ml/data/`; datasets and recordings
live in the team's shared drive. Never commit keys or connection strings: use
`local.settings.json` / `.env` (both git-ignored) locally and GitHub Secrets in CI.

## Team

| Area | Folder | Owner |
|------|--------|-------|
| Hardware & edge integration | `edge/` | Member 1 |
| Data collection & preprocessing | `ml/` | Member 2 |
| Fall detection | `ml/`, `edge/` | Member 3 |
| Behavioural anomaly detection | `ml/`, `edge/` | Member 4 |
| Alerting & monitoring interface | `backend/`, `frontend/` | Member 5 |
