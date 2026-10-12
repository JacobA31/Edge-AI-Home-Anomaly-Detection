# shared/

Files used by more than one part of the system. Changing anything here re-runs the
backend, frontend and edge workflows, so a broken contract is caught before merge.

| File | Used by |
|------|---------|
| `event.schema.json` | Pi (builds events), backend (validates incoming events), frontend (types) |
| `sample-event.json` | Tests and local manual testing (`curl -d @shared/sample-event.json …`) |

The backend workflow copies this folder into `backend/shared/` before deploying, because
Azure only receives the `backend/` folder.
