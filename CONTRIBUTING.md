# How we work

## Branches

- `main` is always deployable. Nobody pushes to it directly; everything goes through a pull request.
- Make a branch per task, named `<area>/<short-description>`:
  `backend/events-endpoint`, `frontend/alert-banner`, `edge/audio-capture`, `ml/yamnet-baseline`, `docs/setup`.
- Keep a branch to one folder when you can. A PR that only touches `frontend/` only runs the frontend checks, so it's faster to review and merge.

## Day-to-day

```bash
git switch main && git pull                 # start from the latest main
git switch -c backend/events-endpoint       # new branch
# ...work...
git add -A && git commit -m "backend: validate events against shared schema"
git push -u origin backend/events-endpoint  # then open a PR on GitHub
```

## Pull requests

1. Open the PR into `main` and fill in the template.
2. Wait for the checks for your folder to pass (green tick).
3. Get one approval from a teammate (CODEOWNERS requests the right person automatically).
4. **Squash and merge**, then delete the branch. The merge triggers the deploy.

## Commit messages

Start with the folder: `backend: …`, `frontend: …`, `edge: …`, `ml: …`, `shared: …`, `docs: …`, `ci: …`.

## Changing the event format

The event format lives in `shared/event.schema.json` and is used by the Pi, the backend and the
dashboard. If you change it, update all three in the **same PR**, and add a DB migration in
`backend/db/migrations/` if a column changes.

## Database changes

Never edit an existing migration after it's merged. Add a new file with the next number,
e.g. `002_add_device_location.sql`. Migrations run automatically after the backend deploys.

## Secrets

Never commit keys, connection strings or publish profiles. Use `backend/local.settings.json`,
`frontend/.env.local` and `edge/config.yaml` locally (all git-ignored) and GitHub Secrets in CI.
