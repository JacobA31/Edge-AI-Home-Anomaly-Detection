# Local cloud timing check

Run this with PostgreSQL, Azurite, the backend and `npm run dev` already running.
Use the backend virtual environment. The backend must be in local Development mode
with `EMAIL_MODE=preview`, and its database must be on this computer. No Azure
resource or real Pi is exercised, and the helper refuses SMTP mode.

## Run

From `backend/`:

```sh
python3 -m verify_timing run
```

1. The helper prints a dashboard URL and credentials for a temporary fictional
   caregiver/device. Open that exact URL, log in with those credentials, and wait
   for the **Local timing check** panel to say **Ready**. Do not use your normal account.
2. Press Enter in the terminal to begin. Return to the dashboard immediately and
   keep it as the active tab. Do not refresh it, navigate away or switch accounts.
3. Five synthetic alerts are created at different offsets relative to the normal
   dashboard poll and notification timer. Wait about 25 seconds. A missing/stopped
   worker may make the check take about a minute and produce a failure.
4. Once the dashboard has recorded five timings, click **Download browser timings**.
   The terminal prints the matching backend report path under `reports/timing/`.
5. After saving the browser file, press Enter in the terminal again. The helper
   deletes only its temporary account, device, alerts, sessions and notification jobs.
6. Combine the two matching files:

```sh
python3 -m verify_timing report \
  --backend-file reports/timing/run-RUN_ID.json \
  --browser-file /path/to/browser-timing-RUN_ID.json
```

The combined report is saved beside the backend report as `summary-RUN_ID.json`.
Generated backend reports are git-ignored and excluded from deployment. The files
contain synthetic IDs and measurements, not login passwords, consent codes or sessions.

Use `--samples 1` for a quick single-alert check. Use `--backend-url` and
`--frontend-url` only for alternate HTTP loopback ports; remote sites are refused.

## What the measurements mean

- **Dashboard render (10 seconds):** time from the helper creating an alert to an
  after-paint observation of the actual alert cards on the active development
  dashboard. A development-only probe runs after two animation frames. This is a
  conservative render bound, not an exact timestamp of the first visible pixel.
  The panel/probe is absent from production builds.
- **API availability:** a separate polling observation. It does not prove the
  dashboard displayed the alert and cannot replace the browser measurements.
- **Notification preview (30 seconds):** creation to completion of the real
  notification worker's preview path. The helper never manually runs the worker
  to make a slow timer pass. Preview completion is not SMTP acceptance.
- **Email-service submission:** explicitly **NOT_MEASURED** in this local check.
  It requires real SMTP configuration and submission evidence in a later run.
  Inbox arrival is a separate measure outside the 30-second submission target.
- **Full Pi-to-cloud targets:** explicitly **NOT_VERIFIED**. Real device/network
  behavior and deployed Azure behavior still need integration measurements.

The backend and browser reports must have the same run and event IDs. Missing
samples, a hidden tab, inconsistent clocks, rejected alerts or exceeded budgets
cannot produce a dashboard PASS. The helper checks the local database clock;
measurements from unsynchronized computers must not be compared this way.

## Interruptions

Ctrl+C performs cleanup. If the process is killed forcibly, use the report/manifest
saved at the start of that run:

```sh
python3 -m verify_timing cleanup --backend-file reports/timing/run-RUN_ID.json
```

Cleanup validates the exact synthetic run/account/device identifiers before deleting
anything. Reports remain available after their database fixtures have been removed.
No existing household records, consent settings or device tokens are changed.
