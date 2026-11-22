# Remote monitoring contract for the Pi

The cloud stores requests and shows their progress. Matthew's edge integration needs to
apply them to the actual capture/monitoring pipeline. Collecting a command alone
does not mean that monitoring changed. No edge execution code is added by this feature.

## Provisioning

Apply migration `006_device_commands` first. From `backend/`, with the virtual
environment active, the database operator runs:

```sh
python3 -m device_commands provision pi-livingroom-01
```

Save the printed control token privately on that specific Pi. Only its SHA-256
hash is stored by the server. Re-running this command rotates the token, immediately
invalidating the previous one, while preserving outstanding requests. This token
is separate from the caregiver linking code and the Azure `post_event` function key.
Never include it in frontend code, commits, telemetry, or URLs.

The routes below use `Authorization: Bearer <device-control-token>`. They do not
require an Azure function key: application authentication scopes the token to the
device ID in the route. A cookie or another device's token grants no access.
Use HTTPS in Azure; HTTP is only for loopback local development.

## Collect, apply, confirm

Poll every five seconds while connected, with backoff during network failures:

```http
POST /api/device-control/{device_id}/next
Authorization: Bearer <device-control-token>
```

Response: `{"command": null}` or `{"command": {...}}`. A command has:

- `command_id`: UUID used for retries and local deduplication.
- `action`: `mute` or `resume` (set the state; never toggle it).
- `status`: `delivered` when returned by pickup.
- `requested_at`, `expires_at`, `delivered_at`, `finished_at`: server timestamps.

Uncollected requests expire after ten minutes. Once delivered, the request stays
outstanding until acknowledged, even after `expires_at`. Do not discard a delivered
request based on that timestamp. Repeated pickup returns the same ID. The cloud
does not deliver an opposite request until this one has a result.

Persist the command ID and its result locally before acknowledging. Replayed IDs
must replay the saved acknowledgement instead of executing again, including after
a Pi restart. `mute` must stop acoustic monitoring; `resume` must restore it only
when local consent and other device safety conditions allow it. If the requested
state is already applied, it can succeed without toggling anything. Keep the control
poller and heartbeats running while muted so resume remains possible.

After applying the request, acknowledge it:

```http
POST /api/device-control/{device_id}/commands/{command_id}/ack
Authorization: Bearer <device-control-token>
Content-Type: application/json

{"succeeded": true, "elapsed_seconds": 0.2}
```

`elapsed_seconds` is the age of the execution result when sending this attempt,
not a counter reset for each HTTP retry. Preserve/increase that age on retries and
across restarts so delayed confirmations cannot overwrite newer monitoring reports.
Use `succeeded: false` when execution fails; never report success merely on receipt.
The cloud marks failed monitoring state unknown unless a newer explicit report
already exists. The Pi should also send its normal monitoring heartbeat/transition
event with the actual state, including after failures or local manual changes.

Retry a lost acknowledgement response with the same ID and outcome. It is safe and
does not apply state twice. An opposite outcome for a finished ID returns 409. Do
not collect/execute a new command until the previous acknowledgement succeeds.
If a delivered request is stuck, repair/reconnect the Pi and report its real result;
the dashboard intentionally does not guess whether it was applied.

The dashboard's latest command outcome describes that request; the separate
“Monitoring (last reported)” field reflects the latest available device report.
Subsequent local changes can differ from the outcome of an older command.

## Local cloud-only check

Request mute/resume in the dashboard, then run from `backend/`:

```sh
python3 -m device_commands simulate pi-livingroom-01
```

Paste the token at its hidden prompt. Pickup leaves the dashboard awaiting
confirmation. Enter `success` or `failure` to simulate the result, or `leave` to
exercise pickup/retry without confirming. This helper contacts only localhost and
does **not** operate a microphone or a real Pi.
