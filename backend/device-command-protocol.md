# Remote monitoring contract for the Pi

The cloud stores requests and shows their progress. Matthew's edge integration needs to
apply them to the actual capture/monitoring pipeline. Collecting a command alone
does not mean that monitoring changed. No edge execution code is added by this feature.

## Provisioning

Apply migrations through `007_monitoring_consent` first. From `backend/`, with the virtual
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

## Resident consent setup

The database operator also runs:

```sh
python3 -m consent provision pi-livingroom-01
```

Give this separate, long-lived consent code privately to the person being monitored,
not to the Pi or additional caregivers. It opens `/consent` without an account and
does not grant access to alerts or device commands. Codes never go in URLs or browser
storage. Anyone with the code can change consent; this prototype does not verify
the holder's identity. Reissuing a code invalidates the old one without changing consent.

The initial caregiver links their account first. The resident reviews the disclosure,
the named initial trusted caregiver and current additional caregivers, and explicitly
agrees. Consent is never inferred from existing memberships or recordings. The
operator must configure the same `AUTH_ALLOWED_ORIGINS` as the caregiver dashboard.

Withdrawal preserves existing history but blocks new monitoring alerts, new caregiver
invitations, and remote resume requests. Giving permission again does not resume
monitoring. After withdrawal, a new grant waits for the Pi to confirm stopping.

## Collect, apply, confirm

Poll every five seconds while connected, with backoff during network failures:

```http
POST /api/device-control/{device_id}/next
Authorization: Bearer <device-control-token>
```

Response: `{"command": null, "consent": {...}}` or both objects populated. Always
process `consent` first, including when there is no ordinary command. It contains
`state` (`not_recorded`, `granted`, `withdrawn`), monotonic `version`,
`monitoring_allowed`, `device_confirmed`, and server timestamps.

Persist this policy on the Pi. Start with monitoring stopped if no valid permission
is known. When permission is absent/withdrawn, stop microphone capture, analysis,
baseline updates, recording and monitoring-data collection. Discard monitoring
events queued under the invalid permission; keep only minimal control/health traffic.
Do not let a resume command override consent or a local privacy stop. Granting
permission merely permits a later resume; it is not itself a request to resume.

After applying the policy, report its current version and actual monitoring state:

```http
POST /api/device-control/{device_id}/consent/ack
Authorization: Bearer <device-control-token>
Content-Type: application/json

{"version": 1, "monitoring_state": "muted", "elapsed_seconds": 0.2}
```

An absent/withdrawn policy can be confirmed only after monitoring has really stopped
(`muted`); failure leaves it unconfirmed. For a granted policy, report the actual
`muted` or `active` state without resuming just to acknowledge it. Retry lost replies
with the same version and increasing execution age. Stale versions return 409:
collect the current policy and obey it. A policy confirmation does not generate an alert email.

A command has:

- `command_id`: UUID used for retries and local deduplication.
- `action`: `mute` or `resume` (set the state; never toggle it).
- `status`: `delivered` when returned by pickup.
- `consent_version`: the policy version under which it was requested.
- `requested_at`, `expires_at`, `delivered_at`, `finished_at`: server timestamps.

Uncollected requests expire after ten minutes. Once delivered, the request stays
outstanding until acknowledged, even after `expires_at`. Do not discard a delivered
request based on that timestamp. Repeated pickup returns the same ID. The cloud
does not deliver an opposite request until this one has a result, except that a
consent change immediately cancels outstanding commands, including delivered ones.
Cancelled commands must not be executed or retried as completed commands.

Persist the command ID and its result locally before acknowledging. Replayed IDs
must replay the saved acknowledgement instead of executing again, including after
a Pi restart. `mute` must stop acoustic monitoring; `resume` must restore it only
when local consent and other device safety conditions allow it. If the requested
state is already applied, it can succeed without toggling anything. Keep the control
poller and heartbeats running while muted so resume remains possible.

Before executing resume, recheck the latest permission and match `consent_version`.
Confirm the current policy before acknowledging resume success. The server rejects
stale/cancelled resume results, but cannot undo a physical action already taken by
an incorrectly integrated Pi. Consent checks must therefore also run at the actual
point of capture/resume. Synchronize policy processing and pipeline changes locally.

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
not execute a new command until the previous acknowledgement succeeds or the
server cancels it. On 409, collect the policy again; do not blindly retry a stale result.
If a delivered request is stuck, repair/reconnect the Pi and report its real result;
the dashboard intentionally does not guess whether it was applied.

The dashboard's latest command outcome describes that request; the separate
“Monitoring (last reported)” field reflects the latest available device report.
Subsequent local changes can differ from the outcome of an older command.

## Events and offline behavior

`POST /api/events` still uses the existing event function key. It rejects new fall,
anomaly and inactivity alerts with 403 while permission is absent/withdrawn, without
storing the alert or queuing email. Discard these permanent rejections rather than
retrying forever. An already saved event ID still returns duplicate without storing
anything again. Queued alerts whose execution age predates the current grant are
also rejected, so retain accurate `elapsed_seconds` across retries/restarts. Existing
history becomes visible after initial consent and remains available after withdrawal.

Minimal heartbeat/mute status can still be received while blocked; arbitrary details
and scores are stripped. An explicit active report while permission is absent invalidates
the last stop confirmation. Send the actual state rather than claiming a stop from receipt alone.

An offline Pi cannot learn about a website withdrawal immediately. The dashboard
therefore distinguishes server permission from device confirmation. Its local privacy
control must stop monitoring without internet and retain that local stop across
restarts. A persisted local withdrawal/stop must not be overridden by a remote resume.
On reconnect, apply current consent before sending queued monitoring data or executing commands.

## Local cloud-only check

Request mute/resume in the dashboard, then run from `backend/`:

```sh
python3 -m device_commands simulate pi-livingroom-01
```

Paste the token at its hidden prompt. Pickup leaves the dashboard awaiting
confirmation. Enter `success` or `failure` to simulate the result, or `leave` to
exercise pickup/retry without confirming. This helper contacts only localhost and
does **not** operate a microphone or a real Pi.

The simulator now handles consent first: missing/withdrawn permission prompts for
a simulated stop; enter `success` only to simulate that result. Granted permission
is synchronized while remaining muted. Ordinary mute/resume requests are then
handled as above. There is no need to reissue the control token for consent checks.
