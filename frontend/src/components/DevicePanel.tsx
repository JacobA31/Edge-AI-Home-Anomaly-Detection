import { useEffect, useState } from "react";
import type { FormEvent } from "react";

import { ApiError } from "../api/client";
import { acceptInvitation, claimDevice, fetchAccess, inviteCaregiver,
  removeCaregiver, revokeInvitation } from "../api/devices";
import type { Device, DeviceAccess } from "../api/devices";

interface Props {
  devices: Device[];
  onChanged: () => void;
  onSignedOut: (message: string) => void;
}

export default function DevicePanel({ devices, onChanged, onSignedOut }: Props) {
  const [linkingCode, setLinkingCode] = useState("");
  const [invitationCode, setInvitationCode] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [managing, setManaging] = useState<string | null>(null);

  async function link(event: FormEvent<HTMLFormElement>, invitation: boolean) {
    event.preventDefault();
    setBusy(true); setError(""); setMessage("");
    try {
      if (invitation) {
        await acceptInvitation(invitationCode.trim()); setInvitationCode("");
        setMessage("Invitation accepted. You now have read-only access to this device.");
      } else {
        await claimDevice(linkingCode.trim()); setLinkingCode("");
        setMessage("Device linked. You are its trusted caregiver and can manage access.");
      }
      onChanged();
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) onSignedOut("Please log in again.");
      else setError(err instanceof Error ? err.message : "Could not link the device.");
    } finally { setBusy(false); }
  }

  return (
    <section className="account-card device-section" aria-labelledby="devices-title">
      <h2 id="devices-title">Your devices</h2>
      {message && <p role="status">{message}</p>}
      {error && <p className="error" role="alert">{error}</p>}
      {devices.length === 0 && <p>No devices linked yet. Link your Pi or accept a caregiver invitation to see events.</p>}
      {devices.map((device) => (
        <article className="device-card" key={device.device_id}>
          <h3>{device.room}</h3>
          <p>{device.device_id}</p>
          <p>{device.role === "trusted_caregiver" ? "Trusted caregiver — you manage access."
            : "Read-only caregiver — you can view events and device information."}</p>
          {device.role === "trusted_caregiver" && (
            <>
              <button className="secondary" onClick={() =>
                setManaging(managing === device.device_id ? null : device.device_id)}>
                {managing === device.device_id ? "Close access settings" : "Manage access"}
              </button>
              {managing === device.device_id && <AccessManager deviceId={device.device_id}
                onChanged={onChanged} onSignedOut={onSignedOut} />}
            </>
          )}
        </article>
      ))}
      <details>
        <summary>Link a device as its first caregiver</summary>
        <p>Use the private, one-time linking code provided during device setup. A device name alone does not grant access.</p>
        <form onSubmit={(event) => link(event, false)}>
          <fieldset disabled={busy}>
            <label>Device linking code
              <input type="password" autoComplete="off" value={linkingCode}
                onChange={(event) => setLinkingCode(event.target.value)} required maxLength={100} />
            </label>
            <button type="submit">Link device</button>
          </fieldset>
        </form>
      </details>
      <details>
        <summary>Accept a caregiver invitation</summary>
        <p>Log in with the email address the invitation was made for, then enter the code shared with you.</p>
        <form onSubmit={(event) => link(event, true)}>
          <fieldset disabled={busy}>
            <label>Caregiver invitation code
              <input type="password" autoComplete="off" value={invitationCode}
                onChange={(event) => setInvitationCode(event.target.value)} required maxLength={100} />
            </label>
            <button type="submit">Accept invitation</button>
          </fieldset>
        </form>
      </details>
    </section>
  );
}

function AccessManager({ deviceId, onChanged, onSignedOut }: {
  deviceId: string; onChanged: () => void; onSignedOut: (message: string) => void;
}) {
  const [access, setAccess] = useState<DeviceAccess | null>(null);
  const [email, setEmail] = useState("");
  const [issued, setIssued] = useState<{ email: string; code: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [revision, setRevision] = useState(0);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    async function load() {
      try {
        const result = await fetchAccess(deviceId);
        if (!cancelled) setAccess(result);
      } catch (err) {
        if (cancelled) return;
        setAccess(null);
        if (err instanceof ApiError && err.status === 401) onSignedOut("Please log in again.");
        else setError(err instanceof Error ? err.message : "Could not load access settings.");
      } finally {
        if (!cancelled) timer = setTimeout(load, 5000);
      }
    }
    void load();
    return () => { cancelled = true; clearTimeout(timer); };
  }, [deviceId, revision, onSignedOut]);

  async function run(action: () => Promise<void>) {
    setBusy(true); setError(""); setMessage("");
    try {
      await action();
      setRevision((value) => value + 1); onChanged();
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) onSignedOut("Please log in again.");
      else setError(err instanceof Error ? err.message : "Could not change access.");
    } finally { setBusy(false); }
  }

  function invite(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void run(async () => {
      const code = await inviteCaregiver(deviceId, email);
      setIssued({ email: email.trim().toLowerCase(), code }); setEmail("");
    });
  }

  return (
    <div className="access-settings">
      <h4>Caregiver access</h4>
      {error && <p className="error" role="alert">{error}</p>}
      {message && <p role="status">{message}</p>}
      {!access && !error && <p role="status">Loading access settings…</p>}
      {access && <>
        <ul className="access-list">
          {access.members.map((member) => (
            <li key={member.account_id}>
              <span>{member.display_name} ({member.email}) — {member.role === "trusted_caregiver"
                ? "Trusted caregiver" : "Read-only caregiver"}</span>
              {member.role === "caregiver" && <button className="secondary" disabled={busy} onClick={() => {
                if (!window.confirm(`Remove ${member.email}'s access to this device and its events?`)) return;
                void run(async () => {
                  await removeCaregiver(deviceId, member.account_id); setIssued(null);
                  setMessage("Caregiver access removed.");
                });
              }}>Remove access</button>}
            </li>
          ))}
        </ul>
        <form onSubmit={invite}>
          <fieldset disabled={busy}>
            <label>Caregiver email to invite
              <input type="email" value={email} onChange={(event) => setEmail(event.target.value)}
                required maxLength={254} />
            </label>
            <p>Invitations grant read-only access. A new invitation to the same email replaces its previous code.</p>
            <button type="submit">Create invitation</button>
          </fieldset>
        </form>
        {issued && !access.members.some((member) => member.email === issued.email)
          && <div className="invitation-code" role="status">
          <p>Share this code privately with <strong>{issued.email}</strong>. It expires in three days and works once.</p>
          <label>Invitation code to share
            <input readOnly value={issued.code} onFocus={(event) => event.target.select()} />
          </label>
          <p>The caregiver can create an account, log in with that email, and choose “Accept a caregiver invitation”.</p>
        </div>}
        {access.invitations.length > 0 && <>
          <h4>Pending invitations</h4>
          <ul className="access-list">
            {access.invitations.map((invitation) => <li key={invitation.invitation_id}>
              <span>{invitation.email} — expires {new Date(invitation.expires_at).toLocaleString()}</span>
              <button disabled={busy} className="secondary" onClick={() => void run(async () => {
                await revokeInvitation(deviceId, invitation.invitation_id); setIssued(null);
                setMessage("Invitation cancelled.");
              })}>Cancel invitation</button>
            </li>)}
          </ul>
        </>}
      </>}
    </div>
  );
}
