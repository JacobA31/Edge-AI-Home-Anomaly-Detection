import { useState } from "react";

import { ApiError } from "../api/client";
import { requestDeviceCommand } from "../api/devices";
import type { Device, DeviceCommand } from "../api/devices";

export default function DeviceControls({ device, onChanged, onSignedOut }: {
  device: Device; onChanged: () => void; onSignedOut: (message: string) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [submitted, setSubmitted] = useState<DeviceCommand | null>(null);
  // Keep the immediate POST response until the dashboard poll catches up.
  // Afterwards the server remains authoritative, including completion/expiry.
  const command = submitted && (!device.latest_command
    || new Date(submitted.requested_at) > new Date(device.latest_command.requested_at))
    ? submitted : device.latest_command;
  const outstanding = command?.status === "pending" || command?.status === "delivered";
  const manager = device.role === "trusted_caregiver";

  async function request(action: "mute" | "resume") {
    if (action === "mute" && !window.confirm(`Request that ${device.room} stop monitoring audio?`)) return;
    setBusy(true); setError("");
    try {
      setSubmitted(await requestDeviceCommand(device.device_id, action));
      onChanged();
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) onSignedOut("Please log in again.");
      else setError(err instanceof Error ? err.message : "Could not request the change.");
      onChanged();
    } finally { setBusy(false); }
  }

  return <div className="device-controls">
    <h4>Remote monitoring</h4>
    {command && <p role="status" className="device-command-status">
      <strong>{command.action === "mute" ? "Mute" : "Resume"} request: </strong>
      {command.status === "pending" && "Pending — waiting for the device to collect it."}
      {command.status === "delivered" && "Received by device — awaiting confirmation."}
      {command.status === "completed" && "Device confirmed success."}
      {command.status === "failed" && "Device reported failure. Check the device before trying again."}
      {command.status === "expired" && "Expired without being collected. No change was confirmed."}
    </p>}
    {manager && <>
      <div className="device-control-actions">
        <button type="button" className="secondary" disabled={busy || outstanding || !device.controls_configured}
          onClick={() => void request("mute")}>Mute monitoring</button>
        <button type="button" className="secondary" disabled={busy || outstanding || !device.controls_configured}
          onClick={() => void request("resume")}>Resume monitoring</button>
      </div>
      <p className="device-status-note">
        {!device.controls_configured ? "Remote controls need to be configured during device setup."
          : outstanding ? "Monitoring changes only when the device applies the request and confirms it."
            : "Only the trusted caregiver can request changes. Uncollected requests expire after 10 minutes."}
      </p>
      {outstanding && device.connection_status !== "online"
        && <p className="device-status-note">The device is offline. A remote request cannot change it while disconnected.</p>}
      {error && <p className="error" role="alert">{error}</p>}
    </>}
  </div>;
}
