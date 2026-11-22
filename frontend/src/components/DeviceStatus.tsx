import type { Device } from "../api/devices";

function StatusTime({ value }: { value: string | null }) {
  return value ? <time dateTime={value}>{new Date(value).toLocaleString()}</time> : <>Not received yet</>;
}

function timeoutLabel(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  return [minutes ? `${minutes}m` : "", remainder ? `${remainder}s` : ""].filter(Boolean).join(" ");
}

export default function DeviceStatus({ device }: { device: Device }) {
  const connected = device.connection_status === "online";
  const connection = connected ? "Online" : device.connection_status === "offline"
    ? "Offline" : "Awaiting first check-in";
  const monitoring = device.monitoring_state === "active" ? "Active" : device.monitoring_state === "muted"
    ? "Muted" : device.monitoring_updated_at ? "Unknown" : "Not reported";
  const monitoringStyle = connected ? device.monitoring_state : "unknown";

  return (
    <div className="device-status">
      <dl className="device-status-grid">
        <div>
          <dt>Connection</dt>
          <dd><span className={`status-badge status-${device.connection_status}`}>
            <span className="status-dot" aria-hidden="true" />{connection}
          </span></dd>
        </div>
        <div>
          <dt>Monitoring (last reported)</dt>
          <dd><span className={`status-badge status-${monitoringStyle}`}>
            <span className="status-dot" aria-hidden="true" />{monitoring}
          </span></dd>
        </div>
        <div>
          <dt>Last check-in</dt>
          <dd><StatusTime value={device.last_seen_at} /></dd>
        </div>
        {device.monitoring_updated_at && <div>
          <dt>Monitoring reported at</dt>
          <dd><StatusTime value={device.monitoring_updated_at} /></dd>
        </div>}
      </dl>
      <p className="device-status-note">
        {device.connection_status === "never_seen" ? "Waiting for the first message from this device."
          : connected ? "The device has communicated recently."
            : "The device has stopped checking in. Current monitoring cannot be confirmed."}
        {connected && device.monitoring_state === "unknown" && " Monitoring activity has not been confirmed by the device."}
      </p>
      <p className="device-status-note">Marked offline after {timeoutLabel(device.offline_after_seconds)} without a new message.</p>
    </div>
  );
}
