import type { ConsentPolicy } from "../api/consent";

export default function ConsentStatus({ consent }: { consent: ConsentPolicy }) {
  const granted = consent.state === "granted";
  return <div className={`consent-status ${granted ? "consent-granted" : "consent-blocked"}`}>
    <p><strong>Monitoring consent: </strong>{granted ? "Granted"
      : consent.state === "withdrawn" ? "Withdrawn" : "Awaiting consent"}</p>
    {consent.state !== "not_recorded" && <p className="device-status-note">
      Updated <time dateTime={consent.updated_at}>{new Date(consent.updated_at).toLocaleString()}</time>
    </p>}
    <p className="device-status-note" role="status">
      {granted ? consent.device_confirmed
        ? "The device confirmed this permission. Monitoring follows its separate mute/resume setting."
        : "Permission recorded; awaiting device confirmation. Giving permission does not resume monitoring."
        : consent.device_confirmed
          ? "The device confirmed monitoring has stopped. Resume and new monitoring alerts are blocked."
          : "Resume and new monitoring alerts are blocked. Stopping on the device is not yet confirmed; if offline, use its local control."}
    </p>
  </div>;
}
