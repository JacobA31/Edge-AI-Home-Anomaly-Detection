import { useEffect, useState } from "react";
import type { FormEvent } from "react";

import { decideConsent, inspectConsent } from "../api/consent";
import type { ConsentDetails } from "../api/consent";
import ConsentStatus from "./ConsentStatus";

export default function ConsentPage() {
  const [entry, setEntry] = useState("");
  const [code, setCode] = useState("");
  const [details, setDetails] = useState<ConsentDetails | null>(null);
  const [agreed, setAgreed] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");

  useEffect(() => {
    if (!code) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    async function refresh() {
      try {
        const result = await inspectConsent(code);
        if (!cancelled) { setDetails((current) => current && current.consent.version > result.consent.version ? current : result); setError(""); }
      } catch (err) {
        if (!cancelled) {
          setDetails(null);
          setError(err instanceof Error ? err.message : "Could not refresh consent information.");
        }
      } finally { if (!cancelled) timer = setTimeout(refresh, 5000); }
    }
    timer = setTimeout(refresh, 5000);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [code]);

  const reviewKey = details ? `${details.consent.version}:${details.disclosure.version}` : "";

  async function open(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setBusy(true); setError(""); setMessage("");
    try {
      const privateCode = entry.trim();
      const result = await inspectConsent(privateCode);
      setCode(privateCode); setEntry(""); setDetails(result); setAgreed("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not open consent settings.");
    } finally { setBusy(false); }
  }

  async function decide(decision: "grant" | "withdraw") {
    if (!details) return;
    if (decision === "withdraw" && !window.confirm("Withdraw permission to monitoring for this device?")) return;
    setBusy(true); setError(""); setMessage("");
    try {
      setDetails(await decideConsent(code, details, decision)); setAgreed("");
      setMessage(decision === "grant" ? "Your permission has been recorded. Monitoring has not been resumed."
        : "Your permission has been withdrawn. Waiting for the device to confirm stopping.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not record your decision.");
      try { setDetails(await inspectConsent(code)); } catch { setDetails(null); }
      setAgreed("");
    } finally { setBusy(false); }
  }

  function close() {
    setCode(""); setEntry(""); setDetails(null); setAgreed(""); setMessage(""); setError("");
  }

  const manager = details?.caregivers.find((caregiver) => caregiver.role === "trusted_caregiver");
  const awaitingStop = details?.consent.state === "withdrawn" && !details.consent.device_confirmed;
  return <>
    <a className="button-link back-link" href="/">← Caregiver login / dashboard</a>
    <section className="account-card consent-page" aria-labelledby="consent-title">
      <h2 id="consent-title">Monitoring consent</h2>
      <p>For the person being monitored. You do not need a caregiver account.</p>
      {error && <p className="error" role="alert">{error}</p>}
      {message && <p role="status">{message}</p>}
      {!code ? <form onSubmit={open}>
        <p>Enter the private consent code provided during device setup. Your caregiver invitation or device linking code will not work here.</p>
        <fieldset disabled={busy}>
          <label>Private consent code
            <input type="password" autoComplete="off" value={entry} required maxLength={100}
              onChange={(event) => setEntry(event.target.value)} />
          </label>
          <button type="submit">Open consent settings</button>
        </fieldset>
      </form> : <>
        <button className="secondary" onClick={close} disabled={busy}>Close private consent settings</button>
        {details && <>
          <h3>{details.room}</h3>
          <p className="alert-device">Device: {details.device_id}</p>
          <ConsentStatus consent={details.consent} />
          <h3>What you are agreeing to</h3>
          {details.disclosure.paragraphs.map((paragraph, index) => <p key={index}>{paragraph}</p>)}
          <h3>Caregivers with access</h3>
          {details.caregivers.length ? <ul>
            {details.caregivers.map((caregiver) => <li key={caregiver.email}>
              {caregiver.display_name} ({caregiver.email}) — {caregiver.role === "trusted_caregiver"
                ? "Initial trusted caregiver" : "Read-only caregiver"}
            </li>)}
          </ul> : <p>The initial caregiver must link their account before you can give permission.</p>}
          {details.consent.state !== "granted" && <fieldset disabled={busy || !manager || awaitingStop}>
            {awaitingStop && <p>Wait for the device to confirm stopping before giving permission again.</p>}
            <label className="consent-check">
              <input type="checkbox" checked={agreed === reviewKey} onChange={(event) => setAgreed(event.target.checked ? reviewKey : "")} />
              <span>I am the person being monitored. I understand the explanation above, agree to acoustic monitoring,
                and authorize the initial trusted caregiver shown here to manage access.</span>
            </label>
            <button type="button" disabled={!reviewKey || agreed !== reviewKey} onClick={() => void decide("grant")}>Give permission to monitoring</button>
          </fieldset>}
          {details.consent.state !== "withdrawn" && <button type="button" className="secondary"
            disabled={busy} onClick={() => void decide("withdraw")}>Withdraw permission</button>}
          {details.history.length > 0 && <details>
            <summary>Recent consent decisions</summary>
            <ul>{details.history.map((record) => <li key={record.recorded_at}>
              {record.decision === "granted" ? "Permission granted" : "Permission withdrawn"} — {new Date(record.recorded_at).toLocaleString()}
            </li>)}</ul>
          </details>}
          <p className="device-status-note">Explanation version: {details.disclosure.version}. The code is kept only in this page’s memory;
            closing these settings or reloading the page clears it.</p>
        </>}
      </>}
    </section>
  </>;
}
