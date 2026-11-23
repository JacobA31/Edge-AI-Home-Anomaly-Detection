import { useEffect, useRef, useState } from "react";

import type { RecordedEvent } from "../types/event";

interface Observation {
  event_id: string;
  detected_at: string;
  observed_at: string;
  seconds: number;
  foreground: boolean;
  clock_consistent: boolean;
}

// This probe is rendered only by Vite development builds with an explicit run ID.
// It measures the actual alert render, never a separate API polling request.
export default function AlertTimingProbe({ events }: { events: RecordedEvent[] }) {
  const [runId] = useState(() => new URLSearchParams(window.location.search).get("timing_run") ?? "");
  const [observations, setObservations] = useState<Observation[]>([]);
  const [showJson, setShowJson] = useState(false);
  const seen = useRef(new Set<string>());
  const validRun = /^[0-9a-f]{8}-[0-9a-f-]{27}$/.test(runId);

  useEffect(() => {
    if (!validRun) return;
    const fresh = events.filter((event) => event.details?.timing_run_id === runId && !seen.current.has(event.event_id));
    if (!fresh.length) return;
    let secondFrame = 0;
    // Two animation frames let the committed alert cards reach a paint. This is
    // a conservative after-paint bound, not a claim to a precise pixel timestamp.
    const firstFrame = requestAnimationFrame(() => {
      secondFrame = requestAnimationFrame(() => {
        const now = Date.now();
        const added = fresh.filter((event) => !seen.current.has(event.event_id)).map((event) => {
          seen.current.add(event.event_id);
          return {
            event_id: event.event_id, detected_at: event.detected_at,
            observed_at: new Date(now).toISOString(), seconds: (now - Date.parse(event.detected_at)) / 1000,
            foreground: document.visibilityState === "visible",
            clock_consistent: Math.abs(now - (performance.timeOrigin + performance.now())) < 250,
          };
        });
        setObservations((previous) => [...previous, ...added]);
      });
    });
    return () => { cancelAnimationFrame(firstFrame); cancelAnimationFrame(secondFrame); };
  }, [events, runId, validRun]);

  if (!validRun) return null;
  const result = { kind: "e3-browser-alert-timing", schema_version: 1, run_id: runId, observations };
  const json = JSON.stringify(result, null, 2);

  function download() {
    const url = URL.createObjectURL(new Blob([json], { type: "application/json" }));
    const link = document.createElement("a");
    link.href = url; link.download = `browser-timing-${runId}.json`; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  return <section className="account-card" aria-label="Local timing check">
    <h2>Local timing check</h2>
    <p>Ready — return to the terminal and start the test. Keep this dashboard tab active and do not refresh it.</p>
    <p>This measures synthetic alerts on this computer. Production builds do not include this panel.</p>
    <p role="status">Recorded {observations.length} dashboard alert timings.</p>
    <ul>{observations.map((item) => <li key={item.event_id}>
      {item.event_id.slice(0, 8)}: {item.seconds.toFixed(3)} seconds — {!item.foreground || !item.clock_consistent || item.seconds < 0
        ? "INVALID (tab hidden or clock changed)" : item.seconds <= 10 ? "PASS (10-second target)" : "FAIL (over 10 seconds)"}
    </li>)}</ul>
    <button type="button" className="secondary" disabled={!observations.length} onClick={download}>Download browser timings</button>
    <button type="button" className="secondary" onClick={() => setShowJson(!showJson)}>Show / hide measurement JSON</button>
    {showJson && <label>Browser timing measurements JSON
      <textarea readOnly rows={12} value={json} style={{ width: "100%" }} />
    </label>}
  </section>;
}
