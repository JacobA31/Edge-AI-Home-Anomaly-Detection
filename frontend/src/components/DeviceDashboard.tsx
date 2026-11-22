import { useCallback, useEffect, useState } from "react";

import { ApiError, fetchEvents } from "../api/client";
import { fetchDevices } from "../api/devices";
import type { Device } from "../api/devices";
import type { E3Event } from "../types/event";
import DevicePanel from "./DevicePanel";

export default function DeviceDashboard({ onSignedOut }: { onSignedOut: (message: string) => void }) {
  const [devices, setDevices] = useState<Device[]>([]);
  const [events, setEvents] = useState<E3Event[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [revision, setRevision] = useState(0);
  const refresh = useCallback(() => setRevision((value) => value + 1), []);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    async function load() {
      try {
        const [linked, saved] = await Promise.all([fetchDevices(), fetchEvents()]);
        if (cancelled) return;
        setDevices(linked); setEvents(saved); setError("");
      } catch (err) {
        if (cancelled) return;
        // Clear cached household information if an access check fails.
        setDevices([]); setEvents([]);
        if (err instanceof ApiError && err.status === 401) {
          onSignedOut("Your session ended. Please log in again.");
        } else { setError(err instanceof Error ? err.message : "Could not load the dashboard."); }
      } finally {
        if (!cancelled) { setLoading(false); timer = setTimeout(load, 5000); }
      }
    }
    void load();
    return () => { cancelled = true; clearTimeout(timer); };
  }, [onSignedOut, revision]);

  return (
    <>
      {error && <p className="error" role="alert">{error}</p>}
      {loading ? <p role="status">Loading your devices…</p> : <>
        <DevicePanel devices={devices} onChanged={refresh} onSignedOut={onSignedOut} />
        {devices.length > 0 && <section aria-labelledby="event-history-title">
          <h2 id="event-history-title">Event history</h2>
          {events.length === 0 ? <p>No events yet.</p> : <ul className="event-list">
            {events.map((event) => <li key={event.event_id}>
              {new Date(event.detected_at).toLocaleString()} · {event.event_type} · {event.room}
            </li>)}
          </ul>}
        </section>}
      </>}
    </>
  );
}
