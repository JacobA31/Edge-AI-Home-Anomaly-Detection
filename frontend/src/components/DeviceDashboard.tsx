import { useCallback, useEffect, useState } from "react";

import { ApiError, fetchAlerts } from "../api/client";
import { fetchDevices } from "../api/devices";
import type { Device } from "../api/devices";
import type { E3Event } from "../types/event";
import { LatestAlert, RecentAlerts } from "./Alerts";
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
        const [linked, firstPage] = await Promise.all([
          fetchDevices(), fetchAlerts(5),
        ]);
        if (cancelled) return;
        setDevices(linked); setEvents(firstPage.events); setError("");
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

  if (error) return <section role="alert">
    <p className="error">Could not load your dashboard: {error}</p>
    <button type="button" onClick={refresh}>Try again</button>
  </section>;

  return (
    <>
      {loading ? <p role="status">Loading your devices…</p> : <>
        {devices.length > 0 && <LatestAlert event={events[0]} />}
        <DevicePanel devices={devices} onChanged={refresh} onSignedOut={onSignedOut} />
        {devices.length > 0 && <RecentAlerts events={events} />}
      </>}
    </>
  );
}
