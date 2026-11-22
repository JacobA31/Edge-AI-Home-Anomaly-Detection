import { useCallback, useEffect, useState } from "react";

import { ApiError, fetchAlerts } from "../api/client";
import { fetchDevices } from "../api/devices";
import type { Device } from "../api/devices";
import type { E3Event } from "../types/event";
import { AlertHistory } from "./Alerts";
import DeviceFilter from "./DeviceFilter";

const PAGE_SIZE = 20;

export default function AlertHistoryPage({ onSignedOut }: { onSignedOut: (message: string) => void }) {
  const [events, setEvents] = useState<E3Event[]>([]);
  const [devices, setDevices] = useState<Device[]>([]);
  const [selectedDevice, setSelectedDevice] = useState(() =>
    new URLSearchParams(window.location.search).get("device_id") ?? "");
  const [visibleCount, setVisibleCount] = useState(PAGE_SIZE);
  const [hasMore, setHasMore] = useState(false);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState("");
  const [revision, setRevision] = useState(0);
  const refresh = useCallback(() => setRevision((value) => value + 1), []);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    async function load() {
      try {
        const [linked, firstPage] = await Promise.all([
          fetchDevices(), fetchAlerts(Math.min(visibleCount, 100), undefined, selectedDevice),
        ]);
        const saved = [...firstPage.events];
        let cursor = firstPage.next_cursor;
        // Recheck the entire visible window so new alerts, late arrivals and
        // removed device permissions are reflected without stale history.
        while (!cancelled && cursor && saved.length < visibleCount) {
          const page = await fetchAlerts(Math.min(visibleCount - saved.length, 100), cursor, selectedDevice);
          saved.push(...page.events);
          cursor = page.next_cursor;
        }
        if (cancelled) return;
        setDevices(linked); setEvents(saved); setHasMore(cursor !== null); setError("");
      } catch (err) {
        if (cancelled) return;
        setDevices([]); setEvents([]); setHasMore(false);
        if (err instanceof ApiError && err.status === 401) {
          onSignedOut("Your session ended. Please log in again.");
        } else { setError(err instanceof Error ? err.message : "Could not load alert history."); }
      } finally {
        if (!cancelled) { setLoading(false); setLoadingMore(false); timer = setTimeout(load, 5000); }
      }
    }
    void load();
    return () => { cancelled = true; clearTimeout(timer); };
  }, [onSignedOut, visibleCount, revision, selectedDevice]);

  function selectDevice(deviceId: string) {
    if (deviceId === selectedDevice) return;
    // A new device starts at its first page, without flashing the previous device's alerts.
    setSelectedDevice(deviceId); setVisibleCount(PAGE_SIZE); setEvents([]);
    setHasMore(false); setLoadingMore(false); setLoading(true); setError("");
    const url = new URL(window.location.href);
    if (deviceId) url.searchParams.set("device_id", deviceId);
    else url.searchParams.delete("device_id");
    window.history.replaceState(null, "", `${url.pathname}${url.search}${url.hash}`);
  }

  return <>
    <div className="history-toolbar">
      <a className="button-link back-link" href="/">
        <span aria-hidden="true">←</span> Back to dashboard
      </a>
      <DeviceFilter devices={devices} value={selectedDevice} onChange={selectDevice} />
    </div>
    {loading ? <p role="status">Loading alert history…</p> : error ? <section role="alert">
      <p className="error">Could not load your alert history: {error}</p>
      <button type="button" onClick={refresh}>Try again</button>
    </section> : selectedDevice && !devices.some((device) => device.device_id === selectedDevice) ? (
      <p role="status">This device is not available to your account. Select another device or “All devices”.</p>
    ) : <AlertHistory events={events} hasMore={hasMore} loadingMore={loadingMore}
      onLoadMore={() => { setLoadingMore(true); setVisibleCount((count) => count + PAGE_SIZE); }} />}
  </>;
}
