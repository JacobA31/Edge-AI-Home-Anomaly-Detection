import { useEffect, useState } from "react";

import { fetchEvents } from "./api/client";
import type { E3Event } from "./types/event";

const POLL_MS = 5000;

export default function App() {
  const [events, setEvents] = useState<E3Event[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = () =>
      fetchEvents()
        .then((e) => !cancelled && (setEvents(e), setError(null)))
        .catch((err: Error) => !cancelled && setError(err.message));
    load();
    const id = setInterval(load, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  return (
    <main style={{ fontFamily: "system-ui, sans-serif", maxWidth: 800, margin: "2rem auto", padding: "0 1rem" }}>
      <h1>E3 Caregiver Dashboard</h1>
      {error && <p role="alert">Can't reach the backend: {error}</p>}
      {/* TODO(member5): alert banner + sound for new falls, device status, acknowledge button */}
      {events.length === 0 ? (
        <p>No events yet.</p>
      ) : (
        <ul>
          {events.map((e) => (
            <li key={e.event_id}>
              {new Date(e.detected_at).toLocaleString()} · {e.event_type} · {e.room}
            </li>
          ))}
        </ul>
      )}
    </main>
  );
}
