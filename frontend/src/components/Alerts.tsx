import type { E3Event } from "../types/event";

const descriptions: Record<string, { title: string; summary: string }> = {
  fall: { title: "Possible fall", summary: "A possible fall was detected" },
  anomaly: { title: "Unusual sound activity", summary: "Unusual sound activity was detected" },
  inactivity: { title: "Inactivity", summary: "An unusual period of inactivity was detected" },
};

const findings: Record<string, string> = {
  prolonged_water: "Water sounds continued longer than usual",
  prolonged_appliance: "Appliance sounds continued longer than usual",
  repeated_impacts: "Repeated impact sounds were detected",
  sustained_unusual_sound: "Unusual sound activity continued for an extended period",
  post_impact_inactivity: "An extended quiet period followed an impact sound",
  post_impact_silence: "A quiet period followed an impact sound",
};

function description(event: E3Event) {
  return descriptions[event.event_type] ?? { title: "Alert", summary: "An alert was reported" };
}

function summary(event: E3Event): string {
  const kind = event.details?.kind;
  const finding = event.event_type !== "fall" && typeof kind === "string" ? findings[kind] : undefined;
  return `${finding ?? description(event).summary} in ${event.room}.`;
}

function AlertTime({ value }: { value: string }) {
  return <time dateTime={value}>{new Date(value).toLocaleString(undefined, {
    dateStyle: "medium", timeStyle: "short",
  })}</time>;
}

export function LatestAlert({ event }: { event?: E3Event }) {
  return (
    <section className={`latest-alert ${event ? `alert-${event.event_type}` : "alert-empty"}`}
      aria-labelledby="latest-alert-title">
      <h2 id="latest-alert-title">Latest alert</h2>
      <div aria-live="polite" aria-atomic="true">
        {event ? <>
          <p className="alert-label">{description(event).title}</p>
          <p className="alert-summary">{summary(event)}</p>
          <dl className="alert-facts">
            <div><dt>Room</dt><dd>{event.room}</dd></div>
            <div><dt>Detected</dt><dd><AlertTime value={event.detected_at} /></dd></div>
          </dl>
          <p className="alert-device">Device: {event.device_id}</p>
        </> : <>
          <p className="alert-label">No alerts yet</p>
          <p>No fall, unusual sound or inactivity alerts have been received for your linked devices.</p>
        </>}
      </div>
    </section>
  );
}

function AlertList({ events }: { events: E3Event[] }) {
  return (
    <ol className="alert-history-list">
          {events.map((event) => <li className={`history-alert alert-${event.event_type}`} key={event.event_id}>
            <article>
              <div className="history-alert-heading">
                <h3>{description(event).title}</h3>
                <AlertTime value={event.detected_at} />
              </div>
              <p className="history-room">{event.room}</p>
              <p>{summary(event)}</p>
              <p className="alert-device">Device: {event.device_id}</p>
            </article>
          </li>)}
    </ol>
  );
}

export function RecentAlerts({ events }: { events: E3Event[] }) {
  return (
    <section className="alert-history" aria-labelledby="recent-alerts-title">
      <h2 id="recent-alerts-title">Recent alerts</h2>
      <p className="history-caption">The five most recent alerts. Times are shown in your local timezone.</p>
      {events.length === 0 ? <p>Your saved alerts will appear here.</p> : <AlertList events={events} />}
      <a className="button-link" href="/history">View all alerts</a>
    </section>
  );
}

export function AlertHistory({ events, hasMore, loadingMore, onLoadMore }: {
  events: E3Event[]; hasMore: boolean; loadingMore: boolean; onLoadMore: () => void;
}) {
  return (
    <section className="alert-history" aria-labelledby="alert-history-title" aria-busy={loadingMore}>
      <h2 id="alert-history-title">Alert history</h2>
      <p className="history-caption">Newest first. Times are shown in your local timezone.</p>
      {events.length === 0 ? <p>Your saved alerts will appear here.</p> : <>
        <p role="status">Showing {events.length} {events.length === 1 ? "alert" : "alerts"}.</p>
        <AlertList events={events} />
        {hasMore ? <button type="button" className="secondary" disabled={loadingMore} onClick={onLoadMore}>
          {loadingMore ? "Loading older alerts…" : "Load older alerts"}
        </button> : <p className="history-caption">You have reached the end of your saved alerts.</p>}
      </>}
    </section>
  );
}
