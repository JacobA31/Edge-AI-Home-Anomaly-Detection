import type { E3Event } from "../types/event";

// Empty in local dev (Vite proxies /api to func start); the Function App URL in production.
const BASE = import.meta.env.VITE_API_BASE_URL ?? "";

export async function fetchEvents(): Promise<E3Event[]> {
  const res = await fetch(`${BASE}/api/events`);
  if (!res.ok) throw new Error(`GET /api/events failed: ${res.status}`);
  const body = (await res.json()) as { events: E3Event[] };
  return body.events;
}
