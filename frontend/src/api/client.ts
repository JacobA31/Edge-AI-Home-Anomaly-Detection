import type { RecordedEvent } from "../types/event";

// Empty in local dev (Vite proxies /api to func start); the Function App URL in production.
const BASE = import.meta.env.VITE_API_BASE_URL ?? "";

export class ApiError extends Error {
  constructor(public readonly status: number, message: string) {
    super(message);
  }
}

export async function apiRequest<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers = new Headers(options.headers);
  headers.set("Accept", "application/json");
  if (options.body !== undefined) headers.set("Content-Type", "application/json");
  const res = await fetch(`${BASE}/api${path}`, { ...options, headers, credentials: "include" });
  const body = await res.json().catch(() => null);
  if (!res.ok) throw new ApiError(res.status, body?.error ?? `Request failed (${res.status})`);
  if (body === null) throw new Error("The server returned an unexpected response.");
  return body as T;
}

export async function fetchEvents(): Promise<RecordedEvent[]> {
  const body = await apiRequest<{ events: RecordedEvent[] }>("/events");
  return body.events;
}

export interface AlertPage {
  events: RecordedEvent[];
  next_cursor: string | null;
}

export async function fetchAlerts(limit: number, cursor?: string, deviceId?: string): Promise<AlertPage> {
  const params = new URLSearchParams({ alerts_only: "true", limit: String(limit) });
  if (cursor) params.set("cursor", cursor);
  if (deviceId) params.set("device_id", deviceId);
  return apiRequest<AlertPage>(`/events?${params}`);
}
