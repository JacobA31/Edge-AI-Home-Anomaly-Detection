import type { E3Event } from "../types/event";

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

export async function fetchEvents(): Promise<E3Event[]> {
  const body = await apiRequest<{ events: E3Event[] }>("/events");
  return body.events;
}
