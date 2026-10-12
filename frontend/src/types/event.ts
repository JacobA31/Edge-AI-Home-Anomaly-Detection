// Mirrors shared/event.schema.json. Update both in the same PR.

export type EventType =
  | "fall"
  | "anomaly"
  | "inactivity"
  | "heartbeat"
  | "device_muted"
  | "device_unmuted";

export interface E3Event {
  event_id: string;
  device_id: string;
  event_type: EventType;
  room: string;
  detected_at: string; // ISO 8601, UTC
  elapsed_seconds: number;
  confidence?: number;
  details?: Record<string, unknown>;
}
