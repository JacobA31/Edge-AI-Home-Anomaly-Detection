// Pi message fields mirror shared/event.schema.json.

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

// Read-only metadata supplied by the backend; it is not part of the Pi message schema.
export interface RecordedEvent extends E3Event {
  received_at: string;
  is_delayed: boolean;
  delivery_delay_seconds: number;
  delivery_delay_label: string;
}
