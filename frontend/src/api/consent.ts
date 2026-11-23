import { apiRequest } from "./client";

export interface ConsentPolicy {
  state: "not_recorded" | "granted" | "withdrawn";
  version: number;
  updated_at: string;
  disclosure_version: string | null;
  first_granted_at: string | null;
  monitoring_allowed: boolean;
  device_confirmed: boolean;
}

export interface ConsentDetails {
  device_id: string;
  room: string;
  consent: ConsentPolicy;
  caregivers: { display_name: string; email: string; role: "trusted_caregiver" | "caregiver" }[];
  history: { decision: "granted" | "withdrawn"; disclosure_version: string; recorded_at: string }[];
  disclosure: { version: string; paragraphs: string[] };
}

export function inspectConsent(code: string): Promise<ConsentDetails> {
  return apiRequest("/consent/inspect", { method: "POST", body: JSON.stringify({ code }) });
}

export function decideConsent(code: string, details: ConsentDetails, decision: "grant" | "withdraw"):
Promise<ConsentDetails> {
  return apiRequest("/consent/decision", { method: "POST", body: JSON.stringify({
    code, decision, expected_version: details.consent.version,
    ...(decision === "grant" ? { disclosure_version: details.disclosure.version, confirmed: true } : {}),
  }) });
}
