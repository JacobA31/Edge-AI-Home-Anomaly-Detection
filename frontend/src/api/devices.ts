import { apiRequest } from "./client";

export interface DeviceCommand {
  command_id: string;
  action: "mute" | "resume";
  status: "pending" | "delivered" | "completed" | "failed" | "expired";
  requested_at: string;
  expires_at: string;
  delivered_at: string | null;
  finished_at: string | null;
}

export interface Device {
  device_id: string;
  room: string;
  role: "trusted_caregiver" | "caregiver";
  connection_status: "online" | "offline" | "never_seen";
  last_seen_at: string | null;
  monitoring_state: "active" | "muted" | "unknown";
  monitoring_updated_at: string | null;
  offline_after_seconds: number;
  controls_configured: boolean;
  latest_command: DeviceCommand | null;
}

export interface DeviceAccess {
  members: { account_id: string; email: string; display_name: string; role: Device["role"] }[];
  invitations: { invitation_id: string; email: string; expires_at: string }[];
}

const devicePath = (deviceId: string) => `/devices/${encodeURIComponent(deviceId)}`;

export async function requestDeviceCommand(deviceId: string, action: "mute" | "resume"): Promise<DeviceCommand> {
  return (await apiRequest<{ command: DeviceCommand }>(`${devicePath(deviceId)}/commands`, {
    method: "POST", body: JSON.stringify({ action }),
  })).command;
}

export async function fetchDevices(): Promise<Device[]> {
  return (await apiRequest<{ devices: Device[] }>("/devices")).devices;
}

export async function claimDevice(linkingCode: string): Promise<void> {
  await apiRequest("/devices/claim", { method: "POST", body: JSON.stringify({ linking_code: linkingCode }) });
}

export async function acceptInvitation(invitationCode: string): Promise<void> {
  await apiRequest("/devices/invitations/accept", {
    method: "POST", body: JSON.stringify({ invitation_code: invitationCode }),
  });
}

export async function fetchAccess(deviceId: string): Promise<DeviceAccess> {
  return apiRequest<DeviceAccess>(`${devicePath(deviceId)}/access`);
}

export async function inviteCaregiver(deviceId: string, email: string): Promise<string> {
  const result = await apiRequest<{ invitation_code: string }>(`${devicePath(deviceId)}/invitations`, {
    method: "POST", body: JSON.stringify({ email }),
  });
  return result.invitation_code;
}

export async function removeCaregiver(deviceId: string, accountId: string): Promise<void> {
  await apiRequest(`${devicePath(deviceId)}/members/${encodeURIComponent(accountId)}`, { method: "DELETE" });
}

export async function revokeInvitation(deviceId: string, invitationId: string): Promise<void> {
  await apiRequest(`${devicePath(deviceId)}/invitations/${encodeURIComponent(invitationId)}`, { method: "DELETE" });
}
