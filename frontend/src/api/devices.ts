import { apiRequest } from "./client";

export interface Device {
  device_id: string;
  room: string;
  role: "trusted_caregiver" | "caregiver";
}

export interface DeviceAccess {
  members: { account_id: string; email: string; display_name: string; role: Device["role"] }[];
  invitations: { invitation_id: string; email: string; expires_at: string }[];
}

const devicePath = (deviceId: string) => `/devices/${encodeURIComponent(deviceId)}`;

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
