import { apiRequest } from "./client";

export interface Account {
  account_id: string;
  email: string;
  display_name: string;
}

export async function currentAccount(): Promise<Account> {
  return (await apiRequest<{ user: Account }>("/auth/me")).user;
}

export async function register(email: string, displayName: string, password: string): Promise<void> {
  await apiRequest("/auth/register", {
    method: "POST",
    body: JSON.stringify({ email, display_name: displayName, password }),
  });
}

export async function login(email: string, password: string): Promise<Account> {
  const result = await apiRequest<{ user: Account }>("/auth/login", {
    method: "POST", body: JSON.stringify({ email, password }),
  });
  return result.user;
}

export async function logout(): Promise<void> {
  await apiRequest("/auth/logout", { method: "POST" });
}

export async function updateName(displayName: string): Promise<Account> {
  const result = await apiRequest<{ user: Account }>("/auth/profile", {
    method: "PATCH", body: JSON.stringify({ display_name: displayName }),
  });
  return result.user;
}

export async function changePassword(currentPassword: string, newPassword: string): Promise<void> {
  await apiRequest("/auth/password", {
    method: "POST",
    body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
  });
}
