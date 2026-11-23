import { useState } from "react";
import type { FormEvent } from "react";

import { currentAccount, login, register } from "../api/auth";
import type { Account } from "../api/auth";

export default function AuthForm({ onLogin, notice }: { onLogin: (user: Account) => void; notice: string }) {
  const [creating, setCreating] = useState(false);
  const [email, setEmail] = useState("");
  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setError("");
    setMessage("");
    try {
      if (creating) {
        await register(email, name, password);
        setPassword("");
        setCreating(false);
        setMessage("Account created. You can now log in.");
      } else {
        const user = await login(email, password);
        // Confirm the browser accepted the session cookie before opening the dashboard.
        await currentAccount();
        setPassword("");
        onLogin(user);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong. Please try again.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="account-card" aria-labelledby="auth-title">
      <h2 id="auth-title">{creating ? "Create your account" : "Log in"}</h2>
      <p>Use your own account to access the caregiver dashboard.</p>
      {notice && <p role="status">{notice}</p>}
      {message && <p role="status">{message}</p>}
      {error && <p className="error" role="alert">{error}</p>}
      <form onSubmit={submit}>
        <fieldset disabled={busy}>
          {creating && (
            <label>Your name
              <input autoComplete="name" value={name} onChange={(e) => setName(e.target.value)}
                required maxLength={100} />
            </label>
          )}
          <label>Email
            <input type="email" autoComplete="username" value={email}
              onChange={(e) => setEmail(e.target.value)} required maxLength={254} />
          </label>
          <label>Password
            <input type="password" autoComplete={creating ? "new-password" : "current-password"}
              value={password} onChange={(e) => setPassword(e.target.value)}
              required minLength={creating ? 15 : 1} maxLength={128}
              aria-describedby={creating ? "password-help" : undefined} />
          </label>
          {creating && <p id="password-help">Use 15–128 characters. A few memorable words work well.</p>}
          <button type="submit">{busy ? "Please wait…" : creating ? "Create account" : "Log in"}</button>
          <button type="button" className="secondary" onClick={() => {
            setCreating(!creating); setError(""); setMessage(""); setPassword("");
          }}>{creating ? "Already have an account? Log in" : "Create an account"}</button>
        </fieldset>
      </form>
    </section>
  );
}
