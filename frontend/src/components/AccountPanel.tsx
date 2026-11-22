import { useState } from "react";
import type { FormEvent } from "react";

import { changePassword, updateName } from "../api/auth";
import type { Account } from "../api/auth";
import { ApiError } from "../api/client";

interface Props {
  user: Account;
  onUpdate: (user: Account) => void;
  onSignedOut: (message: string) => void;
}

export default function AccountPanel({ user, onUpdate, onSignedOut }: Props) {
  const [name, setName] = useState(user.display_name);
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [confirm, setConfirm] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  async function submit(event: FormEvent<HTMLFormElement>, password: boolean) {
    event.preventDefault();
    setError(""); setMessage("");
    if (password && next !== confirm) { setError("The new passwords do not match."); return; }
    setBusy(true);
    try {
      if (password) {
        await changePassword(current, next);
        setCurrent(""); setNext(""); setConfirm("");
        onSignedOut("Password changed. Please log in with your new password.");
      } else {
        onUpdate(await updateName(name));
        setMessage("Your name has been updated.");
      }
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        onSignedOut("Please log in again.");
      } else {
        setError(err instanceof Error ? err.message : "Could not update your account.");
      }
    } finally { setBusy(false); }
  }

  return (
    <section className="account-card" aria-labelledby="account-title">
      <h2 id="account-title">My account</h2>
      <p>{user.email}</p>
      {message && <p role="status">{message}</p>}
      {error && <p className="error" role="alert">{error}</p>}
      <form onSubmit={(e) => submit(e, false)}>
        <fieldset disabled={busy}>
          <label>Your name
            <input autoComplete="name" value={name} onChange={(e) => setName(e.target.value)}
              required maxLength={100} />
          </label>
          <button type="submit">Save name</button>
        </fieldset>
      </form>
      <form onSubmit={(e) => submit(e, true)}>
        <fieldset disabled={busy}>
          <legend>Change password</legend>
          <label>Current password
            <input type="password" autoComplete="current-password" value={current}
              onChange={(e) => setCurrent(e.target.value)} required maxLength={128} />
          </label>
          <label>New password (15–128 characters)
            <input type="password" autoComplete="new-password" value={next}
              onChange={(e) => setNext(e.target.value)} required minLength={15} maxLength={128} />
          </label>
          <label>Confirm new password
            <input type="password" autoComplete="new-password" value={confirm}
              onChange={(e) => setConfirm(e.target.value)} required minLength={15} maxLength={128} />
          </label>
          <p>Changing your password logs you out on all devices.</p>
          <button type="submit">Change password</button>
        </fieldset>
      </form>
    </section>
  );
}
