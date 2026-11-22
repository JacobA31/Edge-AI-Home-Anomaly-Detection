import { useCallback, useEffect, useState } from "react";

import { currentAccount, logout } from "./api/auth";
import type { Account } from "./api/auth";
import { ApiError } from "./api/client";
import AccountPanel from "./components/AccountPanel";
import AuthForm from "./components/AuthForm";
import DeviceDashboard from "./components/DeviceDashboard";
import "./App.css";

export default function App() {
  const [user, setUser] = useState<Account | null>(null);
  const [checking, setChecking] = useState(true);
  const [sessionError, setSessionError] = useState("");
  const [notice, setNotice] = useState("");
  const [showAccount, setShowAccount] = useState(false);
  const [loggingOut, setLoggingOut] = useState(false);

  useEffect(() => {
    let cancelled = false;
    currentAccount()
      .then((account) => { if (!cancelled) setUser(account); })
      .catch((err) => {
        if (!cancelled && !(err instanceof ApiError && err.status === 401)) {
          setSessionError(err instanceof Error ? err.message : "Could not check your login.");
        }
      })
      .finally(() => { if (!cancelled) setChecking(false); });
    return () => { cancelled = true; };
  }, []);

  const signedOut = useCallback((message: string) => {
    setUser(null); setShowAccount(false); setNotice(message); setSessionError("");
  }, []);

  async function signOut() {
    setLoggingOut(true); setSessionError("");
    try {
      await logout();
      signedOut("You have been logged out.");
    } catch (err) {
      setSessionError(err instanceof Error ? err.message : "Could not log out. Please try again.");
    } finally { setLoggingOut(false); }
  }

  return (
    <main>
      <h1>E3 Caregiver Dashboard</h1>
      {sessionError && <p className="error" role="alert">{sessionError}</p>}
      {checking ? <p role="status">Checking your login…</p> : !user ? (
        <AuthForm notice={notice} onLogin={(account) => {
          setUser(account); setNotice(""); setSessionError("");
        }} />
      ) : (
        <>
          <div className="account-toolbar">
            <p>Signed in as <strong>{user.display_name}</strong></p>
            <button className="secondary" onClick={() => setShowAccount(!showAccount)}>
              {showAccount ? "Back to dashboard" : "My account"}
            </button>
            <button onClick={signOut} disabled={loggingOut}>{loggingOut ? "Logging out…" : "Log out"}</button>
          </div>
          {showAccount ? <AccountPanel key={user.account_id} user={user} onUpdate={(updated) => {
            setUser((previous) => previous?.account_id === updated.account_id ? updated : previous);
          }}
            onSignedOut={signedOut} /> : <DeviceDashboard onSignedOut={signedOut} />}
        </>
      )}
    </main>
  );
}
