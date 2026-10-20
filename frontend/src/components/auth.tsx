"use client";

import { useState, useSyncExternalStore } from "react";
import { api, getToken, setToken } from "@/lib/api";

// localStorage isn't observable, so changes are broadcast with a custom event.
const EVENT = "insightforge-auth";

function subscribe(onChange: () => void) {
  window.addEventListener(EVENT, onChange);
  window.addEventListener("storage", onChange);
  return () => {
    window.removeEventListener(EVENT, onChange);
    window.removeEventListener("storage", onChange);
  };
}

export function useToken(): string | null | undefined {
  // undefined during server render / hydration: neither "signed in" nor "signed out" yet.
  return useSyncExternalStore(subscribe, getToken, () => undefined);
}

function saveToken(token: string | null) {
  setToken(token);
  window.dispatchEvent(new Event(EVENT));
}

export function RequireAuth({ children }: { children: React.ReactNode }) {
  const token = useToken();
  if (token === undefined) return null;
  if (!token) return <DevLogin />;
  return <>{children}</>;
}

function DevLogin() {
  const [email, setEmail] = useState("dev@insightforge.local");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const res = await api<{ access_token: string }>("/auth/dev-login", {
        method: "POST",
        body: JSON.stringify({ email }),
      });
      saveToken(res.access_token);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Sign-in failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="mx-auto mt-16 max-w-sm space-y-4 rounded-lg border border-border bg-panel p-6">
      <h1 className="text-lg font-semibold">Sign in</h1>
      <p className="text-sm text-muted">Development sign-in: any email address. Production uses your identity provider.</p>
      <input
        type="email"
        required
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        className="w-full rounded-md border border-border bg-bg px-3 py-2 text-sm"
        aria-label="Email"
      />
      {error && <p className="text-sm text-bad">{error}</p>}
      <button disabled={busy} className="w-full rounded-md bg-accent px-3 py-2 text-sm font-medium text-white disabled:opacity-50">
        {busy ? "Signing in…" : "Sign in"}
      </button>
    </form>
  );
}

export function SignOut() {
  const token = useToken();
  if (!token) return null;
  return (
    <button onClick={() => saveToken(null)} className="text-sm text-muted hover:text-text">
      Sign out
    </button>
  );
}
