"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { AlertView, ChangeSummary, WatchView } from "@/lib/types";
import { Badge, formatDate, Panel } from "./ui";

function Changes({ summary }: { summary: ChangeSummary }) {
  const block = (title: string, items: React.ReactNode[], kind: "ok" | "warn" | "muted") =>
    items.length > 0 && (
      <div>
        <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted">{title} <Badge kind={kind}>{items.length}</Badge></p>
        <ul className="list-disc space-y-1 pl-5 text-sm">{items.map((x, i) => <li key={i}>{x}</li>)}</ul>
      </div>
    );
  return (
    <div className="space-y-3">
      {block("New", summary.added_findings.map((f) => f.text), "ok")}
      {block("Changed", summary.changed_findings.map((f) => (
        <span key={f.after}>{f.after} <span className="text-muted">(was: {f.before})</span></span>
      )), "warn")}
      {block("No longer found", summary.removed_findings.map((f) => f.text), "muted")}
      {summary.note && summary.removed_findings.length > 0 && <p className="text-xs text-muted">{summary.note}</p>}
    </div>
  );
}

function AddItem({ onAdded }: { onAdded: () => void }) {
  const [topic, setTopic] = useState("");
  const [cadence, setCadence] = useState<"daily" | "weekly">("weekly");
  const [webhook, setWebhook] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await api("/watchlist", { method: "POST", body: JSON.stringify({ topic, cadence, webhook_url: webhook || null }) });
      setTopic("");
      setWebhook("");
      onAdded();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not add the topic");
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="grid gap-3 md:grid-cols-[1fr_auto_1fr_auto]">
      <input value={topic} onChange={(e) => setTopic(e.target.value)} maxLength={500} required
             placeholder="Topic or company to monitor" className="rounded-md border border-border bg-bg px-3 py-2 text-sm" />
      <select value={cadence} onChange={(e) => setCadence(e.target.value as "daily" | "weekly")}
              className="rounded-md border border-border bg-bg px-3 py-2 text-sm" aria-label="Cadence">
        <option value="daily">Daily</option>
        <option value="weekly">Weekly</option>
      </select>
      <input value={webhook} onChange={(e) => setWebhook(e.target.value)} type="url"
             placeholder="Webhook URL (optional, https, e.g. Slack)" className="rounded-md border border-border bg-bg px-3 py-2 text-sm" />
      <button disabled={busy} className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-white disabled:opacity-50">Add</button>
      {error && <p className="text-sm text-bad md:col-span-4">{error}</p>}
    </form>
  );
}

export function WatchlistDashboard() {
  const router = useRouter();
  const [items, setItems] = useState<WatchView[] | null>(null);
  const [alerts, setAlerts] = useState<AlertView[]>([]);
  const [diff, setDiff] = useState<AlertView | "none" | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [version, setVersion] = useState(0);
  const reload = () => setVersion((v) => v + 1);

  useEffect(() => {
    let cancelled = false;
    Promise.all([api<WatchView[]>("/watchlist"), api<AlertView[]>("/alerts?limit=30")])
      .then(([w, a]) => {
        if (!cancelled) {
          setItems(w);
          setAlerts(a);
        }
      })
      .catch(() => !cancelled && setItems([]));
    return () => { cancelled = true; };
  }, [version]);

  async function act(fn: () => Promise<unknown>, ok?: string) {
    setMsg(null);
    try {
      await fn();
      if (ok) setMsg(ok);
      reload();
    } catch (err) {
      setMsg(err instanceof Error ? err.message : "Action failed");
    }
  }

  async function showDiff(id: string) {
    const latest = await api<AlertView | null>(`/watchlist/${id}/diff`);
    setDiff(latest ?? "none");
  }

  return (
    <div className="space-y-6">
      <Panel><AddItem onAdded={reload} /></Panel>
      {msg && <p className="text-sm text-muted">{msg}</p>}

      <Panel className="overflow-x-auto p-0">
        <table className="w-full text-left text-sm">
          <thead className="border-b border-border text-xs uppercase tracking-wide text-muted">
            <tr>
              <th className="px-5 py-3">Topic</th><th className="px-3 py-3">Cadence</th>
              <th className="px-3 py-3">Last run</th><th className="px-3 py-3">Next run</th>
              <th className="px-3 py-3">Changes</th><th className="px-5 py-3 text-right">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {items?.map((w) => (
              <tr key={w.id} className={w.active ? "" : "opacity-60"}>
                <td className="max-w-xs px-5 py-3">
                  <span className="block truncate font-medium">{w.topic}</span>
                  {w.webhook_url && <span className="block truncate text-xs text-muted">webhook: {new URL(w.webhook_url).host}</span>}
                </td>
                <td className="px-3 py-3 capitalize">{w.cadence}{!w.active && " (paused)"}</td>
                <td className="px-3 py-3 text-muted">{formatDate(w.last_run)}</td>
                <td className="px-3 py-3 text-muted">{w.active ? formatDate(w.next_run) : "–"}</td>
                <td className="px-3 py-3">
                  <button onClick={() => showDiff(w.id)} className="text-accent hover:underline">
                    {w.latest_alert ? "View diff" : "No changes yet"}
                  </button>
                </td>
                <td className="space-x-3 whitespace-nowrap px-5 py-3 text-right">
                  <button className="text-accent hover:underline" onClick={() => act(async () => {
                    const r = await api<{ run_id: string }>(`/watchlist/${w.id}/run-now`, { method: "POST" });
                    router.push(`/run/${r.run_id}`);
                  })}>Run now</button>
                  <button className="text-muted hover:text-text"
                          onClick={() => act(() => api(`/watchlist/${w.id}`, { method: "PATCH", body: JSON.stringify({ active: !w.active }) }))}>
                    {w.active ? "Pause" : "Resume"}
                  </button>
                  <button className="text-muted hover:text-bad"
                          onClick={() => act(() => api(`/watchlist/${w.id}`, { method: "DELETE" }), "Removed.")}>Remove</button>
                </td>
              </tr>
            ))}
            {items?.length === 0 && (
              <tr><td colSpan={6} className="px-5 py-6 text-center text-muted">No monitored topics yet.</td></tr>
            )}
          </tbody>
        </table>
      </Panel>

      {diff && (
        <Panel>
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-sm font-semibold">
              {diff === "none" ? "No changes yet" : `What changed: ${diff.topic}`}
            </h2>
            <button onClick={() => setDiff(null)} className="text-sm text-muted">Close</button>
          </div>
          {diff === "none" ? (
            <p className="text-sm text-muted">The first run saves a baseline; later runs alert only on material changes.</p>
          ) : (
            <>
              <Changes summary={diff.change_summary} />
              {diff.report_id && <Link href={`/report/${diff.report_id}`} className="mt-3 inline-block text-sm text-accent">Open the new report</Link>}
            </>
          )}
        </Panel>
      )}

      <Panel>
        <h2 className="mb-3 text-sm font-semibold">Alerts</h2>
        {alerts.length === 0 ? <p className="text-sm text-muted">No alerts yet.</p> : (
          <ul className="space-y-4">
            {alerts.map((a) => (
              <li key={a.id} className={`rounded-md border p-3 ${a.read ? "border-border" : "border-accent/40 bg-accent/5"}`}>
                <div className="mb-2 flex flex-wrap items-center gap-2 text-sm">
                  <span className="font-medium">{a.topic}</span>
                  <span className="text-xs text-muted">{formatDate(a.created_at)}</span>
                  {a.webhook_status && <Badge kind={a.webhook_status === "sent" ? "ok" : "bad"}>webhook {a.webhook_status}</Badge>}
                  {!a.read && (
                    <button onClick={() => act(() => api(`/alerts/${a.id}/read`, { method: "POST" }))}
                            className="ml-auto text-xs text-muted hover:text-text">Mark read</button>
                  )}
                </div>
                <Changes summary={a.change_summary} />
              </li>
            ))}
          </ul>
        )}
      </Panel>
    </div>
  );
}
