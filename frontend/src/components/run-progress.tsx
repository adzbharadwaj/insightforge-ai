"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { api, statusStreamUrl } from "@/lib/api";
import type { LogEvent, RunSnapshot, RunStatus } from "@/lib/types";
import { formatDuration, Panel, RunStatusBadge, SubTaskBadge } from "./ui";

const TERMINAL: RunStatus[] = ["COMPLETE", "FAILED", "CANCELLED"];
const MAX_LOG = 400;

// Rough share of a run each stage represents (measured on live runs: research and writing dominate).
function progress(s: RunSnapshot): number {
  const tasks = s.sub_tasks.length || 1;
  const done = s.sub_tasks.filter((t) => t.status === "Done" || t.status === "Failed").length;
  switch (s.status) {
    case "QUEUED": return 2;
    case "PLANNING": case "AWAITING_CLARIFICATION": return 8;
    case "RESEARCHING": return 12 + Math.round((40 * done) / tasks);
    case "SYNTHESIZING": return 60;
    case "WRITING": return 70;
    case "GROUNDING": return 80;
    case "FACT_CHECKING": return 90;
    default: return 100;
  }
}

function useElapsed(snapshot: RunSnapshot | null): number | null {
  const [now, setNow] = useState(() => Date.now());
  const live = snapshot && !TERMINAL.includes(snapshot.status) && snapshot.started_at;
  useEffect(() => {
    if (!live) return;
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [live]);
  if (!snapshot?.started_at) return null;
  const end = snapshot.finished_at ? Date.parse(snapshot.finished_at) : now;
  return (end - Date.parse(snapshot.started_at)) / 1000;
}

export function RunProgress({ runId }: { runId: string }) {
  const [snapshot, setSnapshot] = useState<RunSnapshot | null>(null);
  const [log, setLog] = useState<LogEvent[]>([]);
  const [connError, setConnError] = useState(false);
  const [streamKey, setStreamKey] = useState(0);
  const logEnd = useRef<HTMLDivElement>(null);
  const elapsed = useElapsed(snapshot);

  useEffect(() => {
    // Closed in cleanup: Next keeps hidden routes mounted (Activity) but runs effect cleanups.
    const source = new EventSource(statusStreamUrl(runId));
    source.addEventListener("snapshot", (e) => setSnapshot(JSON.parse((e as MessageEvent).data)));
    source.addEventListener("end", (e) => {
      setSnapshot(JSON.parse((e as MessageEvent).data));
      source.close();
    });
    source.addEventListener("log", (e) => {
      const me = e as MessageEvent;
      const data = JSON.parse(me.data);
      setLog((prev) => [...prev, { id: me.lastEventId, ...data }].slice(-MAX_LOG));
    });
    source.onopen = () => setConnError(false);
    source.onerror = () => setConnError(source.readyState === EventSource.CLOSED);
    return () => source.close();
  }, [runId, streamKey]);

  useEffect(() => logEnd.current?.scrollIntoView({ block: "nearest" }), [log.length]);

  if (!snapshot) {
    return <p className="text-sm text-muted">{connError ? "Could not connect to the run." : "Connecting…"}</p>;
  }
  const pct = progress(snapshot);
  return (
    <div className="space-y-6">
      <Panel>
        <div className="flex flex-wrap items-center gap-3">
          <RunStatusBadge status={snapshot.status} />
          {snapshot.attempts > 1 && <span className="text-xs text-muted">attempt {snapshot.attempts}</span>}
          <span className="ml-auto text-sm text-muted">
            Elapsed {formatDuration(elapsed)} · estimate ~{formatDuration(snapshot.eta_seconds)}
          </span>
        </div>
        <p className="mt-3 text-sm">{snapshot.brief}</p>
        <div className="mt-4 h-2 overflow-hidden rounded-full bg-border" role="progressbar" aria-valuenow={pct}>
          <div className={`h-full transition-all duration-700 ${snapshot.status === "FAILED" ? "bg-bad" : "bg-accent"}`}
               style={{ width: `${pct}%` }} />
        </div>
        <Outcome snapshot={snapshot} onRetried={() => { setLog([]); setStreamKey((k) => k + 1); }} />
      </Panel>

      {snapshot.status === "AWAITING_CLARIFICATION" && snapshot.clarifying_question && (
        <Clarify runId={runId} question={snapshot.clarifying_question} />
      )}

      <Panel>
        <h2 className="mb-3 text-sm font-semibold">Sub-tasks</h2>
        {snapshot.sub_tasks.length === 0 ? (
          <p className="text-sm text-muted">The Planner is breaking the brief into sub-tasks…</p>
        ) : (
          <ul className="divide-y divide-border">
            {snapshot.sub_tasks.map((t) => (
              <li key={t.id} className="flex items-start gap-3 py-2.5 text-sm">
                <span className="w-10 shrink-0 font-mono text-xs text-muted">{t.id}</span>
                <span className="flex-1">
                  {t.query}
                  <span className="ml-2 text-xs text-muted">[{t.source_type}]</span>
                  {t.error && <span className="block text-xs text-bad">{t.error}</span>}
                  {t.status === "Done" && t.summary && <span className="block text-xs text-muted">{t.summary}</span>}
                </span>
                <SubTaskBadge status={t.status} />
              </li>
            ))}
          </ul>
        )}
      </Panel>

      <Panel>
        <h2 className="mb-3 text-sm font-semibold">Activity</h2>
        <div className="max-h-80 overflow-y-auto font-mono text-xs leading-relaxed">
          {log.map((e) => (
            <div key={e.id} className={e.kind === "error" ? "text-bad" : e.kind === "retry" || e.kind === "cut" ? "text-warn" : ""}>
              <span className="text-muted">{new Date(e.at).toLocaleTimeString()} </span>
              <span className="font-semibold">{e.agent}</span>: {e.message}
            </div>
          ))}
          <div ref={logEnd} />
        </div>
      </Panel>
    </div>
  );
}

function Outcome({ snapshot, onRetried }: { snapshot: RunSnapshot; onRetried: () => void }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function retry() {
    setBusy(true);
    setError(null);
    try {
      await api(`/research/${snapshot.run_id}/retry`, { method: "POST" });
      onRetried();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Retry failed");
    } finally {
      setBusy(false);
    }
  }

  if (snapshot.status === "COMPLETE" && snapshot.report_id) {
    return (
      <Link href={`/report/${snapshot.report_id}`}
            className="mt-4 inline-block rounded-md bg-accent px-4 py-2 text-sm font-medium text-white">
        Open report
      </Link>
    );
  }
  if (snapshot.status === "FAILED") {
    return (
      <div className="mt-4 space-y-2">
        <p className="text-sm text-bad">{snapshot.error}</p>
        <button onClick={retry} disabled={busy}
                className="rounded-md border border-border px-4 py-2 text-sm hover:bg-bg disabled:opacity-50">
          {busy ? "Retrying…" : "Retry from the last checkpoint"}
        </button>
        <p className="text-xs text-muted">Finished stages and finished researchers are kept; only what failed runs again.</p>
        {error && <p className="text-sm text-bad">{error}</p>}
      </div>
    );
  }
  if (snapshot.status === "CANCELLED") return <p className="mt-4 text-sm text-muted">{snapshot.error ?? "Cancelled."}</p>;
  return null;
}

function Clarify({ runId, question }: { runId: string; question: string }) {
  const [answer, setAnswer] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await api(`/research/${runId}/clarify`, { method: "POST", body: JSON.stringify({ answer }) });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not send the answer");
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="space-y-3 rounded-lg border border-warn/40 bg-warn/10 p-5">
      <p className="text-sm font-medium">The Planner needs one detail before it starts:</p>
      <p className="text-sm">{question}</p>
      <div className="flex gap-2">
        <input value={answer} onChange={(e) => setAnswer(e.target.value)} maxLength={500} required
               className="flex-1 rounded-md border border-border bg-panel px-3 py-2 text-sm" aria-label="Your answer" />
        <button disabled={busy} className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-white disabled:opacity-50">
          Answer
        </button>
      </div>
      <p className="text-xs text-muted">Unanswered questions cancel the run after 30 minutes.</p>
      {error && <p className="text-sm text-bad">{error}</p>}
    </form>
  );
}
