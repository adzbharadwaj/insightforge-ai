import type { Confidence, RunStatus, SubTaskStatus } from "@/lib/types";

const tone = {
  ok: "bg-ok/15 text-ok border-ok/30",
  warn: "bg-warn/15 text-warn border-warn/30",
  bad: "bg-bad/15 text-bad border-bad/30",
  info: "bg-accent/10 text-accent border-accent/30",
  muted: "bg-muted/10 text-muted border-border",
};

export function Badge({ kind, children }: { kind: keyof typeof tone; children: React.ReactNode }) {
  return (
    <span className={`inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-medium ${tone[kind]}`}>
      {children}
    </span>
  );
}

export function ConfidenceBadge({ value }: { value: Confidence }) {
  const kind = value === "HIGH" ? "ok" : value === "MEDIUM" ? "warn" : "bad";
  return <Badge kind={kind}>Confidence: {value}</Badge>;
}

export function SubTaskBadge({ status }: { status: SubTaskStatus }) {
  const kind = status === "Done" ? "ok" : status === "Failed" ? "bad" : status === "Running" ? "info" : "muted";
  return <Badge kind={kind}>{status}</Badge>;
}

export function RunStatusBadge({ status }: { status: RunStatus }) {
  const kind = status === "COMPLETE" ? "ok" : status === "FAILED" || status === "CANCELLED" ? "bad"
    : status === "AWAITING_CLARIFICATION" ? "warn" : "info";
  return <Badge kind={kind}>{status.replaceAll("_", " ").toLowerCase()}</Badge>;
}

export function Panel({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <section className={`rounded-lg border border-border bg-panel p-5 ${className}`}>{children}</section>;
}

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds == null) return "–";
  const s = Math.max(0, Math.round(seconds));
  return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
}

export function formatDate(iso: string | null | undefined): string {
  if (!iso) return "–";
  return new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}
