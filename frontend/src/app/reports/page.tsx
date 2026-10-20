"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { RequireAuth, useToken } from "@/components/auth";
import { formatDate, Panel } from "@/components/ui";
import { api } from "@/lib/api";
import type { ReportSummary } from "@/lib/types";

function ReportsList() {
  const token = useToken();
  const [reports, setReports] = useState<ReportSummary[] | null>(null);
  useEffect(() => {
    if (token) api<ReportSummary[]>("/reports?limit=50").then(setReports).catch(() => setReports([]));
  }, [token]);
  if (!reports) return <p className="text-sm text-muted">Loading…</p>;
  if (!reports.length) return <p className="text-sm text-muted">No reports yet. <Link href="/" className="text-accent">Run a brief</Link>.</p>;
  return (
    <Panel className="p-0">
      <ul className="divide-y divide-border">
        {reports.map((r) => (
          <li key={r.id}>
            <Link href={`/report/${r.id}`} className="flex items-center gap-4 px-5 py-3 hover:bg-bg">
              <span className="min-w-0 flex-1">
                <span className="block truncate font-medium">{r.title}</span>
                <span className="block truncate text-xs text-muted">{r.brief}</span>
              </span>
              <span className="text-xs text-muted">
                {r.pass_rate != null ? `fact-check ${Math.round(r.pass_rate * 100)}% · ` : ""}{formatDate(r.created_at)}
              </span>
            </Link>
          </li>
        ))}
      </ul>
    </Panel>
  );
}

export default function ReportsPage() {
  return (
    <RequireAuth>
      <h1 className="mb-6 text-2xl font-semibold tracking-tight">Reports</h1>
      <ReportsList />
    </RequireAuth>
  );
}
