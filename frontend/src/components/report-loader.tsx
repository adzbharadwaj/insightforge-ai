"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { ReportView } from "@/lib/types";
import { ReportViewer } from "./report-viewer";

export function ReportLoader({ path, shareToken }: { path: string; shareToken?: string }) {
  const [report, setReport] = useState<ReportView | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    api<ReportView>(path).then(setReport).catch((e) => setError(e instanceof Error ? e.message : "Not found"));
  }, [path]);
  if (error) return <p className="text-sm text-bad">{error}</p>;
  if (!report) return <p className="text-sm text-muted">Loading report…</p>;
  return <ReportViewer report={report} shareToken={shareToken} />;
}
