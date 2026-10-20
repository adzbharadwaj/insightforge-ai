"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { api, download } from "@/lib/api";
import type { ClaimCheck, ReportView, WrittenSentence } from "@/lib/types";
import { EntityGraphView } from "./entity-graph";
import { Badge, ConfidenceBadge, formatDate, Panel } from "./ui";

interface Citation {
  sourceId: string;
  evidenceId: string;
}

function citationsFor(sentence: WrittenSentence, report: ReportView): Citation[] {
  const seen = new Set<string>();
  const out: Citation[] = [];
  for (const eid of sentence.evidence_ids) {
    const sid = report.evidence[eid]?.source_ids[0];
    if (sid && !seen.has(sid)) {
      seen.add(sid);
      out.push({ sourceId: sid, evidenceId: eid });
    }
  }
  return out;
}

function slug(s: string) {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
}

export function ReportViewer({ report, shareToken }: { report: ReportView; shareToken?: string }) {
  const [open, setOpen] = useState<Citation | null>(null);
  const [tab, setTab] = useState<"report" | "entities">("report");
  const openEvidence = (evidenceId: string) => {
    const sourceId = report.evidence[evidenceId]?.source_ids[0];
    if (sourceId) setOpen({ sourceId, evidenceId });
  };
  const checks = useMemo(() => {
    const byClaim = new Map<string, ClaimCheck>();
    report.fact_check?.checks.forEach((c) => byClaim.set(c.claim_id, c));
    const byLoc = new Map<string, ClaimCheck>();
    report.structure?.claims.forEach((c) => {
      const check = byClaim.get(c.claim_id);
      if (check) byLoc.set(c.loc, check);
    });
    return byLoc;
  }, [report]);
  const confidence = new Map(report.sections.map((s) => [s.sub_task_id, s.confidence]));
  const structure = report.structure?.report;
  const toc = structure
    ? ["Executive Summary", ...structure.sections.map((s) => s.heading), "Gaps & Limitations", "Fact-check"]
    : [];

  function Sentence({ s, loc }: { s: WrittenSentence; loc: string }) {
    const check = checks.get(loc);
    const unverified = check && !check.verified;
    const cites = citationsFor(s, report);
    // Citation trace: clicking a cited sentence opens the source paragraph it was drawn from.
    const traceable = cites.length > 0;
    return (
      <>
        <span
          className={`${unverified ? "rounded bg-amber px-0.5" : ""} ${traceable ? "cursor-pointer decoration-dotted underline-offset-4 hover:underline" : ""}`}
          onClick={traceable ? () => setOpen(cites[0]) : undefined}
          title={traceable ? "Show the source paragraph" : undefined}
        >{s.text}</span>
        {cites.map((c) => {
          const src = report.sources[c.sourceId];
          return (
            <button key={c.sourceId} onClick={() => setOpen(c)}
                    className="ml-1 rounded border border-border px-1 py-px align-baseline text-[11px] text-muted hover:border-accent hover:text-accent">
              {src?.label ?? c.sourceId}, {src?.published_date ?? "n.d."}, {src?.credibility?.toFixed(2) ?? "n/a"}
            </button>
          );
        })}
        {unverified && (
          <span title={`Not verified against the source on re-fetch (${check.note || "low similarity"}).`}
                className="ml-1 cursor-help rounded bg-amber px-1 text-[11px] font-semibold text-warn">
            [UNVERIFIED]
          </span>
        )}{" "}
      </>
    );
  }

  return (
    <div className="grid gap-8 lg:grid-cols-[220px_1fr]">
      {structure && (
        <aside className="hidden lg:block">
          <nav className="sticky top-6 space-y-1 text-sm">
            <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted">Contents</p>
            {toc.map((t) => (
              <a key={t} href={`#${slug(t)}`} className="block truncate text-muted hover:text-text">{t}</a>
            ))}
          </nav>
        </aside>
      )}

      <article className="min-w-0 space-y-6">
        <header className="space-y-2">
          <h1 className="text-2xl font-semibold tracking-tight">{report.title}</h1>
          <p className="text-sm text-muted">Brief: {report.brief} · {formatDate(report.created_at)}</p>
          <Exports report={report} shareToken={shareToken} />
          {report.entity_graph && (
            <div className="flex gap-1 pt-2 text-sm" role="tablist">
              {(["report", "entities"] as const).map((t) => (
                <button key={t} role="tab" aria-selected={tab === t} onClick={() => setTab(t)}
                        className={`rounded-md px-3 py-1.5 ${tab === t ? "bg-accent text-white" : "text-muted hover:text-text"}`}>
                  {t === "report" ? "Report" : "Entity graph"}
                </button>
              ))}
            </div>
          )}
        </header>

        {tab === "entities" ? (
          <Panel><EntityGraphView report={report} onOpenEvidence={openEvidence} /></Panel>
        ) : structure ? (
          <>
            <Panel>
              <h2 id="executive-summary" className="mb-2 text-lg font-semibold">Executive Summary</h2>
              <p className="leading-relaxed">{structure.executive_summary.map((s) => s.text).join(" ")}</p>
            </Panel>
            {structure.sections.map((section, si) => (
              <Panel key={section.sub_task_id + si}>
                <h2 id={slug(section.heading)} className="text-lg font-semibold">{section.heading}</h2>
                <div className="mb-3 mt-1"><ConfidenceBadge value={confidence.get(section.sub_task_id) ?? "LOW"} /></div>
                <p className="leading-relaxed">
                  {section.sentences.map((s, i) => <Sentence key={i} s={s} loc={`section:${si}:${i}`} />)}
                </p>
              </Panel>
            ))}
            <Panel>
              <h2 id="gaps-limitations" className="mb-2 text-lg font-semibold">Gaps &amp; Limitations</h2>
              <ul className="list-disc space-y-1 pl-5 text-sm">
                {(structure.gaps.length ? structure.gaps : ["None identified."]).map((g, i) => <li key={i}>{g}</li>)}
              </ul>
            </Panel>
          </>
        ) : (
          <Panel className="prose-sm max-w-none [&_h1]:text-xl [&_h1]:font-semibold [&_h2]:mt-6 [&_h2]:text-lg [&_h2]:font-semibold [&_li]:ml-5 [&_li]:list-disc [&_p]:my-2">
            <ReactMarkdown remarkPlugins={[remarkGfm]} components={{
              strong: ({ children }) => String(children).startsWith("[UNVERIFIED]")
                ? <span className="rounded bg-amber px-1 text-xs font-semibold text-warn">{children}</span>
                : <strong>{children}</strong>,
            }}>{report.markdown}</ReactMarkdown>
          </Panel>
        )}

        <FactCheck report={report} />
      </article>

      {open && <CitationDrawer report={report} citation={open} onClose={() => setOpen(null)} />}
    </div>
  );
}

function FactCheck({ report }: { report: ReportView }) {
  const fc = report.fact_check;
  if (!fc) return null;
  const rate = Math.round(fc.pass_rate * 100);
  return (
    <Panel>
      <h2 id="fact-check" className="mb-3 text-lg font-semibold">Fact-check</h2>
      <div className="flex flex-wrap gap-6 text-sm">
        <span><span className="text-2xl font-semibold">{rate}%</span> <span className="text-muted">pass rate</span></span>
        <span className="text-muted">{fc.passed} of {fc.total_checked} sampled claims verified against a fresh fetch of their source</span>
      </div>
      {fc.checks.some((c) => !c.verified) && (
        <ul className="mt-4 space-y-2 text-sm">
          {fc.checks.filter((c) => !c.verified).map((c) => (
            <li key={c.claim_id} className="rounded-md bg-amber p-2">
              <span className="font-semibold text-warn">[UNVERIFIED]</span> {c.text}
              <span className="block text-xs text-muted">
                similarity {c.similarity?.toFixed(2) ?? "n/a"} (needs 0.85) · checked against {c.checked_against}
              </span>
            </li>
          ))}
        </ul>
      )}
    </Panel>
  );
}

function CitationDrawer({ report, citation, onClose }: { report: ReportView; citation: Citation; onClose: () => void }) {
  const src = report.sources[citation.sourceId];
  const ev = report.evidence[citation.evidenceId];
  const panel = useRef<HTMLElement>(null);
  useEffect(() => {
    panel.current?.focus();  // keyboard users land in the drawer
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div className="fixed inset-0 z-40 flex justify-end bg-black/30" onClick={onClose}>
      <aside ref={panel} tabIndex={-1} role="dialog" aria-modal="true" aria-label="Source details"
             onClick={(e) => e.stopPropagation()}
             className="h-full w-full max-w-md space-y-4 overflow-y-auto border-l border-border bg-panel p-6 shadow-xl outline-none">
        <div className="flex items-start justify-between gap-4">
          <h2 className="text-base font-semibold">{src?.title ?? src?.label ?? "Source"}</h2>
          <button onClick={onClose} className="text-sm text-muted hover:text-text" aria-label="Close">✕</button>
        </div>
        <dl className="grid grid-cols-[110px_1fr] gap-y-1 text-sm">
          <dt className="text-muted">Publisher</dt><dd>{src?.label}</dd>
          <dt className="text-muted">Published</dt><dd>{src?.published_date ?? "unknown"}</dd>
          <dt className="text-muted">Credibility</dt><dd>{src?.credibility?.toFixed(2) ?? "n/a"}</dd>
          {ev && <><dt className="text-muted">Confidence</dt><dd><ConfidenceBadge value={ev.confidence} /></dd></>}
        </dl>
        {ev?.conflict && <Badge kind="warn">Sources disagree on this fact</Badge>}
        {ev && (
          <div>
            <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted">Evidence</p>
            <p className="text-sm">{ev.claim}</p>
          </div>
        )}
        {ev?.passage && (
          <div>
            <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted">Source paragraph</p>
            <blockquote className="border-l-2 border-accent bg-bg p-3 text-sm leading-relaxed">{ev.passage}</blockquote>
          </div>
        )}
        {src?.url && (
          <a href={src.url} target="_blank" rel="noopener noreferrer" className="block break-all text-sm text-accent hover:underline">
            {src.url}
          </a>
        )}
      </aside>
    </div>
  );
}

function Exports({ report, shareToken }: { report: ReportView; shareToken?: string }) {
  const [msg, setMsg] = useState<string | null>(null);
  const base = shareToken ? `/share/${shareToken}` : `/reports/${report.id}`;

  async function exportAs(format: "md" | "pdf") {
    setMsg(null);
    try {
      await download(`${base}/export?format=${format}`, `report.${format}`, !shareToken);
    } catch (err) {
      setMsg(err instanceof Error ? err.message : "Export failed");
    }
  }

  async function share() {
    setMsg(null);
    try {
      const link = await api<{ url: string; expires_at: string }>(`/reports/${report.id}/share`, { method: "POST" });
      await navigator.clipboard.writeText(link.url);
      setMsg(`Share link copied (read-only, expires ${formatDate(link.expires_at)}).`);
    } catch (err) {
      setMsg(err instanceof Error ? err.message : "Could not create a share link");
    }
  }

  const btn = "rounded-md border border-border px-3 py-1.5 text-sm hover:bg-bg";
  return (
    <div className="flex flex-wrap items-center gap-2">
      <button className={btn} onClick={() => exportAs("pdf")}>Download PDF</button>
      <button className={btn} onClick={() => exportAs("md")}>Download .md</button>
      {!shareToken && <button className={btn} onClick={share}>Copy share link</button>}
      {msg && <span className="text-sm text-muted">{msg}</span>}
    </div>
  );
}
