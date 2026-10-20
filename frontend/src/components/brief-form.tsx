"use client";

import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import { formatDuration, Panel } from "./ui";

const MAX_CHARS = 500;
const MAX_FILES = 10;
const MAX_MB = 20;

function Toggle({ label, checked, onChange, disabled }: {
  label: string; checked: boolean; onChange: (v: boolean) => void; disabled?: boolean;
}) {
  return (
    <label className={`flex items-center gap-2 text-sm ${disabled ? "opacity-50" : ""}`}>
      <input type="checkbox" checked={checked} disabled={disabled} onChange={(e) => onChange(e.target.checked)}
             className="h-4 w-4 accent-[var(--accent)]" />
      {label}
    </label>
  );
}

export function BriefForm() {
  const router = useRouter();
  const [brief, setBrief] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [web, setWeb] = useState(true);
  const [memory, setMemory] = useState(true);
  const [eta, setEta] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const docs = files.length > 0;

  useEffect(() => {
    // Sub-task count is decided by the Planner; estimate with the typical 5 until it runs.
    api<{ eta_seconds: number }>(`/research/estimate?sub_tasks=5&pdfs=${files.length}`)
      .then((r) => setEta(r.eta_seconds))
      .catch(() => setEta(null));
  }, [files.length]);

  function addFiles(list: FileList | null) {
    if (!list) return;
    setError(null);
    const next = [...files];
    for (const f of Array.from(list)) {
      if (!f.name.toLowerCase().endsWith(".pdf") && f.type !== "application/pdf") {
        setError(`${f.name} is not a PDF.`);
        continue;
      }
      if (f.size > MAX_MB * 1024 * 1024) {
        setError(`${f.name} is larger than ${MAX_MB} MB.`);
        continue;
      }
      if (next.length >= MAX_FILES) {
        setError(`At most ${MAX_FILES} PDFs per brief.`);
        break;
      }
      if (!next.some((x) => x.name === f.name && x.size === f.size)) next.push(f);
    }
    setFiles(next);
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    const form = new FormData();
    form.set("brief", brief);
    form.set("web", String(web));
    form.set("docs", String(docs));
    form.set("memory", String(memory));
    files.forEach((f) => form.append("files", f));
    try {
      const res = await api<{ run_id: string }>("/research/run", { method: "POST", body: form });
      router.push(`/run/${res.run_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not start the run.");
      setBusy(false);
    }
  }

  const remaining = MAX_CHARS - brief.length;
  return (
    <form onSubmit={submit} className="space-y-6">
      <Panel>
        <label htmlFor="brief" className="mb-2 block text-sm font-medium">Research brief</label>
        <textarea
          id="brief"
          value={brief}
          maxLength={MAX_CHARS}
          onChange={(e) => setBrief(e.target.value)}
          rows={5}
          placeholder="e.g. Competitive landscape for B2B payroll software in Southeast Asia: key players, pricing models and recent funding."
          className="w-full resize-y rounded-md border border-border bg-bg p-3 text-sm leading-relaxed"
        />
        <div className={`mt-1 text-right text-xs ${remaining < 50 ? "text-warn" : "text-muted"}`}>
          {brief.length} / {MAX_CHARS}
        </div>
      </Panel>

      <Panel>
        <div
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => { e.preventDefault(); setDragging(false); addFiles(e.dataTransfer.files); }}
          onClick={() => input.current?.click()}
          className={`cursor-pointer rounded-md border-2 border-dashed p-6 text-center text-sm transition-colors ${
            dragging ? "border-accent bg-accent/5" : "border-border text-muted"}`}
        >
          Drop PDFs here or click to choose · up to {MAX_FILES} files, {MAX_MB} MB each
          <input ref={input} type="file" accept="application/pdf,.pdf" multiple hidden
                 onChange={(e) => { addFiles(e.target.files); e.target.value = ""; }} />
        </div>
        {files.length > 0 && (
          <ul className="mt-3 space-y-1 text-sm">
            {files.map((f) => (
              <li key={f.name + f.size} className="flex items-center justify-between">
                <span className="truncate">{f.name} <span className="text-muted">({(f.size / 1e6).toFixed(1)} MB)</span></span>
                <button type="button" onClick={() => setFiles(files.filter((x) => x !== f))}
                        className="text-xs text-muted hover:text-bad">Remove</button>
              </li>
            ))}
          </ul>
        )}
      </Panel>

      <Panel className="flex flex-wrap items-center gap-6">
        <Toggle label="Web search" checked={web} onChange={setWeb} />
        <Toggle label="Uploaded docs" checked={docs} onChange={() => {}} disabled={!docs} />
        <Toggle label="Prior reports" checked={memory} onChange={setMemory} />
        <span className="ml-auto text-sm text-muted">Estimated runtime: ~{formatDuration(eta)}</span>
      </Panel>

      {error && <p className="rounded-md border border-bad/30 bg-bad/10 p-3 text-sm text-bad">{error}</p>}
      <button
        disabled={busy || brief.trim().length < 10 || (!web && !docs)}
        className="rounded-md bg-accent px-5 py-2.5 text-sm font-medium text-white disabled:opacity-50"
      >
        {busy ? "Starting…" : "Run research"}
      </button>
    </form>
  );
}
