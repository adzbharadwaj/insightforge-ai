"use client";

import { useMemo, useState } from "react";
import type { EntityGraph as Graph, ReportView } from "@/lib/types";
import { Badge } from "./ui";

const MAX_NODES = 40;
const SHOWN_TYPES = new Set(["company", "person", "product"]);
const W = 760;
const H = 460;
const TYPE_COLOR: Record<string, string> = { company: "var(--accent)", person: "var(--ok)", product: "var(--warn)" };

interface Node {
  key: string;
  name: string;
  type: string;
  degree: number;
  conflict: boolean;
  x: number;
  y: number;
}

// Small deterministic force layout (repulsion + springs + centring): enough for ≤ 40 nodes, no dependency.
function layout(nodes: Node[], edges: [number, number][]): Node[] {
  const n = nodes.length;
  const pos = nodes.map((node, i) => {
    const angle = (2 * Math.PI * i) / Math.max(1, n);
    return { ...node, x: W / 2 + 160 * Math.cos(angle), y: H / 2 + 160 * Math.sin(angle) };
  });
  // Spring length from the area per node; repulsion only within a few spring lengths, plus gravity, so that
  // entities with no links settle into an even cloud instead of being pushed against the walls.
  const k = Math.sqrt((W * H) / Math.max(1, n)) * 0.42;
  for (let step = 0, temp = 60; step < 300; step++, temp *= 0.985) {
    const disp = pos.map(() => ({ x: 0, y: 0 }));
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const dx = pos[i].x - pos[j].x, dy = pos[i].y - pos[j].y;
        const d = Math.max(1, Math.hypot(dx, dy));
        if (d > 3 * k) continue;
        const f = (k * k) / d;
        disp[i].x += (dx / d) * f; disp[i].y += (dy / d) * f;
        disp[j].x -= (dx / d) * f; disp[j].y -= (dy / d) * f;
      }
    }
    for (const [a, b] of edges) {
      const dx = pos[a].x - pos[b].x, dy = pos[a].y - pos[b].y;
      const d = Math.max(1, Math.hypot(dx, dy));
      const f = (d * d) / k;
      disp[a].x -= (dx / d) * f; disp[a].y -= (dy / d) * f;
      disp[b].x += (dx / d) * f; disp[b].y += (dy / d) * f;
    }
    for (let i = 0; i < n; i++) {
      disp[i].x += (W / 2 - pos[i].x) * 0.08;
      disp[i].y += (H / 2 - pos[i].y) * 0.12;  // flatter canvas: pull harder vertically
      const d = Math.max(1, Math.hypot(disp[i].x, disp[i].y));
      pos[i].x = Math.min(W - 30, Math.max(30, pos[i].x + (disp[i].x / d) * Math.min(d, temp)));
      pos[i].y = Math.min(H - 20, Math.max(20, pos[i].y + (disp[i].y / d) * Math.min(d, temp)));
    }
  }
  return pos;
}

function buildGraph(graph: Graph) {
  const byName = new Map(graph.entities.map((e) => [e.name.toLowerCase(), e.key]));
  const degree = new Map<string, number>();
  const links: [string, string][] = [];
  for (const r of graph.relationships) {
    degree.set(r.subject, (degree.get(r.subject) ?? 0) + 1);
    const target = byName.get(r.object.toLowerCase());
    if (target && target !== r.subject) {
      links.push([r.subject, target]);
      degree.set(target, (degree.get(target) ?? 0) + 1);
    }
  }
  const conflicted = new Set(graph.conflicts.map((c) => c.subject));
  const chosen = graph.entities
    .filter((e) => SHOWN_TYPES.has(e.type))
    .sort((a, b) => (degree.get(b.key) ?? 0) - (degree.get(a.key) ?? 0))
    .slice(0, MAX_NODES);
  const index = new Map(chosen.map((e, i) => [e.key, i]));
  const nodes: Node[] = chosen.map((e) => ({
    key: e.key, name: e.name, type: e.type, degree: degree.get(e.key) ?? 0, conflict: conflicted.has(e.key), x: 0, y: 0,
  }));
  const edgeSet = new Set<string>();
  const edges: [number, number][] = [];
  for (const [a, b] of links) {
    const ia = index.get(a), ib = index.get(b);
    const id = `${Math.min(ia ?? -1, ib ?? -1)}-${Math.max(ia ?? -1, ib ?? -1)}`;
    if (ia !== undefined && ib !== undefined && !edgeSet.has(id)) {
      edgeSet.add(id);
      edges.push([ia, ib]);
    }
  }
  return { nodes: layout(nodes, edges), edges, total: graph.entities.length };
}

export function EntityGraphView({ report, onOpenEvidence }: {
  report: ReportView; onOpenEvidence: (evidenceId: string) => void;
}) {
  const graph = report.entity_graph;
  const built = useMemo(() => (graph ? buildGraph(graph) : null), [graph]);
  const [selected, setSelected] = useState<string | null>(null);
  if (!graph || !built || built.nodes.length === 0) {
    return <p className="text-sm text-muted">No entity graph for this report.</p>;
  }
  const node = built.nodes.find((n) => n.key === selected);
  const facts = selected ? graph.relationships.filter((r) => r.subject === selected) : [];
  const claims = selected ? Object.values(report.evidence).filter((e) => e.entity_keys.includes(selected)) : [];

  return (
    <div className="grid gap-4 lg:grid-cols-[1fr_300px]">
      <div className="overflow-hidden rounded-md border border-border bg-bg">
        <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label="Entity relationship graph">
          {built.edges.map(([a, b], i) => (
            <line key={i} x1={built.nodes[a].x} y1={built.nodes[a].y} x2={built.nodes[b].x} y2={built.nodes[b].y}
                  stroke="var(--border)" strokeWidth={1.5} />
          ))}
          {built.nodes.map((n) => {
            const r = 6 + Math.min(10, n.degree);
            return (
              <g key={n.key} transform={`translate(${n.x},${n.y})`} onClick={() => setSelected(n.key)}
                 className="cursor-pointer" role="button" aria-label={n.name}>
                {n.conflict && <circle r={r + 4} fill="none" stroke="var(--bad)" strokeWidth={2} />}
                <circle r={r} fill={TYPE_COLOR[n.type] ?? "var(--muted)"} opacity={selected && selected !== n.key ? 0.45 : 0.9}
                        stroke={selected === n.key ? "var(--text)" : "none"} strokeWidth={2} />
                <text y={r + 12} textAnchor="middle" fontSize={11} fill="var(--text)">{n.name.slice(0, 22)}</text>
              </g>
            );
          })}
        </svg>
        <p className="border-t border-border px-3 py-2 text-xs text-muted">
          {built.nodes.length} of {built.total} entities (companies, people, products; most connected first) ·
          <span className="text-bad"> red ring</span> = conflicting facts · click a node for its claims and sources
        </p>
      </div>

      <aside className="space-y-3 text-sm">
        {!node ? <p className="text-muted">Select an entity.</p> : (
          <>
            <div>
              <p className="font-semibold">{node.name}</p>
              <p className="text-xs capitalize text-muted">{node.type}</p>
              {node.conflict && <Badge kind="bad">Sources disagree on a fact</Badge>}
            </div>
            {facts.length > 0 && (
              <div>
                <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted">Facts</p>
                <ul className="space-y-1">
                  {facts.slice(0, 12).map((f, i) => (
                    <li key={i} className={f.conflict ? "text-bad" : ""}>
                      {f.predicate.replaceAll("_", " ")} {f.object}{f.qualifier ? ` (${f.qualifier})` : ""}
                      <span className="block text-xs text-muted">{report.sources[f.source_id]?.label ?? f.source_url ?? f.source_id}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {claims.length > 0 && (
              <div>
                <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted">In this report</p>
                <ul className="space-y-1">
                  {claims.map((c) => (
                    <li key={c.evidence_id}>
                      <button onClick={() => onOpenEvidence(c.evidence_id)} className="text-left hover:text-accent">{c.claim}</button>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </>
        )}
      </aside>
    </div>
  );
}
