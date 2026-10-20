"""Evaluation runner: score finished runs on the sample briefs (deliverable 7).

    uv run insightforge-eval --brief payroll-sea      run the brief live, then score it
    uv run insightforge-eval --run-id <uuid>          score a run that already finished (no research cost)
    uv run insightforge-eval --report                 rebuild eval/report.md from eval/results/*.json

Metrics per run:

- fact-check pass rate   from the run's own Fact-Checker (sampled claims vs re-fetched sources)
- citation accuracy      every claim (not a sample) judged against the full source paragraphs it cites, by
                         the primary model: a different, stronger judge than the in-run grounding check
- synthesis quality      rubric judge (coverage, structure, faithfulness, gaps honesty, clarity; 1-5)
                         plus deterministic checks: required sections, confidence labels, citation format,
                         summary length, and coverage of the brief's must-cover topics
- context engineering    raw corpus tokens vs what the Planner and the Writer actually received
- cost and reliability   tokens per stage, wall time, writer attempts, resume attempts
"""

import argparse
import asyncio
import datetime as dt
import json
import re
import sys
import uuid
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from sqlalchemy import select

from insightforge.config import REPO_ROOT, get_settings
from insightforge.llm.gateway import ModelGateway, Role
from insightforge.llm.tokens import count_tokens
from insightforge.store.db import get_engine, get_sessionmaker
from insightforge.store.models import Claim, Report, Run, RunEvent, RunStatus, SubTask, User
from insightforge.store.scratch import ScratchStore

BRIEFS = REPO_ROOT / "briefs" / "briefs.json"
RESULTS = REPO_ROOT / "eval" / "results"
REPORT = REPO_ROOT / "eval" / "report.md"
EVAL_EMAIL = "eval@insightforge.local"
JUDGE_BATCH = 12

CITATION_PROMPT = """You audit citations in a research report. For each claim you get the source paragraphs it cites.
A claim is supported only if the cited paragraphs state every fact in it (names, numbers, dates, comparisons).
Paraphrase is fine; facts not in the paragraphs, different numbers, or stronger wording are not. Attributing a fact
to the paragraph's listed publisher (or that publisher's name, e.g. "RAC" for rac.co.uk) is fine.
Write a short reason first, then decide. One verdict per claim_id."""

RUBRIC_PROMPT = """You review a market-intelligence report against its brief and the expected structure.
Score each criterion from 1 (poor) to 5 (excellent), with a one-sentence reason:
- coverage: does it answer every part of the brief?
- structure: does it follow the expected sections in a sensible order, with clear headings?
- faithfulness: does it read as evidence-based (specific, dated, attributed; conflicts shown, no hype)?
- gaps_honesty: are limitations and unresolved questions stated plainly?
- clarity: is it concise and readable for an analyst?"""


class CitationVerdict(BaseModel):
    claim_id: str
    reason: str = Field(default="", max_length=300)
    supported: bool


class CitationVerdicts(BaseModel):
    verdicts: list[CitationVerdict]


class Criterion(BaseModel):
    reason: str = Field(max_length=300)
    score: int = Field(ge=1, le=5)


class Rubric(BaseModel):
    coverage: Criterion
    structure: Criterion
    faithfulness: Criterion
    gaps_honesty: Criterion
    clarity: Criterion


def load_briefs() -> dict[str, dict]:
    return {b["id"]: b for b in json.loads(BRIEFS.read_text())}


# --- deterministic checks -----------------------------------------------------------------


def structure_checks(markdown: str, structure: dict | None, brief: dict | None) -> dict:
    report = (structure or {}).get("report") or {}
    sections = report.get("sections", [])
    summary_words = sum(len(s["text"].split()) for s in report.get("executive_summary", []))
    checks = {
        "has_executive_summary": "## Executive Summary" in markdown,
        "summary_within_150_words": summary_words <= 150 if report else None,
        "has_gaps_section": "## Gaps & Limitations" in markdown,
        "every_section_has_confidence": markdown.count("**Confidence: ") >= len(sections) > 0,
        "uses_citation_format": bool(re.search(r"\[Source: [^,\]]+, [^,\]]+, credibility: [\d.]+|n/a\]", markdown)),
        "thematic_sections": len(sections),
    }
    if brief:
        low = markdown.lower()
        hits = [t for t in brief["must_cover"] if t.lower() in low]
        checks["must_cover_hit"] = hits
        checks["must_cover_rate"] = round(len(hits) / len(brief["must_cover"]), 3)
    return checks


# --- judges -------------------------------------------------------------------------------


async def citation_accuracy(gateway: ModelGateway, claims: list[Claim], evidence: dict[str, dict],
                            sources: dict[str, dict] | None = None) -> dict:
    checkable = [c for c in claims if c.evidence_ids]
    sources = sources or {}

    def publisher(e: str) -> str:
        # The report attributes facts to publishers ("RAC reports ..."); the judge must see who published what.
        labels = [sources[s]["label"] for s in evidence[e]["source_ids"] if s in sources]
        return ", ".join(dict.fromkeys(labels)) or "unknown"

    def block(c: Claim) -> str:
        cited = "\n".join(f"  [{e}] (published by {publisher(e)}) "
                          f"{(evidence[e].get('passage') or evidence[e]['claim'])[:900]}"
                          for e in c.evidence_ids if e in evidence)
        return f"claim_id: {c.claim_key}\nclaim: {c.text}\ncited paragraphs:\n{cited or '  (none resolvable)'}"

    batches = [checkable[i:i + JUDGE_BATCH] for i in range(0, len(checkable), JUDGE_BATCH)]
    results = await asyncio.gather(*(
        gateway.structured(Role.SYNTHESIS, CitationVerdicts).ainvoke(
            [SystemMessage(CITATION_PROMPT), HumanMessage("\n\n".join(block(c) for c in batch))])
        for batch in batches))
    asked = {c.claim_key for c in checkable}
    verdicts = {v.claim_id: v for r in results for v in r.verdicts if v.claim_id in asked}
    supported = sum(v.supported for v in verdicts.values())
    unresolvable = sum(1 for c in checkable if not any(e in evidence for e in c.evidence_ids))
    return {
        "claims_checked": len(verdicts),
        "supported": supported,
        "accuracy": round(supported / len(verdicts), 3) if verdicts else None,
        "unresolvable_citations": unresolvable,
        "unsupported": [{"claim_id": k, "reason": v.reason} for k, v in verdicts.items() if not v.supported],
    }


async def synthesis_rubric(gateway: ModelGateway, markdown: str, brief: dict | None, brief_text: str) -> dict:
    expected = "\n".join(f"- {s}" for s in (brief or {}).get("expected_sections", [])) or "(not specified)"
    notes = (brief or {}).get("notes", "")
    body = markdown if count_tokens(markdown) < 12_000 else markdown[:40_000]
    rubric: Rubric = await gateway.structured(Role.SYNTHESIS, Rubric).ainvoke([
        SystemMessage(RUBRIC_PROMPT),
        HumanMessage(f"Brief: {brief_text}\n\nExpected sections:\n{expected}\n\nReviewer notes: {notes}\n\n"
                     f"Report:\n{body}"),
    ])
    scores = rubric.model_dump()
    scores["overall"] = round(sum(c["score"] for c in scores.values()) / 5, 2)
    return scores


# --- one run --------------------------------------------------------------------------------


async def evaluate_run(run_id: uuid.UUID, brief_id: str | None) -> dict:
    maker = get_sessionmaker()
    briefs = load_briefs()
    async with maker() as s:
        run = await s.get(Run, run_id)
        if run is None:
            raise LookupError(f"run {run_id} not found")
        report = await s.scalar(select(Report).where(Report.run_id == run_id))
        claims = list((await s.scalars(select(Claim).where(Claim.report_id == report.id))).all()) if report else []
        tasks = (await s.scalars(select(SubTask).where(SubTask.run_id == run_id))).all()
        token_events = (await s.scalars(select(RunEvent).where(RunEvent.run_id == run_id,
                                                               RunEvent.kind == "tokens"))).all()
    brief = briefs.get(brief_id) if brief_id else next((b for b in briefs.values() if b["brief"] == run.brief), None)
    result: dict = {
        "run_id": str(run_id), "brief_id": brief["id"] if brief else None, "brief": run.brief,
        "evaluated_at": dt.datetime.now(dt.UTC).isoformat(), "models": [get_settings().model_primary,
                                                                      get_settings().model_secondary],
        "status": run.status.value, "attempts": run.attempts, "tokens_used": run.tokens_used,
        "wall_seconds": round((run.finished_at - run.started_at).total_seconds(), 1)
        if run.finished_at and run.started_at else None,
        "error": run.error,
    }
    stages: dict[str, int] = {}
    for e in token_events:
        stages[e.data["stage"]] = stages.get(e.data["stage"], 0) + e.data["tokens"]
    result["stage_tokens"] = stages
    result["writer_attempts"] = sum(1 for e in token_events if e.data["stage"] == "write")
    result["sub_tasks"] = {"total": len(tasks), "done": sum(t.status.value == "Done" for t in tasks)}
    if report is None:
        return result

    bundle = report.evidence_bundle or {}
    evidence = {i["evidence_id"]: i for s in bundle.get("sections", []) for i in s["evidence"]}
    corpus = await ScratchStore(maker, run_id).total_tokens()
    handed_up = sum(count_tokens(json.dumps(t.result)) for t in tasks if t.result)
    result["context"] = {
        "raw_corpus_tokens": corpus, "planner_received_tokens": handed_up,
        "writer_bundle_tokens": bundle.get("token_count"), "bundle_budget": bundle.get("token_budget"),
        "compression_cuts": len(bundle.get("cuts", [])), "evidence_items": len(evidence),
        "sources": len(bundle.get("sources", {})), "notes": bundle.get("notes", []),
    }
    fc = report.factcheck_summary or {}
    result["fact_check"] = {k: fc.get(k) for k in ("total_checked", "passed", "failed", "pass_rate")}
    result["structure"] = structure_checks(report.markdown, report.structure, brief)

    judge = ModelGateway(get_settings())  # outside the run's budget; primary model as an independent judge
    result["citations"] = await citation_accuracy(judge, claims, evidence, bundle.get("sources", {}))
    result["quality"] = await synthesis_rubric(judge, report.markdown, brief, run.brief)
    result["report_id"] = str(report.id)
    return result


async def run_brief(brief_id: str) -> uuid.UUID:
    from insightforge.runtime import Runtime, missing_keys

    briefs = load_briefs()
    if brief_id not in briefs:
        raise SystemExit(f"unknown brief {brief_id!r}; choose from {', '.join(briefs)}")
    settings = get_settings()
    if missing := missing_keys(settings):
        raise SystemExit(f"Missing in .env: {', '.join(missing)}")
    b = briefs[brief_id]
    maker = get_sessionmaker()
    async with maker() as s:
        user = await s.scalar(select(User).where(User.email == EVAL_EMAIL))
        if user is None:
            user = User(email=EVAL_EMAIL)
            s.add(user)
            await s.flush()
        run = Run(user_id=user.id, brief=b["brief"], source_prefs=b["source_prefs"])
        s.add(run)
        await s.commit()
    print(f"run {run.id}: {b['brief']}", flush=True)
    async with Runtime.open(settings) as runtime:
        outcome = await runtime.run_with_auto_resume(await runtime.start(run.id), run.id)
        if outcome.status is RunStatus.AWAITING_CLARIFICATION:  # sample briefs are specific; answer generically
            outcome = await runtime.resume(run.id, "Use the scope as written in the brief.")
    print(f"status: {outcome.status.value}", flush=True)
    return run.id


# --- report ---------------------------------------------------------------------------------


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{x * 100:.0f}%"


def write_report() -> Path:
    results = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("*.json"))]
    briefs = load_briefs()
    lines = ["# Evaluation report", "",
             f"Generated {dt.date.today().isoformat()} from `eval/results/*.json` by `uv run insightforge-eval --report`.",
             "", f"Runs evaluated: {len(results)} of {len(briefs)} sample briefs (see `briefs/briefs.json`).", "",
             "| Brief | Status | Fact-check | Citation accuracy | Quality (1-5) | Must-cover | Tokens | Time |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        fc, cit, q, st = r.get("fact_check", {}), r.get("citations", {}), r.get("quality", {}), r.get("structure", {})
        fc_cell = f"{fc.get('passed')}/{fc.get('total_checked')} ({_pct(fc.get('pass_rate'))})" if fc else "–"
        cit_cell = f"{cit.get('supported')}/{cit.get('claims_checked')} ({_pct(cit.get('accuracy'))})" if cit else "–"
        wall = f"{r['wall_seconds']:.0f} s" if r.get("wall_seconds") else "–"
        lines.append(f"| {r.get('brief_id') or r['brief'][:40]} | {r['status']} | {fc_cell} | {cit_cell} | "
                     f"{q.get('overall', '–')} | {_pct(st.get('must_cover_rate'))} | {r['tokens_used']:,} | {wall} |")
    for r in results:
        lines += ["", f"## {r.get('brief_id') or r['run_id']}", "", f"**Brief:** {r['brief']}", "",
                  (f"Run `{r['run_id']}` · models {', '.join(r.get('models', []))} · attempts {r.get('attempts')} · "
                  f"writer attempts {r.get('writer_attempts')} · sub-tasks {r['sub_tasks']['done']}/{r['sub_tasks']['total']} done")]
        if c := r.get("context"):
            ratio = c["raw_corpus_tokens"] / max(1, c["writer_bundle_tokens"] or 1)
            lines += ["", ("**Context engineering:** raw corpus "
                      f"{c['raw_corpus_tokens']:,} tokens → Planner received {c['planner_received_tokens']:,} → Writer "
                      f"received a {c['writer_bundle_tokens']:,}/{c['bundle_budget']:,}-token evidence bundle "
                      f"({ratio:.0f}x smaller than the corpus; {c['evidence_items']} evidence items from "
                      f"{c['sources']} sources; {c['compression_cuts']} compression cut(s)).")]
            lines += [f"- Excluded by time horizon: {n}" for n in c.get("notes", [])]
        if stg := r.get("stage_tokens"):
            lines += ["", "**Tokens by stage:** " + ", ".join(f"{k} {v:,}" for k, v in stg.items())]
        if q := r.get("quality"):
            lines += ["", "**Synthesis quality (rubric judge):**", ""]
            lines += [f"- {k.replace('_', ' ')}: {v['score']}/5 — {v['reason']}" for k, v in q.items() if k != "overall"]
        if st := r.get("structure"):
            lines += ["", "**Structure checks:** " + ", ".join(
                f"{k.replace('_', ' ')}: {v}" for k, v in st.items() if k not in ("must_cover_hit",))]
        if (cit := r.get("citations")) and cit.get("unsupported"):
            lines += ["", "**Claims the citation judge found unsupported:**", ""]
            lines += [f"- `{u['claim_id']}`: {u['reason']}" for u in cit["unsupported"]]
    missing = [b for b in briefs if b not in {r.get("brief_id") for r in results}]
    if missing:
        lines += ["", "## Not yet run", "", "Run with `uv run insightforge-eval --brief <id>`: " + ", ".join(
            f"`{m}`" for m in missing) + "."]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n")
    return REPORT


async def main_async(args) -> int:
    try:
        if args.report:
            print(f"wrote {write_report()}")
            return 0
        run_id = args.run_id or await run_brief(args.brief)
        result = await evaluate_run(run_id, args.brief)
        RESULTS.mkdir(parents=True, exist_ok=True)
        path = RESULTS / f"{result['brief_id'] or 'adhoc'}-{str(run_id)[:8]}.json"
        path.write_text(json.dumps(result, indent=2, default=str))
        print(f"wrote {path}\nwrote {write_report()}")
        print(json.dumps({k: result.get(k) for k in ("status", "fact_check", "citations", "quality")},
                         indent=2, default=str)[:3000])
        return 0 if result["status"] == "COMPLETE" else 1
    finally:
        await get_engine().dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate InsightForge runs on the sample briefs.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--brief", help="Run this sample brief live, then score it")
    group.add_argument("--run-id", type=uuid.UUID, help="Score a finished run (no research cost)")
    group.add_argument("--report", action="store_true", help="Rebuild eval/report.md from saved results")
    parser.add_argument("--as-brief", dest="brief_for_run", help="With --run-id: which sample brief it answered")
    args = parser.parse_args()
    if args.run_id and args.brief_for_run:
        args.brief = args.brief_for_run
    sys.exit(asyncio.run(main_async(args)))
