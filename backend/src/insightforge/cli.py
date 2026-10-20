"""Command-line vertical slice: run one brief through the real pipeline.

    uv run insightforge "Competitive landscape for B2B payroll software in Southeast Asia"
    uv run insightforge "Summarize pricing in these reports" --pdf a.pdf --pdf b.pdf
    uv run insightforge --resume <run-id>      # continue a failed run from its last checkpoint

Needs OPENROUTER_API_KEY and the search provider's key in .env, plus `docker compose up -d`.
Prints the task graph, per-sub-task results, token use, and the context-engineering
numbers (raw corpus tokens vs. what the Planner actually received).
"""

import argparse
import asyncio
import faulthandler
import signal
import sys
import uuid
from pathlib import Path

from sqlalchemy import select

from insightforge.config import REPO_ROOT, get_settings
from insightforge.graph.state import EntityGraph, ResearchResult
from insightforge.llm.tokens import count_tokens
from insightforge.runtime import Runtime, Upload, missing_keys
from insightforge.store.db import get_engine, get_sessionmaker
from insightforge.store.models import Run, RunStatus, User
from insightforge.store.scratch import ScratchStore
from insightforge.tools.docling_ingest import PdfRejected, save_upload, validate_pdfs

DEV_EMAIL = "dev@insightforge.local"
REPORTS_DIR = REPO_ROOT / "data" / "reports"


async def create_run(maker, brief: str, source_prefs: dict[str, bool]) -> tuple[uuid.UUID, uuid.UUID]:
    async with maker() as session:
        user = await session.scalar(select(User).where(User.email == DEV_EMAIL))
        if user is None:
            user = User(email=DEV_EMAIL)
            session.add(user)
            await session.flush()
        run = Run(user_id=user.id, brief=brief, source_prefs=source_prefs)
        session.add(run)
        await session.commit()
        return user.id, run.id


def print_results(results: list[ResearchResult]) -> None:
    for r in results:
        print(f"\n  {r.sub_task_id}  {r.status.value}" + (f"  ({r.error})" if r.error else ""))
        for ref in r.refs:
            print(f"    - {ref.source_id}  cred={ref.credibility}  {ref.summary}")


def print_entities(graph: EntityGraph | None) -> None:
    if not graph:
        print("\nentity graph: (none)")
        return
    print(f"\nentity graph: {len(graph.entities)} entities, {len(graph.relationships)} facts, "
          f"{len(graph.conflicts)} conflict(s)")
    for c in graph.conflicts:
        values = " vs ".join(f"{v.object} {v.source_ids}" for v in c.values)
        print(f"  CONFLICT {c.subject} {c.predicate.value}" + (f" ({c.qualifier})" if c.qualifier else "")
              + f": {values}")


def ask_conflicts(review: dict | None) -> dict[str, str]:
    """Interactive conflict review: pick an option number per conflict, or Enter to keep both."""
    choices = {}
    for c in (review or {}).get("conflicts", []):
        what = f"{c['subject']} {c['predicate'].replace('_', ' ')}" + (f" ({c['qualifier']})" if c["qualifier"] else "")
        print(f"\nSources disagree on {what}:")
        for i, o in enumerate(c["options"], start=1):
            print(f"  {i}. {o['value']}  [{', '.join(o['sources'])}]")
        pick = input("Use which? (number, Enter = keep both) > ").strip()
        if pick.isdigit() and 1 <= int(pick) <= len(c["options"]):
            choices[c["conflict_id"]] = c["options"][int(pick) - 1]["option_id"]
    return choices


def print_report(state: dict) -> None:
    if not state.get("final_markdown"):
        return
    path = REPORTS_DIR / f"{state['run_id']}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(state["final_markdown"])
    draft, fc = state["draft"], state["fact_check"]
    print(f"\nreport: {draft.report.title}\n  {len(draft.claims)} claim(s), {state['writer_attempts']} writer attempt(s)"
          f"\n  saved to {path}")
    print(f"fact-check: {fc.passed}/{fc.total_checked} passed (pass rate {fc.pass_rate:.0%})")
    for c in fc.checks:
        mark = "ok  " if c.verified else "FAIL"
        print(f"  {mark} {c.similarity or 0:.2f} [{c.checked_against}] {c.text[:110]}")


async def main_async(brief: str | None, source_prefs: dict[str, bool], pdfs: list[Path],
                     resume: uuid.UUID | None = None) -> int:
    settings = get_settings()
    if missing := missing_keys(settings):
        print(f"Missing in .env: {', '.join(missing)} (see .env.example)", file=sys.stderr)
        return 2
    files = [(p.name, p.read_bytes()) for p in pdfs]
    try:
        validate_pdfs(files)
    except PdfRejected as exc:
        print(f"PDF rejected: {exc}", file=sys.stderr)
        return 2

    maker = get_sessionmaker()
    async with Runtime.open(settings) as runtime:
        if resume:
            run_id = resume
            print(f"run {run_id}\nresuming from the last checkpoint…")
            try:
                outcome = await runtime.retry(run_id)
            except (LookupError, ValueError) as exc:
                print(str(exc), file=sys.stderr)
                return 2
        else:
            user_id, run_id = await create_run(maker, brief, source_prefs)
            uploads = [Upload(name, save_upload(user_id, run_id, name, data)) for name, data in files]
            if uploads:
                print(f"ingesting {len(uploads)} PDF(s)…")
            print(f"run {run_id}\nplanning…")
            outcome = await runtime.start(run_id, uploads)
        while outcome.status in (RunStatus.AWAITING_CLARIFICATION, RunStatus.AWAITING_REVIEW):
            if outcome.status is RunStatus.AWAITING_CLARIFICATION:
                answer = input(f"\nPlanner asks: {outcome.question}\n> ").strip()
                outcome = await runtime.resume(run_id, answer)
            else:
                outcome = await runtime.resolve(run_id, ask_conflicts(outcome.review))
    async with maker() as session:
        run = await session.get(Run, run_id)
    print(f"\nstatus: {run.status.value} (attempt {run.attempts})" + (f"  error: {run.error}" if run.error else ""))
    if run.status is RunStatus.FAILED:
        print(f"resume from the last checkpoint with: uv run insightforge --resume {run_id}")
    if outcome.state.get("task_graph"):
        print("\ntask graph:")
        for t in outcome.state["task_graph"].sub_tasks:
            print(f"  {t.id} p{t.priority} [{t.source_type.value}] {t.query}  ({t.time_horizon}; {t.scope})")
    results = outcome.state.get("research_results", [])
    print_results(results)
    print_entities(outcome.state.get("entity_graph"))

    corpus = await ScratchStore(maker, run_id).total_tokens()
    handed_up = sum(count_tokens(r.model_dump_json()) for r in results)
    print_report(outcome.state)
    print(f"\ntokens used (all models): {run.tokens_used:,}")
    for stage, used in outcome.state.get("stage_tokens", {}).items():
        print(f"  {stage:<11} {used:>8,}")
    print(f"context: raw corpus in scratch store ≈ {corpus:,} tokens; "
          f"Planner received ≈ {handed_up:,} tokens of references")
    if bundle := outcome.state.get("evidence_bundle"):
        print(f"         Writer received an evidence bundle of {bundle.token_count:,}/{bundle.token_budget:,} tokens "
              f"({len(bundle.cuts)} compression cut(s))")
    await get_engine().dispose()
    return 0 if run.status is not RunStatus.FAILED else 1


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)  # progress shows up even when output is piped to a file
    faulthandler.register(signal.SIGUSR1, all_threads=True)  # `kill -USR1 <pid>` dumps every stack
    parser = argparse.ArgumentParser(description="Run one research brief through the InsightForge pipeline.")
    parser.add_argument("brief", nargs="?", help="Research brief (max 500 characters)")
    parser.add_argument("--resume", type=uuid.UUID, metavar="RUN_ID",
                        help="Continue a failed run from its last checkpoint instead of starting a new one")
    parser.add_argument("--no-memory", action="store_true", help="Do not load prior session summaries")
    parser.add_argument("--no-web", action="store_true", help="Use only the uploaded PDFs")
    parser.add_argument("--pdf", action="append", type=Path, default=[], help="PDF to research (repeatable, max 10)")
    parser.add_argument("--review-conflicts", action="store_true",
                        help="Pause before writing to choose between conflicting facts")
    args = parser.parse_args()
    if args.resume:
        sys.exit(asyncio.run(main_async(None, {}, [], resume=args.resume)))
    if not args.brief:
        parser.error("a brief is required (or --resume RUN_ID)")
    if len(args.brief) > get_settings().brief_max_chars:
        parser.error(f"brief is longer than {get_settings().brief_max_chars} characters")
    if missing := [p for p in args.pdf if not p.is_file()]:
        parser.error(f"not a file: {', '.join(map(str, missing))}")
    if args.no_web and not args.pdf:
        parser.error("--no-web needs at least one --pdf")
    prefs = {"web": not args.no_web, "docs": bool(args.pdf), "memory": not args.no_memory,
             "review_conflicts": args.review_conflicts}
    sys.exit(asyncio.run(main_async(args.brief, prefs, args.pdf)))


if __name__ == "__main__":
    main()
