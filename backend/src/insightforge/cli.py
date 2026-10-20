"""Command-line vertical slice: run one brief through the real pipeline.

    uv run insightforge "Competitive landscape for B2B payroll software in Southeast Asia"
    uv run insightforge "Summarize pricing in these reports" --pdf a.pdf --pdf b.pdf

Needs OPENROUTER_API_KEY and the search provider's key in .env, plus `docker compose up -d`.
Prints the task graph, per-sub-task results, token use, and the context-engineering
numbers (raw corpus tokens vs. what the Planner actually received).
"""

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

import httpx
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command
from sqlalchemy import select

from insightforge.agents.doc_reader import doc_reader_spec
from insightforge.agents.web_search import web_researcher_spec
from insightforge.config import Settings, get_settings
from insightforge.graph.pipeline import PipelineDeps, RunContext, build_pipeline, execute
from insightforge.graph.state import EntityGraph, ResearchResult
from insightforge.llm.gateway import ModelGateway
from insightforge.llm.tokens import count_tokens
from insightforge.store.db import get_engine, get_sessionmaker
from insightforge.store.entity_store import load_related_entities
from insightforge.store.memory import load_recent_summaries
from insightforge.store.models import Run, RunStatus, User
from insightforge.store.qdrant import VectorStore, make_client
from insightforge.store.run_log import RunLog
from insightforge.store.scratch import ScratchStore
from insightforge.tools.doc_search import DocSearcher, make_doc_search_tool
from insightforge.tools.docling_ingest import PdfRejected, ingest_pdf, save_upload, validate_pdfs
from insightforge.tools.fetch import PageFetcher
from insightforge.tools.search_providers import make_search_provider
from insightforge.tools.web_search import WebResearcher, make_web_search_tool

DEV_EMAIL = "dev@insightforge.local"


def checkpointer_url(settings: Settings) -> str:
    return settings.database_url.replace("postgresql+asyncpg://", "postgresql://", 1)


def missing_keys(settings: Settings) -> list[str]:
    missing = [] if settings.openrouter_api_key else ["OPENROUTER_API_KEY"]
    key = getattr(settings, f"{settings.search_provider}_api_key")
    if key is None:
        missing.append(f"{settings.search_provider.upper()}_API_KEY")
    return missing


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


async def ingest_pdfs(maker, vectors, user_id, run_id, paths: list[Path]) -> int:
    files = [(p.name, p.read_bytes()) for p in paths]
    validate_pdfs(files)
    log, indexed = RunLog(maker, run_id), 0
    for name, data in files:
        print(f"ingesting {name}…")
        try:
            doc = await ingest_pdf(sessionmaker=maker, vectors=vectors, log=log, user_id=user_id,
                                   filename=name, path=save_upload(user_id, run_id, name, data))
            print(f"  {doc.pages} page(s), {doc.chunk_count} chunk(s)")
            indexed += 1
        except PdfRejected as exc:
            print(f"  skipped: {exc}", file=sys.stderr)
    return indexed


async def main_async(brief: str, source_prefs: dict[str, bool], pdfs: list[Path]) -> int:
    settings = get_settings()
    if missing := missing_keys(settings):
        print(f"Missing in .env: {', '.join(missing)} (see .env.example)", file=sys.stderr)
        return 2

    maker = get_sessionmaker()
    user_id, run_id = await create_run(maker, brief, source_prefs)
    gateway = ModelGateway.for_run(settings)
    qdrant = make_client()
    vectors = VectorStore(qdrant)
    await vectors.ensure_collections()
    try:
        has_docs = bool(pdfs) and await ingest_pdfs(maker, vectors, user_id, run_id, pdfs) > 0
    except PdfRejected as exc:
        print(f"PDF rejected: {exc}", file=sys.stderr)
        return 2

    async with httpx.AsyncClient(timeout=20) as http, \
            AsyncPostgresSaver.from_conn_string(checkpointer_url(settings)) as saver:
        await saver.setup()
        provider = make_search_provider(settings, http)
        fetcher = PageFetcher(http)

        def subagents(ctx: RunContext):
            researcher = WebResearcher(provider, fetcher, ctx.scratch, ctx.log,
                                       results_per_subtask=settings.search_results_per_subtask)
            specs = [web_researcher_spec(gateway, make_web_search_tool(researcher),
                                         settings.max_tool_calls_per_subagent)]
            if has_docs:
                searcher = DocSearcher(vectors, ctx.log, ctx.user_id, top_k=settings.doc_reader_top_k,
                                       min_relevance=settings.doc_min_relevance)
                specs.append(doc_reader_spec(gateway, make_doc_search_tool(searcher),
                                             settings.max_tool_calls_per_subagent))
            return specs

        async def memory_loader(uid, n):
            return await load_recent_summaries(vectors, uid, n)

        async def entity_loader(uid, query, n):
            return await load_related_entities(vectors, uid, query, n)

        deps = PipelineDeps(settings, maker, gateway, subagents, memory_loader,
                            vectors=vectors, entity_loader=entity_loader)
        pipeline = build_pipeline(deps, saver)

        print(f"run {run_id}\nplanning…")
        outcome = await execute(pipeline, deps, run_id, {
            "run_id": run_id, "user_id": user_id, "brief": brief, "source_prefs": source_prefs,
        })
        if outcome.status is RunStatus.AWAITING_CLARIFICATION:
            answer = input(f"\nPlanner asks: {outcome.question}\n> ").strip()
            outcome = await execute(pipeline, deps, run_id, Command(resume=answer))

    await qdrant.close()
    async with maker() as session:
        run = await session.get(Run, run_id)
    print(f"\nstatus: {run.status.value}" + (f"  error: {run.error}" if run.error else ""))
    if outcome.state.get("task_graph"):
        print("\ntask graph:")
        for t in outcome.state["task_graph"].sub_tasks:
            print(f"  {t.id} p{t.priority} [{t.source_type.value}] {t.query}  ({t.time_horizon}; {t.scope})")
    results = outcome.state.get("research_results", [])
    print_results(results)
    print_entities(outcome.state.get("entity_graph"))

    corpus = await ScratchStore(maker, run_id).total_tokens()
    handed_up = sum(count_tokens(r.model_dump_json()) for r in results)
    print(f"\ntokens used (all models): {gateway.meter.total:,}")
    print(f"context: raw corpus in scratch store ≈ {corpus:,} tokens; "
          f"Planner received ≈ {handed_up:,} tokens of references")
    await get_engine().dispose()
    return 0 if run.status is not RunStatus.FAILED else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one research brief through the InsightForge pipeline.")
    parser.add_argument("brief", help="Research brief (max 500 characters)")
    parser.add_argument("--no-memory", action="store_true", help="Do not load prior session summaries")
    parser.add_argument("--no-web", action="store_true", help="Use only the uploaded PDFs")
    parser.add_argument("--pdf", action="append", type=Path, default=[], help="PDF to research (repeatable, max 10)")
    args = parser.parse_args()
    if len(args.brief) > get_settings().brief_max_chars:
        parser.error(f"brief is longer than {get_settings().brief_max_chars} characters")
    if missing := [p for p in args.pdf if not p.is_file()]:
        parser.error(f"not a file: {', '.join(map(str, missing))}")
    if args.no_web and not args.pdf:
        parser.error("--no-web needs at least one --pdf")
    prefs = {"web": not args.no_web, "docs": bool(args.pdf), "memory": not args.no_memory}
    sys.exit(asyncio.run(main_async(args.brief, prefs, args.pdf)))


if __name__ == "__main__":
    main()
