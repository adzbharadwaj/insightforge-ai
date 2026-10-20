"""Process-wide runtime: the clients every run shares, and how a run's pipeline is built.

Used by the run worker (arq) and the CLI, so both execute runs identically. One Runtime
per process holds the HTTP client, Qdrant, the search provider (and its circuit breaker),
the page fetcher (and its per-domain limits) and a pooled LangGraph checkpointer. Each
run gets its own ModelGateway and TokenMeter; a resumed run's meter starts from the
tokens it already used, so pausing for clarification never resets the budget.
"""

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from insightforge.agents.doc_reader import doc_reader_spec
from insightforge.agents.web_search import web_researcher_spec
from insightforge.config import Settings, get_settings
from insightforge.graph.pipeline import PipelineDeps, RunContext, RunOutcome, build_pipeline, execute
from insightforge.llm.gateway import ModelGateway, TokenMeter
from insightforge.store.db import get_sessionmaker
from insightforge.store.entity_store import load_related_entities
from insightforge.store.memory import load_recent_summaries
from insightforge.store.models import Document, Run, RunStatus
from insightforge.store.qdrant import VectorStore, make_client
from insightforge.store.run_log import RunLog
from insightforge.tools.doc_search import DocSearcher, make_doc_search_tool
from insightforge.tools.docling_ingest import PdfRejected, ingest_pdf
from insightforge.tools.fetch import PageFetcher
from insightforge.tools.search_providers import make_search_provider
from insightforge.tools.web_search import WebResearcher, make_web_search_tool


def checkpointer_url(settings: Settings) -> str:
    return settings.database_url.replace("postgresql+asyncpg://", "postgresql://", 1)


def missing_keys(settings: Settings) -> list[str]:
    missing = [] if settings.openrouter_api_key else ["OPENROUTER_API_KEY"]
    if getattr(settings, f"{settings.search_provider}_api_key") is None:
        missing.append(f"{settings.search_provider.upper()}_API_KEY")
    return missing


@dataclass
class Upload:
    filename: str
    path: Path


class Runtime:
    def __init__(self, settings: Settings, sessionmaker: async_sessionmaker[AsyncSession], vectors: VectorStore,
                 http: httpx.AsyncClient, saver: AsyncPostgresSaver):
        self.settings = settings
        self.sessionmaker = sessionmaker
        self.vectors = vectors
        self.http = http
        self.saver = saver
        self.provider = make_search_provider(settings, http)
        self.fetcher = PageFetcher(http)

    @classmethod
    @asynccontextmanager
    async def open(cls, settings: Settings | None = None) -> AsyncIterator["Runtime"]:
        settings = settings or get_settings()
        qdrant = make_client()
        vectors = VectorStore(qdrant)
        await vectors.ensure_collections()
        pool = AsyncConnectionPool(checkpointer_url(settings), max_size=settings.max_concurrent_runs + 2, open=False,
                                   kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row})
        await pool.open()
        try:
            async with httpx.AsyncClient(timeout=20) as http:
                saver = AsyncPostgresSaver(pool)
                await saver.setup()
                yield cls(settings, get_sessionmaker(), vectors, http, saver)
        finally:
            await pool.close()
            await qdrant.close()

    # --- per run ------------------------------------------------------------------------

    async def ingest(self, run_id: uuid.UUID, user_id: uuid.UUID, uploads: list[Upload]) -> int:
        """Index the run's uploaded PDFs; a file that can't be read is logged and skipped."""
        log, indexed = RunLog(self.sessionmaker, run_id), 0
        for up in uploads:
            try:
                await ingest_pdf(sessionmaker=self.sessionmaker, vectors=self.vectors, log=log, user_id=user_id,
                                 filename=up.filename, path=up.path)
                indexed += 1
            except PdfRejected as exc:
                await log.emit("doc-ingest", "error", f"{up.filename} skipped: {exc}")
        return indexed

    async def _has_docs(self, run_id: uuid.UUID) -> bool:
        async with self.sessionmaker() as session:
            n = await session.scalar(select(func.count()).select_from(Document).where(Document.run_id == run_id))
        return bool(n)

    def build(self, run: Run, has_docs: bool, fresh_budget: bool = False):
        s = self.settings
        used = run.tokens_used or 0
        # Pausing for clarification continues the same budget. A retry after a failure gets a fresh budget for
        # its attempt (the run's total keeps counting up in tokens_used).
        meter = TokenMeter(used + s.run_token_budget if fresh_budget else s.run_token_budget)
        meter.input_tokens = used
        gateway = ModelGateway(s, meter)

        def subagents(ctx: RunContext):
            researcher = WebResearcher(self.provider, self.fetcher, ctx.scratch, ctx.log,
                                       results_per_subtask=s.search_results_per_subtask)
            specs = [web_researcher_spec(gateway, make_web_search_tool(researcher), s.max_tool_calls_per_subagent)]
            if has_docs:
                searcher = DocSearcher(self.vectors, ctx.log, ctx.user_id, top_k=s.doc_reader_top_k,
                                       min_relevance=s.doc_min_relevance)
                specs.append(doc_reader_spec(gateway, make_doc_search_tool(searcher), s.max_tool_calls_per_subagent))
            return specs

        async def memory_loader(uid, n):
            return await load_recent_summaries(self.vectors, uid, n)

        async def entity_loader(uid, query, n):
            return await load_related_entities(self.vectors, uid, query, n)

        deps = PipelineDeps(s, self.sessionmaker, gateway, subagents, memory_loader, vectors=self.vectors,
                            entity_loader=entity_loader, fetcher=self.fetcher)
        return build_pipeline(deps, self.saver), deps

    async def _load(self, run_id: uuid.UUID) -> Run:
        async with self.sessionmaker() as session:
            run = await session.get(Run, run_id)
        if run is None:
            raise LookupError(f"run {run_id} not found")
        return run

    async def start(self, run_id: uuid.UUID, uploads: list[Upload] | None = None) -> RunOutcome:
        run = await self._load(run_id)
        if uploads:
            await self.ingest(run.id, run.user_id, uploads)
        pipeline, deps = self.build(run, await self._has_docs(run.id))
        return await execute(pipeline, deps, run.id, {
            "run_id": run.id, "user_id": run.user_id, "brief": run.brief, "source_prefs": run.source_prefs,
        })

    async def has_checkpoint(self, run_id: uuid.UUID) -> bool:
        return await self.saver.aget_tuple({"configurable": {"thread_id": str(run_id)}}) is not None

    async def retry(self, run_id: uuid.UUID) -> RunOutcome:
        """Resume a failed run from its last checkpoint: only the stage that failed (and, inside research or
        entity extraction, only the unfinished agents) runs again."""
        run = await self._load(run_id)
        if run.status not in (RunStatus.FAILED, RunStatus.QUEUED):
            raise ValueError(f"run {run_id} is {run.status.value}; only failed runs can be retried")
        async with self.sessionmaker() as session:
            await session.execute(update(Run).where(Run.id == run_id).values(
                attempts=Run.attempts + 1, error=None, finished_at=None))
            await session.commit()
        run = await self._load(run_id)
        if not await self.has_checkpoint(run_id):  # failed before its first stage finished: start over
            return await self.start(run_id)
        await RunLog(self.sessionmaker, run_id).progress(
            "orchestrator", f"attempt {run.attempts}: resuming from the last checkpoint")
        pipeline, deps = self.build(run, await self._has_docs(run.id), fresh_budget=True)
        return await execute(pipeline, deps, run.id, None)

    async def run_with_auto_resume(self, first: RunOutcome, run_id: uuid.UUID) -> RunOutcome:
        """After a retryable failure, resume from the checkpoint up to `max_auto_resumes` times."""
        outcome = first
        for _ in range(self.settings.max_auto_resumes):
            if outcome.status is not RunStatus.FAILED or not outcome.retryable:
                break
            await RunLog(self.sessionmaker, run_id).progress("orchestrator", "retryable failure: auto-resuming")
            outcome = await self.retry(run_id)
        return outcome

    async def resume(self, run_id: uuid.UUID, answer: str) -> RunOutcome:
        run = await self._load(run_id)
        pipeline, deps = self.build(run, await self._has_docs(run.id))
        return await execute(pipeline, deps, run.id, Command(resume=answer))
