"""Entity Extractor (component 09): the second delegation wave.

After the researchers finish, the Planner spawns one entity-extractor subagent via
the `task` tool. Its only input is the list of source ids the researchers kept (no
brief, no summaries). Its `extract_entities` tool resolves each id and runs one
small extraction call per source, so no prompt ever holds more than one source.
The tool output is accumulated in code; the graph, conflict flags and embeddings
are built from that, never retyped by a model.

Each extraction call covers `per_call` sources (default 4), each wrapped separately and
answered per source id: the fixed cost of a structured call (system prompt, schema and
provider scaffolding, ~2.2k tokens) was two thirds of the cost when sources went one by one.
"""

import asyncio
import json
import math

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from insightforge.agents.planner import build_delegating_agent, delegation_message
from insightforge.agents.prompts import ENTITY_EXTRACTOR_PROMPT, EXTRACTION_PROMPT, render_entity_prompt
from insightforge.agents.subagent import isolated_subagent
from insightforge.graph.state import BatchExtraction, EntityExtractionResult, EntityGraph, SourceExtraction
from insightforge.guardrails.untrusted import wrap
from insightforge.llm.gateway import ModelGateway, Role, TokenBudgetExceeded
from insightforge.store.models import EntityExtractionRow, SubTaskStatus
from insightforge.store.run_log import RunLog
from insightforge.store.sources import ResolvedSource, SourceResolver
from insightforge.tools.entities import merge_extractions

TASK_ID = "entities"
NAME = "entity-extractor"
AGENT = "entity_extractor"


class EntityExtraction:
    """Runs the per-source extraction calls and keeps every result, keyed by source id."""

    def __init__(
        self,
        gateway: ModelGateway,
        resolver: SourceResolver,
        log: RunLog,
        assigned: list[str],
        *,
        max_chars: int = 5_000,
        concurrency: int = 4,
        per_call: int = 4,
    ):
        self.gateway = gateway
        self.resolver = resolver
        self.log = log
        self.assigned = list(dict.fromkeys(assigned))
        self.max_chars = max_chars
        self.per_call = per_call
        self._sem = asyncio.Semaphore(concurrency)
        self.collected: dict[str, tuple[ResolvedSource, SourceExtraction]] = {}

    async def _extract_group(self, group: list[ResolvedSource]) -> dict[str, SourceExtraction]:
        async with self._sem:
            text = "\n\n".join(wrap(s.source_id, s.text[: self.max_chars], url=s.label, title=s.title) for s in group)
            out: BatchExtraction = await self.gateway.structured(Role.EXTRACTOR, BatchExtraction).ainvoke(
                [SystemMessage(EXTRACTION_PROMPT), HumanMessage(text)]
            )
        wanted = {s.source_id for s in group}
        return {e.source_id: SourceExtraction(entities=e.entities, facts=e.facts)
                for e in out.sources if e.source_id in wanted}

    async def extract(self, source_ids: list[str]) -> str:
        allowed = set(self.assigned)
        foreign = [i for i in source_ids if i not in allowed]
        done = [i for i in source_ids if i in self.collected]
        todo = [i for i in dict.fromkeys(source_ids) if i in allowed and i not in self.collected]
        sources = await self.resolver.resolve_many(todo)
        missing = [i for i in todo if i not in {s.source_id for s in sources}]

        groups = [sources[i:i + self.per_call] for i in range(0, len(sources), self.per_call)]
        results = await asyncio.gather(*(self._extract_group(g) for g in groups), return_exceptions=True)
        failed = list(missing)
        entities = facts = 0
        for group, result in zip(groups, results, strict=True):
            if isinstance(result, TokenBudgetExceeded):
                raise result  # a budget breach stops the run; it is not a per-source failure
            if isinstance(result, BaseException):
                failed += [s.source_id for s in group]
                await self.log.emit(AGENT, "error", f"extraction failed for {len(group)} source(s)",
                                    error=f"{type(result).__name__}: {result}"[:300])
                continue
            done_now = []
            for source in group:
                ext = result.get(source.source_id)
                if ext is None:  # the model skipped this source in its answer
                    failed.append(source.source_id)
                    continue
                self.collected[source.source_id] = (source, ext)
                done_now.append((source.source_id, ext))
                entities += len(ext.entities)
                facts += len(ext.facts)
            await self._save(done_now)  # checkpoint per batch: a resume never re-extracts these

        processed = len(sources) - (len(failed) - len(missing))
        parts = [f"Processed {processed} source(s): {entities} entities, {facts} facts."]
        if failed:
            parts.append(f"Failed ids: {failed}.")
        if done:
            parts.append(f"Already processed: {done}.")
        if foreign:
            parts.append(f"Not assigned to you, ignored: {foreign}.")
        await self.log.progress(AGENT, f"extract_entities: {processed} processed, {len(failed)} failed, "
                                       f"{len(done)} already done, {len(foreign)} not assigned",
                                failed=failed, foreign=foreign[:20])
        return " ".join(parts)

    async def load_checkpoints(self) -> int:
        """Restore extractions a previous attempt already saved, so a resumed wave skips those sources."""
        async with self.log.sessionmaker() as session:
            rows = (await session.scalars(select(EntityExtractionRow).where(
                EntityExtractionRow.run_id == self.log.run_id,
                EntityExtractionRow.source_id.in_(self.assigned)))).all()
        saved = {r.source_id: SourceExtraction.model_validate(r.extraction) for r in rows}
        for source in await self.resolver.resolve_many(list(saved)):
            self.collected[source.source_id] = (source, saved[source.source_id])
        return len(saved)

    async def _save(self, items: list[tuple[str, SourceExtraction]]) -> None:
        if not items:
            return
        stmt = insert(EntityExtractionRow).values([
            {"run_id": self.log.run_id, "source_id": sid, "extraction": ext.model_dump(mode="json")}
            for sid, ext in items]).on_conflict_do_nothing()
        async with self.log.sessionmaker() as session:
            await session.execute(stmt)
            await session.commit()

    def graph(self) -> EntityGraph:
        ordered = [self.collected[i] for i in self.assigned if i in self.collected]
        return merge_extractions(ordered)


class ExtractEntitiesArgs(BaseModel):
    source_ids: list[str] = Field(min_length=1, description="Source ids to process")


def make_extract_entities_tool(extraction: EntityExtraction, batch_size: int) -> BaseTool:
    async def extract_entities(source_ids: list[str]) -> str:
        if len(source_ids) > batch_size:
            return f"At most {batch_size} source ids per call; split the list and call again."
        return await extraction.extract(source_ids)

    return StructuredTool.from_function(
        coroutine=extract_entities,
        name="extract_entities",
        description=f"Extract entities and facts from up to {batch_size} sources by id. Returns counts only.",
        args_schema=ExtractEntitiesArgs,
    )


def entity_extractor_spec(gateway: ModelGateway, tool: BaseTool, n_sources: int, batch_size: int):
    batches = max(1, math.ceil(n_sources / batch_size))
    return isolated_subagent(
        name=NAME,
        description="Extracts entities, relationships and conflicts from a list of source ids.",
        system_prompt=ENTITY_EXTRACTOR_PROMPT.format(batch_size=batch_size),
        gateway=gateway,
        role=Role.EXTRACTOR,
        tools=[tool],
        max_tool_calls=batches * 2,  # every batch once, plus one retry each
        response_schema=EntityExtractionResult,
    )


class EntityWave:
    """Wave-2 ledger: a single assignment whose prompt is the list of source ids."""

    def __init__(self, extraction: EntityExtraction, log: RunLog):
        self.extraction = extraction
        self.log = log
        self.started: set[str] = set()
        self.result: EntityExtractionResult | None = None

    def assignment(self, task_id: str) -> tuple[str, str] | None:
        return (NAME, render_entity_prompt(self.extraction.assigned)) if task_id == TASK_ID else None

    def valid_ids(self) -> list[str]:
        return [TASK_ID]

    async def mark_running(self, task_id: str) -> None:
        self.started.add(task_id)
        await self.log.progress(AGENT, f"extracting entities from {len(self.extraction.assigned)} source(s)")

    async def complete(self, task_id: str, output: str | None) -> str:
        try:
            claimed = EntityExtractionResult.model_validate(json.loads(output or "{}"))
        except (ValueError, ValidationError):
            claimed = EntityExtractionResult(status=SubTaskStatus.FAILED, error="unparseable extractor output")
        processed = len(self.extraction.collected)  # what the tool actually did, not what the model claims
        status = SubTaskStatus.DONE if processed else SubTaskStatus.FAILED
        self.result = EntityExtractionResult(status=status, processed=processed,
                                             error=None if processed else (claimed.error or "no sources processed"))
        return self.result.model_dump_json()

    async def fail(self, task_id: str, error: str) -> str:
        self.result = EntityExtractionResult(status=SubTaskStatus.FAILED,
                                             processed=len(self.extraction.collected), error=error)
        return self.result.model_dump_json()


async def run_entity_wave(
    gateway: ModelGateway,
    resolver: SourceResolver,
    log: RunLog,
    source_ids: list[str],
    *,
    batch_size: int = 10,
    max_chars: int = 12_000,
    concurrency: int = 4,
    timeout_s: float = 180,
) -> EntityGraph:
    extraction = EntityExtraction(gateway, resolver, log, source_ids, max_chars=max_chars, concurrency=concurrency)
    if not extraction.assigned:
        return EntityGraph()
    if restored := await extraction.load_checkpoints():
        await log.progress(AGENT, f"resumed: {restored} source extraction(s) restored from checkpoint")
    if len(extraction.collected) == len(extraction.assigned):
        return extraction.graph()  # everything was extracted before the interruption
    wave = EntityWave(extraction, log)
    spec = entity_extractor_spec(gateway, make_extract_entities_tool(extraction, batch_size),
                                 len(extraction.assigned), batch_size)
    agent = build_delegating_agent(gateway, wave, [spec], subtask_timeout_s=timeout_s, max_planner_calls=4)
    await agent.ainvoke({"messages": [HumanMessage(delegation_message([(TASK_ID, NAME)]))]})
    if TASK_ID not in wave.started:
        await log.emit(AGENT, "error", "entity extraction was not delegated")

    # Fail-safe: the tool is deterministic, so sources the agent skipped (it answered without calling the
    # tool, or dropped ids) are processed directly rather than silently leaving the graph incomplete.
    missing = [i for i in extraction.assigned if i not in extraction.collected]
    if missing:
        await log.emit(AGENT, "retry", f"agent left {len(missing)} of {len(extraction.assigned)} source(s) "
                                       "unprocessed; processing them directly", source_ids=missing[:50])
        for start in range(0, len(missing), batch_size):
            await extraction.extract(missing[start:start + batch_size])

    graph = extraction.graph()
    await log.progress(AGENT, f"entity graph: {len(graph.entities)} entities, {len(graph.relationships)} facts, "
                              f"{len(graph.conflicts)} conflict(s)",
                       conflicts=[c.model_dump(mode="json") for c in graph.conflicts])
    return graph
