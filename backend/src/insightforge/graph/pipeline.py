"""LangGraph orchestrator (component 05): the control plane.

    load_context → plan ─┬─(question)→ clarify → plan
                         └─(tasks)───→ research → entities → END   (Phase 4 adds synthesis → writing → fact-check)

research = wave 1 (one isolated researcher per sub-task); entities = wave 2 (one
Entity Extractor that receives source ids only).

Every node records the run's status in Postgres. Budgets are enforced around the
graph (time) and inside the model gateway (tokens); a breach ends the run FAILED.
"""

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from deepagents import SubAgent
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from insightforge.agents.delegation import ResearchRecorder
from insightforge.agents.entity_extractor import run_entity_wave
from insightforge.agents.planner import (
    PlanningError,
    allowed_source_types,
    build_delegating_agent,
    persist_task_graph,
    plan,
    run_research,
)
from insightforge.config import Settings
from insightforge.graph.state import PipelineState
from insightforge.llm.gateway import ModelGateway, TokenBudgetExceeded
from insightforge.store.entity_store import save_entity_graph, upsert_entity_embeddings
from insightforge.store.models import Run, RunStatus, SubTaskStatus
from insightforge.store.qdrant import VectorStore
from insightforge.store.run_log import RunLog
from insightforge.store.scratch import ScratchStore
from insightforge.store.sources import SourceResolver

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class RunContext:
    """Per-run handles given to subagent factories. Everything is scoped to one run and user."""

    run_id: uuid.UUID
    user_id: uuid.UUID
    scratch: ScratchStore
    log: RunLog
    resolver: SourceResolver


SubagentFactory = Callable[[RunContext], list[SubAgent]]


@dataclass
class PipelineDeps:
    settings: Settings
    sessionmaker: async_sessionmaker[AsyncSession]
    gateway: ModelGateway
    subagents: SubagentFactory  # wave-1 researchers available for this run
    memory_loader: Callable[[uuid.UUID, int], Awaitable[list[str]]] | None = None  # (user_id, n)
    vectors: VectorStore | None = None  # needed for PDF chunk ids and entity embeddings
    entity_loader: Callable[[uuid.UUID, str, int], Awaitable[list[str]]] | None = None  # (user_id, brief, n)
    extract_entities: bool = True

    def context(self, run_id: uuid.UUID, user_id: uuid.UUID) -> RunContext:
        scratch = ScratchStore(self.sessionmaker, run_id)
        return RunContext(run_id, user_id, scratch, RunLog(self.sessionmaker, run_id),
                          SourceResolver(scratch, self.vectors, user_id, self.settings.uploaded_doc_credibility))


async def set_run(deps: PipelineDeps, run_id: uuid.UUID, **values: Any) -> None:
    async with deps.sessionmaker() as session:
        await session.execute(update(Run).where(Run.id == run_id).values(**values))
        await session.commit()


def build_pipeline(deps: PipelineDeps, checkpointer: BaseCheckpointSaver | None = None):
    s = deps.settings

    async def load_context(state: PipelineState) -> dict:
        await set_run(deps, state.run_id, status=RunStatus.PLANNING, started_at=datetime.now(UTC))
        summaries, entities = [], []
        if state.source_prefs.get("memory", True):
            if deps.memory_loader:
                summaries = await deps.memory_loader(state.user_id, s.memory_summaries_loaded)
            if deps.entity_loader:
                entities = await deps.entity_loader(state.user_id, state.brief, s.related_entities_loaded)
        return {"status": RunStatus.PLANNING, "memory_summaries": summaries[: s.memory_summaries_loaded],
                "related_entities": entities[: s.related_entities_loaded]}

    async def plan_node(state: PipelineState) -> dict:
        ctx = deps.context(state.run_id, state.user_id)
        log = ctx.log
        available = {spec["name"] for spec in deps.subagents(ctx)}
        graph = await plan(
            deps.gateway,
            brief=state.brief,
            allowed=allowed_source_types(state.source_prefs, available),
            memory_summaries=state.memory_summaries,
            related_entities=state.related_entities,
            max_subtasks=s.max_subtasks,
            clarification_answer=state.clarification_answer,
        )
        if graph.clarifying_question:
            await set_run(deps, state.run_id, status=RunStatus.AWAITING_CLARIFICATION,
                          clarifying_question=graph.clarifying_question)
            await log.progress("planner", "asked a clarifying question", question=graph.clarifying_question)
            return {"task_graph": graph, "status": RunStatus.AWAITING_CLARIFICATION}
        await persist_task_graph(deps.sessionmaker, log, graph)
        return {"task_graph": graph}

    async def clarify(state: PipelineState) -> dict:
        # Pauses the run; the API resumes it with Command(resume=<answer>). The 30-minute timeout that
        # cancels an unanswered run is enforced by the run worker (Phase 5).
        answer = interrupt({"question": state.task_graph.clarifying_question})
        await set_run(deps, state.run_id, status=RunStatus.PLANNING, clarification_answer=answer)
        return {"clarification_answer": answer, "status": RunStatus.PLANNING}

    async def research(state: PipelineState) -> dict:
        await set_run(deps, state.run_id, status=RunStatus.RESEARCHING)
        ctx = deps.context(state.run_id, state.user_id)
        recorder = ResearchRecorder(deps.sessionmaker, ctx.resolver, ctx.log, state.task_graph.sub_tasks)
        agent = build_delegating_agent(
            deps.gateway, recorder, deps.subagents(ctx),
            subtask_timeout_s=s.run_time_budget_s / 2,
        )
        results = await run_research(agent, recorder, state.task_graph)
        update_: dict = {"research_results": results, "status": RunStatus.RESEARCHING}
        if not any(r.status is SubTaskStatus.DONE for r in results):
            update_ |= {"status": RunStatus.FAILED, "error": "no sub-task produced usable sources"}
        return update_

    async def entities(state: PipelineState) -> dict:
        ctx = deps.context(state.run_id, state.user_id)
        source_ids = [ref.source_id for r in state.research_results if r.status is SubTaskStatus.DONE
                      for ref in r.refs]
        try:
            graph = await run_entity_wave(
                deps.gateway, ctx.resolver, ctx.log, source_ids,
                batch_size=s.entity_batch_size, max_chars=s.entity_source_max_chars,
                concurrency=s.entity_concurrency, timeout_s=s.run_time_budget_s / 4,
            )
        except TokenBudgetExceeded:
            raise
        except Exception as exc:  # entities enrich the report; losing them degrades it, it doesn't fail the run
            logger.exception("entity wave failed for run %s", state.run_id)
            await ctx.log.emit("entity_extractor", "error", f"entity extraction failed: {type(exc).__name__}")
            return {}
        await save_entity_graph(deps.sessionmaker, state.run_id, graph)
        if deps.vectors is not None:
            await upsert_entity_embeddings(deps.vectors, state.user_id, state.run_id, graph)
        return {"entity_graph": graph}

    def after_plan(state: PipelineState) -> str:
        return "clarify" if state.task_graph and state.task_graph.clarifying_question else "research"

    def after_research(state: PipelineState) -> str:
        ok = state.status is not RunStatus.FAILED
        return "entities" if ok and deps.extract_entities else END

    g = StateGraph(PipelineState)
    g.add_node("load_context", load_context)
    g.add_node("plan", plan_node)
    g.add_node("clarify", clarify)
    g.add_node("research", research)
    g.add_node("entities", entities)
    g.add_edge(START, "load_context")
    g.add_edge("load_context", "plan")
    g.add_conditional_edges("plan", after_plan, ["clarify", "research"])
    g.add_edge("clarify", "plan")
    g.add_conditional_edges("research", after_research, ["entities", END])
    g.add_edge("entities", END)
    return g.compile(checkpointer=checkpointer, name="insightforge_pipeline")


@dataclass
class RunOutcome:
    status: RunStatus
    state: dict
    question: str | None = None


async def execute(
    pipeline,
    deps: PipelineDeps,
    run_id: uuid.UUID,
    graph_input: dict | Command,
) -> RunOutcome:
    """Run (or resume) one pipeline run to its next stop: done, failed, or awaiting clarification.

    Enforces the run time budget and turns budget breaches and planner errors into a FAILED run
    with a clear message, never a hang or an unhandled crash.
    """
    config = {"configurable": {"thread_id": str(run_id)}, "recursion_limit": 25}
    try:
        state = await asyncio.wait_for(pipeline.ainvoke(graph_input, config), deps.settings.run_time_budget_s)
    except TimeoutError:
        return await _fail(deps, run_id, f"run exceeded its time budget of {deps.settings.run_time_budget_s}s")
    except TokenBudgetExceeded as exc:
        return await _fail(deps, run_id, f"token budget exceeded: {exc}")
    except PlanningError as exc:
        return await _fail(deps, run_id, f"planning failed: {exc}")
    except Exception as exc:  # any other error (provider 4xx, bug) ends the run FAILED, never a crash
        logger.exception("run %s failed with an unexpected error", run_id)
        return await _fail(deps, run_id, f"unexpected error: {type(exc).__name__}: {str(exc)[:500]}")

    if "__interrupt__" in state:
        question = state["__interrupt__"][0].value["question"]
        return RunOutcome(RunStatus.AWAITING_CLARIFICATION, state, question)

    status = RunStatus(state["status"])
    if status is RunStatus.FAILED:
        return await _fail(deps, run_id, state.get("error") or "run failed")
    meter = deps.gateway.meter
    await set_run(deps, run_id, status=status, tokens_used=meter.total if meter else 0)
    return RunOutcome(status, state)


async def _fail(deps: PipelineDeps, run_id: uuid.UUID, error: str) -> RunOutcome:
    meter = deps.gateway.meter
    await set_run(deps, run_id, status=RunStatus.FAILED, error=error, finished_at=datetime.now(UTC),
                  tokens_used=meter.total if meter else 0)
    await RunLog(deps.sessionmaker, run_id).emit("orchestrator", "error", error)
    return RunOutcome(RunStatus.FAILED, {}, None)
