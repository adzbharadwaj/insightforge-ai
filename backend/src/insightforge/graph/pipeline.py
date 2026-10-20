"""LangGraph orchestrator (component 05): the control plane.

    load_context → plan ─┬─(question)→ clarify → plan
                         └─(tasks)───→ research → entities → synthesize → review → write ⇄ ground → fact_check → finalize

research = wave 1 (one isolated researcher per sub-task); entities = wave 2 (one
Entity Extractor that receives source ids only). `ground` sends the draft back to
`write` with explicit constraints until it passes or retries run out; then unsupported
claims are dropped. Only `finalize` stores the report: nothing unvalidated is shown.

Every node records the run's status in Postgres. Budgets are enforced around the
graph (time) and inside the model gateway (tokens); a breach ends the run FAILED.
"""

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from deepagents import SubAgent
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import ValidationError
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from insightforge.agents.conflict_review import KEEP_BOTH, apply_decisions, build_reviews
from insightforge.agents.delegation import ResearchRecorder
from insightforge.agents.entity_extractor import run_entity_wave
from insightforge.agents.fact_checker import FactChecker
from insightforge.agents.planner import (
    PlanningError,
    allowed_source_types,
    build_delegating_agent,
    persist_task_graph,
    plan,
    run_research,
)
from insightforge.agents.synthesis import Synthesizer, adaptive_budget
from insightforge.agents.writer import render_report, write_report
from insightforge.config import Settings
from insightforge.graph.state import FactCheckSummary, PipelineState
from insightforge.guardrails.grounding import check_grounding, drop_unsupported, retry_constraints
from insightforge.guardrails.output import redact_pii
from insightforge.llm.gateway import ModelGateway, TokenBudgetExceeded
from insightforge.store.entity_store import save_entity_graph, upsert_entity_embeddings
from insightforge.store.memory import save_summary
from insightforge.store.models import Run, RunStatus, SubTaskStatus
from insightforge.store.qdrant import VectorStore
from insightforge.store.reports import save_report, session_summary
from insightforge.store.run_log import RunLog
from insightforge.store.scratch import ScratchStore
from insightforge.store.sources import SourceResolver
from insightforge.tools.fetch import PageFetcher

logger = logging.getLogger(__name__)
FACT_CHECK_RESERVE_S = 90  # re-fetching and embedding sources; kept free when deciding on a rewrite

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
    fetcher: PageFetcher | None = None  # live re-fetch for the Fact-Checker; None → stored copies only
    deadline: float | None = None  # time.monotonic() when this attempt's time budget ends (set by execute)

    def seconds_left(self) -> float | None:
        return None if self.deadline is None else self.deadline - time.monotonic()

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
                          clarifying_question=graph.clarifying_question, clarification_asked_at=datetime.now(UTC))
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
        refs = {ref.source_id: ref for r in state.research_results if r.status is SubTaskStatus.DONE for ref in r.refs}
        # Cost grows with sources (~3k tokens each); keep the most credible ones so the stage stays bounded.
        ranked = sorted(refs.values(), key=lambda ref: ref.credibility or 0.0, reverse=True)
        source_ids = [ref.source_id for ref in ranked[: s.entity_max_sources]]
        if len(ranked) > len(source_ids):
            await ctx.log.emit("entity_extractor", "cut",
                               f"extracting from the {len(source_ids)} most credible of {len(ranked)} sources",
                               skipped=[ref.source_id for ref in ranked[s.entity_max_sources:]])
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

    async def synthesize(state: PipelineState) -> dict:
        await set_run(deps, state.run_id, status=RunStatus.SYNTHESIZING)
        ctx = deps.context(state.run_id, state.user_id)
        budget = s.evidence_bundle_token_budget
        if s.adaptive_bundle_budget:
            n_sources = len({ref.source_id for r in state.research_results for ref in r.refs})
            budget = adaptive_budget(budget, len(state.task_graph.sub_tasks), n_sources,
                                     s.adaptive_budget_min, s.adaptive_budget_max)
            await ctx.log.progress("synthesis", f"adaptive evidence budget: {budget:,} tokens "
                                                f"({len(state.task_graph.sub_tasks)} sub-tasks, {n_sources} sources)")
        result = await Synthesizer(
            deps.gateway, ctx.resolver, ctx.log, token_budget=budget,
            passages_per_source=s.synthesis_passages_per_source, high_credibility=s.high_credibility_threshold,
        ).run(state.brief, state.task_graph.sub_tasks, state.research_results, state.entity_graph)
        if not result.bundle.sections:
            return {"status": RunStatus.FAILED, "error": "synthesis found no usable evidence"}
        return {"evidence_bundle": result.bundle, "status": RunStatus.SYNTHESIZING}

    async def review(state: PipelineState) -> dict:
        """Opt-in contradiction resolution: pause so the analyst picks a side for each conflicting fact."""
        if not state.source_prefs.get("review_conflicts"):
            return {}
        reviews = build_reviews(state.evidence_bundle, state.entity_graph)
        if not reviews:
            return {}
        payload = {"type": "review", "conflicts": [r.model_dump(mode="json") for r in reviews]}
        await set_run(deps, state.run_id, status=RunStatus.AWAITING_REVIEW, review=payload,
                      review_requested_at=datetime.now(UTC))
        decision = interrupt(payload)  # the API resumes with Command(resume={"choices": {...}})
        choices = (decision or {}).get("choices", {})
        bundle = apply_decisions(state.evidence_bundle, reviews, choices)
        await set_run(deps, state.run_id, status=RunStatus.SYNTHESIZING, review=None)
        resolved = sum(1 for r in reviews if choices.get(r.conflict_id, KEEP_BOTH) != KEEP_BOTH)
        await RunLog(deps.sessionmaker, state.run_id).progress(
            "review", f"analyst resolved {resolved} of {len(reviews)} conflict(s)", choices=choices)
        return {"evidence_bundle": bundle}

    async def write(state: PipelineState) -> dict:
        await set_run(deps, state.run_id, status=RunStatus.WRITING)
        constraints = None
        if state.grounding and not state.grounding.ok and state.draft:
            constraints = retry_constraints(state.grounding, state.draft)
            await RunLog(deps.sessionmaker, state.run_id).emit(
                "writer", "retry", f"rewriting: {len(constraints)} grounding issue(s)", constraints=constraints)
        failed = [r for r in state.research_results if r.status is not SubTaskStatus.DONE]
        out = await write_report(deps.gateway, brief=state.brief, bundle=state.evidence_bundle,
                                 failed=failed, constraints=constraints)
        return {"draft": render_report(out, state.evidence_bundle), "writer_attempts": state.writer_attempts + 1,
                "grounding": None}

    async def ground(state: PipelineState) -> dict:
        await set_run(deps, state.run_id, status=RunStatus.GROUNDING)
        meter = deps.gateway.meter
        before, ground_started = (meter.total if meter else 0), time.monotonic()
        result = await check_grounding(deps.gateway, state.draft, state.evidence_bundle)
        log = RunLog(deps.sessionmaker, state.run_id)
        await log.progress("grounding", f"attempt {state.writer_attempts}: "
                                        f"{len(result.unsupported_claim_ids)} unsupported claim(s)",
                           reasons=result.reasons)
        if result.ok:
            return {"grounding": result}
        if state.writer_attempts <= s.max_writer_retries:
            # Another attempt costs about one write plus one grounding check, in tokens and in time. If either
            # budget can't cover it, degrade (drop the unsupported claims) rather than fail a nearly finished run.
            retry_tokens = (state.stage_tokens.get("write", 0) / max(1, state.writer_attempts)
                            + ((meter.total if meter else 0) - before))
            retry_seconds = (state.stage_seconds.get("write", 0) / max(1, state.writer_attempts)
                             + (time.monotonic() - ground_started))
            left = deps.seconds_left()
            tokens_ok = meter is None or meter.remaining > retry_tokens * 1.2
            time_ok = left is None or left > retry_seconds * 1.2 + FACT_CHECK_RESERVE_S
            if tokens_ok and time_ok:
                return {"grounding": result}
            reason = (f"~{retry_tokens:,.0f} tokens needed, {meter.remaining:,} left" if not tokens_ok else
                      f"~{retry_seconds:.0f}s needed (+{FACT_CHECK_RESERVE_S}s for fact-checking), {left:.0f}s left")
            await log.emit("grounding", "retry", f"skipping rewrite: {reason} in the run budget")
        # Retries exhausted (or unaffordable): fail safe by removing what the evidence does not support.
        cleaned = drop_unsupported(state.draft, result, state.evidence_bundle)
        await log.emit("grounding", "cut", f"removed {len(result.unsupported_claim_ids)} unsupported claim(s)",
                       claim_ids=result.unsupported_claim_ids)
        return {"draft": render_report(cleaned, state.evidence_bundle),
                "grounding": result.model_copy(update={"ok": True})}

    async def fact_check(state: PipelineState) -> dict:
        await set_run(deps, state.run_id, status=RunStatus.FACT_CHECKING)
        if deps.vectors is None:
            return {"fact_check": FactCheckSummary(total_checked=0, passed=0, failed=0, pass_rate=1.0,
                                                   unverified_claims=[])}
        ctx = deps.context(state.run_id, state.user_id)
        summary = await FactChecker(
            deps.vectors, ctx.resolver, ctx.log, fetcher=deps.fetcher, sample_ratio=s.factcheck_sample_ratio,
            min_claims=s.factcheck_min_claims, threshold=s.factcheck_similarity_threshold,
        ).run(state.draft, state.evidence_bundle)
        return {"fact_check": summary}

    async def finalize(state: PipelineState) -> dict:
        unverified = frozenset(c.claim_id for c in state.fact_check.checks if not c.verified)
        draft = render_report(state.draft.report, state.evidence_bundle, unverified)
        markdown, redacted = redact_pii(draft.markdown)
        log = RunLog(deps.sessionmaker, state.run_id)
        if redacted:
            await log.emit("output_guardrail", "redaction", f"redacted personal data: {redacted}", **redacted)
        report_id = await save_report(deps.sessionmaker, run_id=state.run_id, user_id=state.user_id, draft=draft,
                                      markdown=markdown, bundle=state.evidence_bundle, fact_check=state.fact_check)
        if deps.vectors is not None:
            await save_summary(deps.vectors, state.user_id, state.run_id, session_summary(state.brief, draft))
        await set_run(deps, state.run_id, status=RunStatus.COMPLETE, finished_at=datetime.now(UTC))
        await log.progress("orchestrator", "report complete", report_id=str(report_id),
                           stage_tokens=state.stage_tokens)
        return {"draft": draft, "final_markdown": markdown, "report_id": report_id, "status": RunStatus.COMPLETE}

    def after_plan(state: PipelineState) -> str:
        return "clarify" if state.task_graph and state.task_graph.clarifying_question else "research"

    def after_research(state: PipelineState) -> str:
        if state.status is RunStatus.FAILED:
            return END
        return "entities" if deps.extract_entities else "synthesize"

    def after_synthesize(state: PipelineState) -> str:
        return END if state.status is RunStatus.FAILED else "review"

    def after_ground(state: PipelineState) -> str:
        return "fact_check" if state.grounding.ok else "write"

    def staged(name: str, fn):
        """Record the tokens and seconds each stage uses (summed across write/ground retries)."""
        async def node(state: PipelineState) -> dict:
            meter = deps.gateway.meter
            before, started = (meter.total if meter else 0), time.monotonic()
            update = dict(await fn(state) or {})
            used = (meter.total if meter else 0) - before
            update["stage_tokens"] = {name: state.stage_tokens.get(name, 0) + used}
            update["stage_seconds"] = {name: round(state.stage_seconds.get(name, 0) + time.monotonic() - started, 1)}
            # Logged as each stage ends, so a run that fails later still shows where its tokens went.
            await RunLog(deps.sessionmaker, state.run_id).emit(
                "orchestrator", "tokens", f"{name}: {used:,} tokens", stage=name, tokens=used,
                run_total=meter.total if meter else 0)
            return update
        return node

    g = StateGraph(PipelineState)
    g.add_node("load_context", load_context)
    g.add_node("plan", staged("plan", plan_node))
    g.add_node("clarify", clarify)
    g.add_node("research", staged("research", research))
    g.add_node("entities", staged("entities", entities))
    g.add_node("synthesize", staged("synthesize", synthesize))
    g.add_node("review", review)
    g.add_node("write", staged("write", write))
    g.add_node("ground", staged("ground", ground))
    g.add_node("fact_check", staged("fact_check", fact_check))
    g.add_node("finalize", finalize)
    g.add_edge(START, "load_context")
    g.add_edge("load_context", "plan")
    g.add_conditional_edges("plan", after_plan, ["clarify", "research"])
    g.add_edge("clarify", "plan")
    g.add_conditional_edges("research", after_research, ["entities", "synthesize", END])
    g.add_edge("entities", "synthesize")
    g.add_conditional_edges("synthesize", after_synthesize, ["review", END])
    g.add_edge("review", "write")
    g.add_edge("write", "ground")
    g.add_conditional_edges("ground", after_ground, ["write", "fact_check"])
    g.add_edge("fact_check", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer, name="insightforge_pipeline")


@dataclass
class RunOutcome:
    status: RunStatus
    state: dict
    question: str | None = None
    retryable: bool = False
    review: dict | None = None  # pending conflict review (AWAITING_REVIEW)  # a resume from the last checkpoint could succeed (time budget, provider error)


async def execute(
    pipeline,
    deps: PipelineDeps,
    run_id: uuid.UUID,
    graph_input: dict | Command | None,
) -> RunOutcome:
    """Run, resume, or retry one pipeline run to its next stop: done, failed, or awaiting clarification.

    `graph_input=None` resumes from the last checkpoint: every stage saves one when it finishes
    (durability="sync": written before the next stage starts), so a retry re-runs only the stage
    that failed. Enforces the run time budget (per attempt) and turns budget breaches and planner
    errors into a FAILED run with a clear message, never a hang or an unhandled crash.
    """
    config = {"configurable": {"thread_id": str(run_id)}, "recursion_limit": 25}
    budget = deps.settings.run_time_budget_s
    deps.deadline = time.monotonic() + budget
    try:
        state = await asyncio.wait_for(pipeline.ainvoke(graph_input, config, durability="sync"), budget)
    except TimeoutError:
        return await _fail(deps, run_id, f"run exceeded its time budget of {budget}s", retryable=True)
    except TokenBudgetExceeded as exc:
        return await _fail(deps, run_id, f"token budget exceeded: {exc}")
    except PlanningError as exc:
        return await _fail(deps, run_id, f"planning failed: {exc}")
    except Exception as exc:  # any other error (provider 4xx/5xx, bug) ends the run FAILED, never a crash
        logger.exception("run %s failed with an unexpected error", run_id)
        # invalid run state (e.g. malformed source prefs) fails the same way on every attempt: don't auto-resume
        bad_state = isinstance(exc, ValidationError) and exc.title == PipelineState.__name__
        return await _fail(deps, run_id, f"unexpected error: {type(exc).__name__}: {str(exc)[:500]}",
                           retryable=not bad_state)

    if "__interrupt__" in state:
        value = state["__interrupt__"][0].value
        meter = deps.gateway.meter
        if meter:  # the resume may run in another process; it continues from this count
            await set_run(deps, run_id, tokens_used=meter.total)
        if value.get("type") == "review":
            return RunOutcome(RunStatus.AWAITING_REVIEW, state, review=value)
        return RunOutcome(RunStatus.AWAITING_CLARIFICATION, state, value["question"])

    status = RunStatus(state["status"])
    if status is RunStatus.FAILED:
        return await _fail(deps, run_id, state.get("error") or "run failed")
    meter = deps.gateway.meter
    await set_run(deps, run_id, status=status, tokens_used=meter.total if meter else 0)
    return RunOutcome(status, state)


async def _fail(deps: PipelineDeps, run_id: uuid.UUID, error: str, retryable: bool = False) -> RunOutcome:
    meter = deps.gateway.meter
    await set_run(deps, run_id, status=RunStatus.FAILED, error=error, finished_at=datetime.now(UTC),
                  tokens_used=meter.total if meter else 0)
    await RunLog(deps.sessionmaker, run_id).emit("orchestrator", "error", error, retryable=retryable)
    return RunOutcome(RunStatus.FAILED, {}, None, retryable=retryable)
