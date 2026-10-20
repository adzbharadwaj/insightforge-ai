"""Checkpointing: a failure re-runs only what failed, never the whole run.

Stage level: LangGraph saves state after every stage (durability="sync"); `execute(..., None)` resumes.
Agent level: inside research and entity extraction, each finished agent's output is saved, so a stage
interrupted halfway resumes with only its unfinished agents.
"""

import asyncio
import uuid

from fakes import FakeGateway, RuleModel
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from report_fakes import REPORT_FNS
from sqlalchemy import select
from test_pipeline import deps_for, initial, run_row
from test_planner import make_researcher, planner_rule, researcher_rule, spec

from insightforge.agents.web_search import web_researcher_spec
from insightforge.config import Settings
from insightforge.graph.pipeline import build_pipeline, execute
from insightforge.graph.state import TaskGraph
from insightforge.llm.gateway import Role
from insightforge.store.models import RunEvent, RunStatus, SubTask, SubTaskStatus
from insightforge.tools.web_search import make_web_search_tool


def counting(fn, counter: dict, key: str):
    def wrapped(messages):
        counter[key] = counter.get(key, 0) + 1
        return fn(messages)
    return wrapped


async def events(maker, run_id, kind=None) -> list[str]:
    async with maker() as s:
        q = select(RunEvent).where(RunEvent.run_id == run_id)
        if kind:
            q = q.where(RunEvent.kind == kind)
        return [e.message for e in (await s.scalars(q.order_by(RunEvent.id))).all()]


async def test_provider_error_in_writing_resumes_at_writing(committed_run):
    maker, user_id, run_id = committed_run
    calls: dict[str, int] = {}
    flaky = {"fail": True}

    def writer(messages):
        if flaky["fail"]:
            raise RuntimeError("Error code: 503 - provider overloaded")
        return REPORT_FNS[Role.WRITER](messages)

    deps = deps_for(maker, run_id, counting(lambda m: TaskGraph(sub_tasks=[spec(1), spec(2)]), calls, "plan"))
    deps.gateway.structured_fns[Role.WRITER] = counting(writer, calls, "write")
    deps.gateway.structured_fns[Role.SYNTHESIS] = counting(REPORT_FNS[Role.SYNTHESIS], calls, "synthesis")
    researcher = deps.gateway.models[Role.RESEARCHER]
    pipeline = build_pipeline(deps, InMemorySaver())

    failed = await execute(pipeline, deps, run_id, initial(user_id, run_id))
    assert failed.status is RunStatus.FAILED and failed.retryable
    research_calls = len(researcher.seen)

    flaky["fail"] = False
    resumed = await execute(pipeline, deps, run_id, None)
    assert resumed.status is RunStatus.COMPLETE
    assert calls["plan"] == 1 and calls["synthesis"] == 2  # two sub-tasks, synthesized once: not repeated
    assert len(researcher.seen) == research_calls  # no researcher ran again
    assert calls["write"] == 2  # only the failed stage re-ran


async def test_time_budget_mid_run_resumes_from_the_last_finished_stage(committed_run):
    maker, user_id, run_id = committed_run
    calls: dict[str, int] = {}
    slow = {"on": True}

    async def writer(messages):
        if slow["on"]:
            await asyncio.sleep(10)
        return REPORT_FNS[Role.WRITER](messages)

    deps = deps_for(maker, run_id, counting(lambda m: TaskGraph(sub_tasks=[spec(1)]), calls, "plan"),
                    settings=Settings(_env_file=None, run_time_budget_s=3))
    deps.gateway.structured_fns[Role.WRITER] = writer
    deps.gateway.structured_fns[Role.SYNTHESIS] = counting(REPORT_FNS[Role.SYNTHESIS], calls, "synthesis")
    pipeline = build_pipeline(deps, InMemorySaver())

    failed = await execute(pipeline, deps, run_id, initial(user_id, run_id))
    assert failed.status is RunStatus.FAILED and failed.retryable
    assert "time budget" in (await run_row(maker, run_id)).error

    slow["on"] = False
    resumed = await execute(pipeline, deps, run_id, None)  # a fresh time budget for this attempt
    assert resumed.status is RunStatus.COMPLETE and calls == {"plan": 1, "synthesis": 1}


async def test_interrupted_research_reruns_only_unfinished_researchers(committed_run):
    """Agent-level checkpoint: st-1 finished before the research stage died; only st-2 runs again.

    (A merely slow researcher can't kill the stage: its own sub-task timeout records it as Failed and the run
    continues. A budget breach can, because it is re-raised to stop the run.)
    """
    from insightforge.llm.gateway import TokenBudgetExceeded

    maker, user_id, run_id = committed_run
    searched: list[str] = []
    st2_breaks = {"on": True}
    real_tool = make_web_search_tool(make_researcher(maker, run_id))

    async def web_search(sub_task_id: str, query: str, time_horizon: str | None = None) -> str:
        searched.append(sub_task_id)
        if sub_task_id == "st-2" and st2_breaks["on"]:
            await asyncio.sleep(1)  # let st-1 finish and checkpoint first
            raise TokenBudgetExceeded("run used 250,000 of 250,000 tokens")
        return await real_tool.coroutine(sub_task_id=sub_task_id, query=query, time_horizon=time_horizon)

    tool = StructuredTool.from_function(coroutine=web_search, name="web_search", description=real_tool.description,
                                        args_schema=real_tool.args_schema)
    gw = FakeGateway({Role.PLANNER: RuleModel(rule=planner_rule()), Role.RESEARCHER: RuleModel(rule=researcher_rule())},
                     {Role.PLANNER: lambda m: TaskGraph(sub_tasks=[spec(1), spec(2)]), **REPORT_FNS})
    deps = deps_for(maker, run_id, lambda m: None)
    deps.gateway = gw
    deps.subagents = lambda ctx: [web_researcher_spec(gw, tool, max_tool_calls=4)]
    pipeline = build_pipeline(deps, InMemorySaver())

    failed = await execute(pipeline, deps, run_id, initial(user_id, run_id))
    assert failed.status is RunStatus.FAILED and sorted(searched) == ["st-1", "st-2"]
    async with maker() as s:
        rows = {t.task_key: t for t in await s.scalars(select(SubTask).where(SubTask.run_id == run_id))}
    assert rows["st-1"].status is SubTaskStatus.DONE and rows["st-1"].result  # checkpointed
    assert rows["st-2"].result is None

    st2_breaks["on"] = False
    searched.clear()
    resumed = await execute(pipeline, deps, run_id, None)  # e.g. the user's retry after raising the budget
    assert resumed.status is RunStatus.COMPLETE
    assert searched == ["st-2"]  # st-1 was restored, not researched again
    assert {r.sub_task_id: r.status for r in resumed.state["research_results"]} == {
        "st-1": SubTaskStatus.DONE, "st-2": SubTaskStatus.DONE}
    assert any("restored from checkpoint" in m for m in await events(maker, run_id))


async def test_entity_extractions_are_restored_not_redone(committed_run):
    from test_entity_wave import extract_fn, stored_sources

    from insightforge.agents.entity_extractor import EntityExtraction
    from insightforge.store.run_log import RunLog
    from insightforge.store.scratch import ScratchStore
    from insightforge.store.sources import SourceResolver

    maker, user_id, run_id = committed_run
    ids = await stored_sources(maker, run_id)
    calls: dict[str, int] = {}
    gw = FakeGateway({}, {Role.EXTRACTOR: counting(extract_fn([]), calls, "extract")})
    resolver = SourceResolver(ScratchStore(maker, run_id), None, user_id)

    first = EntityExtraction(gw, resolver, RunLog(maker, run_id), ids, per_call=1)
    await first.extract(ids[:2])  # the attempt dies after two of three sources
    assert calls["extract"] == 2

    second = EntityExtraction(gw, resolver, RunLog(maker, run_id), ids, per_call=1)
    assert await second.load_checkpoints() == 2
    await second.extract(ids)
    assert calls["extract"] == 3  # only the missing source was extracted
    assert len(second.graph().conflicts) == 1  # the restored extractions still count


async def test_planning_errors_are_not_retryable(committed_run):
    maker, user_id, run_id = committed_run
    deps = deps_for(maker, run_id, lambda m: TaskGraph(sub_tasks=[spec(1, source_type="docs")]))  # docs not allowed
    outcome = await execute(build_pipeline(deps, InMemorySaver()), deps, run_id, initial(user_id, run_id))
    assert outcome.status is RunStatus.FAILED and not outcome.retryable


async def test_rewrite_is_skipped_when_time_is_short(committed_run):
    from test_pipeline import lying_writer

    maker, user_id, run_id = committed_run
    seen: list[str] = []
    deps = deps_for(maker, run_id, lambda m: TaskGraph(sub_tasks=[spec(1)]),
                    settings=Settings(_env_file=None, run_time_budget_s=60))  # < 90 s fact-check reserve
    deps.gateway.structured_fns[Role.WRITER] = lying_writer(99, seen)
    outcome = await execute(build_pipeline(deps, InMemorySaver()), deps, run_id, initial(user_id, run_id))
    assert outcome.status is RunStatus.COMPLETE and outcome.state["writer_attempts"] == 1
    assert "1 statement(s) were removed" in outcome.state["final_markdown"]
    assert any("for fact-checking" in m for m in await events(maker, run_id, "retry"))


async def test_resume_survives_a_process_restart_with_the_postgres_checkpointer(committed_run):
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    from insightforge.runtime import checkpointer_url

    maker, user_id, run_id = committed_run
    calls: dict[str, int] = {}
    flaky = {"fail": True}

    def writer(messages):
        if flaky["fail"]:
            raise RuntimeError("connection reset")
        return REPORT_FNS[Role.WRITER](messages)

    deps = deps_for(maker, run_id, counting(lambda m: TaskGraph(sub_tasks=[spec(1)]), calls, "plan"))
    deps.gateway.structured_fns[Role.WRITER] = writer
    url = checkpointer_url(deps.settings)
    async with AsyncPostgresSaver.from_conn_string(url) as saver:
        await saver.setup()
        assert (await execute(build_pipeline(deps, saver), deps, run_id, initial(user_id, run_id))).retryable
    flaky["fail"] = False
    async with AsyncPostgresSaver.from_conn_string(url) as saver:  # new process, same thread id
        resumed = await execute(build_pipeline(deps, saver), deps, run_id, None)
        await saver.adelete_thread(str(run_id))
    assert resumed.status is RunStatus.COMPLETE and calls["plan"] == 1


# --- auto-resume and the retry endpoint ----------------------------------------------------------


async def test_worker_auto_resumes_retryable_failures_once():
    from insightforge.graph.pipeline import RunOutcome
    from insightforge.runtime import Runtime

    class Stub:
        settings = Settings(_env_file=None, max_auto_resumes=1)
        sessionmaker = None

        def __init__(self, outcomes):
            self.outcomes, self.retries = outcomes, 0

        async def retry(self, run_id):
            self.retries += 1
            return self.outcomes.pop(0)

    async def run(first, later):
        stub = Stub(later)
        from unittest.mock import patch
        with patch("insightforge.runtime.RunLog") as log:
            log.return_value.progress = lambda *a, **k: asyncio.sleep(0)
            out = await Runtime.run_with_auto_resume(stub, first, uuid.uuid4())
        return out, stub.retries

    failed_retryable = RunOutcome(RunStatus.FAILED, {}, retryable=True)
    out, n = await run(failed_retryable, [RunOutcome(RunStatus.COMPLETE, {})])
    assert out.status is RunStatus.COMPLETE and n == 1
    out, n = await run(failed_retryable, [RunOutcome(RunStatus.FAILED, {}, retryable=True)])
    assert out.status is RunStatus.FAILED and n == 1  # bounded: one automatic resume
    out, n = await run(RunOutcome(RunStatus.FAILED, {}, retryable=False), [])
    assert n == 0  # a planning error would just fail again
