import asyncio
import json
import re
import uuid

import pytest
from fakes import FakeGateway, RuleModel, ai_calls, tool_call
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from sqlalchemy import select
from test_web_search import TODAY, FakeProvider, article, fetcher_for, html

from insightforge.agents.delegation import ResearchRecorder
from insightforge.agents.planner import (
    PlanningError,
    allowed_source_types,
    build_delegating_agent,
    persist_task_graph,
    plan,
    run_research,
)
from insightforge.agents.web_search import web_researcher_spec
from insightforge.graph.state import SubTaskSpec, TaskGraph
from insightforge.llm.gateway import Role
from insightforge.store.models import SourceType, SubTask, SubTaskStatus
from insightforge.store.run_log import RunLog
from insightforge.store.scratch import ScratchStore
from insightforge.store.sources import SourceResolver
from insightforge.tools.search_providers import SearchHit
from insightforge.tools.web_search import WebResearcher, make_web_search_tool

BRIEF = "Summarize the competitive landscape for B2B payroll software in Southeast Asia, focus on pricing."


def spec(i: int, source_type=SourceType.WEB, priority=None) -> SubTaskSpec:
    return SubTaskSpec(id=f"st-{i}", query=f"payroll topic {i}", scope="SEA only", source_type=source_type,
                       time_horizon="last 12 months", priority=priority or i)


# --- step 1: plan() --------------------------------------------------------------


async def test_plan_filters_disallowed_sources_and_caps():
    graph = TaskGraph(sub_tasks=[spec(1), spec(2, SourceType.DOCS), spec(3), spec(4), spec(5, priority=5)])
    gw = FakeGateway({}, {Role.PLANNER: lambda _: graph})
    out = await plan(gw, brief=BRIEF, allowed=[SourceType.WEB], memory_summaries=[], max_subtasks=3)
    assert [t.id for t in out.sub_tasks] == ["st-1", "st-3", "st-4"]


async def test_plan_passes_through_one_clarifying_question_only():
    gw = FakeGateway({}, {Role.PLANNER: lambda _: TaskGraph(clarifying_question="Which region?")})
    out = await plan(gw, brief="payroll", allowed=[SourceType.WEB], memory_summaries=[], max_subtasks=6)
    assert out.clarifying_question == "Which region?"
    with pytest.raises(PlanningError, match="second clarifying question"):
        await plan(gw, brief="payroll", allowed=[SourceType.WEB], memory_summaries=[], max_subtasks=6,
                   clarification_answer="SEA")


def test_allowed_source_types_respects_prefs_and_available_agents():
    prefs = {"web": True, "docs": True, "memory": False}
    assert allowed_source_types(prefs, {"web-researcher"}) == [SourceType.WEB]
    assert allowed_source_types(prefs, {"web-researcher", "doc-reader"}) == [SourceType.WEB, SourceType.DOCS]


# --- step 2: delegation through the deep agent -----------------------------------


def planner_rule(sneak_brief: bool = False, skip: str | None = None):
    """Plans like a well-behaved coordinator; optionally also tries to leak the brief into `task`."""

    def rule(messages):
        human = next(m for m in messages if isinstance(m, HumanMessage)).content
        ids = re.findall(r"([\w-]+) → subagent_type=([\w-]+)", human)
        n_ai = sum(isinstance(m, AIMessage) for m in messages)
        if n_ai == 0:
            return ai_calls(tool_call("write_todos", {"todos": [{"content": i, "status": "pending"} for i, _ in ids]}))
        if n_ai == 1:
            calls = [tool_call("task", {"description": i, "subagent_type": t}) for i, t in ids if i != skip]
            if sneak_brief:
                calls.append(tool_call("task", {"description": f"Context: {BRIEF}", "subagent_type": "web-researcher"}))
                calls.append(tool_call("task", {"description": ids[0][0], "subagent_type": "general-purpose"}))
            return ai_calls(*calls)
        return AIMessage(content="\n".join(f"{i} Done" for i, _ in ids))

    return rule


def researcher_rule(invent_ref: bool = False, as_tool: bool = False):
    def rule(messages):
        last = messages[-1]
        if isinstance(last, HumanMessage):
            sub_task = re.search(r"Sub-task id: (st-\d+)", last.content).group(1)
            query = re.search(r"Query: (.+)", last.content).group(1)
            return ai_calls(tool_call("web_search", {"sub_task_id": sub_task, "query": query}))
        if isinstance(last, ToolMessage) and last.name == "web_search":
            sub_task = re.search(r"Sub-task id: (st-\d+)", next(
                m.content for m in messages if isinstance(m, HumanMessage))).group(1)
            refs = [
                {"source_id": sid, "kind": "web_page", "summary": "Vendor pricing news.", "credibility": float(cred)}
                for sid, cred in re.findall(r"source_id=(src-\w+) .*?credibility=([\d.]+)", last.content)
            ]
            if invent_ref:
                refs.append({"source_id": "src-000000000000", "kind": "web_page", "summary": "made up"})
            result = {"sub_task_id": sub_task, "status": "Done", "refs": refs}
            if as_tool:  # open models answer through a forced structured-output tool call
                return ai_calls(tool_call("ResearchResult", result))
            return AIMessage(content=json.dumps(result))
        return AIMessage(content="done")

    return rule


def make_researcher(maker, run_id, n_per_query=2) -> WebResearcher:
    class PerQueryProvider(FakeProvider):
        async def search(self, query, max_results, days=None):
            slug = re.sub(r"\W+", "-", query)
            return [SearchHit(f"https://news{i}.example.com/{slug}", f"{query} {i}", "") for i in range(n_per_query)]

    routes = {}
    for q in range(1, 7):
        for i in range(n_per_query):
            routes[f"https://news{i}.example.com/payroll-topic-{q}"] = html(article(f"T{q}{i}", f"Fact {q}.{i}."))
    return WebResearcher(PerQueryProvider([]), fetcher_for(routes), ScratchStore(maker, run_id), RunLog(maker, run_id),
                         results_per_subtask=n_per_query, today=TODAY)


async def setup_run(maker, run_id, specs, *, planner, researcher, web_tool=None, timeout_s=30):
    graph = TaskGraph(sub_tasks=specs)
    log, scratch = RunLog(maker, run_id), ScratchStore(maker, run_id)
    await persist_task_graph(maker, log, graph)
    recorder = ResearchRecorder(maker, SourceResolver(scratch, None, uuid.uuid4()), log, specs)
    gw = FakeGateway({Role.PLANNER: planner, Role.RESEARCHER: researcher})
    tool = web_tool or make_web_search_tool(make_researcher(maker, run_id))
    agent = build_delegating_agent(gw, recorder, [web_researcher_spec(gw, tool, max_tool_calls=4)],
                                 subtask_timeout_s=timeout_s)
    return graph, recorder, agent


async def test_planner_delegates_in_parallel_and_collects_refs(committed_run):
    maker, _, run_id = committed_run
    planner, researcher = RuleModel(rule=planner_rule()), RuleModel(rule=researcher_rule())
    graph, recorder, agent = await setup_run(maker, run_id, [spec(1), spec(2)], planner=planner, researcher=researcher)

    results = await run_research(agent, recorder, graph)

    assert [(r.sub_task_id, r.status) for r in results] == [("st-1", SubTaskStatus.DONE), ("st-2", SubTaskStatus.DONE)]
    assert all(len(r.refs) == 2 for r in results)
    rows = await ScratchStore(maker, run_id).get_many([ref.source_id for ref in results[0].refs])
    assert all("Fact 1." in row.body_text for row in rows)
    async with maker() as s:
        statuses = {t.task_key: t.status for t in await s.scalars(select(SubTask).where(SubTask.run_id == run_id))}
    assert statuses == {"st-1": SubTaskStatus.DONE, "st-2": SubTaskStatus.DONE}
    # The Planner only ever sees references, never page bodies.
    planner_inputs = "".join(str(m.content) for call in planner.seen for m in call)
    assert "Fact 1.0" not in planner_inputs


async def test_brief_never_reaches_researchers_and_bad_delegations_are_refused(committed_run):
    maker, _, run_id = committed_run
    planner = RuleModel(rule=planner_rule(sneak_brief=True))
    researcher = RuleModel(rule=researcher_rule())
    graph, recorder, agent = await setup_run(maker, run_id, [spec(1)], planner=planner, researcher=researcher)

    results = await run_research(agent, recorder, graph)

    assert results[0].status is SubTaskStatus.DONE
    researcher_inputs = [str(m.content) for call in researcher.seen for m in call]
    assert not any("Southeast Asia" in text or "competitive landscape" in text for text in researcher_inputs)
    refusals = [m.content for call in planner.seen for m in call if isinstance(m, ToolMessage) and m.status == "error"]
    assert any("Unknown id" in r for r in refusals)
    assert any("must use subagent_type" in r for r in refusals)


async def test_invented_source_ids_are_dropped(committed_run):
    maker, _, run_id = committed_run
    graph, recorder, agent = await setup_run(maker, run_id, [spec(1)], planner=RuleModel(rule=planner_rule()),
                                             researcher=RuleModel(rule=researcher_rule(invent_ref=True)))
    results = await run_research(agent, recorder, graph)
    assert "src-000000000000" not in {r.source_id for r in results[0].refs}
    assert len(results[0].refs) == 2


async def test_slow_subtask_times_out_and_skipped_subtask_fails(committed_run):
    maker, _, run_id = committed_run

    async def slow_search(sub_task_id: str, query: str, time_horizon: str | None = None) -> str:
        await asyncio.sleep(10)
        return ""

    slow_tool = StructuredTool.from_function(coroutine=slow_search, name="web_search", description="slow")
    graph, recorder, agent = await setup_run(
        maker, run_id, [spec(1), spec(2)], planner=RuleModel(rule=planner_rule(skip="st-2")),
        researcher=RuleModel(rule=researcher_rule()), web_tool=slow_tool, timeout_s=0.5,
    )
    results = {r.sub_task_id: r for r in await run_research(agent, recorder, graph)}
    assert results["st-1"].status is SubTaskStatus.FAILED and "timed out" in results["st-1"].error
    assert results["st-2"].status is SubTaskStatus.FAILED and results["st-2"].error == "not delegated"


async def test_researchers_have_no_write_tools(committed_run):
    """Inside a real delegation, the researcher is offered only web_search and read-only read_file:
    nothing that writes, and no `task` tool to spawn further agents. Its output schema arrives as a
    native response_format, not as a tool."""
    maker, _, run_id = committed_run
    offered: set[str] = set()
    formats: list = []

    class Capture(RuleModel):
        def bind_tools(self, tools, **kwargs):
            offered.update(getattr(t, "name", None) or t.get("name") or t["function"]["name"] for t in tools)
            formats.append(kwargs.get("response_format"))
            return self

    graph, recorder, agent = await setup_run(maker, run_id, [spec(1)], planner=RuleModel(rule=planner_rule()),
                                             researcher=Capture(rule=researcher_rule()))
    await run_research(agent, recorder, graph)
    assert offered == {"web_search", "read_file"}
    assert formats and all(f and f["json_schema"]["name"] == "ResearchResult" for f in formats)



async def test_open_model_path_returns_results_through_a_structured_output_tool(committed_run):
    """Open models get ToolStrategy: the researcher's answer arrives as a ResearchResult tool call."""
    maker, _, run_id = committed_run
    graph = TaskGraph(sub_tasks=[spec(1)])
    log, scratch = RunLog(maker, run_id), ScratchStore(maker, run_id)
    await persist_task_graph(maker, log, graph)
    recorder = ResearchRecorder(maker, SourceResolver(scratch, None, uuid.uuid4()), log, graph.sub_tasks)
    gw = FakeGateway({Role.PLANNER: RuleModel(rule=planner_rule()),
                      Role.RESEARCHER: RuleModel(rule=researcher_rule(as_tool=True))}, output_strategy="tool")
    tool = make_web_search_tool(make_researcher(maker, run_id))
    agent = build_delegating_agent(gw, recorder, [web_researcher_spec(gw, tool, max_tool_calls=4)],
                                   subtask_timeout_s=30)
    [result] = await run_research(agent, recorder, graph)
    assert result.status is SubTaskStatus.DONE and len(result.refs) == 2
