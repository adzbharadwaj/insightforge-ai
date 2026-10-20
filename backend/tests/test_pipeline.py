from fakes import FakeGateway, RuleModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from report_fakes import REPORT_FNS
from sqlalchemy import select
from test_planner import BRIEF, make_researcher, planner_rule, researcher_rule, spec

from insightforge.agents.web_search import web_researcher_spec
from insightforge.config import Settings
from insightforge.graph.pipeline import PipelineDeps, build_pipeline, execute
from insightforge.graph.state import TaskGraph
from insightforge.llm.gateway import Role, TokenMeter
from insightforge.store.models import Run, RunStatus, SubTask, SubTaskStatus
from insightforge.tools.web_search import make_web_search_tool


def deps_for(maker, run_id, plan_fn, *, researcher=None, settings=None, meter=None, memory=None):
    gw = FakeGateway(
        {Role.PLANNER: RuleModel(rule=planner_rule()), Role.RESEARCHER: researcher or RuleModel(rule=researcher_rule())},
        {Role.PLANNER: plan_fn, **REPORT_FNS},
    )
    gw.meter = meter

    def subagents(ctx):
        tool = make_web_search_tool(make_researcher(maker, run_id))
        return [web_researcher_spec(gw, tool, max_tool_calls=4)]

    async def memory_loader(user_id, n):
        return memory or []

    return PipelineDeps(settings or Settings(_env_file=None), maker, gw, subagents, memory_loader,
                        extract_entities=False)  # wave 2 has its own test below


def initial(user_id, run_id, **extra):
    return {"run_id": run_id, "user_id": user_id, "brief": BRIEF,
            "source_prefs": {"web": True, "docs": False, "memory": True}, **extra}


async def run_row(maker, run_id) -> Run:
    async with maker() as s:
        return await s.get(Run, run_id)


async def test_plan_then_research_end_to_end(committed_run):
    maker, user_id, run_id = committed_run
    seen_prompts = []

    def plan_fn(messages):
        seen_prompts.append(messages[0].content)
        return TaskGraph(sub_tasks=[spec(1), spec(2)])

    deps = deps_for(maker, run_id, plan_fn, memory=["Last week: Deel pricing in Vietnam."])
    outcome = await execute(build_pipeline(deps, InMemorySaver()), deps, run_id, initial(user_id, run_id))

    assert outcome.status is RunStatus.COMPLETE
    assert [r.sub_task_id for r in outcome.state["research_results"]] == ["st-1", "st-2"]
    assert "Deel pricing in Vietnam" in seen_prompts[0]  # memory summaries reach the Planner
    row = await run_row(maker, run_id)
    assert row.status is RunStatus.COMPLETE and row.started_at is not None
    async with maker() as s:
        tasks = (await s.scalars(select(SubTask).where(SubTask.run_id == run_id))).all()
    assert {t.status for t in tasks} == {SubTaskStatus.DONE}


async def test_clarifying_question_pauses_and_resumes(committed_run):
    maker, user_id, run_id = committed_run
    answers = []

    def plan_fn(messages):
        human = messages[1].content
        if "The user answered" in human:
            answers.append(human)
            return TaskGraph(sub_tasks=[spec(1)])
        return TaskGraph(clarifying_question="Which region should I focus on?")

    deps = deps_for(maker, run_id, plan_fn)
    pipeline = build_pipeline(deps, InMemorySaver())

    paused = await execute(pipeline, deps, run_id, initial(user_id, run_id))
    assert paused.status is RunStatus.AWAITING_CLARIFICATION
    assert paused.question == "Which region should I focus on?"
    row = await run_row(maker, run_id)
    assert row.status is RunStatus.AWAITING_CLARIFICATION and row.clarifying_question == paused.question

    resumed = await execute(pipeline, deps, run_id, Command(resume="Southeast Asia"))
    assert resumed.status is RunStatus.COMPLETE
    assert "Southeast Asia" in answers[0]
    assert (await run_row(maker, run_id)).clarification_answer == "Southeast Asia"


async def test_run_fails_cleanly_when_no_subtask_succeeds(committed_run):
    maker, user_id, run_id = committed_run
    empty = RuleModel(rule=researcher_rule())
    deps = deps_for(maker, run_id, lambda m: TaskGraph(sub_tasks=[spec(9, priority=1)]), researcher=empty)  # no fixture pages
    outcome = await execute(build_pipeline(deps, InMemorySaver()), deps, run_id, initial(user_id, run_id))
    assert outcome.status is RunStatus.FAILED
    row = await run_row(maker, run_id)
    assert row.status is RunStatus.FAILED and "no sub-task produced usable sources" in row.error


async def test_token_budget_breach_fails_the_run(committed_run):
    maker, user_id, run_id = committed_run
    meter = TokenMeter(budget=10)
    meter.input_tokens = 10  # already spent

    def plan_fn(messages):
        meter.on_chat_model_start({}, messages)  # what the real gateway's callback does before each call
        return TaskGraph(sub_tasks=[spec(1)])

    deps = deps_for(maker, run_id, plan_fn, meter=meter)
    outcome = await execute(build_pipeline(deps, InMemorySaver()), deps, run_id, initial(user_id, run_id))
    assert outcome.status is RunStatus.FAILED
    assert "token budget exceeded" in (await run_row(maker, run_id)).error


async def test_time_budget_breach_fails_the_run(committed_run):
    import asyncio

    maker, user_id, run_id = committed_run

    async def slow_plan(messages):
        await asyncio.sleep(5)

    deps = deps_for(maker, run_id, slow_plan, settings=Settings(_env_file=None, run_time_budget_s=1))
    outcome = await execute(build_pipeline(deps, InMemorySaver()), deps, run_id, initial(user_id, run_id))
    assert outcome.status is RunStatus.FAILED
    assert "time budget" in (await run_row(maker, run_id)).error


async def test_clarification_survives_a_real_postgres_checkpoint(committed_run):
    """Pause, drop the pipeline object, rebuild it, and resume from Postgres: state must round-trip."""
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    from insightforge.cli import checkpointer_url

    maker, user_id, run_id = committed_run

    def plan_fn(messages):
        if "The user answered" in messages[1].content:
            return TaskGraph(sub_tasks=[spec(1)])
        return TaskGraph(clarifying_question="Which region?")

    deps = deps_for(maker, run_id, plan_fn)
    url = checkpointer_url(deps.settings)
    async with AsyncPostgresSaver.from_conn_string(url) as saver:
        await saver.setup()
        paused = await execute(build_pipeline(deps, saver), deps, run_id, initial(user_id, run_id))
    assert paused.status is RunStatus.AWAITING_CLARIFICATION

    async with AsyncPostgresSaver.from_conn_string(url) as saver:  # fresh process, same thread_id
        resumed = await execute(build_pipeline(deps, saver), deps, run_id, Command(resume="SEA"))
        await saver.adelete_thread(str(run_id))
    assert resumed.status is RunStatus.COMPLETE
    assert resumed.state["task_graph"].sub_tasks[0].id == "st-1"


async def test_provider_error_fails_the_run_instead_of_crashing(committed_run):
    """A provider 400 (e.g. an unsupported request parameter) must end the run FAILED with the reason."""
    maker, user_id, run_id = committed_run

    def plan_fn(messages):
        raise RuntimeError("Error code: 400 - tool_choice not supported")

    deps = deps_for(maker, run_id, plan_fn)
    outcome = await execute(build_pipeline(deps, InMemorySaver()), deps, run_id, initial(user_id, run_id))
    assert outcome.status is RunStatus.FAILED
    assert "tool_choice not supported" in (await run_row(maker, run_id)).error


def with_entities(deps, extract):
    from dataclasses import replace

    from test_entity_wave import extractor_agent_rule

    deps.gateway.models[Role.EXTRACTOR] = RuleModel(rule=extractor_agent_rule)
    deps.gateway.structured_fns[Role.EXTRACTOR] = extract
    return replace(deps, extract_entities=True)


async def test_entity_wave_runs_after_research_and_is_persisted(committed_run):
    import re

    from insightforge.graph.state import BatchExtraction, EntityType, ExtractedEntity, SourceEntities

    maker, user_id, run_id = committed_run
    seen = []

    def extract(messages):
        seen.append(messages[-1].content)
        ids = re.findall(r'<untrusted_source id="([^"]+)"', messages[-1].content)
        return BatchExtraction(sources=[
            SourceEntities(source_id=i, entities=[ExtractedEntity(name="Salary Hero", type=EntityType.COMPANY)])
            for i in ids])

    deps = with_entities(deps_for(maker, run_id, lambda m: TaskGraph(sub_tasks=[spec(1)])), extract)
    outcome = await execute(build_pipeline(deps, InMemorySaver()), deps, run_id, initial(user_id, run_id))

    assert outcome.status is RunStatus.COMPLETE
    refs = outcome.state["research_results"][0].refs
    assert len(refs) == 2 and len(seen) == 1  # both kept sources in one batched extraction call
    graph = outcome.state["entity_graph"]
    assert "company:salaryhero" in {e.key for e in graph.entities}
    assert (await run_row(maker, run_id)).entity_graph["entities"]


async def test_entity_wave_failure_degrades_instead_of_failing_the_run(committed_run):
    from insightforge.store.models import RunEvent

    maker, user_id, run_id = committed_run

    def broken(messages):
        raise RuntimeError("extractor down")

    deps = with_entities(deps_for(maker, run_id, lambda m: TaskGraph(sub_tasks=[spec(1)])), broken)
    outcome = await execute(build_pipeline(deps, InMemorySaver()), deps, run_id, initial(user_id, run_id))

    assert outcome.status is RunStatus.COMPLETE
    assert not outcome.state["entity_graph"].entities
    async with maker() as s:
        errors = (await s.scalars(select(RunEvent).where(RunEvent.run_id == run_id, RunEvent.kind == "error"))).all()
    assert any("extraction failed" in e.message for e in errors)


def lying_writer(times: int, seen: list[str]):
    """Adds an invented figure to the first section for the first `times` attempts."""
    from report_fakes import writer_fn

    from insightforge.graph.state import WrittenSentence

    def fn(messages):
        seen.append(messages[-1].content)
        out = writer_fn(messages)
        if len(seen) <= times:
            ev = out.sections[0].sentences[0].evidence_ids
            out.sections[0].sentences.append(WrittenSentence(text="The market grew 37% in 2026.", evidence_ids=ev))
        return out

    return fn


async def test_ungrounded_draft_is_retried_with_explicit_constraint(committed_run):
    from insightforge.store.models import RunEvent

    maker, user_id, run_id = committed_run
    seen: list[str] = []
    deps = deps_for(maker, run_id, lambda m: TaskGraph(sub_tasks=[spec(1)]))
    deps.gateway.structured_fns[Role.WRITER] = lying_writer(1, seen)
    outcome = await execute(build_pipeline(deps, InMemorySaver()), deps, run_id, initial(user_id, run_id))

    assert outcome.status is RunStatus.COMPLETE and outcome.state["writer_attempts"] == 2
    assert "Fix all of these" in seen[1] and "37" in seen[1]  # the retry names the bad sentence and why
    assert "37%" not in outcome.state["final_markdown"]
    async with maker() as s:
        retries = (await s.scalars(select(RunEvent).where(RunEvent.run_id == run_id, RunEvent.kind == "retry"))).all()
    assert len(retries) == 1


async def test_claims_still_unsupported_after_retries_are_dropped(committed_run):
    maker, user_id, run_id = committed_run
    seen: list[str] = []
    deps = deps_for(maker, run_id, lambda m: TaskGraph(sub_tasks=[spec(1)]))
    deps.gateway.structured_fns[Role.WRITER] = lying_writer(99, seen)
    outcome = await execute(build_pipeline(deps, InMemorySaver()), deps, run_id, initial(user_id, run_id))

    assert outcome.status is RunStatus.COMPLETE and outcome.state["writer_attempts"] == 3  # 1 + 2 retries
    md = outcome.state["final_markdown"]
    assert "37%" not in md and "1 statement(s) were removed" in md


async def test_full_run_stores_report_fact_check_and_memory(committed_run, vectors):
    from insightforge.store.memory import load_recent_summaries
    from insightforge.store.models import Claim, Report

    maker, user_id, run_id = committed_run
    deps = deps_for(maker, run_id, lambda m: TaskGraph(sub_tasks=[spec(1), spec(2)]))
    deps.vectors = vectors
    outcome = await execute(build_pipeline(deps, InMemorySaver()), deps, run_id, initial(user_id, run_id))

    assert outcome.status is RunStatus.COMPLETE
    fc = outcome.state["fact_check"]
    assert fc.total_checked >= 1  # pass/fail behaviour is covered with realistic text in test_report.py
    row = await run_row(maker, run_id)
    assert row.finished_at is not None and row.tokens_used == 0  # fakes report no usage
    async with maker() as s:
        report = (await s.scalars(select(Report).where(Report.run_id == run_id))).one()
        claims = (await s.scalars(select(Claim).where(Claim.report_id == report.id))).all()
    assert report.markdown == outcome.state["final_markdown"] and "## Gaps & Limitations" in report.markdown
    assert claims and sum(c.sampled for c in claims) == fc.total_checked
    assert set(outcome.state["stage_tokens"]) >= {"plan", "research", "synthesize", "write", "ground", "fact_check"}
    summaries = await load_recent_summaries(vectors, user_id, 3)
    assert summaries and summaries[0].startswith("Brief: ")


async def test_unaffordable_retry_degrades_instead_of_failing(committed_run):
    """A live run hit the token budget on its last rewrite and failed with a nearly finished report."""
    maker, user_id, run_id = committed_run
    seen: list[str] = []
    meter = TokenMeter(budget=1_000)

    def writer(messages):
        meter.input_tokens += 600  # each write costs 600 of the 1,000-token budget
        return lying_writer(99, seen)(messages)

    deps = deps_for(maker, run_id, lambda m: TaskGraph(sub_tasks=[spec(1)]), meter=meter)
    deps.gateway.structured_fns[Role.WRITER] = writer
    outcome = await execute(build_pipeline(deps, InMemorySaver()), deps, run_id, initial(user_id, run_id))

    assert outcome.status is RunStatus.COMPLETE and outcome.state["writer_attempts"] == 1
    assert "37%" not in outcome.state["final_markdown"]
    assert "1 statement(s) were removed" in outcome.state["final_markdown"]
