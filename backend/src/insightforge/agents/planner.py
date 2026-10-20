"""Planner (component 06).

Two steps, both owned by the Planner:

1. `plan()`        brief → typed TaskGraph (or one clarifying question), validated,
                   capped and saved so the UI can render the live progress view.
2. `run_research()` a `create_deep_agent()` root that tracks sub-tasks with
                   `write_todos` and spawns one isolated researcher per sub-task via
                   the `task` tool. Isolation and limits are enforced by
                   DelegationMiddleware, not by the prompt. The same root type runs
                   the second wave (Entity Extractor) with a different ledger.
"""

from deepagents import FilesystemMiddleware, SubAgent, create_deep_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ModelFallbackMiddleware, TodoListMiddleware
from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from insightforge.agents.delegation import SUBAGENT_FOR_SOURCE, DelegationMiddleware, Ledger, ResearchRecorder
from insightforge.agents.prompts import PLANNER_DELEGATE_PROMPT, PLANNER_PLAN_PROMPT
from insightforge.graph.state import ResearchResult, TaskGraph
from insightforge.llm.gateway import ModelGateway, Role
from insightforge.store.models import SourceType, SubTask
from insightforge.store.run_log import RunLog


class PlanningError(RuntimeError):
    pass


def allowed_source_types(source_prefs: dict[str, bool], available_subagents: set[str]) -> list[SourceType]:
    return [
        t for t in SourceType
        if source_prefs.get(t.value, False) and SUBAGENT_FOR_SOURCE[t] in available_subagents
    ]


async def plan(
    gateway: ModelGateway,
    *,
    brief: str,
    allowed: list[SourceType],
    memory_summaries: list[str],
    max_subtasks: int,
    related_entities: list[str] | None = None,
    clarification_answer: str | None = None,
) -> TaskGraph:
    if not allowed:
        raise PlanningError("no source type is enabled and available for this run")
    system = PLANNER_PLAN_PROMPT.format(
        max_subtasks=max_subtasks,
        allowed_sources=", ".join(t.value for t in allowed),
        memory="\n".join(f"- {m}" for m in memory_summaries) or "(none)",
        entities="\n".join(f"- {e}" for e in related_entities or []) or "(none)",
    )
    user = f"Research brief:\n{brief}"
    if clarification_answer:
        user += f"\n\nThe user answered your clarifying question: {clarification_answer}\nDo not ask again; plan now."
    graph: TaskGraph = await gateway.structured(Role.PLANNER, TaskGraph).ainvoke(
        [SystemMessage(system), HumanMessage(user)]
    )
    if graph.clarifying_question:
        if clarification_answer:  # one question only; never loop on clarification
            raise PlanningError("planner asked a second clarifying question")
        return graph
    kept = [t for t in graph.sub_tasks if t.source_type in allowed]
    if not kept:
        raise PlanningError("planner produced no sub-tasks with an allowed source type")
    return graph.model_copy(update={"sub_tasks": kept}).capped(max_subtasks)


async def persist_task_graph(
    sessionmaker: async_sessionmaker[AsyncSession], log: RunLog, graph: TaskGraph
) -> None:
    async with sessionmaker() as session:
        session.add_all(
            SubTask(
                run_id=log.run_id,
                task_key=t.id,
                query=t.query,
                source_type=t.source_type,
                time_horizon=t.time_horizon,
                priority=t.priority,
            )
            for t in graph.sub_tasks
        )
        await session.commit()
    await log.progress("planner", f"planned {len(graph.sub_tasks)} sub-task(s)",
                       task_graph=graph.model_dump(mode="json"))


def build_delegating_agent(
    gateway: ModelGateway,
    ledger: Ledger,
    subagents: list[SubAgent],
    *,
    subtask_timeout_s: float,
    max_planner_calls: int = 8,
):
    return create_deep_agent(
        model=gateway.chat_model(Role.PLANNER),
        tools=[],
        system_prompt=PLANNER_DELEGATE_PROMPT,
        subagents=subagents,
        middleware=[
            FilesystemMiddleware(tools=["read_file"]),  # no write tools for the Planner either
            TodoListMiddleware(),
            DelegationMiddleware(ledger, timeout_s=subtask_timeout_s),
            ModelCallLimitMiddleware(run_limit=max_planner_calls, exit_behavior="end"),
            ModelFallbackMiddleware(gateway.fallback_model(Role.PLANNER)),
        ],
        name="planner",
    )


def delegation_message(assignments: list[tuple[str, str]]) -> str:
    lines = [f"- {task_id} → subagent_type={subagent_type}" for task_id, subagent_type in assignments]
    return "Delegate these sub-tasks:\n" + "\n".join(lines)


async def run_research(agent, recorder: ResearchRecorder, graph: TaskGraph) -> list[ResearchResult]:
    assignments = [(t.id, SUBAGENT_FOR_SOURCE[t.source_type]) for t in graph.sub_tasks]
    await agent.ainvoke({"messages": [HumanMessage(delegation_message(assignments))]})
    return await recorder.finalize()
