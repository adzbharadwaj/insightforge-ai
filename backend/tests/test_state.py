import uuid

import pytest
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from insightforge.graph.state import (
    EvidenceBundle,
    PipelineState,
    ResearchResult,
    SourceRef,
    SubTaskSpec,
    TaskGraph,
)
from insightforge.store.models import SourceKind, SourceType, SubTaskStatus


def _task(i: int, priority: int | None = None) -> SubTaskSpec:
    return SubTaskSpec(
        id=f"st-{i}",
        query=f"query {i}",
        scope="SEA only",
        source_type=SourceType.WEB,
        time_horizon="last 18 months",
        priority=priority or i,
    )


def test_task_graph_needs_question_xor_tasks():
    TaskGraph(sub_tasks=[_task(1)])
    TaskGraph(clarifying_question="Which region?")
    with pytest.raises(ValidationError):
        TaskGraph()
    with pytest.raises(ValidationError):
        TaskGraph(sub_tasks=[_task(1)], clarifying_question="Which region?")


def test_task_graph_never_exceeds_six_and_rejects_duplicates():
    graph = TaskGraph(sub_tasks=[_task(i, priority=min(i, 6)) for i in range(1, 8)])
    assert len(graph.sub_tasks) == 6  # the extra one is cut by priority, not a reason to fail the run
    with pytest.raises(ValidationError):
        TaskGraph(sub_tasks=[_task(1), _task(1)])


def test_capped_keeps_highest_priority():
    graph = TaskGraph(sub_tasks=[_task(1, 3), _task(2, 1), _task(3, 2)])
    assert [t.id for t in graph.capped(2).sub_tasks] == ["st-2", "st-3"]


def test_source_ref_has_no_room_for_raw_text():
    ref = SourceRef(source_id="src-1", kind=SourceKind.WEB_PAGE, summary="x" * 5_000)
    assert len(ref.summary) == 300  # a researcher pasting a whole article only gets one line through
    assert "body_text" not in SourceRef.model_fields


def test_bundle_over_budget_is_invalid():
    with pytest.raises(ValidationError, match="budget"):
        EvidenceBundle(sections=[], token_count=6_001, token_budget=6_000)


def test_parallel_research_results_are_merged_by_reducer():
    """Two branches writing research_results in the same superstep must both survive."""

    def branch(sub_task_id):
        def node(state: PipelineState) -> dict:
            return {"research_results": [ResearchResult(sub_task_id=sub_task_id, status=SubTaskStatus.DONE)]}

        return node

    g = StateGraph(PipelineState)
    g.add_node("a", branch("st-1"))
    g.add_node("b", branch("st-2"))
    g.add_edge(START, "a")
    g.add_edge(START, "b")
    g.add_edge("a", END)
    g.add_edge("b", END)
    out = g.compile().invoke({"run_id": uuid.uuid4(), "user_id": uuid.uuid4(), "brief": "b"})
    assert sorted(r.sub_task_id for r in out["research_results"]) == ["st-1", "st-2"]


def test_model_written_fields_clip_instead_of_failing_the_run():
    """Native JSON schema treats maxLength as a hint; a 350-char scope once failed a live run."""
    raw = {"sub_tasks": [{"id": f"st-{i}", "query": "q" * 400, "scope": "s" * 350, "source_type": "web",
                          "time_horizon": "last 12 months", "priority": p}
                         for i, p in enumerate([6, 5, 4, 3, 2, 1, 1], start=1)]}
    g = TaskGraph.model_validate(raw)
    assert len(g.sub_tasks) == 6 and "st-1" not in {t.id for t in g.sub_tasks}  # lowest priority (6) dropped
    assert {len(t.query) for t in g.sub_tasks} == {300} and {len(t.scope) for t in g.sub_tasks} == {300}
    assert TaskGraph.model_json_schema()["$defs"]["SubTaskSpec"]["properties"]["scope"]["maxLength"] == 300
