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


def test_task_graph_rejects_more_than_six_and_duplicates():
    with pytest.raises(ValidationError):
        TaskGraph(sub_tasks=[_task(i, priority=min(i, 6)) for i in range(1, 8)])
    with pytest.raises(ValidationError):
        TaskGraph(sub_tasks=[_task(1), _task(1)])


def test_capped_keeps_highest_priority():
    graph = TaskGraph(sub_tasks=[_task(1, 3), _task(2, 1), _task(3, 2)])
    assert [t.id for t in graph.capped(2).sub_tasks] == ["st-2", "st-3"]


def test_source_ref_has_no_room_for_raw_text():
    with pytest.raises(ValidationError):
        SourceRef(source_id="src-1", kind=SourceKind.WEB_PAGE, summary="x" * 301)
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
