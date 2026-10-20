import datetime as dt

from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import select
from test_pipeline import deps_for, initial, run_row
from test_planner import spec
from test_report import SOURCES, item

from insightforge.agents.conflict_review import (
    KEEP_BOTH,
    ConflictOption,
    ConflictReview,
    apply_decisions,
    build_reviews,
)
from insightforge.graph.pipeline import build_pipeline, execute
from insightforge.graph.state import (
    Confidence,
    Conflict,
    ConflictValue,
    Entity,
    EntityGraph,
    EntityType,
    EvidenceBundle,
    EvidenceSection,
    Predicate,
    TaskGraph,
)
from insightforge.llm.gateway import Role
from insightforge.store.models import Run, RunEvent, RunStatus

A = item(1, claim="Allego operates more than 34,000 charge points.", sources=("src-a",), conf=Confidence.LOW)
B = item(2, claim="Allego operates more than 35,000 charging points.", sources=("src-b",), conf=Confidence.LOW)
OTHER = item(3, claim="Fastned operates 300 stations.", sources=("src-c",), conf=Confidence.MEDIUM)
A.conflict = B.conflict = True
GRAPH = EntityGraph(
    entities=[Entity(key="company:allego", name="Allego", type=EntityType.COMPANY)],
    conflicts=[Conflict(subject="company:allego", predicate=Predicate.EMPLOYEE_COUNT,
                        values=[ConflictValue(object="34,000", source_ids=["src-a"]),
                                ConflictValue(object="35,000", source_ids=["src-b"])])],
)


def bundle() -> EvidenceBundle:
    return EvidenceBundle(sections=[EvidenceSection(theme="Operators", sub_task_id="st-1", evidence=[A, B, OTHER],
                                                    confidence=Confidence.LOW)],
                          sources=SOURCES, token_count=100, token_budget=6000)


def test_reviews_list_each_side_with_its_publishers():
    [review] = build_reviews(bundle(), GRAPH)
    assert review.subject == "Allego" and [o.value for o in review.options] == ["34,000", "35,000"]
    assert review.options[0].evidence_ids == ["ev-1"] and review.options[0].sources == ["e27.co"]
    one_sided = bundle()
    one_sided.sections[0].evidence = [A, OTHER]  # the other side never made it into the bundle
    assert build_reviews(one_sided, GRAPH) == []
    assert build_reviews(bundle(), None) == []


def test_choosing_a_side_keeps_it_confirmed_and_drops_the_other():
    [review] = build_reviews(bundle(), GRAPH)
    out = apply_decisions(bundle(), [review], {review.conflict_id: review.options[0].option_id})
    kept = {i.evidence_id: i for s in out.sections for i in s.evidence}
    assert set(kept) == {"ev-1", "ev-3"}
    assert kept["ev-1"].conflict is False and kept["ev-1"].confidence is Confidence.MEDIUM
    assert out.cuts[-1].evidence_ids == ["ev-2"] and "analyst chose 34,000" in out.cuts[-1].reason
    assert "used 34,000 (e27.co) over 35,000 (techsauce.co)" in out.notes[-1]

    kept_both = apply_decisions(bundle(), [review], {review.conflict_id: KEEP_BOTH})
    assert {i.evidence_id for s in kept_both.sections for i in s.evidence} == {"ev-1", "ev-2", "ev-3"}
    assert "kept side by side" in kept_both.notes[-1]


async def test_run_pauses_for_review_and_the_writer_sees_only_the_chosen_side(committed_run, monkeypatch):
    import insightforge.graph.pipeline as pipeline_mod

    maker, user_id, run_id = committed_run

    def fake_reviews(b, graph):  # a conflict between the first two evidence items the fakes produce
        ids = list(b.items())[:2]
        return [ConflictReview(conflict_id="k1", subject="Vendor", predicate="priced_at", options=[
            ConflictOption(option_id=f"k1-v{i}", value=f"value {i}", evidence_ids=[e], claims=[b.items()[e].claim],
                           sources=["x"]) for i, e in enumerate(ids, start=1)])]

    monkeypatch.setattr(pipeline_mod, "build_reviews", fake_reviews)
    seen: list[str] = []
    deps = deps_for(maker, run_id, lambda m: TaskGraph(sub_tasks=[spec(1)]))
    real_writer = deps.gateway.structured_fns[Role.WRITER]
    deps.gateway.structured_fns[Role.WRITER] = lambda m: (seen.append(m[-1].content), real_writer(m))[1]
    pipeline = build_pipeline(deps, InMemorySaver())
    prefs = {"web": True, "docs": False, "memory": True, "review_conflicts": True}

    paused = await execute(pipeline, deps, run_id, initial(user_id, run_id, source_prefs=prefs))
    assert paused.status is RunStatus.AWAITING_REVIEW and paused.review["conflicts"][0]["conflict_id"] == "k1"
    row = await run_row(maker, run_id)
    assert row.status is RunStatus.AWAITING_REVIEW and row.review["conflicts"][0]["options"][1]["option_id"] == "k1-v2"
    assert not seen  # nothing written before the analyst decides
    rejected = paused.review["conflicts"][0]["options"][1]["claims"][0]

    done = await execute(pipeline, deps, run_id, pipeline_mod.Command(resume={"choices": {"k1": "k1-v1"}}))
    assert done.status is RunStatus.COMPLETE
    assert rejected not in seen[0] and "Analyst resolved" in seen[0]
    assert (await run_row(maker, run_id)).review is None


async def test_review_is_skipped_unless_requested(committed_run):
    maker, user_id, run_id = committed_run
    deps = deps_for(maker, run_id, lambda m: TaskGraph(sub_tasks=[spec(1)]))
    outcome = await execute(build_pipeline(deps, InMemorySaver()), deps, run_id, initial(user_id, run_id))
    assert outcome.status is RunStatus.COMPLETE


async def test_unanswered_review_continues_with_both_sides(committed_run):
    from insightforge.store.db import get_engine
    from insightforge.worker import continue_unreviewed

    await get_engine().dispose(close=False)
    maker, user_id, _ = committed_run
    async with maker() as s:
        old = Run(user_id=user_id, brief="b", status=RunStatus.AWAITING_REVIEW,
                  review_requested_at=dt.datetime.now(dt.UTC) - dt.timedelta(minutes=31))
        s.add(old)
        await s.commit()

    class Queue:
        def __init__(self):
            self.jobs = []

        async def enqueue_job(self, fn, *args, _job_id=None):
            self.jobs.append((fn, args))

    q = Queue()
    assert await continue_unreviewed({"redis": q}) >= 1
    assert ("resolve_run", (str(old.id), {})) in q.jobs  # empty choices = keep both
    async with maker() as s:
        assert (await s.get(Run, old.id)).status is RunStatus.SYNTHESIZING
        msgs = [e.message for e in await s.scalars(select(RunEvent).where(RunEvent.run_id == old.id))]
    assert any("both versions kept" in m for m in msgs)
