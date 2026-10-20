import json
import re
import uuid

import pytest
from fakes import FakeGateway, RuleModel, ai_calls, tool_call
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from qdrant_client import models
from test_planner import planner_rule

from insightforge.agents.entity_extractor import EntityExtraction, run_entity_wave
from insightforge.graph.state import (
    BatchExtraction,
    EntityType,
    ExtractedEntity,
    ExtractedFact,
    Predicate,
    SourceEntities,
)
from insightforge.llm.gateway import Role, TokenBudgetExceeded
from insightforge.store.entity_store import load_related_entities, save_entity_graph, upsert_entity_embeddings
from insightforge.store.models import Run, SourceKind
from insightforge.store.qdrant import ENTITIES
from insightforge.store.run_log import RunLog
from insightforge.store.scratch import ScratchStore
from insightforge.store.sources import SourceResolver

PAGES = {
    "https://a.example.com/1": "Salary Hero raised US$7.6 million in a Series A led by Cento Ventures.",
    "https://b.example.com/2": "Thai startup Salary Hero closed a $7.6M Series A round this week.",
    "https://c.example.com/3": "Salary Hero's Series A was USD 12 million, a person familiar said.",
}


_BLOCK = re.compile(r'<untrusted_source id="([^"]+)"[^>]*>(.*?)</untrusted_source>', re.DOTALL)


def extract_fn(seen: list[str]):
    """Stands in for the extraction model: answers per source block, reading only that block."""

    def fn(messages):
        text = messages[-1].content
        seen.append(text)
        out = []
        for source_id, body in _BLOCK.findall(text):
            amount = re.search(r"(US\$7\.6 million|\$7\.6M|USD 12 million)", body).group(1)
            out.append(SourceEntities(
                source_id=source_id,
                entities=[ExtractedEntity(name="Salary Hero", type=EntityType.COMPANY),
                          ExtractedEntity(name="Cento Ventures", type=EntityType.COMPANY)],
                facts=[ExtractedFact(subject="Salary Hero", predicate=Predicate.RAISED, object=amount,
                                     qualifier="Series A", date="2026-03")],
            ))
        return BatchExtraction(sources=out)

    return fn


def extractor_agent_rule(messages):
    last = messages[-1]
    if isinstance(last, HumanMessage):
        ids = re.findall(r"^(src-\w+|chk-\w+)$", last.content, flags=re.MULTILINE)
        return ai_calls(tool_call("extract_entities", {"source_ids": ids}))
    if isinstance(last, ToolMessage):
        return AIMessage(content=json.dumps({"status": "Done", "processed": 99}))  # a wrong count, on purpose
    return AIMessage(content="done")


async def stored_sources(maker, run_id) -> list[str]:
    scratch = ScratchStore(maker, run_id)
    return [await scratch.put(kind=SourceKind.WEB_PAGE, key=url, url=url, body_text=body)
            for url, body in PAGES.items()]


async def test_entity_wave_isolated_one_source_per_call_and_flags_conflict(committed_run):
    maker, user_id, run_id = committed_run
    ids = await stored_sources(maker, run_id)
    seen_by_extraction: list[str] = []
    extractor = RuleModel(rule=extractor_agent_rule)
    gw = FakeGateway({Role.PLANNER: RuleModel(rule=planner_rule()), Role.EXTRACTOR: extractor},
                     {Role.EXTRACTOR: extract_fn(seen_by_extraction)})
    resolver = SourceResolver(ScratchStore(maker, run_id), None, user_id)

    graph = await run_entity_wave(gw, resolver, RunLog(maker, run_id), ids, batch_size=10)

    # The extractor agent got ids only; the three sources went in one batched call, each wrapped separately.
    agent_inputs = "".join(str(m.content) for call in extractor.seen for m in call if isinstance(m, HumanMessage))
    assert all(i in agent_inputs for i in ids) and "Salary Hero" not in agent_inputs
    assert len(seen_by_extraction) == 1
    assert [sid for sid, _ in _BLOCK.findall(seen_by_extraction[0])] == ids

    assert {e.key for e in graph.entities} == {"company:salaryhero", "company:centoventures"}
    [conflict] = graph.conflicts
    assert conflict.predicate is Predicate.RAISED and len(conflict.values) == 2
    assert graph.relationships[0].date.isoformat() == "2026-03-01"


async def test_extraction_ignores_unassigned_ids_and_reports_failures(committed_run):
    maker, user_id, run_id = committed_run
    ids = await stored_sources(maker, run_id)
    calls = []

    def flaky(messages):  # one source per call here, so a failure hits exactly one source
        calls.append(1)
        if "USD 12 million" in messages[-1].content:
            raise RuntimeError("model hiccup")
        return extract_fn([])(messages)

    gw = FakeGateway({}, {Role.EXTRACTOR: flaky})
    resolver = SourceResolver(ScratchStore(maker, run_id), None, user_id)
    extraction = EntityExtraction(gw, resolver, RunLog(maker, run_id), ids[:3], per_call=1)
    out = await extraction.extract([*ids, "src-notassigned1"])
    assert "Processed 2 source(s)" in out and f"Failed ids: ['{ids[2]}']" in out
    assert "Not assigned to you, ignored: ['src-notassigned1']" in out
    again = await extraction.extract(ids[:1])
    assert "Already processed" in again and len(calls) == 3  # no second model call for a done source


async def test_budget_breach_inside_extraction_propagates(committed_run):
    maker, user_id, run_id = committed_run
    ids = await stored_sources(maker, run_id)

    def over_budget(messages):
        raise TokenBudgetExceeded("run used 250000 tokens")

    gw = FakeGateway({}, {Role.EXTRACTOR: over_budget})
    extraction = EntityExtraction(gw, SourceResolver(ScratchStore(maker, run_id), None, user_id),
                                  RunLog(maker, run_id), ids)
    with pytest.raises(TokenBudgetExceeded):
        await extraction.extract(ids)


async def test_graph_is_persisted_and_entities_are_recalled_per_user(committed_run, vectors):
    maker, user_id, run_id = committed_run
    ids = await stored_sources(maker, run_id)
    gw = FakeGateway({}, {Role.EXTRACTOR: extract_fn([])})
    extraction = EntityExtraction(gw, SourceResolver(ScratchStore(maker, run_id), None, user_id),
                                  RunLog(maker, run_id), ids)
    await extraction.extract(ids)
    graph = extraction.graph()
    try:
        await save_entity_graph(maker, run_id, graph)
        async with maker() as s:
            stored = (await s.get(Run, run_id)).entity_graph
        assert stored["conflicts"][0]["subject"] == "company:salaryhero"

        assert await upsert_entity_embeddings(vectors, user_id, run_id, graph) == 2
        assert await upsert_entity_embeddings(vectors, user_id, run_id, graph) == 2  # same keys: updated in place
        recalled = await load_related_entities(vectors, user_id, "Salary Hero funding in Thailand", n=5)
        assert any(line.startswith("Salary Hero (company): raised") for line in recalled)
        assert "[conflicting reports]" in next(line for line in recalled if line.startswith("Salary Hero"))
        assert await load_related_entities(vectors, uuid.uuid4(), "Salary Hero funding", n=5) == []
        points, _ = await vectors.client.scroll(ENTITIES, scroll_filter=models.Filter(must=[
            models.FieldCondition(key="user_id", match=models.MatchValue(value=str(user_id)))]), limit=10)
        assert len(points) == 2
    finally:
        await vectors.client.delete(ENTITIES, points_selector=models.FilterSelector(filter=models.Filter(must=[
            models.FieldCondition(key="user_id", match=models.MatchValue(value=str(user_id)))])))


async def test_sources_the_agent_skips_are_processed_directly(committed_run):
    """Live runs showed the extractor agent sometimes answering without calling its tool."""
    from sqlalchemy import select

    from insightforge.store.models import RunEvent

    maker, user_id, run_id = committed_run
    ids = await stored_sources(maker, run_id)
    lazy = RuleModel(rule=lambda messages: AIMessage(content=json.dumps({"status": "Done", "processed": 3})))
    gw = FakeGateway({Role.PLANNER: RuleModel(rule=planner_rule()), Role.EXTRACTOR: lazy},
                     {Role.EXTRACTOR: extract_fn([])})
    graph = await run_entity_wave(gw, SourceResolver(ScratchStore(maker, run_id), None, user_id),
                                  RunLog(maker, run_id), ids)
    assert len(graph.conflicts) == 1  # all three sources were extracted despite the lazy agent
    async with maker() as s:
        retries = (await s.scalars(select(RunEvent).where(RunEvent.run_id == run_id, RunEvent.kind == "retry"))).all()
    assert "left 3 of 3 source(s) unprocessed" in retries[0].message


async def test_sources_the_model_leaves_out_of_a_batch_count_as_failed(committed_run):
    maker, user_id, run_id = committed_run
    ids = await stored_sources(maker, run_id)

    def drops_last(messages):
        out = extract_fn([])(messages)
        return BatchExtraction(sources=out.sources[:-1])

    extraction = EntityExtraction(FakeGateway({}, {Role.EXTRACTOR: drops_last}),
                                  SourceResolver(ScratchStore(maker, run_id), None, user_id), RunLog(maker, run_id), ids)
    out = await extraction.extract(ids)
    assert "Processed 2 source(s)" in out and f"Failed ids: ['{ids[2]}']" in out
