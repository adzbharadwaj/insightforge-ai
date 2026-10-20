import uuid
from datetime import date

from insightforge.store.models import SourceKind
from insightforge.store.scratch import ScratchStore, make_source_id
from insightforge.store.sources import SourceResolver
from insightforge.tools.read_source import make_read_source_tool


async def test_put_get_and_same_url_dedupes(committed_run):
    maker, _, run_id = committed_run
    store = ScratchStore(maker, run_id)
    sid = await store.put(
        kind=SourceKind.WEB_PAGE,
        key="https://example.com/a",
        url="https://example.com/a",
        body_text="first version",
        summary="Deel enters Vietnam",
        published_date=date(2025, 3, 1),
        credibility=0.8,
        sub_task_key="st-1",
    )
    again = await store.put(kind=SourceKind.WEB_PAGE, key="https://example.com/a", body_text="second version")
    assert sid == again == make_source_id(run_id, "https://example.com/a")

    row = await store.get(sid)
    assert row.body_text == "second version" and row.token_count > 0
    assert ScratchStore.to_ref(row).source_id == sid


async def test_sources_are_scoped_to_their_run(committed_run):
    maker, _, run_id = committed_run
    sid = await ScratchStore(maker, run_id).put(kind=SourceKind.WEB_PAGE, key="k", body_text="text")
    assert await ScratchStore(maker, uuid.uuid4()).get(sid) is None


async def test_read_source_pages_and_wraps(committed_run):
    maker, _, run_id = committed_run
    store = ScratchStore(maker, run_id)
    body = "A" * 50 + "B" * 50 + "C" * 10
    sid = await store.put(kind=SourceKind.WEB_PAGE, key="long", body_text=body, url="https://e.com/x")
    tool = make_read_source_tool(SourceResolver(store, None, uuid.uuid4()), page_chars=50)

    first = await tool.ainvoke({"source_id": sid})
    assert "page=1/3" in first and "A" * 50 in first and "B" not in first
    assert f'<untrusted_source id="{sid}" url="https://e.com/x">' in first

    last = await tool.ainvoke({"source_id": sid, "page": 2})
    assert "page=3/3" in last and "C" * 10 in last
    assert "does not exist" in await tool.ainvoke({"source_id": sid, "page": 3})
    assert "No source" in await tool.ainvoke({"source_id": "src-unknown"})
