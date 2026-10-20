from datetime import date

import httpx
import pytest
from sqlalchemy import select

from insightforge.guardrails.net import BlockedDestination, check_public_url
from insightforge.store.models import RunEvent
from insightforge.store.run_log import RunLog
from insightforge.store.scratch import ScratchStore
from insightforge.tools.credibility import credibility_score, domain_category, recency_score
from insightforge.tools.fetch import FetchedPage, PageFetcher, Skipped
from insightforge.tools.search_providers import (
    CircuitBreaker,
    SearchHit,
    SearchUnavailable,
    TavilySearch,
    horizon_to_days,
)
from insightforge.tools.web_search import WebResearcher, format_for_agent

TODAY = date(2026, 10, 18)
PUBLIC = lambda host: ["93.184.216.34"]


def article(title: str, body: str, *, author: str | None = "Jane Tan", extra: str = "") -> str:
    meta = f'<meta name="author" content="{author}">' if author else ""
    paras = "".join(f"<p>{body} Sentence {i} adds more reporting detail for length.</p>" for i in range(8))
    return (
        f'<html><head><title>{title}</title>{meta}<meta property="article:published_time" '
        f'content="2026-06-01"></head><body><article><h1>{title}</h1>{paras}{extra}</article></body></html>'
    )


# --- credibility ------------------------------------------------------------


def test_domain_categories():
    assert domain_category("https://www.mom.gov.sg/payroll") == "institutional"
    assert domain_category("https://www.ox.ac.uk/x") == "institutional"
    assert domain_category("https://www.reuters.com/tech/x") == "major_press"
    assert domain_category("https://someone.substack.com/p/x") == "blog"
    assert domain_category("https://acme.io/blog/launch") == "blog"
    assert domain_category("https://random-site.xyz/a") == "unknown"


def test_recency_decay_and_ordering():
    assert recency_score(TODAY, TODAY) == 1.0
    assert recency_score(date(2025, 10, 18), TODAY) == pytest.approx(0.5, abs=0.01)
    gov_new = credibility_score("https://x.gov/a", TODAY, True, TODAY)
    press_old = credibility_score("https://reuters.com/a", date(2023, 1, 1), True, TODAY)
    blog_no_byline = credibility_score("https://a.medium.com/x", TODAY, False, TODAY)
    assert gov_new > press_old and gov_new > blog_no_byline
    assert 0 <= blog_no_byline <= 1


# --- network guard -------------------------------------------------------------


@pytest.mark.parametrize(
    "url,addr",
    [("http://169.254.169.254/latest", "169.254.169.254"), ("http://internal.corp/x", "10.0.0.5"),
     ("http://localhost:8000", "127.0.0.1"), ("http://[::1]/", "::1")],
)
def test_private_destinations_are_blocked(url, addr):
    with pytest.raises(BlockedDestination):
        check_public_url(url, resolve=lambda host: [addr])


def test_non_http_and_https_only():
    with pytest.raises(BlockedDestination):
        check_public_url("file:///etc/passwd", resolve=PUBLIC)
    with pytest.raises(BlockedDestination):
        check_public_url("http://hooks.example.com", resolve=PUBLIC, require_https=True)
    check_public_url("https://hooks.example.com", resolve=PUBLIC, require_https=True)


# --- fetcher -----------------------------------------------------------------------


def fetcher_for(routes: dict[str, httpx.Response], resolve=PUBLIC) -> PageFetcher:
    def handler(request: httpx.Request) -> httpx.Response:
        return routes.get(str(request.url), httpx.Response(404))

    return PageFetcher(httpx.AsyncClient(transport=httpx.MockTransport(handler)), resolve=resolve)


def html(body: str, status: int = 200, **headers) -> httpx.Response:
    return httpx.Response(status, text=body, headers={"content-type": "text/html; charset=utf-8", **headers})


async def test_fetch_parses_article_with_metadata():
    f = fetcher_for({"https://news.example.com/a": html(article("Deel in Vietnam", "Deel launched payroll."))})
    page = await f.fetch("https://news.example.com/a")
    assert isinstance(page, FetchedPage)
    assert "Deel launched payroll." in page.body_text
    assert page.title == "Deel in Vietnam" and page.author == "Jane Tan"
    assert page.published_date == date(2026, 6, 1)


async def test_fetch_skips_paywalls_and_402():
    jsonld = '<script type="application/ld+json">{"isAccessibleForFree": false}</script>'
    f = fetcher_for({
        "https://ft.example.com/a": html(article("Paywalled", "Teaser text.", extra=jsonld)),
        "https://wsj.example.com/b": html("payment required", status=402),
        "https://login.example.com/c": html("<p>Subscribe to continue reading.</p><p>Short teaser.</p>"),
    })
    for url in ("https://ft.example.com/a", "https://wsj.example.com/b", "https://login.example.com/c"):
        result = await f.fetch(url)
        assert isinstance(result, Skipped), url
        assert "paywall" in result.reason


async def test_redirect_to_private_address_is_blocked():
    def resolve(host):
        return ["10.0.0.7"] if host == "evil-internal.example" else ["93.184.216.34"]

    f = fetcher_for(
        {"https://news.example.com/r": httpx.Response(302, headers={"location": "http://evil-internal.example/admin"})},
        resolve=resolve,
    )
    result = await f.fetch("https://news.example.com/r")
    assert isinstance(result, Skipped) and result.reason.startswith("blocked_destination")


async def test_non_html_is_skipped():
    f = fetcher_for({"https://x.example.com/r.pdf": httpx.Response(200, content=b"%PDF",
                                                                   headers={"content-type": "application/pdf"})})
    result = await f.fetch("https://x.example.com/r.pdf")
    assert isinstance(result, Skipped) and "unsupported_content_type" in result.reason



async def test_parser_crash_skips_only_that_page(monkeypatch):
    import insightforge.tools.fetch as fetch_mod

    def boom(*a, **k):
        raise AttributeError("'NoneType' object has no attribute 'remove'")

    monkeypatch.setattr(fetch_mod, "html_to_text", boom)
    f = fetcher_for({"https://odd.example.com/a": html(article("Odd", "Body."))})
    result = await f.fetch("https://odd.example.com/a")
    assert isinstance(result, Skipped) and result.reason == "parse_error: AttributeError"

# --- providers + circuit breaker ----------------------------------------------------


def test_horizon_parsing():
    assert horizon_to_days("last 18 months") == 540
    assert horizon_to_days("past 2 years") == 730
    assert horizon_to_days("recent") is None


async def test_tavily_request_and_parsing():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        seen["body"] = request.content
        return httpx.Response(200, json={"results": [
            {"url": "https://a.com/x", "title": "A", "content": "snippet", "published_date": "2026-05-01"}]})

    provider = TavilySearch(httpx.AsyncClient(transport=httpx.MockTransport(handler)), "tvly-key")
    hits = await provider.search("payroll SEA", 10, days=90)
    assert seen["auth"] == "Bearer tvly-key" and b'"time_range":"year"' in seen["body"]
    assert hits == [SearchHit("https://a.com/x", "A", "snippet", date(2026, 5, 1))]


async def test_circuit_breaker_opens_after_repeated_failures():
    calls = {"n": 0}

    class Flaky:
        name = "flaky"

        async def search(self, query, max_results, days=None):
            calls["n"] += 1
            raise httpx.ConnectError("down")

    breaker = CircuitBreaker(Flaky(), threshold=2, cooldown_s=60)
    for _ in range(2):
        with pytest.raises(SearchUnavailable):
            await breaker.search("q", 5)
    with pytest.raises(SearchUnavailable, match="circuit open"):
        await breaker.search("q", 5)
    assert calls["n"] == 2  # third call never reached the provider


# --- end to end: WebResearcher -----------------------------------------------------


class FakeProvider:
    name = "fake"

    def __init__(self, hits):
        self.hits = hits

    async def search(self, query, max_results, days=None):
        return self.hits[:max_results]


async def test_researcher_stores_top_n_replaces_skipped_and_logs(committed_run):
    maker, _, run_id = committed_run
    urls = [f"https://site{i}.example.com/a" for i in range(6)]
    routes = {u: html(article(f"Story {i}", f"Fact number {i} about payroll.")) for i, u in enumerate(urls)}
    routes[urls[1]] = html("blocked", status=403)  # skipped → replaced by a later candidate
    researcher = WebResearcher(
        FakeProvider([SearchHit(u, f"Story {i}", "") for i, u in enumerate(urls)]),
        fetcher_for(routes),
        ScratchStore(maker, run_id),
        RunLog(maker, run_id),
        results_per_subtask=3,
        today=TODAY,
    )
    out = await researcher.research("st-1", "payroll vendors", "last 12 months")

    assert [s.url for s in out.sources] == [urls[0], urls[2], urls[3]]
    assert [s.url for s in out.skipped] == [urls[1]]
    row = await ScratchStore(maker, run_id).get(out.sources[0].source_id)
    assert "Fact number 0" in row.body_text and row.sub_task_key == "st-1" and 0 < row.credibility <= 1

    async with maker() as session:
        kinds = [e.kind for e in await session.scalars(select(RunEvent).where(RunEvent.run_id == run_id))]
    assert kinds.count("skipped_url") == 1 and "progress" in kinds

    text = format_for_agent(out)
    assert text.count("<untrusted_source") == 3
    assert len(text) < 3 * 700  # leads only, never full bodies
