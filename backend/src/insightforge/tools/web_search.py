"""Web research pipeline and the `web_search` tool used by the Web Search agent (07).

search → fetch full bodies (top N that succeed) → score credibility → store in the
scratch store (17) → hand back IDs, metadata and a short lead excerpt.

The agent sees at most `results_per_subtask` short excerpts per call; full bodies
stay in the scratch store for Synthesis.
"""

import asyncio
from dataclasses import dataclass
from datetime import date

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from insightforge.guardrails.untrusted import wrap
from insightforge.store.models import SourceKind
from insightforge.store.run_log import RunLog
from insightforge.store.scratch import ScratchStore
from insightforge.tools.credibility import credibility_score, matches_domain, registered_host
from insightforge.tools.fetch import FetchedPage, PageFetcher, Skipped
from insightforge.tools.search_providers import SearchProvider, SearchUnavailable, horizon_to_days

LEAD_CHARS = 400
CANDIDATE_MULTIPLIER = 2  # ask the provider for extra results to replace skipped ones
AGENT = "web_search"
DEPRIORITIZED_FACTOR = 0.5


@dataclass
class StoredSource:
    source_id: str
    url: str
    domain: str
    title: str | None
    published_date: date | None
    credibility: float
    lead: str


@dataclass
class WebResearchOutput:
    sources: list[StoredSource]
    skipped: list[Skipped]
    error: str | None = None


class WebResearcher:
    def __init__(
        self,
        provider: SearchProvider,
        fetcher: PageFetcher,
        scratch: ScratchStore,
        log: RunLog,
        *,
        results_per_subtask: int = 5,
        today: date | None = None,
        blocked_domains: list[str] | None = None,
        deprioritized_domains: list[str] | None = None,
    ):
        self.provider = provider
        self.fetcher = fetcher
        self.scratch = scratch
        self.log = log
        self.n = results_per_subtask
        self.today = today
        # Source credibility tuning (set per run by the user): blocked domains are never fetched; deprioritized
        # ones are fetched but their credibility is halved, which pushes them down in ranking and confidence.
        self.blocked = set(blocked_domains or [])
        self.deprioritized = set(deprioritized_domains or [])

    async def research(self, sub_task_id: str, query: str, time_horizon: str | None = None) -> WebResearchOutput:
        try:
            hits = await self.provider.search(query, self.n * CANDIDATE_MULTIPLIER, horizon_to_days(time_horizon))
        except SearchUnavailable as exc:
            await self.log.emit(AGENT, "error", str(exc), sub_task_id=sub_task_id)
            return WebResearchOutput([], [], error=str(exc))

        seen: set[str] = set()
        candidates = [h for h in hits if not (h.url in seen or seen.add(h.url))]
        if self.blocked:
            blocked = [h for h in candidates if matches_domain(h.url, self.blocked)]
            candidates = [h for h in candidates if h not in blocked]
            for h in blocked:
                await self.log.emit(AGENT, "skipped_url", f"blocked by your source settings: {h.url}", url=h.url,
                                    reason="blocked_domain", sub_task_id=sub_task_id)
        pages: list[FetchedPage] = []
        skipped: list[Skipped] = []
        # Fetch in batches of the remaining need, in rank order, until we have N bodies or run out.
        while candidates and len(pages) < self.n:
            batch, candidates = candidates[: self.n - len(pages)], candidates[self.n - len(pages) :]
            for hit in batch:
                await self.log.progress(AGENT, f"fetching {registered_host(hit.url)}…", sub_task_id=sub_task_id)
            results = await asyncio.gather(*(self.fetcher.fetch(h.url) for h in batch))
            for hit, result in zip(batch, results, strict=True):
                if isinstance(result, Skipped):
                    skipped.append(result)
                    continue
                if result.published_date is None:
                    result.published_date = hit.published_date
                if result.title is None:
                    result.title = hit.title
                pages.append(result)

        for s in skipped:
            await self.log.emit(AGENT, "skipped_url", f"skipped {s.url}", url=s.url, reason=s.reason,
                                sub_task_id=sub_task_id)
        stored = [await self._store(sub_task_id, page) for page in pages]
        return WebResearchOutput(stored, skipped)

    async def _store(self, sub_task_id: str, page: FetchedPage) -> StoredSource:
        score = credibility_score(page.final_url, page.published_date, bool(page.author), today=self.today)
        if self.deprioritized and matches_domain(page.final_url, self.deprioritized):
            score = round(score * DEPRIORITIZED_FACTOR, 3)
        source_id = await self.scratch.put(
            kind=SourceKind.WEB_PAGE,
            key=page.final_url,
            url=page.final_url,
            title=page.title,
            published_date=page.published_date,
            body_text=page.body_text,
            summary=page.title,
            credibility=score,
            sub_task_key=sub_task_id,
            meta={"author": page.author, "injection_flags": page.injection_flags},
        )
        if page.injection_flags:
            await self.log.emit(AGENT, "injection_flag", f"instruction-like text in {page.final_url}",
                                source_id=source_id, flags=page.injection_flags)
        return StoredSource(
            source_id=source_id,
            url=page.final_url,
            domain=registered_host(page.final_url),
            title=page.title,
            published_date=page.published_date,
            credibility=score,
            lead=page.body_text[:LEAD_CHARS],
        )


class WebSearchArgs(BaseModel):
    sub_task_id: str = Field(description='The sub-task id you were given, e.g. "st-2"')
    query: str = Field(min_length=3, max_length=300, description="Search query")
    time_horizon: str | None = Field(default=None, description='e.g. "last 18 months"')


def format_for_agent(out: WebResearchOutput) -> str:
    if out.error:
        return f"Search failed: {out.error}. Report this sub-task as Failed."
    lines = [f"{len(out.sources)} source(s) stored; {len(out.skipped)} skipped."]
    for s in out.sources:
        lines.append(
            f"- source_id={s.source_id} domain={s.domain} published={s.published_date or 'unknown'} "
            f"credibility={s.credibility}\n{wrap(s.source_id, s.lead, url=s.url, title=s.title)}"
        )
    return "\n".join(lines)


def make_web_search_tool(researcher: WebResearcher) -> BaseTool:
    async def web_search(sub_task_id: str, query: str, time_horizon: str | None = None) -> str:
        return format_for_agent(await researcher.research(sub_task_id, query, time_horizon))

    return StructuredTool.from_function(
        coroutine=web_search,
        name="web_search",
        description=(
            "Search the web, fetch the full text of the top results, and store them. "
            "Returns source ids with a short lead excerpt of each; full text stays in storage."
        ),
        args_schema=WebSearchArgs,
    )
