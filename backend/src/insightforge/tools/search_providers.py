"""Web search providers behind one interface, chosen by SEARCH_PROVIDER.

Each provider returns `SearchHit`s (URL, title, snippet, date when known). The
fetcher then pulls full article bodies; snippets alone are never used as evidence.
A circuit breaker stops hammering a provider that keeps failing.
"""

import re
import time
from dataclasses import dataclass
from datetime import date
from typing import Protocol

import httpx

from insightforge.config import Settings
from insightforge.tools.fetch import parse_date


@dataclass(frozen=True)
class SearchHit:
    url: str
    title: str
    snippet: str
    published_date: date | None = None


class SearchUnavailable(RuntimeError):
    pass


class SearchProvider(Protocol):
    name: str

    async def search(self, query: str, max_results: int, days: int | None = None) -> list[SearchHit]: ...


def horizon_to_days(time_horizon: str | None) -> int | None:
    """'last 18 months' → 540. Returns None when the horizon can't be parsed (no date filter)."""
    if not time_horizon:
        return None
    m = re.search(r"(\d+)\s*(day|week|month|year)", time_horizon.lower())
    if not m:
        return None
    n, unit = int(m.group(1)), m.group(2)
    return n * {"day": 1, "week": 7, "month": 30, "year": 365}[unit]


class TavilySearch:
    name = "tavily"
    URL = "https://api.tavily.com/search"

    def __init__(self, client: httpx.AsyncClient, api_key: str):
        self.client, self.api_key = client, api_key

    async def search(self, query: str, max_results: int, days: int | None = None) -> list[SearchHit]:
        body: dict = {"query": query, "max_results": max_results, "search_depth": "advanced"}
        if days is not None and days <= 365:
            body["time_range"] = "day" if days <= 1 else "week" if days <= 7 else "month" if days <= 31 else "year"
        resp = await self.client.post(self.URL, json=body, headers={"Authorization": f"Bearer {self.api_key}"})
        resp.raise_for_status()
        return [
            SearchHit(r["url"], r.get("title", ""), r.get("content", ""), parse_date(r.get("published_date")))
            for r in resp.json().get("results", [])
        ]


class BraveSearch:
    name = "brave"
    URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, client: httpx.AsyncClient, api_key: str):
        self.client, self.api_key = client, api_key

    async def search(self, query: str, max_results: int, days: int | None = None) -> list[SearchHit]:
        params: dict = {"q": query, "count": min(max_results, 20)}
        if days is not None:
            params["freshness"] = "pd" if days <= 1 else "pw" if days <= 7 else "pm" if days <= 31 else "py"
        resp = await self.client.get(
            self.URL, params=params, headers={"X-Subscription-Token": self.api_key, "Accept": "application/json"}
        )
        resp.raise_for_status()
        results = resp.json().get("web", {}).get("results", [])
        return [
            SearchHit(r["url"], r.get("title", ""), r.get("description", ""), parse_date(r.get("page_age")))
            for r in results
        ]


class SerpApiSearch:
    name = "serpapi"
    URL = "https://serpapi.com/search.json"

    def __init__(self, client: httpx.AsyncClient, api_key: str):
        self.client, self.api_key = client, api_key

    async def search(self, query: str, max_results: int, days: int | None = None) -> list[SearchHit]:
        params: dict = {"engine": "google", "q": query, "num": max_results, "api_key": self.api_key}
        if days is not None:
            params["tbs"] = "qdr:d" if days <= 1 else "qdr:w" if days <= 7 else "qdr:m" if days <= 31 else "qdr:y"
        resp = await self.client.get(self.URL, params=params)
        resp.raise_for_status()
        return [
            SearchHit(r["link"], r.get("title", ""), r.get("snippet", ""), parse_date(r.get("date")))
            for r in resp.json().get("organic_results", [])
        ]


class CircuitBreaker:
    """Opens after `threshold` consecutive failures; stays open for `cooldown_s`."""

    def __init__(self, provider: SearchProvider, threshold: int = 3, cooldown_s: float = 60.0):
        self.provider = provider
        self.name = provider.name
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if time.monotonic() - self._opened_at >= self.cooldown_s:
            self._opened_at = None  # half-open: let the next call try
            self._failures = self.threshold - 1
            return False
        return True

    async def search(self, query: str, max_results: int, days: int | None = None) -> list[SearchHit]:
        if self.is_open:
            raise SearchUnavailable(f"{self.name} circuit open after {self.threshold} failures")
        try:
            hits = await self.provider.search(query, max_results, days)
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            self._failures += 1
            if self._failures >= self.threshold:
                self._opened_at = time.monotonic()
            raise SearchUnavailable(f"{self.name} search failed: {type(exc).__name__}") from exc
        self._failures = 0
        return hits


def make_search_provider(settings: Settings, client: httpx.AsyncClient) -> CircuitBreaker:
    providers = {
        "tavily": (TavilySearch, settings.tavily_api_key),
        "brave": (BraveSearch, settings.brave_api_key),
        "serpapi": (SerpApiSearch, settings.serpapi_api_key),
    }
    cls, key = providers[settings.search_provider]
    if key is None:
        raise RuntimeError(f"{settings.search_provider.upper()}_API_KEY is not set (see .env.example)")
    return CircuitBreaker(cls(client, key.get_secret_value()))
