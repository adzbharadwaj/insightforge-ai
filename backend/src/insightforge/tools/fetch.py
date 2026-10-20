"""Full-article fetcher for the Web Search agent (07) and Fact-Checker (12).

- Only public destinations; every redirect hop is re-checked (guardrails/net.py).
- Paywalls and login walls are detected and skipped, never scraped around.
- At most `per_domain` concurrent requests per host (politeness + rate limits).
- Bodies go through the untrusted-content wrapper's cleaning (03c) before anyone sees them.
"""

import asyncio
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from urllib.parse import urljoin, urlsplit

import httpx
import trafilatura

from insightforge.guardrails.net import BlockedDestination, Resolver, _system_resolve, acheck_public_url
from insightforge.guardrails.untrusted import html_to_text, scan_injection

logger = logging.getLogger(__name__)

USER_AGENT = "InsightForgeResearchBot/0.1 (+https://github.com/insightforge-agent)"
MAX_REDIRECTS = 5
MIN_BODY_CHARS = 300

_PAYWALL_JSONLD = re.compile(r'"isAccessibleForFree"\s*:\s*"?(false|False)"?')
_PAYWALL_PHRASES = re.compile(
    r"subscribe to (continue|read)|subscribers only|sign in to (continue|read)|log in to (continue|read)|"
    r"create a free account to (continue|read)|already a subscriber|this (article|content) is for (paid )?subscribers|"
    r"to continue reading",
    re.IGNORECASE,
)


@dataclass
class FetchedPage:
    url: str
    final_url: str
    title: str | None
    published_date: date | None
    author: str | None
    body_text: str
    injection_flags: dict[str, int] = field(default_factory=dict)


@dataclass
class Skipped:
    url: str
    reason: str


class PageFetcher:
    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        resolve: Resolver = _system_resolve,
        max_bytes: int = 3_000_000,
        per_domain: int = 2,
        timeout_s: float = 15.0,
    ):
        self.client = client
        self.resolve = resolve
        self.max_bytes = max_bytes
        self.timeout_s = timeout_s
        self._per_domain = per_domain
        self._domain_locks: dict[str, asyncio.Semaphore] = defaultdict(lambda: asyncio.Semaphore(self._per_domain))

    async def fetch(self, url: str) -> FetchedPage | Skipped:
        host = urlsplit(url).hostname or ""
        async with self._domain_locks[host]:
            try:
                return await asyncio.wait_for(self._fetch(url), self.timeout_s)
            except BlockedDestination as exc:
                return Skipped(url, f"blocked_destination: {exc}")
            except TimeoutError:
                return Skipped(url, "timeout")
            except httpx.HTTPError as exc:
                return Skipped(url, f"network_error: {type(exc).__name__}")
            except Exception as exc:  # malformed markup can break lxml/trafilatura: skip this page, not the sub-task
                logger.warning("parse failed for %s", url, exc_info=True)
                return Skipped(url, f"parse_error: {type(exc).__name__}")

    async def _fetch(self, url: str) -> FetchedPage | Skipped:
        current = url
        for _ in range(MAX_REDIRECTS + 1):
            await acheck_public_url(current, resolve=self.resolve)
            async with self.client.stream(
                "GET", current, headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*;q=0.5"},
                follow_redirects=False,
            ) as resp:
                if resp.is_redirect and resp.headers.get("location"):
                    current = urljoin(current, resp.headers["location"])
                    continue
                if resp.status_code in (401, 402, 403):
                    return Skipped(url, f"paywall_or_blocked: HTTP {resp.status_code}")
                if resp.status_code >= 400:
                    return Skipped(url, f"http_{resp.status_code}")
                ctype = resp.headers.get("content-type", "")
                if "html" not in ctype and "xml" not in ctype:
                    return Skipped(url, f"unsupported_content_type: {ctype.split(';')[0] or 'unknown'}")
                raw = await self._read_capped(resp)
                return self._parse(url, current, raw)
        return Skipped(url, "too_many_redirects")

    async def _read_capped(self, resp: httpx.Response) -> str:
        chunks, size = [], 0
        async for chunk in resp.aiter_bytes():
            chunks.append(chunk)
            size += len(chunk)
            if size >= self.max_bytes:
                break
        return b"".join(chunks).decode(resp.encoding or "utf-8", errors="replace")

    def _parse(self, url: str, final_url: str, raw_html: str) -> FetchedPage | Skipped:
        body = html_to_text(raw_html, url=final_url)
        if _PAYWALL_JSONLD.search(raw_html) or (_PAYWALL_PHRASES.search(raw_html) and len(body) < 1_500):
            return Skipped(url, "paywall_or_login_wall")
        if len(body) < MIN_BODY_CHARS:
            return Skipped(url, "too_little_text")
        meta = trafilatura.extract_metadata(raw_html, default_url=final_url)
        return FetchedPage(
            url=url,
            final_url=final_url,
            title=getattr(meta, "title", None),
            published_date=parse_date(getattr(meta, "date", None)),
            author=getattr(meta, "author", None) or None,
            body_text=body,
            injection_flags=scan_injection(body).flags,
        )


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None
