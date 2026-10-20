"""Outbound webhook guardrail (03d): deliver change summaries safely.

- HTTPS only, and the host must resolve to public addresses only (no localhost, private,
  link-local or metadata IPs). Checked when the URL is saved and again before every send,
  because DNS can change in between. Redirects are never followed (a redirect to an
  internal address would bypass the check).
- Every request is signed: `X-InsightForge-Signature: sha256=<hmac(body)>` with the
  WEBHOOK_SIGNING_SECRET, plus a timestamp header, so receivers can verify origin and
  reject replays.
- 10 s timeout, 3 attempts with exponential backoff on network errors, 429 and 5xx;
  4xx means the receiver rejected the payload and is not retried.
- Slack incoming-webhook URLs get Slack's message format; other URLs get the JSON summary.
"""

import asyncio
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

from insightforge.config import Settings, get_settings, require_secret
from insightforge.guardrails.net import BlockedDestination, Resolver, _system_resolve, acheck_public_url

TIMEOUT_S = 10.0
ATTEMPTS = 3
BACKOFF_S = 1.0
MAX_ITEMS_IN_SLACK = 5


class WebhookRejected(ValueError):
    """The URL can't be used as a webhook. Safe to show the user."""


async def validate_webhook_url(url: str, resolve: Resolver = _system_resolve) -> str:
    url = url.strip()
    if len(url) > 2_000:
        raise WebhookRejected("Webhook URL is too long.")
    try:
        await acheck_public_url(url, resolve=resolve, require_https=True)
    except BlockedDestination as exc:
        raise WebhookRejected(f"Webhook URL not allowed: {exc}") from exc
    return url


def is_slack(url: str) -> bool:
    return (urlsplit(url).hostname or "").lower() == "hooks.slack.com"


def slack_payload(topic: str, summary: dict, link: str | None) -> dict:
    def lines(label: str, items: list, fmt) -> list[str]:
        if not items:
            return []
        shown = [f"• {fmt(i)}" for i in items[:MAX_ITEMS_IN_SLACK]]
        more = len(items) - MAX_ITEMS_IN_SLACK
        return [f"*{label} ({len(items)})*", *shown, *([f"…and {more} more"] if more > 0 else [])]

    body = [
        *lines("New", summary.get("added_findings", []), lambda i: i["text"]),
        *lines("Changed", summary.get("changed_findings", []), lambda i: f"{i['after']} (was: {i['before']})"),
        *lines("No longer found", summary.get("removed_findings", []), lambda i: i["text"]),
    ]
    text = f"InsightForge: changes for *{topic}*\n" + "\n".join(body) + (f"\n<{link}|Open report>" if link else "")
    return {"text": text}


def sign(body: bytes, timestamp: str, secret: str) -> str:
    mac = hmac.new(secret.encode(), f"{timestamp}.".encode() + body, hashlib.sha256)
    return f"sha256={mac.hexdigest()}"


@dataclass
class Delivery:
    status: str  # "sent" | "failed" | "blocked"
    attempts: int
    detail: str = ""


async def deliver(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
    *,
    settings: Settings | None = None,
    resolve: Resolver = _system_resolve,
    backoff_s: float = BACKOFF_S,
) -> Delivery:
    s = settings or get_settings()
    try:
        await acheck_public_url(url, resolve=resolve, require_https=True)
    except BlockedDestination as exc:
        return Delivery("blocked", 0, str(exc))
    body = json.dumps(payload, separators=(",", ":"), default=str).encode()
    timestamp = str(int(time.time()))
    headers = {"Content-Type": "application/json", "User-Agent": "InsightForge-Webhook/1.0",
               "X-InsightForge-Timestamp": timestamp,
               "X-InsightForge-Signature": sign(body, timestamp, require_secret(s.webhook_signing_secret,
                                                                                "WEBHOOK_SIGNING_SECRET"))}
    detail = ""
    for attempt in range(1, ATTEMPTS + 1):
        try:
            resp = await client.post(url, content=body, headers=headers, timeout=TIMEOUT_S, follow_redirects=False)
            if resp.is_success:
                return Delivery("sent", attempt)
            detail = f"HTTP {resp.status_code}"
            if resp.status_code < 500 and resp.status_code != 429:
                return Delivery("failed", attempt, detail)  # the receiver rejected it; retrying won't help
        except httpx.HTTPError as exc:
            detail = type(exc).__name__
        if attempt < ATTEMPTS:
            await asyncio.sleep(backoff_s * 2 ** (attempt - 1))
    return Delivery("failed", ATTEMPTS, detail)
