"""Source credibility rubric for the Web Search agent (07).

score = 0.5 · domain + 0.3 · recency + 0.2 · byline, in [0, 1]

- domain:  .gov / .edu (and national equivalents) > major press > blogs > unknown
- recency: exponential decay, half-life one year; unknown date scores low
- byline:  1 if an author is named, else 0
"""

import re
from datetime import date
from urllib.parse import urlsplit

WEIGHTS = {"domain": 0.5, "recency": 0.3, "byline": 0.2}
RECENCY_HALF_LIFE_DAYS = 365
UNKNOWN_DATE_RECENCY = 0.3

DOMAIN_SCORES = {"institutional": 1.0, "major_press": 0.85, "blog": 0.5, "unknown": 0.35}

MAJOR_PRESS = frozenset({
    "reuters.com", "bloomberg.com", "ft.com", "wsj.com", "nytimes.com", "economist.com", "apnews.com",
    "bbc.com", "bbc.co.uk", "cnbc.com", "forbes.com", "fortune.com", "techcrunch.com", "theinformation.com",
    "axios.com", "businessinsider.com", "theguardian.com", "washingtonpost.com", "asia.nikkei.com",
    "scmp.com", "straitstimes.com", "businesstimes.com.sg", "theedgemalaysia.com", "dealstreetasia.com",
    "techinasia.com", "e27.co", "kr-asia.com", "restofworld.org", "gartner.com", "mckinsey.com",
})
BLOG_HOSTS = frozenset({"medium.com", "substack.com", "blogspot.com", "wordpress.com", "tumblr.com", "ghost.io"})
_INSTITUTIONAL_SUFFIXES = (".gov", ".edu", ".mil", ".int")
_INSTITUTIONAL_LABELS = {"gov", "edu", "ac", "go", "gob", "govt"}  # e.g. gov.sg, ac.uk, go.id, edu.my


def registered_host(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return host.removeprefix("www.")


_DOMAIN = re.compile(r"^(?=.{1,253}$)([a-z0-9-]{1,63}\.)+[a-z]{2,63}$")


def normalize_domains(raw: str | list[str] | None, limit: int = 50) -> list[str]:
    """User-entered domains ("https://www.Example.com/x", "example.com") → registered hosts; invalid ones dropped."""
    items = re.split(r"[\s,;]+", raw) if isinstance(raw, str) else list(raw or [])
    out = []
    for item in items:
        item = item.strip().lower()
        if not item:
            continue
        host = registered_host(item if "://" in item else f"https://{item}")
        if _DOMAIN.match(host) and host not in out:
            out.append(host)
    return out[:limit]


def matches_domain(url: str, domains: set[str]) -> bool:
    """True if the URL's host is one of the domains or a subdomain of one."""
    host = registered_host(url)
    return any(host == d or host.endswith(f".{d}") for d in domains)


def domain_category(url: str) -> str:
    host = registered_host(url)
    labels = host.split(".")
    if host.endswith(_INSTITUTIONAL_SUFFIXES) or (len(labels) >= 3 and labels[-2] in _INSTITUTIONAL_LABELS):
        return "institutional"
    if any(host == d or host.endswith("." + d) for d in MAJOR_PRESS):
        return "major_press"
    path = urlsplit(url).path.lower()
    if any(host == d or host.endswith("." + d) for d in BLOG_HOSTS) or labels[0] == "blog" or "/blog" in path:
        return "blog"
    return "unknown"


def recency_score(published: date | None, today: date | None = None) -> float:
    if published is None:
        return UNKNOWN_DATE_RECENCY
    age_days = max(0, ((today or date.today()) - published).days)
    return 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)


def credibility_score(url: str, published: date | None, has_byline: bool, today: date | None = None) -> float:
    score = (
        WEIGHTS["domain"] * DOMAIN_SCORES[domain_category(url)]
        + WEIGHTS["recency"] * recency_score(published, today)
        + WEIGHTS["byline"] * (1.0 if has_byline else 0.0)
    )
    return round(min(1.0, max(0.0, score)), 3)
