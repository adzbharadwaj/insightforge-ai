"""Source credibility rubric for the Web Search agent (07).

score = 0.5 · domain + 0.3 · recency + 0.2 · byline, in [0, 1]

- domain:  .gov / .edu (and national equivalents) > major press > blogs > unknown
- recency: exponential decay, half-life one year; unknown date scores low
- byline:  1 if an author is named, else 0
"""

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
