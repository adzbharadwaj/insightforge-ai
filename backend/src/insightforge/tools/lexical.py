"""Cheap keyword-overlap scoring, used to shortlist text before embedding it.

Local embedding costs ~60–80 ms per 1,000-character passage on CPU, so embedding every
paragraph of every source dominated a run's wall time. A keyword shortlist first, then
embeddings on the shortlist only, keeps the semantic ranking where it matters.
"""

import re

_WORD = re.compile(r"[a-z0-9$€£¥%.]+")
_STOP = frozenset(["a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "into", "is", "it", "its", "of", "on", "or", "that", "the", "their", "this", "to", "was", "were", "will", "with", "what", "which", "who", "how", "than", "then", "there", "these", "they", "our", "your", "about", "over", "more", "most", "per"])


def terms(text: str) -> set[str]:
    return {w.strip(".") for w in _WORD.findall(text.lower()) if len(w.strip(".")) >= 3 and w not in _STOP}


def overlap(query_terms: set[str], text: str) -> float:
    """Share of the query's terms that appear in the text (0–1)."""
    if not query_terms:
        return 0.0
    return len(query_terms & terms(text)) / len(query_terms)


def shortlist(query: str, texts: list[str], k: int) -> list[int]:
    """Indices of the k texts with the highest overlap (ties keep document order)."""
    q = terms(query)
    ranked = sorted(range(len(texts)), key=lambda i: -overlap(q, texts[i]))
    return sorted(ranked[:k])
