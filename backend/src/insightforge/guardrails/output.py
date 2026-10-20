"""Output guardrails (01b), last step before a report is stored and shown.

Grounding and fact-checking run earlier in the graph; this module covers what is left:
personal data that slipped in from sources. Emails and international-format phone
numbers are redacted. Phones must start with "+" so that dates, amounts and IDs are
never mistaken for phone numbers (local-format numbers are a known gap).
"""

import re

EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE = re.compile(r"(?<![\w+])\+\d{1,3}[\s.-]?(?:\(\d{1,4}\)[\s.-]?)?\d{1,4}(?:[\s.-]\d{2,4}){1,3}\b")


def redact_pii(text: str) -> tuple[str, dict[str, int]]:
    counts = {}
    text, counts["email"] = EMAIL.subn("[redacted email]", text)
    text, counts["phone"] = PHONE.subn("[redacted phone]", text)
    return text, {k: v for k, v in counts.items() if v}
