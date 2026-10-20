"""Entity normalization, merging and conflict detection (component 09). Pure functions.

The extraction model works on one source at a time; this module turns those
per-source extractions into one EntityGraph:

- entity keys are normalized (`company:salaryhero` for "Salary Hero Pte. Ltd."), so the
  same company from two sources is one node;
- facts become relationships keyed by entity, predicate and qualifier;
- for single-valued predicates (funding amount for a round, HQ, price of a plan, ...)
  values from different sources are compared after normalization (money is parsed,
  ±5% tolerance) and disagreements are flagged `conflict: true` and listed.

Deterministic on purpose: conflict flags drive Synthesis confidence, so they must not
depend on a model's mood.
"""

import datetime as dt
import re
from collections import defaultdict
from dataclasses import dataclass

from insightforge.graph.state import (
    SINGLE_VALUED,
    Conflict,
    ConflictValue,
    Entity,
    EntityGraph,
    EntityType,
    Predicate,
    Relationship,
    SourceExtraction,
)
from insightforge.store.sources import ResolvedSource

MONEY_TOLERANCE = 0.05
MONEY_PREDICATES = frozenset({Predicate.RAISED, Predicate.VALUED_AT, Predicate.REVENUE, Predicate.PRICED_AT})
# Without a qualifier these can't be compared: two "raised" facts may be different rounds.
NEEDS_QUALIFIER = frozenset({Predicate.RAISED, Predicate.PRICED_AT, Predicate.REVENUE})

_COMPANY_WORDS = re.compile(
    r"\b(inc|incorporated|ltd|limited|llc|plc|pte|pvt|private|corp|corporation|co|gmbh|sdn|bhd|tbk|pt|jsc|"
    r"holdings?|group)\b\.?",
    re.IGNORECASE,
)

_CURRENCIES = {
    "us$": "USD", "usd": "USD", "$": "USD", "s$": "SGD", "sgd": "SGD", "rm": "MYR", "myr": "MYR",
    "rp": "IDR", "idr": "IDR", "php": "PHP", "₱": "PHP", "thb": "THB", "฿": "THB", "vnd": "VND",
    "€": "EUR", "eur": "EUR", "£": "GBP", "gbp": "GBP", "₹": "INR", "inr": "INR", "a$": "AUD", "aud": "AUD",
}
_SCALES = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mn": 1e6, "mm": 1e6, "million": 1e6,
           "b": 1e9, "bn": 1e9, "billion": 1e9, "t": 1e12, "trillion": 1e12}
_CUR = "|".join(re.escape(c) for c in sorted(_CURRENCIES, key=len, reverse=True))
_SCALE = "|".join(sorted(_SCALES, key=len, reverse=True))
_MONEY = re.compile(
    rf"(?P<cur1>{_CUR})?\s*(?P<num>\d[\d,]*(?:\.\d+)?)\s*(?P<scale>{_SCALE})?\b\.?\s*(?P<cur2>{_CUR})?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Money:
    currency: str | None
    amount: float


def parse_money(text: str) -> Money | None:
    """'US$7.6 million' → Money('USD', 7_600_000). Returns None if no number is present."""
    m = _MONEY.search(text.strip())
    if not m:
        return None
    try:
        amount = float(m["num"].replace(",", ""))
    except ValueError:
        return None
    if m["scale"]:
        amount *= _SCALES[m["scale"].lower()]
    cur = m["cur1"] or m["cur2"]
    return Money(_CURRENCIES[cur.lower()] if cur else None, amount)


_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def numbers_in(text: str) -> set[float]:
    """Every number in the text, plus scaled money amounts ("$7.6M" → 7.6 and 7_600_000)."""
    out: set[float] = set()
    for m in _NUM.finditer(text):
        try:
            out.add(float(m.group().replace(",", "")))
        except ValueError:
            continue
    for m in _MONEY.finditer(text):
        if m["num"] and m["scale"]:
            try:
                out.add(float(m["num"].replace(",", "")) * _SCALES[m["scale"].lower()])
            except ValueError:
                continue
    return out


def entity_key(name: str, type_: EntityType) -> str:
    base = name.lower()
    if type_ is EntityType.COMPANY:
        base = _COMPANY_WORDS.sub(" ", base)
    slug = re.sub(r"[^a-z0-9]", "", base) or re.sub(r"[^a-z0-9]", "", name.lower()) or "unknown"
    return f"{type_.value}:{slug}"


def normalize_qualifier(q: str | None) -> str | None:
    if not q:
        return None
    q = re.sub(r"\b(round|funding|financing)\b", " ", q.lower())
    q = re.sub(r"\s+", " ", q).strip(" .-")
    return q or None


def parse_fact_date(value: str | None) -> dt.date | None:
    if not value:
        return None
    parts = value.strip()[:10].split("-")
    try:
        nums = [int(p) for p in parts if p]
    except ValueError:
        return None
    if not nums:
        return None
    try:
        return dt.date(nums[0], nums[1] if len(nums) > 1 else 1, nums[2] if len(nums) > 2 else 1)
    except ValueError:
        return None


def _norm_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def same_value(a: str, b: str, predicate: Predicate) -> bool:
    """Do two object values say the same thing? Values that can't be compared count as the same."""
    if predicate in MONEY_PREDICATES or predicate is Predicate.EMPLOYEE_COUNT:
        ma, mb = parse_money(a), parse_money(b)
        if ma and mb:
            if ma.currency and mb.currency and ma.currency != mb.currency:
                return True  # different currencies: not comparable without FX rates, so not a conflict
            hi = max(ma.amount, mb.amount)
            return hi == 0 or abs(ma.amount - mb.amount) / hi <= MONEY_TOLERANCE
    na, nb = _norm_text(a), _norm_text(b)
    return na == nb or (bool(na) and bool(nb) and (na in nb or nb in na))


def merge_extractions(extractions: list[tuple[ResolvedSource, SourceExtraction]]) -> EntityGraph:
    entities: dict[str, Entity] = {}
    relationships: list[Relationship] = []
    seen: set[tuple] = set()

    for source, ext in extractions:
        names: dict[str, str] = {}  # name slug → entity key, for resolving fact subjects in this source
        for e in ext.entities:
            key = entity_key(e.name, e.type)
            names[_norm_text(e.name)] = key
            ent = entities.setdefault(key, Entity(key=key, name=e.name.strip(), type=e.type))
            if source.source_id not in ent.source_ids:
                ent.source_ids.append(source.source_id)
        for f in ext.facts:
            subject = names.get(_norm_text(f.subject))
            if subject is None:  # the model used a subject it didn't list: treat it as a company
                subject = entity_key(f.subject, EntityType.COMPANY)
                ent = entities.setdefault(subject, Entity(key=subject, name=f.subject.strip(), type=EntityType.COMPANY))
                if source.source_id not in ent.source_ids:
                    ent.source_ids.append(source.source_id)
            qualifier = normalize_qualifier(f.qualifier)
            dedupe = (subject, f.predicate, qualifier, _norm_text(f.object), source.source_id)
            if dedupe in seen:
                continue
            seen.add(dedupe)
            relationships.append(Relationship(
                subject=subject, predicate=f.predicate, object=f.object.strip(), qualifier=qualifier,
                source_id=source.source_id, source_url=source.url or source.label, date=parse_fact_date(f.date),
            ))

    conflicts = _flag_conflicts(relationships)
    return EntityGraph(entities=list(entities.values()), relationships=relationships, conflicts=conflicts)


def _flag_conflicts(relationships: list[Relationship]) -> list[Conflict]:
    groups: dict[tuple, list[Relationship]] = defaultdict(list)
    for r in relationships:
        if r.predicate in SINGLE_VALUED and not (r.predicate in NEEDS_QUALIFIER and r.qualifier is None):
            groups[(r.subject, r.predicate, r.qualifier)].append(r)

    conflicts = []
    for (subject, predicate, qualifier), rels in groups.items():
        clusters: list[list[Relationship]] = []
        for r in rels:
            for cluster in clusters:
                if same_value(cluster[0].object, r.object, predicate):
                    cluster.append(r)
                    break
            else:
                clusters.append([r])
        sources = {r.source_id for r in rels}
        if len(clusters) < 2 or len(sources) < 2:
            continue
        for r in rels:
            r.conflict = True
        conflicts.append(Conflict(
            subject=subject, predicate=predicate, qualifier=qualifier,
            values=[ConflictValue(object=c[0].object, source_ids=sorted({r.source_id for r in c})) for c in clusters],
        ))
    return conflicts
