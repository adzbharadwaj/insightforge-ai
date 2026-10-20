"""Contradiction resolution (bonus): the analyst picks which version of a conflicting fact the report uses.

Opt-in per run. After Synthesis, every entity conflict with at least two sides present in the evidence bundle
becomes a review item: one option per value, with its evidence ids and publishers. The run pauses
(AWAITING_REVIEW) through the same LangGraph interrupt as clarification. On resume:

- chosen option   its evidence stays, marked analyst-confirmed (conflict cleared, MEDIUM confidence); the other
                  options' evidence is dropped and logged as a cut with the reason;
- "keep_both"     nothing changes: the Writer presents the disagreement, as it does without review.

Decisions are applied in code, so the Writer only ever sees the evidence the analyst kept.
"""

from pydantic import BaseModel

from insightforge.agents.synthesis import render_bundle, section_confidence
from insightforge.graph.state import Confidence, Cut, EntityGraph, EvidenceBundle
from insightforge.llm.tokens import count_tokens

KEEP_BOTH = "keep_both"


class ConflictOption(BaseModel):
    option_id: str
    value: str
    evidence_ids: list[str]
    claims: list[str]
    sources: list[str]  # publisher labels


class ConflictReview(BaseModel):
    conflict_id: str
    subject: str
    predicate: str
    qualifier: str | None = None
    options: list[ConflictOption]


def build_reviews(bundle: EvidenceBundle, graph: EntityGraph | None) -> list[ConflictReview]:
    if not graph or not graph.conflicts:
        return []
    names = {e.key: e.name for e in graph.entities}
    items = bundle.items()
    reviews = []
    for ci, conflict in enumerate(graph.conflicts, start=1):
        options = []
        for vi, value in enumerate(conflict.values, start=1):
            ids = [eid for eid, item in items.items()
                   if item.conflict and set(item.source_ids) & set(value.source_ids)]
            if not ids:
                continue
            labels = sorted({bundle.sources[s].label for e in ids for s in items[e].source_ids if s in bundle.sources})
            options.append(ConflictOption(option_id=f"k{ci}-v{vi}", value=value.object, evidence_ids=ids,
                                          claims=[items[e].claim for e in ids], sources=labels))
        if len(options) >= 2:  # only disagreements the Writer would otherwise present side by side
            reviews.append(ConflictReview(conflict_id=f"k{ci}", subject=names.get(conflict.subject, conflict.subject),
                                          predicate=conflict.predicate.value, qualifier=conflict.qualifier,
                                          options=options))
    return reviews


def apply_decisions(bundle: EvidenceBundle, reviews: list[ConflictReview],
                    choices: dict[str, str]) -> EvidenceBundle:
    bundle = bundle.model_copy(deep=True)
    drop: set[str] = set()
    confirm: set[str] = set()
    notes, cuts = list(bundle.notes), list(bundle.cuts)
    for review in reviews:
        choice = choices.get(review.conflict_id, KEEP_BOTH)
        chosen = next((o for o in review.options if o.option_id == choice), None)
        what = f"{review.subject} {review.predicate.replace('_', ' ')}" + (f" ({review.qualifier})" if review.qualifier else "")
        if chosen is None:
            notes.append(f"Conflicting reports on {what} were kept side by side at the analyst's request.")
            continue
        others = [o for o in review.options if o is not chosen]
        confirm |= set(chosen.evidence_ids)
        rejected = {e for o in others for e in o.evidence_ids} - set(chosen.evidence_ids)
        drop |= rejected
        notes.append(f"Analyst resolved conflicting reports on {what}: used {chosen.value} "
                     f"({', '.join(chosen.sources)}) over " + "; ".join(f"{o.value} ({', '.join(o.sources)})" for o in others) + ".")
        if rejected:
            cuts.append(Cut(evidence_ids=sorted(rejected), action="dropped", tokens_saved=0,
                            reason=f"analyst chose {chosen.value} for {what}"))
    for section in bundle.sections:
        section.evidence = [i for i in section.evidence if i.evidence_id not in drop]
        for item in section.evidence:
            if item.evidence_id in confirm:
                item.conflict = False
                item.confidence = Confidence.MEDIUM  # one side confirmed by the analyst, not by independent sources
        section.confidence = section_confidence(section.evidence)
    bundle.sections = [s for s in bundle.sections if s.evidence]
    bundle.notes, bundle.cuts = notes, cuts
    bundle.token_count = count_tokens(render_bundle(bundle.sections, bundle.sources))
    return bundle
