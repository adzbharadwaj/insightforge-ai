"""Diff Agent (component 20): what materially changed since the last report on a watched topic.

Compares the new report's claims with the previous report's claims:

1. embed both sets (local) and match each new claim to its most similar old claim;
2. similar claims whose numbers differ → changed (decided in code: embeddings rate
   "raised $7.6M" and "raised $9M" as near-identical, so they can't be trusted with numbers);
3. near-identical claims with the same numbers → the same fact rephrased, ignored;
4. what's left (new claims with no close match, old claims nobody restated, loosely similar
   pairs) goes to one batched model call that keeps only material changes, so rephrasing
   and trivia don't page anyone.

Removed findings mean "not found in this run", not "no longer true": research sampling
varies run to run, which the summary says.
"""

import re
from typing import Literal

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from insightforge.llm.embeddings import embed_documents
from insightforge.llm.gateway import ModelGateway, Role
from insightforge.tools.entities import numbers_in

SAME = 0.92  # at or above: same fact (if numbers agree)
RELATED = 0.75  # at or above: about the same thing; below: unrelated

# Names that say where, not who: sharing "Indonesia" doesn't make two funding rounds the same deal.
_NOT_A_SUBJECT = frozenset(["The", "A", "An", "In", "On", "For", "With", "And", "Of", "To", "By", "At", "As", "From", "Its", "Their", "This", "That", "These", "Those", "It", "Southeast", "South", "East", "North", "West", "Asia", "Asian", "ASEAN", "SEA", "APAC", "Singapore", "Malaysia", "Indonesia", "Indonesian", "Philippines", "Filipino", "Thailand", "Thai", "Vietnam", "Vietnamese", "Cambodia", "Laos", "Myanmar", "Brunei", "US", "USD", "SGD", "MYR", "IDR", "PHP", "THB", "VND", "Series", "Seed"])
_NAME = re.compile(r"\b[A-Z][A-Za-z0-9&.'-]{2,}")


def subjects(text: str) -> set[str]:
    """Distinctive capitalised names in a claim (companies, products, people), minus places and filler."""
    return {m.rstrip(".'s").rstrip("'") for m in _NAME.findall(text)} - _NOT_A_SUBJECT


def same_subject(a: str, b: str) -> bool:
    return bool(subjects(a) & subjects(b))

DIFF_PROMPT = """You decide which differences between two research reports on the same topic are material.

Material: a new company, product, deal, funding round, price, regulation or figure; a number or status that
changed; a fact that matters to a competitive-intelligence analyst. Not material: rephrasing, the same fact
with different wording or rounding, generic statements, trivia.
Classify each candidate (write a short reason first):
- "changed": the new statement updates the previous one: the same fact about the same company, product or
  regulation, with a different value or status (e.g. funding amount, price, headcount).
- "new": material information not in the previous report, including a different fact about the same company
  (a [changed] candidate that is really a new fact is "new").
- "removed": a material previous finding (only for [removed] candidates).
- "not_material": rephrasing, rounding, generic statements or trivia.
"""


class Finding(BaseModel):
    text: str


class ChangedFinding(BaseModel):
    before: str
    after: str
    reason: str = ""


class ChangeSummary(BaseModel):
    added_findings: list[Finding] = Field(default_factory=list)
    removed_findings: list[Finding] = Field(default_factory=list)
    changed_findings: list[ChangedFinding] = Field(default_factory=list)
    note: str = "Removed findings were not found in this run; that does not prove they stopped being true."

    @property
    def empty(self) -> bool:
        return not (self.added_findings or self.removed_findings or self.changed_findings)

    @property
    def alertable(self) -> bool:
        """Removals alone don't page anyone: research samples sources differently run to run."""
        return bool(self.added_findings or self.changed_findings)


class Candidate(BaseModel):
    id: str
    kind: str  # added | removed | changed
    text: str
    before: str | None = None


class Materiality(BaseModel):
    id: str
    reason: str = Field(default="", max_length=300)
    classification: Literal["changed", "new", "removed", "not_material"]


class MaterialityVerdicts(BaseModel):
    verdicts: list[Materiality]


def numbers_differ(a: str, b: str) -> bool:
    na, nb = numbers_in(a), numbers_in(b)
    return bool(na) and bool(nb) and na != nb


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return a @ b.T


async def diff_claims(gateway: ModelGateway, old: list[str], new: list[str]) -> ChangeSummary:
    old, new = list(dict.fromkeys(old)), list(dict.fromkeys(new))
    if not old:  # first report for this topic: it is the baseline
        return ChangeSummary()
    if not new:
        return ChangeSummary()
    sims = _cos(np.array(await embed_documents(new)), np.array(await embed_documents(old)))  # new × old

    changed: list[ChangedFinding] = []
    candidates: list[Candidate] = []
    restated: set[int] = set()
    for i, text in enumerate(new):
        j = int(sims[i].argmax())
        best = float(sims[i, j])
        about_same = best >= RELATED and same_subject(text, old[j])
        if best >= SAME and not numbers_differ(text, old[j]):
            restated.add(j)
            continue  # same fact, rephrased
        if about_same:
            restated.add(j)
            if best >= SAME and numbers_differ(text, old[j]):
                # Near-identical sentence, different figures ("raised $7.6M" → "raised $9M"): decided in code.
                changed.append(ChangedFinding(before=old[j], after=text, reason="figures differ"))
            else:  # same subject, looser match: the judge decides whether it's an update or a new fact
                candidates.append(Candidate(id=f"x{i}", kind="changed", text=text, before=old[j]))
        else:
            # Similar wording about a different subject (e.g. another startup's seed round) is new, not a change.
            candidates.append(Candidate(id=f"x{i}", kind="added", text=text))
    for j, text in enumerate(old):
        if j not in restated and float(sims[:, j].max()) < RELATED:
            candidates.append(Candidate(id=f"r{j}", kind="removed", text=text))

    summary = ChangeSummary(changed_findings=changed)
    if not candidates:
        return summary
    listing = "\n".join(
        f"{c.id} [{c.kind}] {c.text}" + (f"\n    previously: {c.before}" if c.before else "") for c in candidates)
    verdicts: MaterialityVerdicts = await gateway.structured(Role.DIFF, MaterialityVerdicts).ainvoke(
        [SystemMessage(DIFF_PROMPT), HumanMessage(listing)])
    verdict = {v.id: v for v in verdicts.verdicts}
    for c in candidates:
        v = verdict.get(c.id)
        if v is None or v.classification == "not_material":
            continue
        if c.kind == "removed":
            if v.classification == "removed":
                summary.removed_findings.append(Finding(text=c.text))
        elif v.classification == "changed" and c.kind == "changed":
            summary.changed_findings.append(ChangedFinding(before=c.before or "", after=c.text, reason=v.reason))
        elif v.classification in ("new", "changed"):  # a "change" with no matching previous fact is new
            summary.added_findings.append(Finding(text=c.text))
    return summary
