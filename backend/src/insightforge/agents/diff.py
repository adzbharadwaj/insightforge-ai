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

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from insightforge.llm.embeddings import embed_documents
from insightforge.llm.gateway import ModelGateway, Role
from insightforge.tools.entities import numbers_in

SAME = 0.92  # at or above: same fact (if numbers agree)
RELATED = 0.75  # at or above: about the same thing; below: unrelated

DIFF_PROMPT = """You decide which differences between two research reports on the same topic are material.

Material: a new company, product, deal, funding round, price, regulation or figure; a number or status that
changed; a fact that matters to a competitive-intelligence analyst. Not material: rephrasing, the same fact
with different wording or rounding, generic statements, trivia.

For each candidate, write a short reason first, then decide.
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


class Candidate(BaseModel):
    id: str
    kind: str  # added | removed | changed
    text: str
    before: str | None = None


class Materiality(BaseModel):
    id: str
    reason: str = Field(default="", max_length=300)
    material: bool


class MaterialityVerdicts(BaseModel):
    verdicts: list[Materiality]


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
        if best >= RELATED:
            restated.add(j)
        if best >= RELATED and numbers_in(text) and numbers_in(old[j]) and numbers_in(text) != numbers_in(old[j]):
            changed.append(ChangedFinding(before=old[j], after=text, reason="figures differ"))
        elif best >= SAME:
            continue  # same fact, rephrased
        elif best >= RELATED:
            candidates.append(Candidate(id=f"x{i}", kind="changed", text=text, before=old[j]))
        else:
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
    material = {v.id: v for v in verdicts.verdicts if v.material}
    for c in candidates:
        if c.id not in material:
            continue
        if c.kind == "added":
            summary.added_findings.append(Finding(text=c.text))
        elif c.kind == "removed":
            summary.removed_findings.append(Finding(text=c.text))
        else:
            summary.changed_findings.append(ChangedFinding(before=c.before or "", after=c.text,
                                                           reason=material[c.id].reason))
    return summary
