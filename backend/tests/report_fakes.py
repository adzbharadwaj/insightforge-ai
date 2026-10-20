"""Deterministic stand-ins for the Phase 4 model roles (synthesis, writer, grounding, summarizer)."""

import re

from insightforge.agents.synthesis import CompressedClaim, PassageClaim, SubTaskClaims
from insightforge.graph.state import (
    ClaimVerdict,
    EntailmentVerdicts,
    WriterOutput,
    WrittenSection,
    WrittenSentence,
)
from insightforge.llm.gateway import Role

_PASSAGE = re.compile(r'<untrusted_source id="(p-\d+)"[^>]*>\n?(.*?)\n?</untrusted_source>', re.DOTALL)


def synthesis_fn(messages):
    """One claim per passage: the passage's first sentence, verbatim."""
    passages = _PASSAGE.findall(messages[-1].content)
    claims = [PassageClaim(passage_id=pid, claim=text.strip().split(". ")[0].strip()[:300])
              for pid, text in passages if text.strip()]
    return SubTaskClaims(theme="Findings", claims=claims)


def parse_bundle(text: str) -> list[tuple[str, list[tuple[str, str]]]]:
    """[(sub_task_id, [(evidence_id, claim), ...]), ...] from the rendered bundle the Writer sees."""
    sections = []
    for block in re.split(r"\n(?=## )", text.split("Evidence bundle:\n", 1)[1].split("\n\nResearch notes")[0]):
        m = re.search(r"\(sub-task (st-\d+),", block)
        if not m:
            continue
        items = re.findall(r"^(ev-\d+) \|.*\n  (.+)$", block, flags=re.MULTILINE)
        sections.append((m.group(1), items))
    return sections


def writer_fn(messages):
    """Copies every evidence claim verbatim with its id: always grounded."""
    sections = parse_bundle(messages[-1].content)
    first = [(eid, claim) for _, items in sections for eid, claim in items[:1]]
    return WriterOutput(
        title="Payroll software in Southeast Asia",
        executive_summary=[WrittenSentence(text=claim, evidence_ids=[eid]) for eid, claim in first[:2]],
        sections=[WrittenSection(sub_task_id=sid, heading=f"Section {sid}",
                                 sentences=[WrittenSentence(text=c, evidence_ids=[e]) for e, c in items])
                  for sid, items in sections],
        gaps=["Pricing for enterprise tiers was not public."],
    )


def grounding_fn(messages):
    ids = re.findall(r"^claim_id: (c-\d+)$", messages[-1].content, flags=re.MULTILINE)
    return EntailmentVerdicts(verdicts=[ClaimVerdict(claim_id=i, supported=True) for i in ids])


def summarizer_fn(messages):
    return CompressedClaim(summary="Several smaller findings: " + messages[-1].content.replace("\n", " ")[:200])


REPORT_FNS = {Role.SYNTHESIS: synthesis_fn, Role.WRITER: writer_fn, Role.GROUNDING: grounding_fn,
              Role.SUMMARIZER: summarizer_fn}
