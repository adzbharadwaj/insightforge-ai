import json

from fakes import FakeGateway
from test_report import FUNDING, bundle_of, report

from insightforge import evaluation
from insightforge.agents.writer import render_report
from insightforge.graph.state import WrittenSentence
from insightforge.llm.gateway import Role
from insightforge.store.models import Claim

BRIEF = {"id": "payroll-sea", "must_cover": ["Salary Hero", "pricing", "Vietnam"], "expected_sections": ["Funding"],
         "notes": ""}


def test_sample_brief_library_is_complete():
    briefs = evaluation.load_briefs()
    assert len(briefs) == 5
    assert len({b["industry"] for b in briefs.values()}) == 5
    assert all(len(b["brief"]) <= 500 and b["expected_sections"] and b["must_cover"] for b in briefs.values())


def test_structure_checks_on_a_rendered_report():
    draft = render_report(report(WrittenSentence(text="Salary Hero raised US$7.6 million.", evidence_ids=["ev-1"])),
                          bundle_of(FUNDING))
    checks = evaluation.structure_checks(draft.markdown, {"report": draft.report.model_dump()}, BRIEF)
    assert checks["has_executive_summary"] and checks["has_gaps_section"] and checks["uses_citation_format"]
    assert checks["every_section_has_confidence"] and checks["summary_within_150_words"]
    assert checks["must_cover_hit"] == ["Salary Hero", "pricing"]  # "pricing" is in the Gaps section
    assert checks["must_cover_rate"] == 0.667


async def test_citation_accuracy_counts_every_claim():
    claims = [Claim(claim_key=f"c-{i}", section="S", text=t, evidence_ids=ids)
              for i, (t, ids) in enumerate([("A", ["ev-1"]), ("B", ["ev-1"]), ("C", ["ev-9"]), ("D", [])], start=1)]
    evidence = {"ev-1": {"claim": "x", "passage": "Salary Hero raised US$7.6M.", "source_ids": ["src-a"]}}

    def judge(messages):
        ids = [line.split(": ", 1)[1] for line in messages[-1].content.splitlines() if line.startswith("claim_id: ")]
        return evaluation.CitationVerdicts(verdicts=[
            evaluation.CitationVerdict(claim_id=i, supported=i == "c-1", reason="ok" if i == "c-1" else "no") for i in ids])

    seen = []
    real_judge = judge

    def judge(messages):
        seen.append(messages[-1].content)
        return real_judge(messages)

    out = await evaluation.citation_accuracy(FakeGateway({}, {Role.SYNTHESIS: judge}), claims, evidence,
                                             {"src-a": {"label": "e27.co"}})
    assert "(published by e27.co)" in seen[0]  # attributions can be checked
    assert (out["claims_checked"], out["supported"], out["accuracy"]) == (3, 1, 0.333)  # D has no citations
    assert out["unresolvable_citations"] == 1 and {u["claim_id"] for u in out["unsupported"]} == {"c-2", "c-3"}


def test_report_lists_results_and_missing_briefs(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(evaluation, "REPORT", tmp_path / "report.md")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "payroll-sea-1.json").write_text(json.dumps({
        "run_id": "r1", "brief_id": "payroll-sea", "brief": "b", "status": "COMPLETE", "tokens_used": 1000,
        "wall_seconds": 200, "attempts": 1, "writer_attempts": 1, "models": ["m"], "sub_tasks": {"total": 5, "done": 5},
        "fact_check": {"passed": 7, "total_checked": 8, "pass_rate": 0.875},
        "citations": {"supported": 40, "claims_checked": 44, "accuracy": 0.909, "unsupported": []},
        "quality": {"overall": 4.2, "coverage": {"score": 4, "reason": "r"}},
        "structure": {"must_cover_rate": 1.0},
        "context": {"raw_corpus_tokens": 100000, "planner_received_tokens": 1500, "writer_bundle_tokens": 4000,
                    "bundle_budget": 6000, "compression_cuts": 0, "evidence_items": 50, "sources": 20, "notes": []},
    }))
    text = evaluation.write_report().read_text()
    assert "| payroll-sea | COMPLETE | 7/8 (88%) | 40/44 (91%) | 4.2 | 100% | 1,000 | 200 s |" in text
    assert "25x smaller than the corpus" in text and "`ev-charging-eu`" in text  # not yet run
