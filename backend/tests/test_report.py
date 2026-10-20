import datetime as dt

import numpy as np
from fakes import FakeGateway
from report_fakes import REPORT_FNS
from sqlalchemy import select

from insightforge.agents.fact_checker import FactChecker, sample_claims, sentence_windows
from insightforge.agents.synthesis import (
    Candidate,
    Synthesizer,
    confidence_for,
    dedupe,
    rank_key,
    select_passages,
    split_passages,
)
from insightforge.agents.writer import render_report
from insightforge.graph.state import (
    ClaimRef,
    Confidence,
    Conflict,
    ConflictValue,
    Draft,
    Entity,
    EntityGraph,
    EntityType,
    EvidenceBundle,
    EvidenceItem,
    EvidenceSection,
    Predicate,
    ResearchResult,
    SourceMeta,
    SourceRef,
    SubTaskSpec,
    WriterOutput,
    WrittenSection,
    WrittenSentence,
)
from insightforge.guardrails.grounding import REPORT_KEY, check_grounding, drop_unsupported, structural_issues
from insightforge.guardrails.output import redact_pii
from insightforge.llm.gateway import Role
from insightforge.store.models import Claim, Report, RunEvent, SourceKind, SourceType, SubTaskStatus
from insightforge.store.qdrant import factcheck_collection
from insightforge.store.reports import save_report, session_summary
from insightforge.store.run_log import RunLog
from insightforge.store.scratch import ScratchStore
from insightforge.store.sources import SourceResolver

# --- passages --------------------------------------------------------------------------


def test_split_passages_respects_bounds():
    text = "Short line.\nAnother short line.\n\n" + ("A long sentence about payroll. " * 80)
    parts = split_passages(text)
    assert parts[0].startswith("Short line. Another short line.")  # tiny paragraphs are merged
    assert all(len(p) <= 1_200 for p in parts)


async def test_select_passages_keeps_the_relevant_paragraph():
    from insightforge.store.sources import ResolvedSource

    paragraphs = [
        "The company was founded by two engineers who previously worked at a bank in Bangkok years ago.",
        "Salary Hero raised US$7.6 million in a Series A round led by Cento Ventures to expand into Malaysia.",
        "The office has a rooftop garden and a coffee bar that employees describe as their favourite spot.",
    ]
    body = "\n\n".join(paragraphs)
    src = ResolvedSource(source_id="src-aaaaaaaaaaaa", kind=SourceKind.WEB_PAGE, text=body)
    [p] = await select_passages([src], "Salary Hero funding round amount", per_source=1)
    assert "US$7.6 million" in p.text and p.source_id == src.source_id and p.passage_id == "p-1"


# --- confidence, dedupe, ranking ----------------------------------------------------


def meta(sid, url, cred, date=None):
    return SourceMeta(source_id=sid, kind=SourceKind.WEB_PAGE, label=url.split("/")[2], url=url,
                      published_date=date, credibility=cred)


SOURCES = {
    "src-a": meta("src-a", "https://e27.co/a", 0.92, dt.date(2026, 9, 21)),
    "src-b": meta("src-b", "https://techsauce.co/b", 0.67, dt.date(2026, 9, 20)),
    "src-c": meta("src-c", "https://www.e27.co/c", 0.70),
    "src-d": meta("src-d", "https://randomblog.example/d", 0.35),
}


def cand(claim, sids, vec=(1.0, 0.0), keys=("company:salaryhero",)):
    return Candidate("st-1", claim, claim, list(sids), np.array(vec), list(keys))


def test_confidence_rules():
    assert confidence_for(cand("x", ["src-a", "src-b"]), SOURCES, False, 0.6) is Confidence.HIGH
    assert confidence_for(cand("x", ["src-a", "src-c"]), SOURCES, False, 0.6) is Confidence.MEDIUM  # same publisher
    assert confidence_for(cand("x", ["src-d"]), SOURCES, False, 0.6) is Confidence.LOW
    assert confidence_for(cand("x", ["src-a", "src-b"]), SOURCES, True, 0.6) is Confidence.LOW  # conflict wins


def test_dedupe_merges_same_entity_same_claim_and_keeps_most_credible_first():
    low = cand("Salary Hero raised $7.6M.", ["src-d"], vec=(1.0, 0.01))
    high = cand("Salary Hero raised US$7.6 million.", ["src-a"], vec=(1.0, 0.0))
    other_entity = cand("Talenox raised $7.6M.", ["src-b"], vec=(1.0, 0.0), keys=("company:talenox",))
    kept = dedupe([low, high, other_entity], SOURCES)
    assert len(kept) == 2
    merged = next(k for k in kept if "company:salaryhero" in k.entity_keys)
    assert merged.claim == "Salary Hero raised US$7.6 million." and merged.source_ids == ["src-a", "src-d"]


def item(n, date=None, cred=0.5, rel=0.5, conf=Confidence.MEDIUM, claim=None, sources=("src-a",), passage=None):
    return EvidenceItem(evidence_id=f"ev-{n}", claim=claim or f"Claim {n}.", source_ids=list(sources),
                        published_date=date, credibility=cred, relevance=rel, confidence=conf, passage=passage)


def test_ranking_is_recency_month_then_credibility_then_relevance():
    items = [item(1, dt.date(2026, 9, 1), cred=0.4), item(2, dt.date(2026, 9, 28), cred=0.9),
             item(3, dt.date(2026, 10, 2), cred=0.3), item(4, None, cred=1.0), item(5, dt.date(2026, 9, 3), 0.9, 0.9)]
    assert [i.evidence_id for i in sorted(items, key=rank_key, reverse=True)] == ["ev-3", "ev-5", "ev-2", "ev-1", "ev-4"]


# --- synthesis end to end (with compression) --------------------------------------


async def stored(maker, run_id, n=6):
    scratch = ScratchStore(maker, run_id)
    ids = []
    for i in range(n):
        facts = [f"Vendor {i} item {j} costs SGD {j + 3} per employee per month in Singapore, "
                 f"according to its {2020 + j} price list." for j in range(4)]
        filler = [f"Paragraph {k} describes the company's office culture, team offsites and the history of its "
                  f"founders in some detail, none of which concerns pricing." for k in range(20)]
        body = "\n\n".join(facts + filler)
        ids.append(await scratch.put(kind=SourceKind.WEB_PAGE, key=f"https://news{i}.example.com/a",
                                     url=f"https://news{i}.example.com/a", body_text=body,
                                     published_date=dt.date(2026, 1 + i, 1), credibility=0.5 + i / 20))
    return ids


def results_for(ids, task="st-1"):
    return [ResearchResult(sub_task_id=task, status=SubTaskStatus.DONE,
                           refs=[SourceRef(source_id=i, kind=SourceKind.WEB_PAGE, summary="s") for i in ids])]


SPEC = SubTaskSpec(id="st-1", query="payroll pricing per employee", scope="SEA", source_type=SourceType.WEB,
                   time_horizon="last 12 months", priority=1)


async def test_synthesis_builds_ranked_bundle_with_sources(committed_run):
    maker, user_id, run_id = committed_run
    ids = await stored(maker, run_id, n=3)
    gw = FakeGateway({}, REPORT_FNS)
    result = await Synthesizer(gw, SourceResolver(ScratchStore(maker, run_id), None, user_id),
                               RunLog(maker, run_id)).run("payroll pricing", [SPEC], results_for(ids), None)
    bundle = result.bundle
    [section] = bundle.sections
    assert section.sub_task_id == "st-1" and section.theme == "Findings"
    dates = [i.published_date for i in section.evidence]
    assert dates == sorted(dates, reverse=True)  # newest first
    assert all(i.passage and i.source_ids[0] in bundle.sources for i in section.evidence)
    assert bundle.token_count <= bundle.token_budget and not bundle.cuts
    assert result.passage_tokens < result.corpus_tokens  # passage selection sent less than the raw corpus


async def test_over_budget_bundle_is_compressed_and_cuts_are_logged(committed_run):
    maker, user_id, run_id = committed_run
    ids = await stored(maker, run_id, n=6)
    gw = FakeGateway({}, REPORT_FNS)
    result = await Synthesizer(gw, SourceResolver(ScratchStore(maker, run_id), None, user_id), RunLog(maker, run_id),
                               token_budget=250).run("payroll pricing", [SPEC], results_for(ids), None)
    bundle = result.bundle
    assert bundle.token_count <= 250 and bundle.cuts
    assert any(c.action == "summarized" for c in bundle.cuts)
    assert any(i.compressed for s in bundle.sections for i in s.evidence)
    async with maker() as s:
        logged = (await s.scalars(select(RunEvent).where(RunEvent.run_id == run_id, RunEvent.kind == "cut"))).all()
    assert len(logged) == len(bundle.cuts) and all("lowest-ranked" in e.message or "after" in e.message
                                                   for e in logged)


async def test_conflicting_claim_is_marked_low(committed_run):
    maker, user_id, run_id = committed_run
    scratch = ScratchStore(maker, run_id)
    a = await scratch.put(kind=SourceKind.WEB_PAGE, key="https://e27.co/a", url="https://e27.co/a", credibility=0.9,
                          body_text="Salary Hero raised US$7.6 million in its Series A, led by Cento Ventures.")
    b = await scratch.put(kind=SourceKind.WEB_PAGE, key="https://x.example/b", url="https://x.example/b",
                          credibility=0.5, body_text="Salary Hero raised USD 12 million in its Series A this year.")
    graph = EntityGraph(
        entities=[Entity(key="company:salaryhero", name="Salary Hero", type=EntityType.COMPANY)],
        conflicts=[Conflict(subject="company:salaryhero", predicate=Predicate.RAISED, qualifier="series a",
                            values=[ConflictValue(object="US$7.6 million", source_ids=[a]),
                                    ConflictValue(object="USD 12 million", source_ids=[b])])],
    )
    result = await Synthesizer(FakeGateway({}, REPORT_FNS), SourceResolver(scratch, None, user_id),
                               RunLog(maker, run_id)).run("Salary Hero funding", [SPEC], results_for([a, b]), graph)
    items = result.bundle.sections[0].evidence
    assert items and all(i.conflict and i.confidence is Confidence.LOW for i in items)
    assert result.bundle.sections[0].confidence is Confidence.LOW


# --- rendering + grounding ----------------------------------------------------------


def bundle_of(*items, conf=Confidence.HIGH):
    return EvidenceBundle(
        sections=[EvidenceSection(theme="Funding", sub_task_id="st-1", evidence=list(items), confidence=conf)],
        sources=SOURCES, token_count=100, token_budget=6000,
    )


FUNDING = item(1, dt.date(2026, 9, 21), 0.92, claim="Salary Hero raised US$7.6 million in a Series A.",
               passage="Salary Hero raised US$7.6M in a Series A led by Cento Ventures.")


def report(*sentences, summary=None, sub_task="st-1"):
    return WriterOutput(
        title="SEA payroll", executive_summary=summary or [WrittenSentence(text="Funding is active.", evidence_ids=["ev-1"])],
        sections=[WrittenSection(sub_task_id=sub_task, heading="Funding", sentences=list(sentences))],
        gaps=["Enterprise pricing is not public."],
    )


def test_render_formats_citations_labels_and_unverified():
    draft = render_report(report(WrittenSentence(text="Salary Hero raised US$7.6 million.", evidence_ids=["ev-1"]),
                                 WrittenSentence(text="Investors remain interested.")), bundle_of(FUNDING))
    md = draft.markdown
    assert md.index("## Executive Summary") < md.index("## Funding") < md.index("## Gaps & Limitations")
    assert "**Confidence: HIGH**" in md
    assert "Salary Hero raised US$7.6 million. [Source: e27.co, 2026-09-21, credibility: 0.92]" in md
    assert [c.text for c in draft.claims] == ["Funding is active.", "Salary Hero raised US$7.6 million."]

    marked = render_report(draft.report, bundle_of(FUNDING), frozenset({"c-2"})).markdown
    assert "credibility: 0.92] **[UNVERIFIED]**" in marked


def test_structural_grounding_rules():
    b = bundle_of(FUNDING)
    invented = WrittenSentence(text="Salary Hero raised US$9 million.", evidence_ids=["ev-1"])
    unknown = WrittenSentence(text="Talenox is growing fast.", evidence_ids=["ev-99"])
    uncited = WrittenSentence(text="Prices rose 40% last year.")
    fine = WrittenSentence(text="Salary Hero raised US$7.6 million in its Series A.", evidence_ids=["ev-1"])
    draft = render_report(report(invented, unknown, uncited, fine), b)
    reasons = structural_issues(draft, b)
    by_text = {c.text: reasons.get(c.claim_id) for c in draft.claims}
    assert "do not appear" in by_text[invented.text] and "9" in by_text[invented.text]
    assert "do not exist" in by_text[unknown.text]
    assert "cites no evidence" in by_text[uncited.text]
    assert by_text[fine.text] is None

    long_summary = [WrittenSentence(text="word " * 160, evidence_ids=["ev-1"])]
    wrong_section = render_report(report(fine, summary=long_summary, sub_task="st-9"), b)
    assert "150" in structural_issues(wrong_section, b)[REPORT_KEY]
    assert "sections must be" in structural_issues(wrong_section, b)[REPORT_KEY]


async def test_entailment_failures_and_drop_fallback():
    b = bundle_of(FUNDING)
    claim = WrittenSentence(text="Salary Hero is the market leader after its Series A.", evidence_ids=["ev-1"])

    def judge(messages):
        from insightforge.graph.state import ClaimVerdict, EntailmentVerdicts
        return EntailmentVerdicts(verdicts=[ClaimVerdict(claim_id="c-2", supported=False, reason="'market leader'")])

    draft = render_report(report(claim), b)
    result = await check_grounding(FakeGateway({}, {Role.GROUNDING: judge}), draft, b)
    assert not result.ok and result.unsupported_claim_ids == ["c-2"]

    cleaned = drop_unsupported(draft, result, b)
    assert cleaned.sections == []  # its only sentence was removed, so the section goes too
    assert "1 statement(s) were removed" in cleaned.gaps[-1]


# --- fact-checker ------------------------------------------------------------------------


def test_sampling_takes_twenty_percent_with_a_floor_of_five():
    claims = [ClaimRef(claim_id=f"c-{i}", section="S", text="t", evidence_ids=["ev-1"], loc=f"section:0:{i}")
              for i in range(1, 41)]
    claims.append(ClaimRef(claim_id="c-99", section="Executive Summary", text="t", evidence_ids=["ev-1"], loc="summary:0"))
    assert len(sample_claims(claims, 0.2, 5, "seed")) == 8
    assert len(sample_claims(claims[:10], 0.2, 5, "seed")) == 5
    assert sample_claims(claims, 0.2, 5, "seed") == sample_claims(claims, 0.2, 5, "seed")  # reproducible
    assert all(c.section != "Executive Summary" for c in sample_claims(claims, 1.0, 5, "seed"))


def test_sentence_windows():
    w = sentence_windows("Salary Hero raised US$7.6M in March. Cento Ventures led the round. Ok.")
    assert "Salary Hero raised US$7.6M in March." in w
    assert "Salary Hero raised US$7.6M in March. Cento Ventures led the round." in w
    assert "Ok." not in w  # too short to be meaningful


async def test_fact_checker_verifies_against_source_and_cleans_up(committed_run, vectors):
    maker, user_id, run_id = committed_run
    scratch = ScratchStore(maker, run_id)
    sid = await scratch.put(kind=SourceKind.WEB_PAGE, key="https://e27.co/a", url="https://e27.co/a", credibility=0.9,
                            body_text="Bangkok-based Salary Hero raised US$7.6 million in a Series A round led by "
                                      "Cento Ventures. The startup will use the funds to enter Malaysia.")
    ev = item(1, claim="Salary Hero raised US$7.6M.", sources=[sid])
    b = EvidenceBundle(sections=[EvidenceSection(theme="F", sub_task_id="st-1", evidence=[ev],
                                                 confidence=Confidence.HIGH)],
                       sources={sid: meta(sid, "https://e27.co/a", 0.9)}, token_count=10, token_budget=6000)
    good = "Bangkok-based Salary Hero raised US$7.6 million in a Series A round led by Cento Ventures."
    bad = "Rippling acquired a German payroll company for two billion euros last spring."
    claims = [ClaimRef(claim_id=f"c-{i}", section="Funding", text=t, evidence_ids=["ev-1"], loc=f"section:0:{i}")
              for i, t in enumerate([good, bad], start=1)]
    draft = Draft(report=report(), markdown="", claims=claims)

    summary = await FactChecker(vectors, SourceResolver(scratch, vectors, user_id), RunLog(maker, run_id),
                                fetcher=None).run(draft, b)

    by_id = {c.claim_id: c for c in summary.checks}
    assert by_id["c-1"].verified and by_id["c-1"].similarity >= 0.85 and by_id["c-1"].checked_against == "stored_copy"
    assert not by_id["c-2"].verified and "best match" in by_id["c-2"].note
    assert (summary.total_checked, summary.passed, summary.failed, summary.pass_rate) == (2, 1, 1, 0.5)
    assert summary.unverified_claims == [bad]
    assert not await vectors.client.collection_exists(factcheck_collection(run_id))


# --- output guardrail + persistence -------------------------------------------------------


def test_pii_is_redacted_but_figures_and_dates_are_not():
    text = ("Contact jane.tan@salaryhero.co or +66 2 123 4567. Raised US$7,600,000 on 2026-09-21; "
            "ID 1234-5678.")
    out, counts = redact_pii(text)
    assert "[redacted email]" in out and "[redacted phone]" in out
    assert "US$7,600,000" in out and "2026-09-21" in out and "1234-5678" in out
    assert counts == {"email": 1, "phone": 1}


async def test_report_and_claims_are_persisted(committed_run):
    from insightforge.graph.state import ClaimCheck, FactCheckSummary

    maker, user_id, run_id = committed_run
    b = bundle_of(FUNDING)
    draft = render_report(report(WrittenSentence(text="Salary Hero raised US$7.6 million.", evidence_ids=["ev-1"])), b)
    fc = FactCheckSummary(total_checked=1, passed=0, failed=1, pass_rate=0.0, unverified_claims=[draft.claims[1].text],
                          checks=[ClaimCheck(claim_id="c-2", text=draft.claims[1].text, verified=False,
                                             similarity=0.7, checked_against="live", note="low")])
    report_id = await save_report(maker, run_id=run_id, user_id=user_id, draft=draft, markdown=draft.markdown,
                                  bundle=b, fact_check=fc)
    async with maker() as s:
        row = await s.get(Report, report_id)
        claims = {c.claim_key: c for c in await s.scalars(select(Claim).where(Claim.report_id == report_id))}
    assert row.factcheck_summary["failed"] == 1 and row.evidence_bundle["sections"][0]["sub_task_id"] == "st-1"
    assert claims["c-2"].sampled and claims["c-2"].verified is False and claims["c-1"].verified is None
    assert session_summary("brief text", draft).startswith("Brief: brief text | Report: SEA payroll | Funding is active.")



def test_dedupe_never_merges_claims_with_different_numbers():
    a = cand("Vendor 1 costs SGD 5 per employee.", ["src-a"], keys=())
    b = cand("Vendor 1 costs SGD 3 per employee.", ["src-b"], keys=())
    assert len(dedupe([a, b], SOURCES)) == 2


def test_keyword_shortlist():
    from insightforge.tools.lexical import shortlist

    texts = ["The office has a garden.", "Salary Hero raised a Series A.", "Payroll pricing per employee in Singapore."]
    assert shortlist("Salary Hero Series A funding", texts, 1) == [1]
    assert shortlist("payroll pricing Singapore", texts, 1) == [2]
    assert shortlist("payroll pricing Singapore", texts, 5) == [0, 1, 2]  # k > len: all, in document order


def test_three_sentence_windows():
    w = sentence_windows("Salary Hero raised US$7.6M in March. Cento Ventures led the round. "
                         "The funds will support expansion into Malaysia.")
    assert ("Salary Hero raised US$7.6M in March. Cento Ventures led the round. "
            "The funds will support expansion into Malaysia.") in w


def test_claim_sized_word_windows_isolate_a_clause():
    from insightforge.agents.fact_checker import word_windows

    text = ("The deal marks the ninth investment from Apis Growth Fund III, which closed at US$1.23 billion. "
            "BIPO, founded in 2010 and headquartered in Singapore, serves 460,000 employees globally.")
    ws = word_windows(text, sizes=(12,), stride=4)
    assert all(len(w.split()) == 12 for w in ws[:-1])
    assert any("ninth investment from Apis Growth Fund III" in w and "BIPO" not in w for w in ws)


async def test_entailment_judge_sees_publishers():
    seen = []

    def judge(messages):
        from insightforge.graph.state import EntailmentVerdicts
        seen.append(messages[-1].content)
        return EntailmentVerdicts(verdicts=[])

    b = bundle_of(FUNDING)
    draft = render_report(report(WrittenSentence(text="e27 reports Salary Hero raised US$7.6 million.",
                                                 evidence_ids=["ev-1"])), b)
    await check_grounding(FakeGateway({}, {Role.GROUNDING: judge}), draft, b)
    assert "ev-1 [published by e27.co]" in seen[0]
