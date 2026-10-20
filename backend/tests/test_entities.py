from insightforge.graph.state import EntityType, ExtractedEntity, ExtractedFact, Predicate, SourceExtraction
from insightforge.store.models import SourceKind
from insightforge.store.sources import ResolvedSource
from insightforge.tools.entities import (
    Money,
    entity_key,
    merge_extractions,
    normalize_qualifier,
    parse_fact_date,
    parse_money,
    same_value,
)


def test_parse_money_variants():
    assert parse_money("US$7.6 million") == Money("USD", 7_600_000)
    assert parse_money("$7.6M") == Money("USD", 7_600_000)
    assert parse_money("USD 8 mn") == Money("USD", 8_000_000)
    assert parse_money("SGD 4 per employee per month") == Money("SGD", 4)
    assert parse_money("RM 1.2bn") == Money("MYR", 1_200_000_000)
    assert parse_money("1,200 employees") == Money(None, 1200)
    assert parse_money("undisclosed") is None


def test_entity_keys_merge_legal_name_variants():
    a = entity_key("Salary Hero Pte. Ltd.", EntityType.COMPANY)
    assert a == entity_key("SalaryHero", EntityType.COMPANY) == "company:salaryhero"
    assert entity_key("PT Mekari", EntityType.COMPANY) == "company:mekari"
    assert entity_key("Jane Tan", EntityType.PERSON) == "person:janetan"


def test_qualifier_and_date_normalization():
    assert normalize_qualifier("Series A round") == normalize_qualifier("series a") == "series a"
    assert normalize_qualifier("  ") is None
    assert parse_fact_date("2026-05").isoformat() == "2026-05-01"
    assert parse_fact_date("sometime") is None


def test_same_value_rules():
    assert same_value("US$7.6 million", "$7.6M", Predicate.RAISED)
    assert same_value("US$7.6 million", "USD 7.8 million", Predicate.RAISED)  # within 5%
    assert not same_value("US$7.6 million", "USD 9 million", Predicate.RAISED)
    assert same_value("SGD 10M", "USD 8M", Predicate.RAISED)  # different currencies: not comparable
    assert same_value("Singapore", "Singapore, SG", Predicate.HEADQUARTERED_IN)
    assert not same_value("Singapore", "Jakarta", Predicate.HEADQUARTERED_IN)


def src(i: int) -> ResolvedSource:
    return ResolvedSource(source_id=f"src-{i:012d}", kind=SourceKind.WEB_PAGE, text="", url=f"https://n{i}.example.com")


def ext(*facts, entities=(("Salary Hero", EntityType.COMPANY),)) -> SourceExtraction:
    return SourceExtraction(
        entities=[ExtractedEntity(name=n, type=t) for n, t in entities],
        facts=[ExtractedFact(subject=s, predicate=p, object=o, qualifier=q) for s, p, o, q in facts],
    )


def test_conflicting_funding_amounts_are_flagged():
    graph = merge_extractions([
        (src(1), ext(("Salary Hero", Predicate.RAISED, "US$7.6 million", "Series A"))),
        (src(2), ext(("Salary Hero", Predicate.RAISED, "$7.6M", "Series A round"),
                     entities=(("Salary Hero Pte Ltd", EntityType.COMPANY),))),
        (src(3), ext(("Salary Hero", Predicate.RAISED, "USD 12 million", "series a"))),
    ])
    assert [e.key for e in graph.entities] == ["company:salaryhero"]
    assert graph.entities[0].source_ids == [src(1).source_id, src(2).source_id, src(3).source_id]
    assert all(r.conflict for r in graph.relationships)
    [conflict] = graph.conflicts
    assert conflict.subject == "company:salaryhero" and conflict.qualifier == "series a"
    assert [(v.object, len(v.source_ids)) for v in conflict.values] == [("US$7.6 million", 2), ("USD 12 million", 1)]


def test_agreement_multivalued_and_unqualified_facts_are_not_conflicts():
    graph = merge_extractions([
        (src(1), ext(("Salary Hero", Predicate.RAISED, "US$7.6 million", "Series A"),
                     ("Salary Hero", Predicate.OPERATES_IN, "Thailand", None),
                     ("Salary Hero", Predicate.RAISED, "$2M", None))),
        (src(2), ext(("Salary Hero", Predicate.RAISED, "USD 7.6M", "Series A"),
                     ("Salary Hero", Predicate.OPERATES_IN, "Malaysia", None),
                     ("Salary Hero", Predicate.RAISED, "$9M", None))),
    ])
    assert graph.conflicts == []
    assert not any(r.conflict for r in graph.relationships)
    assert len(graph.relationships) == 6  # corroborating facts from different sources are all kept


def test_one_source_disagreeing_with_itself_is_not_a_cross_source_conflict():
    graph = merge_extractions([(src(1), ext(("Salary Hero", Predicate.HEADQUARTERED_IN, "Bangkok", None),
                                            ("Salary Hero", Predicate.HEADQUARTERED_IN, "Singapore", None)))])
    assert graph.conflicts == []


def test_unlisted_subject_becomes_a_company_entity():
    graph = merge_extractions([(src(1), ext(("Sprout Solutions", Predicate.LAUNCHED, "Sprout Payroll", None),
                                            entities=()))])
    assert graph.entities[0].key == "company:sproutsolutions"
    assert graph.relationships[0].subject == "company:sproutsolutions"
    assert graph.relationships[0].source_url == "https://n1.example.com"
