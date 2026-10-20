"""Typed schemas for every agent handoff, plus the LangGraph pipeline state.

Each model is the contract on one edge of the pipeline:

    Planner ──TaskGraph──▶ Researchers ──ResearchResult(SourceRef[])──▶ Synthesis
    Synthesis ──EvidenceBundle──▶ Writer ──Draft(claims → evidence_ids)──▶ Grounding
    Grounding ──GroundingResult──▶ Fact-Checker ──FactCheckSummary──▶ COMPLETE

Researchers hand back references (IDs + one-line summaries), never raw text.
"""

import datetime as dt
import enum
import operator
import uuid
from typing import Annotated

from pydantic import BaseModel, Field, field_validator, model_validator

from insightforge.store.models import RunStatus, SourceKind, SourceType, SubTaskStatus

MAX_SUBTASKS = 6  # hard ceiling from the assignment; config may lower it, never raise it


class Confidence(enum.StrEnum):
    HIGH = "HIGH"  # 2+ independent sources agree
    MEDIUM = "MEDIUM"  # single high-credibility source
    LOW = "LOW"  # single low-credibility source, or conflicting signals


# --- Planner → Researchers ------------------------------------------------


class SubTaskSpec(BaseModel):
    id: str = Field(pattern=r"^st-\d{1,2}$", description='Sub-task id, e.g. "st-1"')
    query: str = Field(min_length=3, max_length=300)
    scope: str = Field(max_length=300, description="What is in and out of bounds for this sub-task")
    source_type: SourceType
    time_horizon: str = Field(max_length=64, description='e.g. "last 18 months"')
    priority: int = Field(ge=1, le=MAX_SUBTASKS, description="1 = most important")


class TaskGraph(BaseModel):
    """Planner output. Either a clarifying question or 1–6 sub-tasks, never both."""

    sub_tasks: list[SubTaskSpec] = Field(default_factory=list, max_length=MAX_SUBTASKS)
    clarifying_question: str | None = None

    @model_validator(mode="after")
    def _question_xor_tasks(self) -> "TaskGraph":
        if bool(self.clarifying_question) == bool(self.sub_tasks):
            raise ValueError("provide either a clarifying_question or at least one sub-task, not both")
        ids = [t.id for t in self.sub_tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("sub-task ids must be unique")
        return self

    def capped(self, limit: int) -> "TaskGraph":
        """Keep the `limit` highest-priority sub-tasks (config can only tighten the cap)."""
        keep = sorted(self.sub_tasks, key=lambda t: t.priority)[: min(limit, MAX_SUBTASKS)]
        return self.model_copy(update={"sub_tasks": keep})


# --- Researchers → Synthesis ----------------------------------------------


class SourceRef(BaseModel):
    """A pointer into the scratch store (17) or Qdrant (18). Carries no raw text."""

    source_id: str = Field(max_length=64)
    kind: SourceKind
    summary: str = Field(max_length=300, description="One line, written by the researcher")
    credibility: float | None = Field(default=None, ge=0, le=1)


class ResearchResult(BaseModel):
    sub_task_id: str
    status: SubTaskStatus
    refs: list[SourceRef] = Field(default_factory=list)
    skipped_urls: list[str] = Field(default_factory=list)
    error: str | None = None


# --- Entity Extractor -------------------------------------------------------


class EntityType(enum.StrEnum):
    COMPANY = "company"
    PERSON = "person"
    PRODUCT = "product"
    FUNDING_ROUND = "funding_round"
    MONEY = "money"
    DATE = "date"
    REGION = "region"


class Predicate(enum.StrEnum):
    RAISED = "raised"
    VALUED_AT = "valued_at"
    REVENUE = "revenue"
    PRICED_AT = "priced_at"
    EMPLOYEE_COUNT = "employee_count"
    HEADQUARTERED_IN = "headquartered_in"
    FOUNDED_IN = "founded_in"
    OPERATES_IN = "operates_in"
    ACQUIRED = "acquired"
    INVESTED_IN = "invested_in"
    PARTNERED_WITH = "partnered_with"
    COMPETES_WITH = "competes_with"
    LAUNCHED = "launched"
    LEADS = "leads"


# A subject can have only one true value for these (per qualifier), so different values from
# different sources are a conflict. Multi-valued predicates (operates_in, partnered_with, ...) never are.
SINGLE_VALUED = frozenset({
    Predicate.RAISED, Predicate.VALUED_AT, Predicate.REVENUE, Predicate.PRICED_AT,
    Predicate.EMPLOYEE_COUNT, Predicate.HEADQUARTERED_IN, Predicate.FOUNDED_IN,
})


class ExtractedEntity(BaseModel):
    """LLM output for one source (the extraction call). Normalized into Entity by code."""

    name: str = Field(max_length=200)
    type: EntityType


class ExtractedFact(BaseModel):
    subject: str = Field(max_length=200, description="An entity name exactly as listed in entities")
    predicate: Predicate
    object: str = Field(max_length=200)
    qualifier: str | None = Field(default=None, max_length=100)
    date: str | None = Field(default=None, description="YYYY-MM-DD if the source states when")


class SourceExtraction(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list, max_length=25)
    facts: list[ExtractedFact] = Field(default_factory=list, max_length=25)


class Entity(BaseModel):
    key: str = Field(description='Normalized id, e.g. "company:salaryhero"')
    name: str
    type: EntityType
    source_ids: list[str] = Field(default_factory=list)


class Relationship(BaseModel):
    subject: str = Field(description="Entity key")
    predicate: Predicate
    object: str
    qualifier: str | None = None
    source_id: str
    source_url: str | None = None
    date: dt.date | None = None
    conflict: bool = False


class ConflictValue(BaseModel):
    object: str
    source_ids: list[str]


class Conflict(BaseModel):
    """Sources disagree on a single-valued fact (e.g. two funding amounts for one round)."""

    subject: str
    predicate: Predicate
    qualifier: str | None = None
    values: list[ConflictValue] = Field(min_length=2)


class EntityGraph(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)


class EntityExtractionResult(BaseModel):
    """The Entity Extractor agent's final answer. The graph itself is built in code from tool output."""

    status: SubTaskStatus
    processed: int = 0
    error: str | None = None


# --- Synthesis → Writer ----------------------------------------------------


class EvidenceItem(BaseModel):
    evidence_id: str = Field(pattern=r"^ev-\d+$")
    claim: str
    source_ids: list[str] = Field(min_length=1)
    published_date: dt.date | None = None
    credibility: float = Field(ge=0, le=1)
    relevance: float = Field(ge=0, le=1)
    confidence: Confidence
    compressed: bool = False  # True when this item is a summary of lower-priority items


class EvidenceSection(BaseModel):
    theme: str
    sub_task_id: str
    evidence: list[EvidenceItem]
    confidence: Confidence


class Cut(BaseModel):
    """A record of evidence the compressor summarized or dropped, and why (goes to the run log)."""

    evidence_ids: list[str]
    action: str = Field(pattern=r"^(summarized|dropped)$")
    reason: str
    tokens_saved: int


class EvidenceBundle(BaseModel):
    sections: list[EvidenceSection]
    token_count: int
    token_budget: int
    cuts: list[Cut] = Field(default_factory=list)

    @model_validator(mode="after")
    def _within_budget(self) -> "EvidenceBundle":
        if self.token_count > self.token_budget:
            raise ValueError(f"bundle is {self.token_count} tokens, budget is {self.token_budget}")
        return self

    def evidence_ids(self) -> set[str]:
        return {item.evidence_id for s in self.sections for item in s.evidence}


# --- Writer → Grounding → Fact-Checker -------------------------------------


class ClaimRef(BaseModel):
    claim_id: str = Field(pattern=r"^c-\d+$")
    section: str
    text: str
    evidence_ids: list[str] = Field(min_length=1)


class Draft(BaseModel):
    markdown: str
    claims: list[ClaimRef]


class GroundingResult(BaseModel):
    ok: bool
    unsupported_claim_ids: list[str] = Field(default_factory=list)
    reasons: dict[str, str] = Field(default_factory=dict)


class FactCheckSummary(BaseModel):
    total_checked: int
    passed: int
    failed: int
    pass_rate: float
    unverified_claims: list[str]

    @field_validator("pass_rate")
    @classmethod
    def _rate(cls, v: float) -> float:
        return round(v, 4)


# --- Pipeline state (LangGraph) --------------------------------------------


class PipelineState(BaseModel):
    """State owned by the LangGraph orchestrator (05). Validated on every node transition.

    Holds references and typed payloads only: raw source text stays in the scratch store.
    """

    run_id: uuid.UUID
    user_id: uuid.UUID
    brief: str
    source_prefs: dict[str, bool] = Field(default_factory=lambda: {"web": True, "docs": True, "memory": True})
    memory_summaries: list[str] = Field(default_factory=list)
    related_entities: list[str] = Field(default_factory=list)  # past-run entities (Qdrant `entities`)
    clarification_answer: str | None = None

    status: RunStatus = RunStatus.QUEUED
    task_graph: TaskGraph | None = None
    # Parallel researcher branches append here; the reducer merges their results.
    research_results: Annotated[list[ResearchResult], operator.add] = Field(default_factory=list)
    entity_graph: EntityGraph | None = None
    evidence_bundle: EvidenceBundle | None = None
    draft: Draft | None = None
    grounding: GroundingResult | None = None
    writer_attempts: int = 0
    fact_check: FactCheckSummary | None = None
    final_markdown: str | None = None
    error: str | None = None
