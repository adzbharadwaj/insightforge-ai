"""Postgres schema (component 16) and the scratch store table (component 17).

Every user-owned row carries `user_id`, and queries filter on it: identity is
enforced at the data layer, not in prompts.
"""

import enum
import uuid
from datetime import date, datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _enum(cls: type[enum.Enum], name: str) -> Enum:
    return Enum(cls, name=name, values_callable=lambda e: [m.value for m in e])


# --- Enums ---------------------------------------------------------------


class RunStatus(enum.StrEnum):
    QUEUED = "QUEUED"
    PLANNING = "PLANNING"
    AWAITING_CLARIFICATION = "AWAITING_CLARIFICATION"
    RESEARCHING = "RESEARCHING"
    SYNTHESIZING = "SYNTHESIZING"
    WRITING = "WRITING"
    GROUNDING = "GROUNDING"
    FACT_CHECKING = "FACT_CHECKING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class SubTaskStatus(enum.StrEnum):
    QUEUED = "Queued"
    RUNNING = "Running"
    DONE = "Done"
    FAILED = "Failed"


class SourceType(enum.StrEnum):
    WEB = "web"
    DOCS = "docs"
    MEMORY = "memory"


class SourceKind(enum.StrEnum):
    WEB_PAGE = "web_page"
    PDF_CHUNK = "pdf_chunk"
    ENTITY = "entity"


class Cadence(enum.StrEnum):
    DAILY = "daily"
    WEEKLY = "weekly"


# --- Tables --------------------------------------------------------------


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


def _created() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now())


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = _uuid_pk()
    email: Mapped[str] = mapped_column(String(320), unique=True)
    created_at: Mapped[datetime] = _created()


class Run(Base):
    """One research run, manual or scheduled. Status mirrors the LangGraph state machine."""

    __tablename__ = "runs"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    watchlist_item_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("watchlist_items.id", ondelete="SET NULL"), index=True
    )
    brief: Mapped[str] = mapped_column(Text)
    source_prefs: Mapped[dict] = mapped_column(JSONB, default=dict)  # {"web": bool, "docs": bool, "memory": bool}
    status: Mapped[RunStatus] = mapped_column(_enum(RunStatus, "run_status"), default=RunStatus.QUEUED)
    clarifying_question: Mapped[str | None] = mapped_column(Text)
    clarification_answer: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    entity_graph: Mapped[dict | None] = mapped_column(JSONB)  # EntityGraph from wave 2 (component 09)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = _created()
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    sub_tasks: Mapped[list["SubTask"]] = relationship(back_populates="run", cascade="all, delete-orphan")


class SubTask(Base):
    """A node of the Planner's task graph; one row per UI progress row."""

    __tablename__ = "sub_tasks"
    __table_args__ = (UniqueConstraint("run_id", "task_key"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    task_key: Mapped[str] = mapped_column(String(32))  # e.g. "st-1", as emitted by the Planner
    query: Mapped[str] = mapped_column(Text)
    source_type: Mapped[SourceType] = mapped_column(_enum(SourceType, "source_type"))
    time_horizon: Mapped[str] = mapped_column(String(64))
    priority: Mapped[int] = mapped_column(Integer)
    status: Mapped[SubTaskStatus] = mapped_column(
        _enum(SubTaskStatus, "sub_task_status"), default=SubTaskStatus.QUEUED
    )
    summary: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    run: Mapped[Run] = relationship(back_populates="sub_tasks")


class RunEvent(Base):
    """Append-only run log. Feeds the SSE activity stream and records cuts, skips and retries."""

    __tablename__ = "run_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    agent: Mapped[str] = mapped_column(String(64))  # "planner", "web_search", "synthesis", ...
    kind: Mapped[str] = mapped_column(String(32))  # "progress" | "cut" | "skipped_url" | "retry" | "error"
    message: Mapped[str] = mapped_column(Text)
    data: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = _created()


class ScratchSource(Base):
    """Scratch store (17): raw fetched content lives here, never in conversation history.

    Agents pass `source_id` references; only Synthesis and the Fact-Checker resolve full text.
    """

    __tablename__ = "scratch_sources"

    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True)
    source_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    kind: Mapped[SourceKind] = mapped_column(_enum(SourceKind, "source_kind"))
    sub_task_key: Mapped[str | None] = mapped_column(String(32))
    url: Mapped[str | None] = mapped_column(Text)
    title: Mapped[str | None] = mapped_column(Text)
    published_date: Mapped[date | None] = mapped_column(Date)
    body_text: Mapped[str] = mapped_column(Text)
    summary: Mapped[str | None] = mapped_column(Text)  # the one-line summary handed back up
    credibility: Mapped[float | None] = mapped_column(Float)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    meta: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = _created()


class Document(Base):
    """A user-uploaded PDF; its chunks live in Qdrant `doc_chunks`."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = _uuid_pk()
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    filename: Mapped[str] = mapped_column(String(512))
    storage_path: Mapped[str] = mapped_column(Text)
    size_bytes: Mapped[int] = mapped_column(Integer)
    pages: Mapped[int | None] = mapped_column(Integer)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = _created()


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[uuid.UUID] = _uuid_pk()
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), unique=True)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(Text)
    markdown: Mapped[str] = mapped_column(Text)
    evidence_bundle: Mapped[dict] = mapped_column(JSONB)
    factcheck_summary: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = _created()

    claims: Mapped[list["Claim"]] = relationship(back_populates="report", cascade="all, delete-orphan")


class Claim(Base):
    """One factual claim from the Writer, with the evidence it cites (evidence travels with the draft)."""

    __tablename__ = "claims"
    __table_args__ = (UniqueConstraint("report_id", "claim_key"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    report_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("reports.id", ondelete="CASCADE"), index=True)
    claim_key: Mapped[str] = mapped_column(String(32))
    section: Mapped[str] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text)
    evidence_ids: Mapped[list[str]] = mapped_column(ARRAY(String(64)))
    sampled: Mapped[bool] = mapped_column(Boolean, default=False)
    verified: Mapped[bool | None] = mapped_column(Boolean)  # None = not fact-checked
    similarity: Mapped[float | None] = mapped_column(Float)
    verify_note: Mapped[str | None] = mapped_column(Text)

    report: Mapped[Report] = relationship(back_populates="claims")


class WatchlistItem(Base):
    __tablename__ = "watchlist_items"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    topic: Mapped[str] = mapped_column(Text)
    cadence: Mapped[Cadence] = mapped_column(_enum(Cadence, "cadence"))
    webhook_url: Mapped[str | None] = mapped_column(Text)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_run: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    next_run: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = _created()


class Alert(Base):
    __tablename__ = "alerts"
    __table_args__ = (Index("ix_alerts_user_created", "user_id", "created_at"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    watchlist_item_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("watchlist_items.id", ondelete="CASCADE"))
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    change_summary: Mapped[dict] = mapped_column(JSONB)  # {added_findings, removed_findings, changed_findings}
    read: Mapped[bool] = mapped_column(Boolean, default=False)
    webhook_status: Mapped[str | None] = mapped_column(String(32))  # "sent" | "failed" | None
    created_at: Mapped[datetime] = _created()


class RevokedShareLink(Base):
    """`jti` denylist so a 7-day share link can be revoked early."""

    __tablename__ = "revoked_share_links"

    jti: Mapped[str] = mapped_column(String(64), primary_key=True)
    revoked_at: Mapped[datetime] = _created()
