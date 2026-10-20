"""One way to resolve any source id a researcher hands back.

Two kinds of id travel through the pipeline:

- `src-<12 hex>`  a web page in the scratch store (17), scoped to the run
- `chk-<32 hex>`  a PDF chunk in Qdrant `doc_chunks` (18); the hex is the point UUID

Every consumer (the recorder's invented-id check, `read_source`, the Entity
Extractor, Synthesis) goes through SourceResolver, so the ACL is applied in one
place: a chunk resolves only for the run's own user and run.
"""

import uuid
from dataclasses import dataclass
from datetime import date

from insightforge.store.models import SourceKind
from insightforge.store.qdrant import DOC_CHUNKS, VectorStore
from insightforge.store.scratch import ScratchStore

CHUNK_PREFIX = "chk-"


def chunk_source_id(point_id: uuid.UUID | str) -> str:
    return CHUNK_PREFIX + uuid.UUID(str(point_id)).hex


def chunk_point_id(source_id: str) -> str | None:
    """The Qdrant point id behind a `chk-` source id, or None if it isn't one."""
    if not source_id.startswith(CHUNK_PREFIX):
        return None
    try:
        return str(uuid.UUID(hex=source_id.removeprefix(CHUNK_PREFIX)))
    except ValueError:
        return None


@dataclass(frozen=True)
class ResolvedSource:
    source_id: str
    kind: SourceKind
    text: str
    title: str | None = None
    url: str | None = None  # web pages only
    filename: str | None = None  # PDF chunks only
    page_number: int | None = None
    section_heading: str | None = None
    published_date: date | None = None
    credibility: float | None = None

    @property
    def label(self) -> str:
        """Human-readable origin, used as the `url` attribute when wrapping untrusted text."""
        if self.filename:
            return f"{self.filename} p.{self.page_number or '?'}"
        return self.url or self.source_id


class SourceResolver:
    def __init__(
        self,
        scratch: ScratchStore,
        vectors: VectorStore | None,
        user_id: uuid.UUID,
        doc_credibility: float = 0.7,
    ):
        self.scratch = scratch
        self.vectors = vectors
        self.user_id = user_id
        self.doc_credibility = doc_credibility

    @property
    def run_id(self) -> uuid.UUID:
        return self.scratch.run_id

    async def resolve_many(self, source_ids: list[str]) -> list[ResolvedSource]:
        """Resolve ids in the order given; unknown, foreign or malformed ids are left out."""
        web_ids = [i for i in source_ids if i.startswith("src-")]
        points = {i: p for i in source_ids if (p := chunk_point_id(i))}
        found: dict[str, ResolvedSource] = {}

        for row in await self.scratch.get_many(web_ids):
            found[row.source_id] = ResolvedSource(
                source_id=row.source_id, kind=row.kind, text=row.body_text, title=row.title, url=row.url,
                published_date=row.published_date, credibility=row.credibility,
            )
        if points and self.vectors is not None:
            hits = await self.vectors.retrieve(DOC_CHUNKS, list(points.values()), user_id=self.user_id)
            for hit in hits:
                if hit.payload.get("run_id") != str(self.run_id):
                    continue  # the user's own chunk, but from another run's uploads
                sid = chunk_source_id(hit.id)
                found[sid] = ResolvedSource(
                    source_id=sid, kind=SourceKind.PDF_CHUNK, text=hit.payload.get("text", ""),
                    title=hit.payload.get("section_heading"), filename=hit.payload.get("source_filename"),
                    page_number=hit.payload.get("page_number"), section_heading=hit.payload.get("section_heading"),
                    credibility=self.doc_credibility,
                )
        return [found[i] for i in source_ids if i in found]

    async def resolve(self, source_id: str) -> ResolvedSource | None:
        found = await self.resolve_many([source_id])
        return found[0] if found else None

    async def known(self, source_ids: list[str]) -> set[str]:
        return {s.source_id for s in await self.resolve_many(source_ids)}
