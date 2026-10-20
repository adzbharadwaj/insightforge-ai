"""Semantic search over the run's uploaded PDFs, and the `search_documents` tool (08).

Given a sub-task query, returns the top-k chunks from Qdrant `doc_chunks`, filtered
to this user and this run. The agent sees each chunk's id, file, page, section,
relevance score and a short excerpt; the full chunk text is resolved later by
Synthesis through SourceResolver.
"""

import uuid
from dataclasses import dataclass

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from insightforge.guardrails.untrusted import wrap
from insightforge.llm.embeddings import embed_query
from insightforge.store.qdrant import DOC_CHUNKS, VectorStore
from insightforge.store.run_log import RunLog
from insightforge.store.sources import chunk_source_id

EXCERPT_CHARS = 400
AGENT = "doc_reader"


@dataclass(frozen=True)
class ChunkHit:
    chunk_id: str
    text: str
    source: str
    page: int | None
    section: str | None
    relevance_score: float


class DocSearcher:
    def __init__(
        self,
        vectors: VectorStore,
        log: RunLog,
        user_id: uuid.UUID,
        *,
        top_k: int = 8,
        min_relevance: float = 0.3,
    ):
        self.vectors = vectors
        self.log = log
        self.user_id = user_id
        self.top_k = top_k
        self.min_relevance = min_relevance

    async def search(self, sub_task_id: str, query: str) -> list[ChunkHit]:
        hits = await self.vectors.search(
            DOC_CHUNKS, await embed_query(query), user_id=self.user_id, limit=self.top_k,
            match={"run_id": str(self.log.run_id)}, score_threshold=self.min_relevance,
        )
        chunks = [
            ChunkHit(
                chunk_id=chunk_source_id(h.id),
                text=h.payload.get("text", ""),
                source=h.payload.get("source_filename", "document.pdf"),
                page=h.payload.get("page_number"),
                section=h.payload.get("section_heading"),
                relevance_score=round(h.score, 3),
            )
            for h in hits
        ]
        await self.log.progress(AGENT, f"{sub_task_id}: {len(chunks)} chunk(s) matched",
                                sub_task_id=sub_task_id, query=query, chunk_ids=[c.chunk_id for c in chunks])
        return chunks


class DocSearchArgs(BaseModel):
    sub_task_id: str = Field(description='The sub-task id you were given, e.g. "st-2"')
    query: str = Field(min_length=3, max_length=300, description="What to look for in the uploaded documents")


def format_for_agent(chunks: list[ChunkHit]) -> str:
    if not chunks:
        return "No relevant passages in the uploaded documents."
    lines = [f"{len(chunks)} chunk(s), most relevant first."]
    for c in chunks:
        origin = f"{c.source} p.{c.page or '?'}"
        lines.append(
            f"- chunk_id={c.chunk_id} source={c.source} page={c.page or 'unknown'} "
            f"section={c.section or 'unknown'} relevance_score={c.relevance_score}\n"
            f"{wrap(c.chunk_id, c.text[:EXCERPT_CHARS], url=origin, title=c.section)}"
        )
    return "\n".join(lines)


def make_doc_search_tool(searcher: DocSearcher) -> BaseTool:
    async def search_documents(sub_task_id: str, query: str) -> str:
        return format_for_agent(await searcher.search(sub_task_id, query))

    return StructuredTool.from_function(
        coroutine=search_documents,
        name="search_documents",
        description=(
            "Semantic search over the PDFs the user uploaded for this run. Returns chunk ids with file, "
            "page, section, relevance score and a short excerpt; full text stays in storage."
        ),
        args_schema=DocSearchArgs,
    )
