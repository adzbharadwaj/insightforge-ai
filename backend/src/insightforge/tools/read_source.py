"""`read_source` tool: on-demand, paged access to any source id of this run.

Resolves both web pages (`src-…`, scratch store 17) and PDF chunks (`chk-…`, Qdrant 18)
through SourceResolver, so an agent can only read its own run's sources. Content
comes back wrapped as untrusted data (03c) and in pages, so a single call can never
flood the agent's context with a whole article.
"""

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from insightforge.guardrails.untrusted import wrap
from insightforge.store.sources import SourceResolver

PAGE_CHARS = 4_000  # ~1k tokens per page


class ReadSourceArgs(BaseModel):
    source_id: str = Field(description='A source id, e.g. "src-1a2b3c4d5e6f" or "chk-…"')
    page: int = Field(default=0, ge=0, description="Page of the source body to read, starting at 0")


def make_read_source_tool(resolver: SourceResolver, page_chars: int = PAGE_CHARS) -> BaseTool:
    async def read_source(source_id: str, page: int = 0) -> str:
        src = await resolver.resolve(source_id)
        if src is None:
            return f"No source with id {source_id!r} in this run."
        pages = max(1, -(-len(src.text) // page_chars))
        if page >= pages:
            return f"Source {source_id} has {pages} page(s); page {page} does not exist."
        chunk = src.text[page * page_chars : (page + 1) * page_chars]
        header = (
            f"source_id={src.source_id} page={page + 1}/{pages} "
            f"published={src.published_date or 'unknown'} credibility={src.credibility}"
        )
        return f"{header}\n{wrap(src.source_id, chunk, url=src.label, title=src.title)}"

    return StructuredTool.from_function(
        coroutine=read_source,
        name="read_source",
        description=(
            "Read the stored text of one source by its source_id, one page at a time. "
            "Use it only for sources you need; do not read every source."
        ),
        args_schema=ReadSourceArgs,
    )
