"""PDF ingest (component 19): validate → Docling → semantic chunks → Qdrant `doc_chunks`.

- Input guardrail (01a): at most `max_pdfs_per_run` files, each ≤ `max_pdf_mb`, and the
  bytes must start with the PDF magic number (the extension and MIME type are not trusted).
- Docling extracts layout-aware text and tables; HybridChunker splits it into sections
  sized for the embedding model, keeping the heading path and page number of each chunk.
- Chunk text is untrusted (03c): cleaned of smuggling characters and scanned for
  injection before it is stored. Flags are kept in the payload for the run log.
- Every point carries `user_id` and `run_id`; chunk ids (`chk-…`) are what the Doc Reader
  hands back, never the text.
"""

import asyncio
import re
import threading
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from insightforge.config import Settings, get_settings
from insightforge.guardrails.untrusted import clean_text, scan_injection
from insightforge.llm.embeddings import embed_documents
from insightforge.store.models import Document
from insightforge.store.qdrant import DOC_CHUNKS, VectorStore
from insightforge.store.run_log import RunLog
from insightforge.store.sources import chunk_source_id

PDF_MAGIC = b"%PDF-"
_CHUNK_NS = uuid.UUID("5f0c1b8e-6a43-4a4f-9a51-2d3c7a1e9b10")


class PdfRejected(ValueError):
    """A file failed validation or could not be parsed. The message is safe to show the user."""


@dataclass(frozen=True)
class DocChunk:
    index: int
    text: str  # heading path + section text, as embedded
    page_number: int | None
    section_heading: str | None


@dataclass(frozen=True)
class ParsedPdf:
    pages: int
    chunks: list[DocChunk]


Parser = Callable[[Path], Awaitable[ParsedPdf]]


def safe_filename(name: str) -> str:
    base = Path(name).name
    base = re.sub(r"[^\w.\- ()]", "_", base).strip(" .") or "document.pdf"
    return base[:200]


def validate_pdfs(files: list[tuple[str, bytes]], settings: Settings | None = None) -> None:
    s = settings or get_settings()
    if len(files) > s.max_pdfs_per_run:
        raise PdfRejected(f"at most {s.max_pdfs_per_run} PDFs per research run (got {len(files)})")
    limit = s.max_pdf_mb * 1024 * 1024
    for name, data in files:
        if len(data) > limit:
            raise PdfRejected(f"{safe_filename(name)} is larger than {s.max_pdf_mb} MB")
        if not data.startswith(PDF_MAGIC):
            raise PdfRejected(f"{safe_filename(name)} is not a PDF")


def save_upload(user_id: uuid.UUID, run_id: uuid.UUID, filename: str, data: bytes,
                settings: Settings | None = None) -> Path:
    """Store under a server-chosen name; the user's filename is kept only as metadata."""
    s = settings or get_settings()
    folder = s.upload_dir / str(user_id) / str(run_id)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{uuid.uuid4().hex}.pdf"
    path.write_bytes(data)
    return path


# --- Docling -------------------------------------------------------------------------

_docling_lock = threading.Lock()  # converter instances are not documented as thread-safe


@lru_cache
def _converter_and_chunker(ocr: bool, max_tokens: int, tokenizer_model: str):
    # Imported lazily: Docling pulls in PyTorch, which the rest of the app never needs.
    from docling.chunking import HybridChunker
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
    from transformers import AutoTokenizer

    converter = DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=PdfPipelineOptions(do_ocr=ocr)),
    })
    tokenizer = HuggingFaceTokenizer(tokenizer=AutoTokenizer.from_pretrained(tokenizer_model), max_tokens=max_tokens)
    return converter, HybridChunker(tokenizer=tokenizer, merge_peers=True)


def parse_pdf_sync(path: Path, settings: Settings | None = None) -> ParsedPdf:
    s = settings or get_settings()
    converter, chunker = _converter_and_chunker(s.docling_ocr, s.doc_chunk_max_tokens, s.embedding_model)
    with _docling_lock:
        try:
            doc = converter.convert(str(path)).document
        except Exception as exc:  # Docling raises many types for corrupt/encrypted files
            raise PdfRejected(f"could not read PDF: {type(exc).__name__}") from exc
        chunks = []
        for i, ch in enumerate(chunker.chunk(doc)):
            pages = sorted({p.page_no for item in ch.meta.doc_items for p in item.prov})
            headings = ch.meta.headings or []
            chunks.append(DocChunk(
                index=i,
                text=chunker.contextualize(ch),
                page_number=pages[0] if pages else None,
                section_heading=headings[-1] if headings else None,
            ))
    return ParsedPdf(pages=doc.num_pages(), chunks=chunks)


async def parse_pdf(path: Path) -> ParsedPdf:
    return await asyncio.to_thread(parse_pdf_sync, path)


# --- Ingest ----------------------------------------------------------------------------


def chunk_point(document_id: uuid.UUID, index: int) -> str:
    return str(uuid.uuid5(_CHUNK_NS, f"{document_id}:{index}"))


async def ingest_pdf(
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    vectors: VectorStore,
    log: RunLog,
    user_id: uuid.UUID,
    filename: str,
    path: Path,
    parse: Parser = parse_pdf,
) -> Document:
    """Parse one stored PDF and index its chunks. Raises PdfRejected if it yields no text."""
    name = safe_filename(filename)
    parsed = await parse(path)
    chunks = [(c, clean_text(c.text)) for c in parsed.chunks]
    chunks = [(c, text) for c, text in chunks if text.strip()]
    if not chunks:
        raise PdfRejected(f"{name} has no extractable text (scanned PDFs need OCR, which is off)")

    document = Document(run_id=log.run_id, user_id=user_id, filename=name, storage_path=str(path),
                        size_bytes=path.stat().st_size, pages=parsed.pages, chunk_count=len(chunks))
    async with sessionmaker() as session:
        session.add(document)
        await session.commit()
        await session.refresh(document)

    payloads, flagged = [], 0
    for c, text in chunks:
        flags = scan_injection(text).flags
        flagged += bool(flags)
        payloads.append({
            "run_id": str(log.run_id),
            "document_id": str(document.id),
            "source_filename": name,
            "page_number": c.page_number,
            "section_heading": c.section_heading,
            "chunk_index": c.index,
            "text": text,
            "injection_flags": flags,
        })
    vecs = await embed_documents([text for _, text in chunks])
    ids = [chunk_point(document.id, c.index) for c, _ in chunks]
    await vectors.upsert(DOC_CHUNKS, user_id=user_id, vectors=vecs, payloads=payloads, ids=ids)

    if flagged:
        await log.emit("doc-ingest", "injection_flag", f"{name}: {flagged} chunk(s) look like instructions",
                       document_id=str(document.id), chunk_ids=[chunk_source_id(i) for i in ids][:20])
    await log.progress("doc-ingest", f"indexed {name}: {parsed.pages} page(s), {len(chunks)} chunk(s)",
                       document_id=str(document.id))
    return document
