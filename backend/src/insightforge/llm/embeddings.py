"""Local embeddings via FastEmbed (no API key, runs on CPU).

One model for every vector in the system (document chunks, entities, report
summaries, fact-check pages), so similarity scores are comparable. Changing the
model means re-indexing every collection.
"""

import asyncio
import threading
from functools import lru_cache

from fastembed import TextEmbedding

from insightforge.config import get_settings


@lru_cache
def _model() -> TextEmbedding:
    return TextEmbedding(model_name=get_settings().embedding_model)


# ONNX already uses every core for one call; concurrent calls from parallel sub-tasks only thrash the CPU
# (a live run spent 133 s in sys time). One inference at a time.
_lock = threading.Lock()


def embed_documents_sync(texts: list[str]) -> list[list[float]]:
    with _lock:
        return [vec.tolist() for vec in _model().embed(texts)]


def embed_query_sync(text: str) -> list[float]:
    # query_embed applies the model's query instruction (bge uses one), which improves retrieval.
    with _lock:
        return next(iter(_model().query_embed(text))).tolist()


async def embed_documents(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return await asyncio.to_thread(embed_documents_sync, texts)


async def embed_query(text: str) -> list[float]:
    return await asyncio.to_thread(embed_query_sync, text)
