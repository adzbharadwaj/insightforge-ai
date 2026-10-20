"""Typed settings for every service, budget and limit.

Values come from environment variables (or a .env file). Defaults are the
starting points from the architecture plan; tune them from eval results.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # --- Infrastructure ---
    database_url: str = "postgresql+asyncpg://insightforge:insightforge@localhost:5432/insightforge"
    redis_url: str = "redis://localhost:6379/0"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None

    # --- Model gateway (13) — OpenRouter ---
    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_app_url: str = "http://localhost:3000"
    openrouter_app_name: str = "InsightForge AI"
    # Slugs verified against openrouter.ai/api/v1/models (both support tool calling).
    model_primary: str = "anthropic/claude-sonnet-5.5"
    model_secondary: str = "anthropic/claude-haiku-5.5"
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2

    # --- Embeddings (local FastEmbed) ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # --- Web search (07) ---
    search_provider: Literal["tavily", "brave", "serpapi"] = "tavily"
    tavily_api_key: SecretStr | None = None
    brave_api_key: SecretStr | None = None
    serpapi_api_key: SecretStr | None = None
    search_results_per_subtask: int = 5

    # --- Run limits (05) ---
    max_subtasks: int = Field(6, ge=1, le=6)
    max_tool_calls_per_subagent: int = 12
    run_time_budget_s: int = 8 * 60
    run_token_budget: int = 250_000
    max_writer_retries: int = 2
    max_concurrent_runs: int = 4
    clarification_timeout_s: int = 30 * 60

    # --- Context engineering ---
    evidence_bundle_token_budget: int = 6_000
    memory_summaries_loaded: int = 3
    doc_reader_top_k: int = 8
    doc_min_relevance: float = 0.3  # chunks below this cosine score are not shown to the Doc Reader
    entity_batch_size: int = 10  # source ids per extract_entities call
    entity_source_max_chars: int = 12_000  # per-source text sent to the extraction model
    entity_concurrency: int = 4
    related_entities_loaded: int = 5  # past-run entities shown to the Planner

    # --- Fact-checking (12) ---
    factcheck_sample_ratio: float = 0.20
    factcheck_min_claims: int = 5
    factcheck_similarity_threshold: float = 0.85

    # --- Uploads ---
    max_pdfs_per_run: int = 10
    max_pdf_mb: int = 20
    upload_dir: Path = REPO_ROOT / "data" / "uploads"
    docling_ocr: bool = False  # OCR adds a large model download; text-based PDFs don't need it
    doc_chunk_max_tokens: int = 512  # bge-small's input window
    uploaded_doc_credibility: float = 0.7  # user-provided, provenance unknown
    brief_max_chars: int = 500

    # --- Security ---
    jwt_secret: SecretStr = SecretStr("change-me-in-env")
    share_link_ttl_days: int = 7
    webhook_signing_secret: SecretStr = SecretStr("change-me-in-env")


@lru_cache
def get_settings() -> Settings:
    return Settings()
