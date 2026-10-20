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
    # Slugs verified against openrouter.ai/api/v1/models and probed with the pipeline's call patterns
    # (JSON-schema output; tool-calling agent). Cheap open-weight alternative, also verified:
    # deepseek/deepseek-v4.1-flash / z-ai/glm-5.3-flash (agent_output_strategy "auto" handles both).
    model_primary: str = "anthropic/claude-sonnet-5.5"
    model_secondary: str = "anthropic/claude-haiku-5.5"
    # How agents with tools return their typed answer. "provider" = native JSON schema (needed for
    # Claude, which rejects forced tool_choice); "tool" = a forced structured-output tool call (open
    # models: with native JSON alongside tools, GLM skipped its tools and DeepSeek hung). "auto" picks
    # per model: provider for anthropic/*, tool otherwise.
    agent_output_strategy: Literal["auto", "provider", "tool"] = "auto"
    llm_max_retries: int = 1  # per-role timeouts live in llm/gateway.py ROLE_SPECS

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
    # Measured 2026-10-20: a clean run ≈ 185k (research ~75k, entities ~50k, synthesis ~24k, write ~14k,
    # grounding ~20k); each writer retry adds ~34k. 300k fits the 2 allowed retries with margin.
    run_token_budget: int = 300_000
    max_writer_retries: int = 2
    # A failed run resumes from its last checkpoint (the stage that failed re-runs; finished stages and
    # finished researchers/extractions are kept). The worker does this automatically for retryable failures
    # (time budget, provider errors) this many times; users can retry up to max_run_attempts in total.
    max_auto_resumes: int = 1
    max_run_attempts: int = 3
    max_concurrent_runs: int = 4
    clarification_timeout_s: int = 30 * 60

    # --- Context engineering ---
    evidence_bundle_token_budget: int = 6_000
    synthesis_passages_per_source: int = 3  # most query-relevant paragraphs taken from each source
    high_credibility_threshold: float = 0.6  # single source at/above this → MEDIUM confidence
    memory_summaries_loaded: int = 3
    doc_reader_top_k: int = 8
    doc_min_relevance: float = 0.3  # chunks below this cosine score are not shown to the Doc Reader
    entity_batch_size: int = 10  # source ids per extract_entities call
    entity_max_sources: int = 20  # most credible sources only: the stage costs ~3k tokens per source
    entity_source_max_chars: int = 5_000  # per-source text sent to the extraction model (cost driver)
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
    access_token_ttl_hours: int = 24
    share_link_ttl_days: int = 7
    # Local development only: POST /auth/dev-login issues a token for any email. Off in production,
    # where tokens come from the identity provider.
    allow_dev_login: bool = True
    app_base_url: str = "http://localhost:3000"  # frontend origin, used in share links and CORS

    # --- Watchlist (scheduler) ---
    watchlist_interval_override_s: int | None = None  # set e.g. 60 for demos: every item runs every minute
    scheduler_reconcile_s: int = 60  # how often the scheduler syncs its jobs with the watchlist table
    scheduler_misfire_grace_s: int = 3600  # a run missed while the scheduler was down still fires within this

    # --- API limits ---
    runs_per_hour_per_user: int = 10
    max_active_runs_per_user: int = 2
    webhook_signing_secret: SecretStr = SecretStr("change-me-in-env")


PLACEHOLDER_SECRETS = {"", "change-me-in-env"}
MIN_SECRET_CHARS = 32


def require_secret(value: SecretStr, name: str) -> str:
    """A signing secret that is actually secret. Checked where it is used (API startup, webhooks), so the
    CLI and worker run without one."""
    secret = value.get_secret_value()
    if secret in PLACEHOLDER_SECRETS or len(secret) < MIN_SECRET_CHARS:
        raise RuntimeError(
            f"{name} must be set in .env to at least {MIN_SECRET_CHARS} random characters. Generate one with:\n"
            "  openssl rand -base64 48")
    return secret


@lru_cache
def get_settings() -> Settings:
    return Settings()
