# Build Status — InsightForge AI

Single source of truth for where the build is. Update after every component.

- **Plan:** `~/.claude/plans/git-init-eager-snowflake.md`
- **Architecture diagram:** `docs/architecture.html` · https://claude.ai/artifact/52efizCUMZiLyaEzeRotdt
- **Style rules:** `architecture-style-guide.md` (git-ignored, local only)

## Resume here

**Current phase:** 3 complete **and verified live** → Phase 4 — Synthesis → report
**Next step:** 4.1 Synthesis agent [10]: resolve refs via `SourceResolver`, dedupe by entity + claim using `state.entity_graph`, HIGH/MEDIUM/LOW confidence (conflicts → LOW), rank recency → credibility → relevance, compress to 6k tokens with logged cuts.
Live slice: `cd backend && uv run insightforge "<brief>" [--pdf file.pdf ...] [--no-web]`

```bash
docker compose up -d                 # Postgres, Redis, Qdrant (from repo root)
cd backend
uv sync                              # Python 3.12 venv + deps
uv run alembic upgrade head          # apply DB migrations
uv run pytest -q                     # all tests should pass (DB tests skip if Postgres is down)
uv run uvicorn insightforge.main:app --reload   # API on :8000, check /health
uv run insightforge "<brief>"                    # CLI: plan + research one brief (needs keys)
```

## Decisions locked in

| Topic | Decision |
|---|---|
| LLM access | OpenRouter, via `langchain-openai` `ChatOpenAI(base_url=...)`, behind our model gateway (13) |
| Models | `anthropic/claude-sonnet-5.5` primary, `anthropic/claude-haiku-5.5` secondary (verified on OpenRouter 2026-10-18, both support tools) |
| Embeddings | Local FastEmbed `BAAI/bge-small-en-v1.5`, 384-d |
| Python | 3.12 pinned with uv (system Python is 3.14; Docling/onnxruntime wheels lag) |
| Package layout | `backend/src/insightforge/` (one package; plan's folders become subpackages) |
| Repo root | This folder is the repo (plan called it `insightforge-agent/`) |
| Structured output | Always provider-native JSON schema (`method="json_schema"` / `ProviderStrategy`); OpenRouter's Claude models reject forced `tool_choice` |
| Source ids | `src-<12hex>` = web page in scratch store; `chk-<32hex>` = PDF chunk in Qdrant (hex = point UUID). Always resolve via `store/sources.py::SourceResolver` (applies user + run ACL) |
| Delegation | `DelegationMiddleware` + a Ledger per wave (wave 1 `ResearchRecorder`, wave 2 `EntityWave`); subagents built with `agents/subagent.py::isolated_subagent` |
| `memory` source type | Not delegated (no memory-reader); past runs reach the Planner as summaries + related entities |
| Docling | OCR off (`DOCLING_OCR`), HybridChunker on the bge tokenizer (512 tokens); imported lazily so the API doesn't load PyTorch |

## Progress

Legend: ✅ done · 🔨 in progress · ⬜ not started. Numbers in brackets are diagram component IDs.

### Phase 0 — Diagram
- ✅ 0.1 Architecture diagram (`docs/architecture.html`, artifact published)

### Phase 1 — Skeleton
- ✅ 1.1 Config + env — `config.py` (all budgets/limits typed), `.env.example`, `.gitignore` · tests: `tests/test_config.py`
- ✅ 1.2 docker-compose — `postgres:16-alpine`, `redis:7-alpine`, `qdrant/qdrant:v1.19.1`, all with health checks
- ✅ 1.3 Postgres models + Alembic [16] — `store/models.py`, `store/db.py`, migration `initial schema` (upgrade/downgrade tested); 12 tables incl. `scratch_sources` [17] and `run_events` (run log) · tests: `tests/test_models.py` (rolls back, skips if DB down)
- ✅ 1.4 Qdrant client + FastEmbed wrapper [18] — `store/qdrant.py` (`VectorStore`: user_id required on every search/retrieve, persistent collections protected from drop, temp `factcheck_{run}`), `llm/embeddings.py` (bge-small, async via thread) · tests: `tests/test_vector_store.py` (first run downloads the model, ~60s)
- ✅ 1.5 Model gateway [13] — `llm/gateway.py`: `Role` → tier + max_tokens (`ROLE_SPECS`), `chat_model()` for frameworks, `runnable()` with fallback to other tier, thread-safe `TokenMeter` (checks budget before each call; breach is never rescued by fallback) · tests: `tests/test_gateway.py` (live smoke test skips until `OPENROUTER_API_KEY` is in `.env`)
- ✅ 1.6 FastAPI app skeleton [02] — `main.py`: lifespan creates Qdrant collections + Redis client; `/health` pings Postgres, Redis, Qdrant (2s timeout each), 503 + `degraded` if any is down · tests: `tests/test_health.py`; verified live with Redis stopped

### Phase 2 — Pipeline core
- ✅ 2.1 Typed state schemas — `graph/state.py`: one Pydantic model per handoff (`TaskGraph`, `SourceRef`, `ResearchResult`, `EntityGraph`, `EvidenceBundle` (rejects over-budget), `Draft`/`ClaimRef`, `GroundingResult`, `FactCheckSummary`) + `PipelineState` with a merge reducer for parallel research results · tests: `tests/test_state.py`
- ✅ 2.2 Scratch store + `read_source` tool [17] — `store/scratch.py` (stable `src-…` ids per run+URL, upsert, run-scoped reads), `tools/read_source.py` (built per run, paged 4k chars, output wrapped by 03c), `llm/tokens.py` (cl100k estimate +10% margin) · tests: `tests/test_scratch.py`
- ✅ 2.3 Untrusted-content wrapper [03c] — `guardrails/untrusted.py`: drops scripts/comments/hidden elements before extraction, strips zero-width/bidi/tag chars, flags injection phrases, wraps in `<untrusted_source>` with breakout escaping · tests: `tests/test_untrusted.py`
- ✅ 2.4 Web Search tooling [07] — `tools/search_providers.py` (Tavily/Brave/SerpAPI + circuit breaker), `tools/fetch.py` (full bodies, paywall/402/login-wall skip, 5-hop redirect re-check, 2 per domain, 3 MB cap), `tools/credibility.py` (0.5 domain + 0.3 recency half-life 1y + 0.2 byline), `tools/web_search.py` (`WebResearcher`: top 5 that succeed, replaces skipped, stores to 17, logs `skipped_url`/`injection_flag`; agent sees ≤400-char wrapped leads only), `guardrails/net.py` (public-IP guard, reused by 03d), `store/run_log.py` · tests: `tests/test_web_search.py` (all offline via mock transport). Subagent spec is built in 2.5.
- ✅ 2.5 Planner deep agent + `task` subagents [06] — `agents/planner.py` (`plan()`: structured `TaskGraph`, drops disallowed source types, caps at 6, one clarifying question max; `persist_task_graph()`; `build_research_agent()` = `create_deep_agent` + `write_todos` + call limit + model fallback), `agents/delegation.py` (`SubTaskDelegationMiddleware`: Planner delegates by id only, middleware swaps in the fixed prompt, refuses unknown/duplicate/wrong-type incl. general-purpose, per-sub-task timeout; `ResearchRecorder`: drops invented source ids, mirrors status to `sub_tasks`, marks undelegated as Failed), `agents/web_search.py` (subagent spec: web_search + read-only read_file, tool-call limit, fallback, `ToolStrategy(ResearchResult)`), `agents/prompts.py`, `gateway.structured()` · tests: `tests/test_planner.py` with rule-based fake LLMs in `tests/fakes.py` (brief never reaches researchers; Planner never sees page bodies)
- ✅ 2.6 LangGraph pipeline PLANNING → RESEARCHING + CLI [05] — `graph/pipeline.py` (`load_context` → `plan` → `clarify` (interrupt/resume, once) → `research`; run status mirrored to `runs`; `execute()` enforces the run time budget and turns token/time/planning failures into FAILED with a message), `store/memory.py` (last-3 summaries via ordered Qdrant scroll; `save_summary` for Phase 4), `cli.py` (`uv run insightforge "<brief>"`, prints task graph, refs, tokens, corpus vs. handed-up tokens) · tests: `tests/test_pipeline.py` incl. pause → new process → resume through the real Postgres checkpointer · **verified live** (see log)

### Phase 3 — Docs + entities
- ✅ 3.1 Docling ingest → Qdrant [19] — `tools/docling_ingest.py`: 01a checks (≤10 files, ≤20 MB, `%PDF` magic, filename sanitized, server-chosen storage name), Docling → HybridChunker → chunks cleaned + injection-scanned (03c) → `doc_chunks` with `{source_filename, page_number, section_heading, run_id, document_id, text, injection_flags}`; `Document` row; corrupt/empty PDFs → `PdfRejected` (run continues) · `store/sources.py` SourceResolver (src + chk ids, ACL) · tests: `tests/test_docs.py` incl. a real Docling parse of a generated PDF
- ✅ 3.2 Doc Reader agent [08] — `tools/doc_search.py` (`search_documents`: top-k=8, run + user filter, min relevance 0.3, returns chunk_id/source/page/section/relevance_score + 400-char wrapped excerpt), `agents/doc_reader.py`; registered only when the run has ingested PDFs; recorder fills kind + credibility from storage · tests: delegation end to end, invented chunk ids dropped, chunk text never reaches the Planner
- ✅ 3.3 Entity Extractor agent [09] — wave 2 (`entities` pipeline node): Planner delegates `entities` via `task`; subagent receives **source ids only**; `extract_entities` tool does one extraction call per source (`SourceExtraction`, fixed predicate set), accumulates in code. `tools/entities.py` merges deterministically: normalized keys (`company:salaryhero`), money parsing, ±5% tolerance, conflicts on single-valued predicates per qualifier → `conflict: true` + `EntityGraph.conflicts`. Graph saved to `runs.entity_graph` (migration `run entity graph`); company/person/product embeddings upserted to Qdrant `entities` per (user, key); Planner gets related past entities (`load_related_entities`). Wave failure degrades (logged), budget breach still fails the run · tests: `tests/test_entities.py`, `tests/test_entity_wave.py`, pipeline tests

### Phase 4 — Synthesis → report
- ⬜ 4.1 Synthesis + budget compressor [10]
- ⬜ 4.2 Writer + grounding check + retry [11, 03b]
- ⬜ 4.3 Fact-Checker [12]

### Phase 5 — API + SSE
- ⬜ 5.1 Input guardrails [03a]
- ⬜ 5.2 Run queue (arq) [04]
- ⬜ 5.3 `/research/run`, `/research/status` (SSE), clarification resume [15]
- ⬜ 5.4 `/reports` list/get/export (PDF, md) + share links

### Phase 6 — Scheduler
- ⬜ 6.1 Watchlist CRUD + scheduler worker [20]
- ⬜ 6.2 Diff Agent [21] + alerts
- ⬜ 6.3 Webhook guard + Slack delivery [03d]

### Phase 7 — Frontend (Next.js)
- ⬜ 7.1 Brief input panel
- ⬜ 7.2 Live progress view
- ⬜ 7.3 Report viewer + citation drawer
- ⬜ 7.4 Watchlist dashboard + alert feed
- ⬜ 7.5 Export + share

### Phase 8 — Evals + README
- ⬜ 8.1 Five sample briefs
- ⬜ 8.2 Eval runner + report
- ⬜ 8.3 README + context-engineering writeup

### Phase 9 — Bonuses
- ⬜ LangSmith · contradiction-resolution UI · citation trace

## Log

- 2026-10-18 — Phase 1 started. Backend package initialized with uv. 1.1 config done (3 tests passing).
- 2026-10-18 — 1.2 docker-compose up and healthy. 1.3 schema + migration done (4 tests passing).
- 2026-10-18 — Qdrant server bumped to v1.19.1 to match qdrant-client 1.19. 1.4 vector store + embeddings done (7 tests passing).
- 2026-10-18 — 1.5 model gateway done (14 passing, 1 live test skipped). Note for 2.5: deepagents needs a real chat model, so pass `gateway.chat_model(role)` and add fallback via agent middleware there.
- 2026-10-18 — 1.6 FastAPI skeleton done. **Phase 1 complete** (15 passing, 1 live test skipped; ruff clean).
- 2026-10-18 — Phase 2 started. Added langgraph 1.2, deepagents 0.7, trafilatura, tiktoken. 2.1–2.3 done (28 passing). Design note for 2.5: deepagents copies parent non-message state into subagents and the Planner LLM writes the `task` description, so the Planner will delegate by sub-task id and middleware will rewrite it into a fixed prompt from the stored `SubTaskSpec` (isolation enforced in code, not by prompt).
- 2026-10-18 — 2.4 web search tooling done (43 passing; ruff clean). Removed literal invisible characters from source that the Write step had produced; the regex now uses explicit escapes.
- 2026-10-18 — 2.5 Planner + delegation done (51 passing). deepagents insists on keeping `read_file` (used to offload huge tool results); all write tools removed.
- 2026-10-18 — 2.6 pipeline + CLI done; **Phase 2 complete offline** (57 passing, 1 live test skipped; ruff clean). Added psycopg[binary] for the LangGraph Postgres checkpointer. Live run blocked on API keys.
- 2026-10-18 — **Live check passed** (61 passing incl. live OpenRouter test; ruff clean). Payroll-SEA brief: 5 sub-tasks, all Done, 19 refs, 53 s wall, 83k tokens; raw corpus in scratch store ≈ 109,731 tokens vs. ≈ 1,421 handed to the Planner (~77× reduction; use this in the README context-engineering writeup). Fixes found by the live run:
  - OpenRouter's Claude models reject forced `tool_choice` ("any"/"tool"). Gateway `structured()` now uses `method="json_schema"`; researcher uses `ProviderStrategy(ResearchResult)` instead of `ToolStrategy` (probed both tiers: tools + native response_format works). **Every future agent with a response schema must use ProviderStrategy.**
  - `html_to_text` crashed on comments before `<html>` (parentless node) → guarded; hidden root returns "". Fetcher now turns any parser exception into `Skipped("parse_error")` so one bad page can't fail a sub-task.
  - `execute()` catch-all: unexpected errors (provider 4xx, bugs) end the run FAILED with the reason instead of crashing. Delegation now logs subagent tracebacks.
- 2026-10-18 — **Phase 3 complete and verified live** (85 passing; ruff clean). Added docling 2.140 (+ PyTorch, venv ≈ 1.4 GB) and fpdf2 (dev, test PDFs). Refactored delegation into a generic Ledger so wave 2 reuses the same isolation enforcement.
  - Live: brief + synthetic analyst PDF (`USD 9M` Series A planted vs. web's `US$7.6M`). Planner created a `docs` sub-task; 4/4 sub-tasks Done; entity graph 96 entities / 120 facts / **1 conflict, exactly the planted one** (4 web sources vs. 1 chunk). 106 s, 142k tokens.
  - **Watch in Phase 4:** tokens rose 83k → 142k (entity extraction over ~14 sources × ≤12k chars). Budget is 250k and synthesis/writer/fact-check still to come: measure per-stage tokens first; levers are `ENTITY_SOURCE_MAX_CHARS` and skipping low-credibility sources.
  - Alembic autogenerate tried to DROP LangGraph's `checkpoint_*` tables; `migrations/env.py` now excludes them (`include_object`). Always review autogenerated migrations.
