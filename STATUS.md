# Build Status — InsightForge AI

Single source of truth for where the build is. Update after every component.

- **Plan:** `~/.claude/plans/git-init-eager-snowflake.md`
- **Architecture diagram:** `docs/architecture.html` · https://claude.ai/artifact/52efizCUMZiLyaEzeRotdt
- **Style rules:** `architecture-style-guide.md` (git-ignored, local only)

## Resume here

**Current phase:** 9 — Bonuses **done** (all phases complete). 183 tests passing; bonuses browser-checked on real data.
**Next step (optional):** run the other 4 evaluation briefs (`uv run insightforge-eval --brief <id>`), README screenshots.
**Run the UI:** `cd frontend && npm run dev` (http://localhost:3000; API at `NEXT_PUBLIC_API_URL`, default http://localhost:8000) — start the API, worker and (optionally) scheduler first. Whole stack in Docker: `docker compose --profile app up -d --build`.
**Resume a failed run:** `uv run insightforge --resume <run-id>` · API `POST /research/{id}/retry` · the worker auto-resumes retryable failures once.
**Models:** back on Claude (`anthropic/claude-sonnet-5.5` / `-haiku-5.5`). Cheap verified alternative in `.env` comments: `deepseek/deepseek-v4.1-flash` / `z-ai/glm-5.3-flash`.
**Optional:** `brew install pango` to enable PDF export (otherwise `/export?format=pdf` returns 501 with this hint).

```bash
docker compose up -d                 # Postgres, Redis, Qdrant (from repo root)
cd backend
uv sync                              # Python 3.12 venv + deps
uv run alembic upgrade head          # apply DB migrations
uv run pytest -q                     # all tests should pass (DB tests skip if Postgres is down)
uv run uvicorn insightforge.main:app --reload   # API on :8000 (/docs, /health); needs JWT_SECRET
uv run arq insightforge.worker.WorkerSettings    # run worker (separate process): executes queued runs
uv run insightforge-scheduler                    # watchlist scheduler (separate process)
uv run insightforge "<brief>" [--pdf f.pdf]      # CLI: one full run in-process (needs keys)
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
- ✅ 4.1 Synthesis + budget compressor [10] — `agents/synthesis.py`: resolve refs → per-source passage selection by embedding similarity to the sub-task (3/source) → one claim-extraction call per sub-task (`SubTaskClaims`, claim ↔ passage) → dedupe (same entities + same numbers + cosine ≥ 0.9; most credible source first) → confidence (conflict → LOW, 2+ independent publishers → HIGH, single ≥ 0.6 → MEDIUM) → rank (month → credibility → relevance to brief) → compress tail of largest section via SUMMARIZER until ≤ 6k tokens (drop as last resort), each cut logged (`kind="cut"`). Bundle carries `sources` (citation metadata) and each item's exact `passage` (citation drawer, never shown to the Writer)
- ✅ 4.2 Writer + grounding check + retry [11, 03b] — `agents/writer.py`: model returns `WriterOutput` (sentences + evidence ids); code renders Markdown, `[Source: domain, date, credibility: x]`, `**Confidence: X**` from the bundle, Gaps. `guardrails/grounding.py`: deterministic (ids exist, numbers ⊆ cited evidence, uncited numbers, sections match, summary ≤ 150 words) + LLM entailment in batches of 15; retry with explicit constraints (max 2), then drop + note in Gaps. `guardrails/output.py` PII redaction (emails, +intl phones)
- ✅ 4.3 Fact-Checker [12] (live: 6/8 on run 15; target ≥ 80% assessed in Phase 8 eval) — `agents/fact_checker.py`: sample max(20%, 5) section claims (seeded), live re-fetch (stored copy fallback, recorded), 1–2 sentence windows → temp Qdrant `factcheck_{run}` → best cosine vs the claim's own sources ≥ 0.85, `[UNVERIFIED]` inline, collection always dropped. `store/reports.py` saves report + claims; session summary → `report_summaries`
- Pipeline: `entities → synthesize → write ⇄ ground → fact_check → finalize`; per-stage tokens in `state.stage_tokens` and `orchestrator/tokens` run events

### Phase 5 — API + SSE
- ✅ 5.1 Input guardrails [03a] — `guardrails/input.py`: brief 10–500 chars after `clean_text`, injection scan → 422, source-pref sanity (docs ⇔ files, ≥ 1 source), PDF checks reused; LLM scope/policy check (`ScopeVerdict`: off-topic / private person / harmful) on a meter-less gateway; if the check can't run → 503, nothing queued (fail closed). Also: clarify answers are cleaned + injection-scanned
- ✅ 5.2 Run queue (arq) [04] — `runtime.py` (shared by worker + CLI: pooled checkpointer, shared HTTP/Qdrant/provider/fetcher, per-run gateway whose meter resumes from `runs.tokens_used`), `worker.py` (`start_run` ingests uploads then runs, `resume_run`, cron `cancel_stale_clarifications` every minute using `runs.clarification_asked_at`; `max_jobs = max_concurrent_runs`, job timeout = budget + 120 s; skips runs cancelled while queued)
- ✅ 5.3 Research API + SSE [15] — `api/research.py`: `POST /research/run` (multipart, 202 + ETA; per-user active-run cap 2 + 10/h Redis limit), `GET /research/{id}`, `GET /research/status/{id}` (SSE via `api/sse.py`: `snapshot` / `log` / `end`, resumes with `Last-Event-ID`, `?token=` for EventSource, progress only), `POST /{id}/clarify` (conditional update → one answer wins, 409 otherwise), `POST /{id}/cancel` (queued/waiting only). `api/auth.py` JWT (user scope, 24 h) + `POST /auth/dev-login` (dev only, `ALLOW_DEV_LOGIN`)
- ✅ 5.4 Reports [15] — `api/reports.py`: list, get (`ReportView`: markdown + `structure` (sentences with evidence ids, for clickable citations) + sources + evidence passages + fact-check), export md / pdf (`export/render.py`, WeasyPrint lazy → 501 without pango), share links (`report:read` JWT, 7 days, `jti` revocation → 410; login tokens rejected on /share). Migration adds `reports.structure`, `runs.clarification_asked_at`
- Not done here (Phase 8): Dockerfiles + compose services for api/worker/frontend

### Phase 6 — Scheduler
- ✅ 6.0 Live E2E on Claude (see log)
- ✅ 6.1 Watchlist CRUD + scheduler worker [20]
- ✅ 6.2 Diff Agent [21] + alerts
- ✅ 6.3 Webhook guard + Slack delivery [03d]

### Phase 7 — Frontend (Next.js)
- ✅ 7.1 Brief input panel
- ✅ 7.2 Live progress view
- ✅ 7.3 Report viewer + citation drawer
- ✅ 7.4 Watchlist dashboard + alert feed
- ✅ 7.5 Export + share

### Phase 8 — Evals + README
- ✅ 8.1 Five sample briefs
- ✅ 8.2 Eval runner + report
- ✅ 8.3 README + context-engineering writeup

### Phase 9 — Bonuses
- ✅ LangSmith · contradiction-resolution UI · citation trace

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
- 2026-10-20 — Committed phases 1–3 (`10841f8`). Phase 4 built (106 passing). Live tuning findings:
  - Claude 5.5 on OpenRouter reasons adaptively by default; hidden reasoning ate 1,358 of grounding's 2,000 max_tokens → LengthFinishReasonError. Sonnet rejects `reasoning.enabled=false` ("mandatory"), so narrow roles use `reasoning.effort="low"` (accepted by both); Planner/Writer keep the default. Grounding max_tokens 5k, batches of 15.
  - Entity extraction cost 150k tokens for 23 sources (6.5k/source, 220 mostly-noise entities) → per-source text 12k → 5k chars, ≤15 entities/facts.
  - Synthesis stalled 6.5 min: 60 s timeout × 3 attempts × 2 models on long Sonnet outputs. Now per-role timeouts (`RoleSpec.timeout_s`, Writer 240 s), 1 retry, claim extraction on Haiku.
  - After the 480 s budget fired, the CLI process stayed alive (cause not yet found). Added `kill -USR1 <pid>` stack dumps (faulthandler) and line-buffered output to diagnose if it recurs.
  - deepagents adds SummarizationMiddleware to the Planner and every subagent by default (assignment's auto-summarization requirement).
  - Model-written fields now clip to their limits instead of failing (`llm/schema.py::clip`): native JSON schema treats maxLength as a hint; a 350-char scope failed a run.
  - Embedding was the wall-time hog (78 ms per 1,000-char passage on CPU; 5 parallel calls thrashed: 133 s sys). Keyword shortlist before embedding (`tools/lexical.py`, 8 paragraphs/source in synthesis, 60 windows/source in fact-check) + one ONNX inference at a time → synthesis 2 min 46 s → 6 s.
  - Entity agent sometimes answered without calling its tool (0 entities). Fail-safe: unprocessed assigned sources are extracted directly after the agent returns (logged as `retry`).
  - Extraction calls had ~2.2k tokens fixed overhead vs ~1.1k of source → 4 sources per call (`BatchExtraction`, results keyed by source id), max 20 most credible sources per run.
  - Run budget 250k → 300k, derived from measured stage costs (see `config.py`). Writer retries are budget-aware: if a rewrite can't be afforded, unsupported claims are dropped instead of failing the run.
  - Grounding judge: reason before verdict (it had produced "unsupported" with reasons saying supported); sees each item's publisher (Writer attributes facts to publishers); superlatives the evidence doesn't state are unsupported. Writer: state facts directly, no "the evidence shows / one item states".
  - Fact-check: true atomic claims scored 0.82–0.85 against whole sentences. Claim-sized word windows (12/20/30 words) lifted 4 of 5 to 0.86–0.91 (measured offline on run `cf23ad40…` sources: 5/10 → 9/10). Threshold unchanged at 0.85.
  - Live results: run 11 COMPLETE 196 s / 231k tokens / fact-check 6/8; run 12 COMPLETE 256 s / 315k / 5/10 (before the window fix); run 13 FAILED on budget during a rewrite (→ budget-aware retry); run 14 stopped by OpenRouter 402 (credits).
- 2026-10-20 — Committed Phase 4 (`39ec025`). **Phase 5 done** (132 passing). Smoke test with real processes: health ok; dev-login as the CLI user; 3 stored reports listed; report view with 86 evidence passages; md export; pdf → 501 (no pango); share link opens without login; SSE replay of a finished run (92 log + snapshot + end); arq worker boots with 3 functions; API and worker stop cleanly.
  - `.env` had empty `JWT_SECRET` / `WEBHOOK_SIGNING_SECRET` (copied from `.env.example`). `config.require_secret` now rejects empty/placeholder/short secrets; the API checks at startup. Tests set their own secret (autouse fixture).
  - Pausing for clarification now saves `tokens_used`, so a resume in another worker process continues the same budget.
  - email-validator rejects reserved domains like `.local` (the CLI's dev user) → dev-login uses a format check instead.
- 2026-10-20 — Secrets set in `.env` (64 chars each; API boots with real config). OpenRouter topped up. **Run 15 (Phase 4 live check):** COMPLETE 189 s, 230k tokens (plan 3k, research 77k, entities 70k, synthesis 20k, write 22k, ground 38k), 2 writer attempts, 94 entities, fact-check 6/8 (fails at 0.849 / 0.84, both paraphrases → `[UNVERIFIED]`). **README context-engineering example:** raw corpus ≈ 202,171 tokens in the scratch store; Planner received ≈ 1,561 tokens of references; Writer received a 3,666-token evidence bundle (budget 6,000). Report: `data/reports/fb6c74b1-5c5e-4408-ac27-29859d841a5c.md`.
- 2026-10-20 — **Model experiment (open weights).** Probed OpenRouter: free models blocked by the account's privacy/ZDR guardrail (404). With native JSON alongside tools, GLM 5.3 Flash skipped its tool and invented an id; DeepSeek V4.1 Flash hung. Both work with a forced structured-output tool call → `Settings.agent_output_strategy` ("auto": provider for anthropic/*, tool otherwise). Run 16 on DeepSeek/GLM: COMPLETE 185 s, 140k tokens, 1 writer attempt, fact-check 7/8, ~1.5 ¢. But on the watchlist E2E, DeepSeek's writer took ~2.5 min per call; two rewrites hit the 480 s budget → FAILED (173k tokens). **Switched back to Claude.** `provider.require_parameters` is applied to open models only: on this account (zero data retention) it leaves no Claude host (404).
- 2026-10-20 — **Phase 6 built** (152 passing): `guardrails/webhooks.py` (03d: HTTPS + public-IP check at save and at send, no redirects, HMAC-SHA256 `X-InsightForge-Signature` over `timestamp.body`, 10 s timeout, 3 attempts with backoff on 5xx/429/network, Slack format for hooks.slack.com), `agents/diff.py` (Diff Agent: embedding match; changed figures caught in code; ≥ 0.92 same numbers = rephrase; rest → one batched materiality call; first report = baseline), `watchlist/service.py` (start run per item with busy guard; `process_watch_result` → alert + webhook, called by the worker after a COMPLETE watchlist run), `watchlist/scheduler.py` (`insightforge-scheduler`: APScheduler, Postgres job store `apscheduler_jobs`, coalesce + max_instances=1 + misfire grace, reconcile every 60 s, new items fire immediately), `api/watchlist.py` (CRUD with brief guardrails + webhook check, run-now, latest diff, alert feed, mark read). Alembic ignores `apscheduler_jobs`. Fixed a test fixture that turned any OSError inside a test into a skip.
  - Live check so far: scheduler booted, reconciled, picked up an API-created item and queued its run; the next firing was correctly skipped while that run was busy; SIGTERM shutdown clean. The run itself failed on the time budget (DeepSeek writer, see above), so the diff/alert step has not run live yet.
  - **Open issue:** writer retries are budget-aware for tokens but not for time. (Fixed below.)
- 2026-10-20 — **Switched back to Claude.** `provider.require_parameters` now applies to open models only (on this zero-data-retention account it left no Claude host → 404). Live smoke test passes.
- 2026-10-20 — **Checkpointing (user requirement: a failed agent must not restart the run).**
  - Stage level: LangGraph saves state after every stage; `execute(..., None)` resumes from the last one, `durability="sync"` (written before the next stage starts). Failures are classified: time budget / provider errors → `retryable`; planning errors, no evidence, token budget → not auto-retried.
  - Agent level: each researcher's Done `ResearchResult` → `sub_tasks.result`; each source's extraction → `entity_extractions`. A resumed research stage re-delegates only unfinished/failed sub-tasks; a resumed entity wave extracts only missing sources. `persist_task_graph` is idempotent.
  - Entry points: `Runtime.retry` (fresh time + token budget for the attempt; `runs.attempts` counts up; no checkpoint → start over), worker auto-resume (`max_auto_resumes=1`) and `retry_run` job, `POST /research/{id}/retry` (FAILED only, ≤ `max_run_attempts`=3, conditional update), CLI `--resume RUN_ID`.
  - Time-aware rewrites: `execute` sets a deadline; stages record `stage_seconds`; `ground` skips a rewrite when another write + grounding check (+ 90 s fact-check reserve) won't fit, dropping unsupported claims instead.
  - **Live proof (run 17, Claude):** attempt 1 with `RUN_TIME_BUDGET_S=100` died during writing after plan/research/entities/synthesis (164k tokens of work). `--resume` → attempt 2 re-ran only write/ground/fact_check (3 writer attempts, 0 unsupported at the end) → COMPLETE in 125 s; finished stages untouched (identical stage token counts), ~62% of the run's tokens reused.
- 2026-10-20 — **Phase 6 live E2E on Claude** (API + worker + scheduler, interval override 300 s): scheduler queued the topic; run 1 COMPLETE (284k tokens) → Diff Agent "saved as the baseline", no alert; next firing skipped while busy; run 2 COMPLETE (273k) → diff → alert created (webhook not configured: needs a public HTTPS receiver). Startup double-fire (stale job misfire + interval-change replace) was absorbed by the busy guard; reconcile now anchors rescheduled jobs to `last_run + interval`.
  - **Diff Agent fixes from real data:** "changed" paired *different companies* (Gajiku vs GIMO seed rounds) and *different facts* about one company (PayrollPanda funding vs customer count). Now: a change needs a shared distinctive subject (places/filler excluded); code decides only near-identical sentences (≥ 0.92) with different figures; looser same-subject pairs go to the judge, which classifies new / changed / removed / not_material (a "change" that is really a new fact becomes an addition). Removals alone no longer alert. On the same two live reports: changed 10 → 1. The remaining 25 added / 23 removed reflect research variance between runs.
- 2026-10-20 — **Phase 7 frontend** (`frontend/`, Next.js 16.4 + React 19.3 + Tailwind 4, Cache Components on): sign-in gate (dev login), brief panel (500-char counter, drag-drop PDFs with client checks, source toggles, ETA from `/research/estimate`), live run view (SSE snapshots + activity log, sub-task rows, progress bar, elapsed vs estimate, inline clarification, retry-from-checkpoint), report viewer from `reports.structure` (sticky TOC, confidence badges, clickable citation chips → drawer with title/date/credibility/exact source paragraph/URL, amber `[UNVERIFIED]` with tooltip, fact-check panel, md/pdf export, copy share link; markdown fallback for older reports), public `/share/[token]`, reports list, watchlist dashboard (add/pause/remove/run-now, one-click diff, alert feed, mark read). Dynamic routes: server page + `<Suspense>` + `use(params)` in a client component (Cache Components rule); EventSource closed in effect cleanup (Activity keeps hidden routes mounted). `npm run build` + lint clean. Browser-checked against live data: watchlist + diff + alert feed, report with drawer, progress replay, no console errors. Fixed from the check: drawer closes on Escape and takes focus.
- **Phase 8 backlog (quality, found live):** (1) time horizon isn't enforced: a "last 12 months" report cited a 2022 article (the Writer flagged it under Gaps) → drop/flag evidence older than the sub-task horizon in synthesis; (2) diff noise from research variance → compare additions against all earlier reports for the item, not just the last; (3) ETA formula under-estimates Claude runs (est ~3 min, actual ~4 min) → recalibrate from eval runs; (4) fact-check pass rate across the 5 briefs (single runs: 75–100%).
- 2026-10-20 — **Phase 8.** Time horizons enforced in synthesis (dated sources older than horizon × 1.25 excluded, logged as cuts, passed to the Writer as notes for Gaps). Sample brief library `briefs/briefs.json` (5 industries / question types, expected sections, must-cover topics). Evaluation runner `insightforge-eval` (`evaluation.py`): fact-check pass rate, citation accuracy over every claim (primary-model judge, sees publishers), rubric synthesis quality + deterministic structure/coverage checks, context and cost figures; results in `eval/results/`, report `eval/report.md`.
  - **One live evaluation run (user's choice): `ev-charging-eu`** run `ca38f566…`: COMPLETE 218 s, 217k tokens, fact-check 6/7 (86%), citation accuracy 37/37 (100%), quality 3.8/5, must-cover 100%; corpus 121,382 → Planner 1,230 → Writer 2,518 tokens; 2 sources excluded by time horizon. First scoring showed 34/37: all 3 were attributions ("RAC reports…") the judge couldn't verify without publisher names → evaluator now shows publishers; re-scored the same run (no new research run).
  - README (setup, keys, env reference, Mermaid architecture + docs/architecture.html, walkthrough, context-engineering writeup with run 15's 202k → 1.6k → 3.7k example, multi-agent design, guardrails, evaluation, limitations), `docs/sample-report.md`. Dockerfiles (backend: pango + libGL for PDF export/Docling; frontend), compose `app` profile (api applies migrations; worker/scheduler wait for api health), `.dockerignore`.
- 2026-10-20 — **Docker verified.** Backend image was 21.3 GB (PyPI's Linux PyTorch bundles CUDA) → torch/torchvision from the CPU-only index on Linux (`[tool.uv.sources]`, macOS unaffected) → **5.35 GB**. `docker compose --profile app up`: migrations applied, API healthy, UI 200, worker + scheduler up, PDF export works in the container (pango installed).
  - **Incident during the smoke test:** the container worker picked up (a) the watchlist E2E's third run, whose worker I had killed earlier, re-delivered by arq 40 min later and restarted from scratch, and (b) a run fired by the scheduler from a persisted job that still had the E2E's 300 s interval (APScheduler fired the misfire before reconcile ran). Stopped both within ~2 min; cancelled them (one had used 0 tokens; the other's 300k were mostly from its earlier attempt).
  - **Fixes:** worker jobs act only on runs in the state they were queued for (`start_run` QUEUED, `resume_run` PLANNING, `retry_run` QUEUED/FAILED), so a re-delivered job never restarts a run; scheduler starts paused → reconcile → resume; new cron `fail_orphaned_runs` (every 5 min) marks runs with no run-log activity for budget + 5 min as FAILED (retry resumes from checkpoint; never auto-resumed). The test watchlist topic was paused so it can't fire daily Claude runs. Note: housekeeping tests act on the shared dev DB (they failed the genuinely orphaned Oct-18 dev runs).


- 2026-10-20 — **Phase 9 bonuses** (no paid runs; tests + browser on real data):
  - Contradiction resolution: opt-in `review_conflicts` (form toggle / `--review-conflicts`); `review` node after synthesize builds options from entity conflicts with ≥ 2 sides in the bundle (`agents/conflict_review.py`), interrupts (`AWAITING_REVIEW`, new enum value + `runs.review`/`review_requested_at`), `POST /research/{id}/resolve` (validated, conditional update) → worker `resolve_run` (status-guarded); chosen side kept as analyst-confirmed (MEDIUM, conflict cleared), others dropped as cuts, notes for Gaps; cron `continue_unreviewed` keeps both after 30 min. Browser-checked with a staged paused run (Allego 34,000 vs 35,000 from the eval report), then the staged run and its job were deleted.
  - Entity graph visualizer (report tab, dependency-free force layout; first layout pushed unlinked nodes to the walls → local repulsion + stronger gravity), node panel with facts + publishers + evidence that opens the drawer. Report API now returns `entity_graph` and evidence `entity_keys`.
  - Source credibility tuning: blocked domains never fetched (logged), deprioritized credibility × 0.5; normalized from free text; stored in `source_prefs`.
  - Citation trace: cited sentences are clickable → drawer with the exact source paragraph. Also visible in the check: amber `[UNVERIFIED]` on a real failed claim.
  - Adaptive bundle budget (opt-in), LangSmith documented.
  - Test infra: autouse fixture resets the global engine pool per test (fixed order-dependent "different loop" failures).
