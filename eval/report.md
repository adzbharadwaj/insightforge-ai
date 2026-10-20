# Evaluation report

Generated 2026-10-20 from `eval/results/*.json` by `uv run insightforge-eval --report`.

Runs evaluated: 1 of 5 sample briefs (see `briefs/briefs.json`).

| Brief | Status | Fact-check | Citation accuracy | Quality (1-5) | Must-cover | Tokens | Time |
|---|---|---|---|---|---|---|---|
| ev-charging-eu | COMPLETE | 6/7 (86%) | 37/37 (100%) | 3.8 | 100% | 217,379 | 218 s |

## ev-charging-eu

**Brief:** European EV fast-charging networks: the largest operators by number of high-power chargers, typical pricing per kWh, and expansion plans announced in the last 12 months.

Run `ca38f566-8158-4574-b4c8-b694a6a76fae` · models anthropic/claude-sonnet-5.5, anthropic/claude-haiku-5.5 · attempts 1 · writer attempts 3 · sub-tasks 4/4 done

**Context engineering:** raw corpus 121,382 tokens → Planner received 1,230 → Writer received a 2,518/6,000-token evidence bundle (48x smaller than the corpus; 39 evidence items from 15 sources; 0 compression cut(s)).
- Excluded by time horizon: st-2: excluded 1 source(s) published before 2025-07-27 (outside 'last 12 months'): numerama.com
- Excluded by time horizon: st-4: excluded 1 source(s) published before 2024-12-14 (outside 'last 18 months'): transportenvironment.org

**Tokens by stage:** entities 62,423, ground 49,007, research 59,295, plan 2,063, synthesize 15,341, write 29,250, fact_check 0

**Synthesis quality (rubric judge):**

- coverage: 3/5 — Covers operator counts, per-kWh pricing and expansions and adds AFIR, but operator ranking by high-power chargers is not answered and several operators lack data.
- structure: 4/5 — Follows the expected sections in order with clear headings and a dedicated gaps section, though the policy section is titled differently from the brief's expected label.
- faithfulness: 3/5 — Most claims carry dates, sources and credibility scores, and conflicting figures are shown side by side, but some sources are weak and a few claims (e.g. the 180% exceedance and 79% figure) are single-source with no corroboration.
- gaps honesty: 5/5 — The gaps section plainly states missing operator rankings, unresolved Spain and Hungary pricing conflicts, undated EnBW target, and excluded out-of-window sources.
- clarity: 4/5 — Concise and readable with confidence labels per section, though the executive summary repeats body content and some paragraphs are dense with inline citations.

**Structure checks:** has executive summary: True, summary within 150 words: True, has gaps section: True, every section has confidence: True, uses citation format: True, thematic sections: 4, must cover rate: 1.0

## Not yet run

Run with `uv run insightforge-eval --brief <id>`: `payroll-sea`, `bnpl-regulation`, `glp1-market`, `vector-db`.
