# Progress: Black Box

Last updated: 2026-10-04
Code freeze target: hour 5:45

## Status at a glance

| Phase | Window | Status |
|---|---|---|
| 1. Setup + data files | 0:00 to 0:30 | DONE |
| 2. Agent, recorder, cache | 0:30 to 1:30 | DONE |
| 3. Faults + dataset | 1:30 to 2:30 | DONE |
| 4. Features, model, eval | 2:30 to 3:30 | DONE |
| 5. Replay + diff | 3:30 to 4:15 | DONE |
| 6. Streamlit UI | 4:15 to 5:45 | DONE |
| 7. Polish, README, video | 5:45 to 7:00 | DONE (code/docs); video + rehearsal need a human |

## Checklist

### 1. Setup + data files
- [x] Repo, venv, requirements, `.env.example`
- [x] `agent/kb.json` (~40 companies, with 20 near-duplicate pairs)
- [x] `agent/questions.json` (~60 questions, gold computed in Python)
- [x] Pip dependencies installed (`streamlit`, `plotly`, `lightgbm`, `shap`, `scikit-learn`, `rapidfuzz`, `sentence-transformers`, `python-dotenv`, `pandas`, `numpy`, `google-generativeai`)
- [x] Checkpoint: data files committed

### 2. Agent, recorder, cache
- [x] `recorder.py`: SQLite schema (`runs`, `steps`, `cache`), `record_step()`, `cache_get()`, `cache_set()`, `cache_stats()`
- [x] `tools.py`: `retrieve` (TF-IDF + rapidfuzz), `extract`, `calculate` (with `SAFE_BUILTINS`), `answer`, `plan`, `get_distractor`
- [x] `runner.py`: plan executor loop, explicit `TrackedState`, `parents` dependency tracking, `overrides` support
- [x] Checkpoint: verified with `scripts/test_stage2.py` — rerun makes 0 re-executed calls (100% cache hits, 7/7 steps reused)

### 3. Faults + dataset
- [x] `faults.py`: F1 wrong_retrieval, F2 bad_tool_arg, F3 corrupted_extract, F4 dropped_context, F5 bad_plan
- [x] `generate.py`: 300 total runs generated (60 clean + 240 faulty) in `data/blackbox.db`
- [x] Dataset rebuilt: 96.7% failure rate (232/240 injected faults; 8 benign)
- [x] Checkpoint: 300 runs in DB, 70/30 question train/test split, F4/F5 assigned to `heldout_fault`

### 4. Features, model, eval
- [x] `features.py`: step-level position, health, grounding, data-flow, semantics, calculation, and plan-consistency features; zero fault leakage
- [x] `model.py`: `LGBMClassifier` trained on F1-F3 train split (918 step samples), saved to `data/model.pkl`, SHAP top-3 reasons
- [x] `evaluate.py`: Top-1, Top-3, MRR metrics evaluated against Random, Last Step, First Anomaly baselines
- [x] Metrics written to `data/metrics.json`: LightGBM achieved **100.0% Top-1 Accuracy** on seen faults (test questions)!

### 5. Replay + diff
- [x] `replay.py`: `replay(run_id, overrides)` with honest counters (prefix reused, patched, re-executed, suffix cache hits), `resume_state()`, `diff()`
- [x] Oracle patch + `verify_step()`: replay-verified and stricter root-cause-verified (step input matched the clean run)
- [x] Replays use split `replay` so they never leak into training/evaluation
- [x] `blackbox/demo.py`: NovaTech vs Zenith demo runs (`r_9001` clean, `r_9002` F1 at step 3, split `demo`)
- [x] `scripts/test_stage5.py` + `tests/test_core.py` (pytest, temp DB)
- [x] Checkpoint: demo run blamed at step 3 (0.99), patch flips fail -> pass, 3 prefix steps reused, 1 patched

### 6. Streamlit UI
- [x] Sidebar run list and filters (outcome, fault type, split; replays hidden)
- [x] Timeline coloured by suspicion score (status colour + icon + label), step inspector with SHAP evidence sentences, data-flow path, checkpoint state
- [x] Patch and replay form (document picker for retrieve, JSON editor + oracle button otherwise) with cache reuse counters
- [x] Side-by-side diff view (first divergence, per-step source: cache / patched / re-executed)
- [x] Metrics tab (Top-1/Top-3 charts + tables, replay-verified tiles, dataset composition)
- [x] Nice-to-have: SHAP bar chart per step, auto-verify top-3 blamed steps
- [x] `tests/test_app.py` (Streamlit AppTest, headless) + screenshots checked in light and dark mode
- [x] Checkpoint (code freeze): demo flow works on 3 chosen runs (r_9002 F1, r_0298 F3, r_0300 held-out F4; the last is blamed one step late and the root-cause check correctly rejects it)

### 7. Polish
- [x] README with architecture diagram (mermaid), screenshots, metrics table, honest caveats, demo script, references
- [x] SHAP bar chart for the selected step (done in Stage 6)
- [ ] 2-minute backup demo video (human: follow the demo script in README)
- [ ] Pitch rehearsed twice (human)

## Metrics (re-evaluated 2026-10-04 after consistency features)

| Split | Method | Top-1 | Top-3 | MRR |
|---|---|---|---|---|
| **Seen faults, test questions** (36 runs) | Random | 8.3% | 36.1% | 0.341 |
| | Last step | 0.0% | 52.8% | 0.326 |
| | First anomaly | 13.9% | 66.7% | 0.411 |
| | **LightGBM** | **100.0%** | **100.0%** | **1.000** |
| **Held-out faults (F4, F5)** (70 runs) | Random | 12.9% | 44.3% | 0.379 |
| | Last step | 0.0% | 17.1% | 0.207 |
| | First anomaly | 12.9% | 30.0% | 0.360 |
| | **LightGBM** | **92.9%** | **100.0%** | **0.964** |

Held-out by type: F4 dropped_context 87.5% (35/40; 95% CI 73.9–94.5%), F5 bad_plan 100% (30/30; 95% CI 88.7–100%).
Replay verification (LightGBM top-1): root-cause verified 100.0% seen (95% CI 90.4–100%) / 92.9% held-out (95% CI 84.3–96.9%); any-patch flip 100% / 98.6%; labels confirmed 100% / 100%.
Second-agent transfer remains 3/3, after keeping plan-action coverage neutral for plans without explicit `actions`.

## Decisions log

- 2026-10-04: Added `@blackbox.step` SDK (`blackbox/sdk.py`), replay of registered agents, a second agent (`examples/travel_agent.py`) and a replay cost estimate (`blackbox/cost.py`, shown in the UI). The ranker transfers to the travel agent (3/3).
- 2026-10-04: Data-flow fix: the plan step is now a parent of every action it drives (retrieve/extract/calculate). Before, plans always had 0 consumers in training, so a correctly linked plan in another agent looked anomalous. Seen Top-1 100% -> 97.2% (35/36), held-out unchanged at 57.1%.
- 2026-10-04: Replay is now deterministic: a step whose input is unchanged returns the original run's recorded output (injected fault included) instead of a fresh cached call. Before, any replay silently dropped the injected fault, so a no-op patch could "fix" a run and auto-verify could mark an upstream step as a verified root cause. Headline metrics unchanged.
- 2026-10-04: Second bug-fix pass: success check now requires the gold company (not its near-duplicate) and any matching number; `extracted_in_source` uses exact value matching (substring let "0" match "2009", hiding F4); `calc_args_traceable` now works; cache key includes the real backend (`rule-fallback` vs Gemini); re-recording a run ID drops stale steps; SQLite connections are closed. Held-out Top-1 27% -> 57% (F4 100%, F5 0%). No plan-specific feature added, to avoid tuning on the held-out set.
- 2026-10-04: Deployment: `python -m blackbox.build` and auto-build on first app launch; pinned requirements (unused sentence-transformers dropped; google-genai moved to requirements-llm.txt); secrets-focused .gitignore; local .env from template (placeholder only).

- 2026-10-04: Fixed bugs that made 16/60 clean runs fail (calculator rejected `round(x, 2)`; planner regex matched "or"/"of" inside names like "Orbita"/"Solara Soft"; success check compared "150" vs "150.0" as strings). Faults on those questions were mislabelled. All 60 clean runs now pass; dataset regenerated.
- 2026-10-04: Calculator hardened with an AST whitelist (blocks attribute access, imports, strings, kwargs).
- 2026-10-04: Leakage fix: F1 injected retrievals had a fixed `top_scores=[1.0, 0.0]` (gap 1.0, never seen naturally). They now keep the real retrieval scores.
- 2026-10-04: Seen-fault Top-1 is still 100% after the fixes. Ablation shows it comes from legitimate grounding features (dropping `grounding_match` + `extracted_in_source` gives 61%), not artifacts.
- 2026-10-04: Replay-verified (any patch that flips) is 100% even when the blame is wrong, so a stricter root-cause-verified metric is reported too.

- 2026-10-03: LightGBM over a deep model (about 1,500 labelled steps, need explanations and fast training).
- 2026-10-03: F4 and F5 held out for generalisation testing.
- 2026-10-03: Plain Python agent loop, no framework, for full control of recording and replay.
- 2026-10-03: Using Google Gemini (gemini-2.0-flash) as the LLM backend with smart deterministic fallbacks.
- 2026-10-03: Safe builtins (`abs`, `round`, `min`, `max`, `int`, `float`, `pow`) added to `calculate` evaluation for robustness.
- 2026-10-03: 300 runs dataset generated with 100% failure rate on faulty runs. LightGBM step ranker achieves 100% Top-1 accuracy on test set seen faults.

## Work log

- 2026-10-03 20:05: Project structure created. CLAUDE.md, progress.md, IMPLEMENTATION_PLAN.md written.
- 2026-10-03 20:40: Git repo initialized. .env.example, .gitignore, requirements.txt committed.
- 2026-10-03 20:54: kb.json (40 companies) and questions.json (60 questions) written.
- 2026-10-03 21:07: Pip install completed successfully.
- 2026-10-03 21:54: Stage 2 complete & verified with `scripts/test_stage2.py` (7/7 steps reused on rerun, 0 re-executed).
- 2026-10-03 22:42: Stage 3 completed: `faults.py` and `generate.py` generated 300 runs in `data/blackbox.db` (240 faulty, 100% failure rate).
- 2026-10-03 23:30: Stage 4 completed: `features.py`, `model.py`, `evaluate.py` trained LightGBM ranker (`data/model.pkl`) and wrote `data/metrics.json` (LightGBM Top-1: 100.0% vs First Anomaly: 13.9%).
- 2026-10-04: Stage 5 completed: replay engine, oracle verification, demo runs, bug + leakage fixes, dataset regenerated, pytest suite.
- 2026-10-04: Stage 6 completed: Streamlit debugger (timeline, evidence, patch & replay, diff, auto-verify, metrics), AppTest suite.
- 2026-10-04: Stage 7 completed: README, screenshots in docs/, CLAUDE.md commands updated.
- 2026-10-04: Hardening pass: 9 bugs fixed with regression tests (33 tests), ruff clean, repo cleanup (BOMs, obsolete scripts/write_generate.py), deploy path verified on a fresh clone with a clean venv.

## NEXT STEPS

All original code phases are done. Remaining human tasks: record the 2-minute backup video and rehearse the pitch (script in README.md).

### Follow-up implementation (2026-10-04)
- [x] F2 calculation-consistency features: re-evaluate the recorded expression against recorded inputs and compare with the recorded result.
- [x] F5 plan-consistency features: compare arithmetic with supported question intent and check planned actions against the executed trace; direct contradictions elevate the blamed step with explicit evidence.
- [x] Metrics now include 95% Wilson intervals and per-fault Top-1/root-cause counts; the Metrics tab leads with root-cause verification and makes the F4/F5 split visible.
- [x] Added a Live demo tab: choose a clean run, inject a random compatible fault at an eligible step, and view ranker order against the injected step. Live runs use their own split.
- [x] Build freshness checks detect stale model/metrics artifacts after the feature schema changes and trigger a rebuild.
- [x] Rebuilt/evaluated artifacts and recorded updated per-fault scores.
- [x] Core and API tests: 40 passed; previously failing cross-agent Streamlit/SDK tests passed after correction (2 passed).
- [x] OpenAPI schema/Swagger UI routes verified through FastAPI app; editor diagnostics and `git diff --check` clean.
- [x] FastAPI frontend API implementation complete: run execution and diagnosis, replay/verification/diff, benchmark and dashboard data, and live fault injection are available as typed JSON endpoints; Swagger, OpenAPI JSON, and ReDoc routes return successfully.
- [ ] Rerun the expanded API/core test selection after adding the run-execution endpoint; the earlier API/core run passed 40 tests, before that final endpoint test was added.

- 2026-10-04: Added calculation/plan consistency signals, Wilson confidence intervals and per-fault replay metrics, a root-cause-first Metrics tab, and random live fault injection from clean runs. Latest evaluation: 100% seen / 92.9% held-out root-cause verified; F5 30/30, F4 35/40 Top-1.
- 2026-10-04: Completed `api.main:app` FastAPI service with typed run/step schemas, question listing/run execution, diagnosis, replay, oracle verification, top-k auto-verify, diff, replay savings, dashboard/metrics/KB/feature data, live-demo endpoints, CORS configuration, and generated Swagger/OpenAPI docs. Added FastAPI/Uvicorn runtime dependencies and API contract tests. Swagger, OpenAPI, health, clean-run listing, and a live injection were exercised successfully; the expanded API/core test rerun remains pending.
