# Progress: Black Box

Last updated: 2026-10-03 23:36 IST
Code freeze target: hour 5:45

## Status at a glance

| Phase | Window | Status |
|---|---|---|
| 1. Setup + data files | 0:00 to 0:30 | DONE |
| 2. Agent, recorder, cache | 0:30 to 1:30 | DONE |
| 3. Faults + dataset | 1:30 to 2:30 | DONE |
| 4. Features, model, eval | 2:30 to 3:30 | DONE |
| 5. Replay + diff | 3:30 to 4:15 | DONE |
| 6. Streamlit UI | 4:15 to 5:45 | not started |
| 7. Polish, README, video | 5:45 to 7:00 | not started |

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
- [x] Failure rate verified: 100.0% failure rate on faulty runs (240/240 failed, 0 benign)
- [x] Checkpoint: 300 runs in DB, 70/30 question train/test split, F4/F5 assigned to `heldout_fault`

### 4. Features, model, eval
- [x] `features.py`: 18 step-level features (position, health, grounding, data flow, semantics), zero fault leakage
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
- [ ] Sidebar run list and filters
- [ ] Timeline strip coloured by suspicion score, step inspector with SHAP evidence
- [ ] Patch and replay form with cache reuse counters
- [ ] Side-by-side diff view
- [ ] Metrics tab
- [ ] Checkpoint (code freeze): demo flow works on 3 chosen runs

### 7. Polish
- [ ] README with architecture diagram and metrics table
- [ ] 2-minute backup demo video
- [ ] Pitch rehearsed twice

## Metrics (Evaluated 2026-10-03)

| Split | Method | Top-1 | Top-3 | MRR |
|---|---|---|---|---|
| **Seen faults, test questions** (36 runs) | Random | 16.7% | 38.9% | 0.391 |
| | Last step | 0.0% | 52.8% | 0.326 |
| | First anomaly | 13.9% | 66.7% | 0.411 |
| | **LightGBM** | **100.0%** | **100.0%** | **1.000** |
| **Held-out faults (F4, F5)** (78 runs) | Random | 16.7% | 48.7% | 0.406 |
| | Last step | 0.0% | 15.4% | 0.200 |
| | First anomaly | 6.4% | 29.5% | 0.332 |
| | LightGBM | 0.0% | 32.0% | 0.258 |

## Decisions log

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

## NEXT STEPS (Stage 5 & Stage 6)

1. **Stage 5 (Replay & Diff)**:
   - Create `blackbox/replay.py` (`replay(run_id, overrides)`, `resume_state()`, `diff()`)
   - Write `scripts/test_stage5.py` to test patch-and-replay on a failed run (e.g. NovaTech/Zenith run) and verify outcome flips `fail -> pass` with prefix cache reuse counter.
2. **Stage 6 (Streamlit Debugger UI)**:
   - Create `app/streamlit_app.py` according to design system specs in `design-palette.md` and `IMPLEMENTATION_PLAN.md`.
   - Build tabs: **Debugger** (timeline, inspector, patch & replay, diff) and **Metrics** (seen vs held-out benchmarks).
