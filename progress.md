# Progress: Black Box

Last updated: 2026-10-03 21:54 IST
Code freeze target: hour 5:45

## Status at a glance

| Phase | Window | Status |
|---|---|---|
| 1. Setup + data files | 0:00 to 0:30 | DONE |
| 2. Agent, recorder, cache | 0:30 to 1:30 | DONE |
| 3. Faults + dataset | 1:30 to 2:30 | IN PROGRESS |
| 4. Features, model, eval | 2:30 to 3:30 | not started |
| 5. Replay + diff | 3:30 to 4:15 | not started |
| 6. Streamlit UI | 4:15 to 5:45 | not started |
| 7. Polish, README, video | 5:45 to 7:00 | not started |

## Checklist

### 1. Setup + data files
- [x] Repo, venv, requirements, `.env.example`
- [x] `agent/kb.json` (~40 companies, with near-duplicate names) — 40 companies written
- [x] `agent/questions.json` (~60 questions, gold computed in Python) — 60 questions written
- [x] Dependencies installed via pip (`streamlit`, `plotly`, `lightgbm`, `shap`, `scikit-learn`, `rapidfuzz`, `sentence-transformers`, `python-dotenv`, `pandas`, `numpy`, `google-generativeai`)
- [x] Checkpoint: data files committed

### 2. Agent, recorder, cache
- [x] `recorder.py`: SQLite schema (`runs`, `steps`, `cache`), `record_step()`, tuple-safe `cache_get()`, `cache_set()`, `cache_stats()`
- [x] `tools.py`: `retrieve` (TF-IDF + rapidfuzz), `extract`, `calculate` (with `SAFE_BUILTINS`), `answer`, `plan`, distractor helper
- [x] `runner.py`: plan executor loop, explicit `TrackedState`, `parents` dependency tracking, `overrides` support
- [x] Checkpoint: verified with `scripts/test_stage2.py` — rerun makes 0 re-executed calls (100% cache hits, 7/7 steps reused, answer correct: `NovaTech is older by 5 years.`)

### 3. Faults + dataset
- [ ] `faults.py`: F1 wrong_retrieval, F2 bad_tool_arg, F3 corrupted_extract, F4 dropped_context, F5 bad_plan
- [ ] `generate.py`: 1 clean + 4 faulty runs per question (~300 total runs)
- [ ] Label verification: clean-output replay flips each labelled fault
- [ ] Checkpoint: at least 250 runs, at least 60% of faulty runs fail; DB committed

### 4. Features, model, eval
- [ ] `features.py` (position, health, grounding, data flow, semantics)
- [ ] `model.py`: train on F1 to F3, SHAP top-3 reasons, save model.pkl
- [ ] `evaluate.py`: Top-1, Top-3, MRR, seen vs held-out, baselines
- [ ] Leakage check: no feature reads fault columns
- [ ] Checkpoint: `data/metrics.json` written

### 5. Replay + diff
- [ ] `replay()` with reused / cache-hit / re-executed counters
- [ ] `resume_state()` checkpoint view
- [ ] `diff()` first divergence and outcome change
- [ ] Replay-verified rate computed
- [ ] Checkpoint: NovaTech / Zenith demo flips fail to pass

### 6. Streamlit UI
- [ ] Run list and filters
- [ ] Timeline coloured by suspicion, step inspector
- [ ] Patch and replay form with counters
- [ ] Diff view
- [ ] Metrics tab
- [ ] Checkpoint (code freeze): demo flow works on 3 runs

### 7. Polish
- [ ] README with architecture image and metrics table
- [ ] 2-minute backup demo video
- [ ] Pitch rehearsed twice

### Stretch
- [ ] LLM-judge baseline
- [ ] Auto-verify top-3 blames
- [ ] SHAP waterfall
- [ ] `@blackbox.step` decorator
- [ ] Second task domain

## Metrics (fill in after phase 4)

| Split | Method | Top-1 | Top-3 | MRR |
|---|---|---|---|---|
| Seen faults, test questions | random | | | |
| | last step | | | |
| | first anomaly | | | |
| | LightGBM | | | |
| Held-out (F4, F5) | LightGBM | | | |

Replay-verified rate: _
Mean steps reused per replay: _

## Decisions log

- 2026-10-03: LightGBM over a deep model (about 1,500 labelled steps, need explanations and fast training).
- 2026-10-03: F4 and F5 held out for generalisation testing.
- 2026-10-03: Plain Python agent loop, no framework, for full control of recording and replay.
- 2026-10-03: Using Google Gemini (gemini-2.0-flash) as the LLM backend with smart deterministic fallbacks.
- 2026-10-03: Safe builtins (`abs`, `round`, `min`, `max`, `int`, `float`, `pow`) added to `calculate` evaluation for robustness.

## Work log

- 2026-10-03 20:05: Project structure created. CLAUDE.md, progress.md, IMPLEMENTATION_PLAN.md written.
- 2026-10-03 20:37: Stage 1 starting — venv, requirements, env setup.
- 2026-10-03 20:40: Git repo initialized. .env.example, .gitignore, requirements.txt committed.
- 2026-10-03 20:54: kb.json (40 companies with 20 near-duplicate pairs), questions.json (60 questions, gold computed in Python) written.
- 2026-10-03 21:07: Pip install completed successfully for all dependencies.
- 2026-10-03 21:20: `blackbox/recorder.py`, `agent/tools.py`, `agent/runner.py` created.
- 2026-10-03 21:54: Stage 2 complete & verified with `scripts/test_stage2.py` (7/7 steps reused on rerun, 0 re-executed).

## NEXT STEPS (Stage 3)

1. Create `blackbox/faults.py` (F1 to F5 fault injectors)
2. Create `blackbox/generate.py` (~300 runs generation script)
3. Run dataset generation into `data/blackbox.db`
4. Verify label quality and failure rate (target >= 60% failure rate on faulty runs)
