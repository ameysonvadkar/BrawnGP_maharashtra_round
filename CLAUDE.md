# CLAUDE.md: Black Box

Black Box is a debugger for AI agents. It records agent runs, learns which step caused a failure (LightGBM ranker), and verifies the diagnosis by patching that step and replaying only the later steps from a content-hashed cache. Hackathon project, one person, 7-hour build.

Read `IMPLEMENTATION_PLAN.md` for specs and `progress.md` for current state before starting work. Update `progress.md` at the end of every task.

## Commands

```bash
.venv\Scripts\activate
pip install -r requirements.txt
python -m blackbox.generate        # build the labelled dataset into data/blackbox.db
python -m blackbox.model           # train and save data/model.pkl
python -m blackbox.evaluate        # write data/metrics.json
streamlit run app/streamlit_app.py
```

## Architecture in one paragraph

`agent/` runs a plan, retrieve, extract, calculate, answer loop over a fictional company KB. `blackbox/recorder.py` wraps every step, storing input, output, state snapshot, parents and a cache key in SQLite. `faults.py` injects one fault per run (F1 to F5) to give ground-truth labels. `features.py` and `model.py` rank steps by blame with SHAP reasons. `replay.py` re-runs from step k with a patched output, with earlier steps served from cache. `evaluate.py` reports Top-1, Top-3 and MRR on seen versus held-out fault types.

## Hard rules

- All LLM and tool calls go through `record_step()`. Never call the LLM directly.
- Temperature is 0. Cache key is `sha1(type + prompt_version + model + canonical_json(input))`.
- Features must never read `fault_type`, `fault_step`, or any injected flag. If accuracy looks perfect, assume leakage.
- Split by question (70/30) first. F1 to F3 are for training, F4 and F5 are held out for testing only.
- Gold answers are computed in Python from `agent/kb.json`, never by an LLM.
- Benign fault runs (fault injected but the run still succeeds) are excluded from ranker training.
- Do not report "replay-verified" without also reporting exact-match localisation accuracy.
- Never commit API keys. Use `.env` (see `.env.example`).

## Conventions

- Python 3.11+, type hints on public functions, small modules, no frameworks for the agent loop.
- Step types are exactly: `plan`, `retrieve`, `extract`, `calculate`, `answer`.
- Run IDs look like `r_0142`. Replayed runs set `parent_run_id`.
- Keep functions pure where possible so cached replay stays deterministic.
- Prefer clear and boring code over clever code. A working smaller version beats a broken bigger one.

## Workflow

1. Check `progress.md` for the next unchecked item.
2. Implement it against the spec and acceptance criteria in `IMPLEMENTATION_PLAN.md`.
3. Verify with the relevant command, or a quick script for the checkpoint.
4. Tick the item in `progress.md`, add a line to the log, and commit.

## Out of scope

Multi-agent frameworks, production agents, deep models (GNN or transformer), auth, deployment beyond Streamlit Cloud.
