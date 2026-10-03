# Black Box: Implementation Plan

One agent, five injected fault types, ~300 labelled runs, a LightGBM step ranker, cached suffix-only replay, and a Streamlit debugger. Build order: data, model, replay, UI. Code freeze at 5:45.

## 0. Ground rules (apply everywhere)

1. Every LLM and tool call goes through `record_step()`. Nothing calls the LLM directly.
2. Temperature 0. Cache key = `sha1(step_type + prompt_version + model + canonical_json(input))`.
3. Features never read `fault_type`, `fault_step`, or any injected flag.
4. Split by question first, then by fault type. F4 and F5 never appear in training.
5. Gold answers are computed in Python from `kb.json`, never by an LLM.
6. Commit and push at every checkpoint.

## 1. Repo layout

agent-blackbox/
  agent/      kb.json  questions.json  tools.py  runner.py
  blackbox/   recorder.py  faults.py  generate.py  features.py
              model.py  replay.py  evaluate.py
  app/        streamlit_app.py
  data/       blackbox.db  model.pkl  metrics.json   (gitignored except a demo DB)
  CLAUDE.md   progress.md   IMPLEMENTATION_PLAN.md   README.md

## 2. Schedule

| Window | Deliverable |
|---|---|
| 0:00-0:30 | repo, kb.json, questions.json |
| 0:30-1:30 | agent, recorder, cache |
| 1:30-2:30 | faults, dataset |
| 2:30-3:30 | features, model, eval |
| 3:30-4:15 | replay, diff |
| 4:15-5:45 | Streamlit UI |
| 5:45 | code freeze |
| 5:45-7:00 | polish, README, video, rehearsal |
