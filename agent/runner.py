"""runner.py — Agent executor for Black Box.

Runs a plan → retrieve → extract → calculate → answer loop.
All steps go through record_step(); nothing calls the LLM directly.
"""
from __future__ import annotations

import copy
import json
from typing import Any

from blackbox.recorder import record_step, save_run, update_run_counts, init_db
from agent.tools import retrieve, extract, calculate, answer, plan

GEMINI_MODEL = "gemini-2.0-flash"


# ── Tracked state dict ───────────────────────────────────────────────────────

class TrackedState(dict):
    """A dict that records which keys are read, enabling parent tracking."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reads: set[str] = set()

    def __getitem__(self, key):
        self._reads.add(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self._reads.add(key)
        return super().get(key, default)

    def reset_reads(self) -> set[str]:
        reads = self._reads.copy()
        self._reads.clear()
        return reads


# ── Main runner ──────────────────────────────────────────────────────────────


def run(
    question: str,
    question_id: str,
    run_id: str,
    gold: str,
    split: str,
    fault_type: str | None = None,
    fault_step: int | None = None,
    overrides: dict[int, Any] | None = None,
    parent_run_id: str | None = None,
) -> dict:
    """Execute one agent run and return a result dict.

    Args:
        question: The natural-language question.
        question_id: ID from questions.json.
        run_id: Unique run ID (e.g. 'r_0001').
        gold: Gold answer string.
        split: 'train' | 'test' | 'heldout_fault'.
        fault_type: Fault label (None for clean runs).
        fault_step: Step index of the injected fault.
        overrides: {step_idx: output} to replace step outputs.
        parent_run_id: Set when this is a replay of another run.

    Returns:
        dict with run_id, final_answer, success, n_steps, state.
    """
    init_db()
    overrides = overrides or {}
    state = TrackedState()
    # Maps state_key -> step_idx that wrote it
    key_written_by: dict[str, int] = {}

    step_idx = 0
    n_reused = 0
    n_reexecuted = 0
    final_answer: str | None = None

    # ── Step 0: plan ────────────────────────────────────────────────────────
    plan_input = {"question": question}
    state.reset_reads()
    plan_output = record_step(
        run_id=run_id,
        idx=step_idx,
        step_type="plan",
        input_data=plan_input,
        fn=plan,
        state_before=dict(state),
        parents=[],
        model=GEMINI_MODEL,
        override=overrides.get(step_idx),
    )
    if plan_output is None:
        plan_output = {"actions": []}

    from blackbox.recorder import get_steps  # local import to avoid circular
    steps_so_far = get_steps(run_id)
    last_step = steps_so_far[-1] if steps_so_far else None
    if last_step and last_step["cache_hit"]:
        n_reused += 1
    else:
        n_reexecuted += 1

    actions = plan_output.get("actions", [])
    step_idx += 1

    # ── Steps 1+: execute actions ────────────────────────────────────────────
    for action in actions:
        act = action.get("action")
        state.reset_reads()

        if act == "retrieve":
            name = action.get("name", "")
            key = action.get("key", f"doc_{step_idx}")
            inp = {"name": name}
            override = overrides.get(step_idx)
            out = record_step(
                run_id=run_id, idx=step_idx, step_type="retrieve",
                input_data=inp, fn=retrieve,
                state_before=copy.deepcopy(dict(state)),
                parents=[], model="tool", override=override,
            )
            if out:
                state[key] = out
                key_written_by[key] = step_idx

        elif act == "extract":
            doc_key = action.get("doc_key", "")
            field = action.get("field", "")
            key = action.get("key", f"val_{step_idx}")
            reads_before = state.reset_reads()
            doc_data = state.get(doc_key, {})
            company = doc_data.get("company", {}) if isinstance(doc_data, dict) else {}
            reads_before = state.reset_reads()
            parents = [key_written_by[k] for k in reads_before if k in key_written_by]
            inp = {"company": company, "field": field}
            override = overrides.get(step_idx)
            out = record_step(
                run_id=run_id, idx=step_idx, step_type="extract",
                input_data=inp, fn=extract,
                state_before=copy.deepcopy(dict(state)),
                parents=parents, model=GEMINI_MODEL, override=override,
            )
            if out:
                val = out.get("value", "") if isinstance(out, dict) else str(out)
                state[key] = val
                key_written_by[key] = step_idx

        elif act == "calculate":
            expr = action.get("expr", "")
            key = action.get("key", f"calc_{step_idx}")
            reads_before = state.reset_reads()
            relevant_state = {k: state.get(k) for k in state if k in expr}
            reads_before = state.reset_reads()
            parents = [key_written_by[k] for k in reads_before if k in key_written_by]
            inp = {"expr": expr, "state": dict(state)}
            override = overrides.get(step_idx)
            out = record_step(
                run_id=run_id, idx=step_idx, step_type="calculate",
                input_data=inp, fn=calculate,
                state_before=copy.deepcopy(dict(state)),
                parents=parents, model="tool", override=override,
            )
            if out:
                res = out.get("result", 0) if isinstance(out, dict) else out
                state[key] = res
                key_written_by[key] = step_idx

        elif act == "answer":
            reads_before = state.reset_reads()
            all_state = dict(state)
            reads_before2 = state.reset_reads()
            parents = [key_written_by[k] for k in all_state if k in key_written_by]
            inp = {"question": question, "state": all_state}
            override = overrides.get(step_idx)
            out = record_step(
                run_id=run_id, idx=step_idx, step_type="answer",
                input_data=inp, fn=answer,
                state_before=copy.deepcopy(dict(state)),
                parents=parents, model=GEMINI_MODEL, override=override,
            )
            if out:
                final_answer = out.get("answer", "") if isinstance(out, dict) else str(out)
                state["final"] = final_answer
                key_written_by["final"] = step_idx

        all_steps = get_steps(run_id)
        this_step = next((s for s in all_steps if s["step_idx"] == step_idx), None)
        if this_step:
            if this_step["cache_hit"]:
                n_reused += 1
            else:
                n_reexecuted += 1

        step_idx += 1

    success = _check_success(final_answer, gold)

    save_run(
        run_id=run_id,
        question_id=question_id,
        question=question,
        gold=gold,
        final_answer=final_answer,
        success=success,
        split=split,
        fault_type=fault_type,
        fault_step=fault_step,
        parent_run_id=parent_run_id,
    )
    update_run_counts(run_id, n_reused, n_reexecuted)

    return {
        "run_id": run_id,
        "final_answer": final_answer,
        "success": success,
        "n_steps": step_idx,
        "state": dict(state),
        "n_reused": n_reused,
        "n_reexecuted": n_reexecuted,
    }


def _check_success(final_answer: str | None, gold: str) -> bool:
    """Check if the final answer matches the gold answer."""
    if not final_answer:
        return False
    import re
    gold_nums = re.findall(r'[\d]+(?:\.\d+)?', gold)
    ans_nums = re.findall(r'[\d]+(?:\.\d+)?', final_answer)
    if gold_nums and ans_nums:
        # Compare numerically so "150" matches "150.0"
        return abs(float(gold_nums[0]) - float(ans_nums[0])) < 0.005
    return gold.lower() in final_answer.lower() or final_answer.lower() in gold.lower()
