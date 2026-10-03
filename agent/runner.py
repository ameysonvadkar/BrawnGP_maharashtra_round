"""runner.py — Agent executor for Black Box.

Runs a plan → retrieve → extract → calculate → answer loop.
All steps go through record_step(); nothing calls the LLM directly.
"""
from __future__ import annotations

import copy
import re
from typing import Any

from agent.tools import answer, calculate, extract, get_kb, llm_model_id, plan, retrieve
from blackbox.recorder import cache_stats, delete_run, init_db, record_step, save_run, update_run_counts

TOOL_MODEL = "tool"
_NAME = re.compile(r"[A-Za-z_]\w*")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


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
        run_id: Unique run ID (e.g. 'r_0001'). An existing run with this ID is replaced.
        gold: Gold answer string.
        split: 'train' | 'test' | 'heldout_fault' | 'replay' | 'demo'.
        fault_type: Fault label (None for clean runs).
        fault_step: Step index of the injected fault.
        overrides: {step_idx: output} to replace step outputs.
        parent_run_id: Set when this is a replay of another run.

    Returns:
        dict with run_id, final_answer, success, n_steps, state, n_reused, n_reexecuted.
    """
    init_db()
    delete_run(run_id)  # never mix steps from an earlier recording of the same ID
    overrides = overrides or {}
    llm_model = llm_model_id()
    state = TrackedState()
    key_written_by: dict[str, int] = {}  # state key -> step that wrote it
    final_answer: str | None = None

    def step(idx: int, step_type: str, inp: Any, fn, parents: list[int], model: str) -> Any:
        return record_step(
            run_id=run_id, idx=idx, step_type=step_type, input_data=inp, fn=fn,
            state_before=copy.deepcopy(dict(state)), parents=parents, model=model,
            override=overrides.get(idx),
        )

    def parents_of(keys: set[str]) -> list[int]:
        return sorted({key_written_by[k] for k in keys if k in key_written_by})

    # ── Step 0: plan ────────────────────────────────────────────────────────
    plan_output = step(0, "plan", {"question": question}, plan, [], llm_model) or {"actions": []}
    actions = plan_output.get("actions", []) if isinstance(plan_output, dict) else []

    # ── Steps 1+: execute actions ────────────────────────────────────────────
    step_idx = 1
    for action in actions:
        act = action.get("action")
        state.reset_reads()

        if act == "retrieve":
            key = action.get("key", f"doc_{step_idx}")
            out = step(step_idx, "retrieve", {"name": action.get("name", "")}, retrieve, [], TOOL_MODEL)
            if out:
                state[key] = out
                key_written_by[key] = step_idx

        elif act == "extract":
            key = action.get("key", f"val_{step_idx}")
            doc = state.get(action.get("doc_key", ""), {})
            company = doc.get("company", {}) if isinstance(doc, dict) else {}
            parents = parents_of(state.reset_reads())
            inp = {"company": company, "field": action.get("field", "")}
            out = step(step_idx, "extract", inp, extract, parents, llm_model)
            if out:
                state[key] = out.get("value", "") if isinstance(out, dict) else str(out)
                key_written_by[key] = step_idx

        elif act == "calculate":
            key = action.get("key", f"calc_{step_idx}")
            expr = action.get("expr", "")
            used = set(_NAME.findall(expr)) & set(state)
            parents = parents_of(used)
            out = step(step_idx, "calculate", {"expr": expr, "state": dict(state)}, calculate, parents, TOOL_MODEL)
            if out:
                state[key] = out.get("result", 0) if isinstance(out, dict) else out
                key_written_by[key] = step_idx

        elif act == "answer":
            all_state = dict(state)
            parents = parents_of(set(all_state))
            out = step(step_idx, "answer", {"question": question, "state": all_state}, answer, parents, llm_model)
            if out:
                final_answer = out.get("answer", "") if isinstance(out, dict) else str(out)
                state["final"] = final_answer
                key_written_by["final"] = step_idx

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
    stats = cache_stats(run_id)
    update_run_counts(run_id, stats["reused"], stats["reexecuted"])

    return {
        "run_id": run_id,
        "final_answer": final_answer,
        "success": success,
        "n_steps": step_idx,
        "state": dict(state),
        "n_reused": stats["reused"],
        "n_reexecuted": stats["reexecuted"],
    }


# ── Scoring ──────────────────────────────────────────────────────────────────


def _check_success(final_answer: str | None, gold: str) -> bool:
    """Check the final answer against the gold answer.

    Numeric golds: the gold number must appear in the answer (compared numerically,
    so "150" matches "150.0"), and if the gold names a company ("Zenith is older by
    11 years") that company must be named too, not its near-duplicate.
    """
    if not final_answer:
        return False
    gold_nums = _NUMBER.findall(gold.replace(",", ""))
    if not gold_nums:
        return gold.lower() in final_answer.lower()

    target = float(gold_nums[0])
    ans_nums = [float(n) for n in _NUMBER.findall(final_answer.replace(",", ""))]
    if not any(abs(n - target) < 0.005 for n in ans_nums):
        return False

    m = re.match(r"^(.+?) (?:is older by|has) ", gold)
    return _mentions_company(final_answer, m.group(1)) if m else True


def _mentions_company(text: str, name: str) -> bool:
    """True if `name` is mentioned, not counting longer KB names that contain it."""
    lowered = text.lower()
    for other in (c["name"] for c in get_kb()):
        if other.lower() != name.lower() and name.lower() in other.lower():
            lowered = lowered.replace(other.lower(), " ")
    return re.search(rf"\b{re.escape(name.lower())}\b", lowered) is not None
