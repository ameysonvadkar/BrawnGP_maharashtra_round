"""faults.py — Injected fault types F1 to F5 for Black Box dataset generation.

Each fault injector is a function:
  inject_fault(step_type, clean_output, state_before, kb) -> faulty_output
applied via the runner overrides path.
"""
from __future__ import annotations

import copy
import random
from typing import Any

from agent.tools import get_distractor, retrieve_by_id


# ── Fault Types ─────────────────────────────────────────────────────────────
# F1: wrong_retrieval (step: retrieve) -> return distractor doc (train)
# F2: bad_tool_arg    (step: calculate) -> swap operand for another fact in state (train)
# F3: corrupted_extract (step: extract) -> off-by-N year / wrong field value (train)
# F4: dropped_context (step: extract/state) -> return empty/corrupted value (held out)
# F5: bad_plan        (step: plan)      -> flip math operation (+ to -) or skip step (held out)

FAULT_TYPES = ["wrong_retrieval", "bad_tool_arg", "corrupted_extract", "dropped_context", "bad_plan"]
TRAIN_FAULTS = ["wrong_retrieval", "bad_tool_arg", "corrupted_extract"]
HELDOUT_FAULTS = ["dropped_context", "bad_plan"]


def inject_fault(
    fault_type: str,
    step_type: str,
    clean_output: Any,
    state_before: dict,
    kb: list[dict],
) -> Any:
    """Inject a specific fault into a step's output.

    Returns the faulty output dict/data to pass as an override.
    """
    if fault_type == "wrong_retrieval":
        return _inject_f1_wrong_retrieval(clean_output, kb)
    elif fault_type == "bad_tool_arg":
        return _inject_f2_bad_tool_arg(clean_output, state_before)
    elif fault_type == "corrupted_extract":
        return _inject_f3_corrupted_extract(clean_output)
    elif fault_type == "dropped_context":
        return _inject_f4_dropped_context(clean_output)
    elif fault_type == "bad_plan":
        return _inject_f5_bad_plan(clean_output)
    else:
        raise ValueError(f"Unknown fault_type: {fault_type}")


def _inject_f1_wrong_retrieval(clean_output: dict, kb: list[dict]) -> dict:
    """F1: Return the near-duplicate distractor document."""
    if not isinstance(clean_output, dict):
        return clean_output
    doc_id = clean_output.get("doc_id", "")
    distractor_id = get_distractor(doc_id)
    if not distractor_id:
        # Fallback distractor if no exact pair: pick a random different doc
        other = [c for c in kb if c["id"] != doc_id]
        if not other:
            return clean_output
        distractor_id = random.choice(other)["id"]
    faulty = retrieve_by_id(distractor_id)
    # Keep the real retrieval scores: retrieve_by_id's fixed [1.0, 0.0] would leak the label
    faulty["top_scores"] = clean_output.get("top_scores", faulty["top_scores"])
    return faulty


def _inject_f2_bad_tool_arg(clean_output: dict, state_before: dict) -> dict:
    """F2: Swap one calculation operand or result for another fact from state."""
    if not isinstance(clean_output, dict):
        return clean_output
    out = copy.deepcopy(clean_output)
    expr = out.get("expr", "")
    res = out.get("result", 0)

    # Distort result by multiplying or subtracting an arbitrary offset from state
    nums = [v for k, v in state_before.items() if isinstance(v, (int, float)) and v > 0]
    if nums:
        offset = random.choice(nums)
        # Flip sign or shift result
        new_res = res + offset if res > 0 else res - offset
        if new_res == res:
            new_res = res + 10
        out["result"] = new_res
    else:
        out["result"] = res + 5 if isinstance(res, (int, float)) else 0
    return out


def _inject_f3_corrupted_extract(clean_output: dict) -> dict:
    """F3: Off-by-N year or wrong field value."""
    if not isinstance(clean_output, dict):
        return clean_output
    out = copy.deepcopy(clean_output)
    val = out.get("value", "")
    field = out.get("field", "")

    try:
        num = float(str(val).replace(',', '').replace('$', '').replace('K', ''))
        if field == "founded" or num > 1900:
            # Shift year by off-by-N (e.g. 2009 -> 2015)
            shift = random.choice([-6, -5, 5, 6, 7])
            new_val = int(num + shift)
        else:
            shift = random.choice([100, 250, 500])
            new_val = int(num + shift)
        out["value"] = new_val
    except (ValueError, TypeError):
        out["value"] = str(val) + "_corrupted"
    return out


def _inject_f4_dropped_context(clean_output: dict) -> dict:
    """F4: Return empty / null output, dropping collected fact."""
    if isinstance(clean_output, dict):
        out = copy.deepcopy(clean_output)
        if "value" in out:
            out["value"] = 0  # Zero out extracted value
        elif "result" in out:
            out["result"] = 0
        elif "doc_id" in out:
            out["doc_id"] = "unknown"
            out["title"] = "Unknown Entity"
            out["company"] = {}
        return out
    return ""


def _inject_f5_bad_plan(clean_output: dict) -> dict:
    """F5: Replaces calculation operation (+ -> - or abs -> sum) or skips lookup."""
    if not isinstance(clean_output, dict):
        return clean_output
    out = copy.deepcopy(clean_output)
    actions = out.get("actions", [])
    if not actions:
        return out
    
    new_actions = []
    for act in actions:
        act_copy = copy.deepcopy(act)
        if act_copy.get("action") == "calculate":
            expr = act_copy.get("expr", "")
            if "-" in expr:
                act_copy["expr"] = expr.replace("-", "+")
            elif "+" in expr:
                act_copy["expr"] = expr.replace("+", "-")
            elif "abs" in expr:
                act_copy["expr"] = expr.replace("abs", "")
        new_actions.append(act_copy)
    
    out["actions"] = new_actions
    return out
