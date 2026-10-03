"""replay.py — Checkpointed replay engine and trace diff for Black Box.

Provides:
  - replay(run_id, overrides): re-run with patched step outputs, reusing cached prefix.
  - resume_state(run_id, step_idx): return snapshot of agent state before step_idx.
  - diff(run_a_id, run_b_id): side-by-side comparison of original vs replayed run.
"""
from __future__ import annotations

import json
from typing import Any

from agent.runner import run
from blackbox.recorder import get_run, get_steps, cache_stats


def replay(
    run_id: str,
    overrides: dict[int, Any],
    new_run_id: str | None = None,
) -> dict[str, Any]:
    """Re-execute a run with patched step outputs, serving earlier steps from cache.

    Args:
        run_id: Parent run ID to replay.
        overrides: {step_idx: patched_output} to replace at suspect steps.
        new_run_id: Optional ID for replayed run (auto-generated if None).

    Returns:
        dict with new_run_id, final_answer, success, n_reused, n_reexecuted.
    """
    parent = get_run(run_id)
    if not parent:
        raise ValueError(f"Run {run_id} not found")

    if new_run_id is None:
        import uuid
        new_run_id = f"rp_{run_id}_{uuid.uuid4().hex[:4]}"

    # Execute replayed run with overrides and parent_run_id set
    res = run(
        question=parent["question"],
        question_id=parent["question_id"],
        run_id=new_run_id,
        gold=parent["gold"],
        split=parent["split"],
        fault_type=None,  # Replays are counterfactual tests
        fault_step=None,
        overrides=overrides,
        parent_run_id=run_id,
    )

    stats = cache_stats(new_run_id)

    return {
        "run_id": new_run_id,
        "parent_run_id": run_id,
        "final_answer": res["final_answer"],
        "success": res["success"],
        "n_reused": stats["reused"],
        "n_reexecuted": stats["reexecuted"],
        "outcome_changed": (res["success"] != parent["success"]),
    }


def resume_state(run_id: str, step_idx: int) -> dict[str, Any]:
    """Return the state_before snapshot for step_idx (checkpoint view)."""
    steps = get_steps(run_id)
    target = next((s for s in steps if s["step_idx"] == step_idx), None)
    if not target:
        return {}
    return json.loads(target["state_before_json"]) if target["state_before_json"] else {}


def diff(run_a_id: str, run_b_id: str) -> dict[str, Any]:
    """Compare original run (run_a) and replayed run (run_b) side by side.

    Returns:
        first_divergent_step: index of first step whose output changed
        step_diffs: list of per-step comparisons
        outcome_change: e.g. "fail -> pass" or "no change"
    """
    run_a = get_run(run_a_id)
    run_b = get_run(run_b_id)
    steps_a = get_steps(run_a_id)
    steps_b = get_steps(run_b_id)

    first_div = None
    step_diffs = []
    max_len = max(len(steps_a), len(steps_b))

    for i in range(max_len):
        sa = steps_a[i] if i < len(steps_a) else None
        sb = steps_b[i] if i < len(steps_b) else None

        out_a = json.loads(sa["output_json"]) if (sa and sa["output_json"]) else None
        out_b = json.loads(sb["output_json"]) if (sb and sb["output_json"]) else None

        changed = (out_a != out_b)
        if changed and first_div is None:
            first_div = i

        step_diffs.append({
            "step_idx": i,
            "type": sa["type"] if sa else (sb["type"] if sb else "unknown"),
            "output_a": out_a,
            "output_b": out_b,
            "cache_hit_a": sa["cache_hit"] if sa else 0,
            "cache_hit_b": sb["cache_hit"] if sb else 0,
            "changed": changed,
        })

    succ_a = "pass" if (run_a and run_a["success"]) else "fail"
    succ_b = "pass" if (run_b and run_b["success"]) else "fail"
    outcome_str = f"{succ_a} -> {succ_b}" if succ_a != succ_b else f"{succ_a} (no change)"

    return {
        "run_a_id": run_a_id,
        "run_b_id": run_b_id,
        "first_divergent_step": first_div,
        "outcome_change": outcome_str,
        "step_diffs": step_diffs,
    }
