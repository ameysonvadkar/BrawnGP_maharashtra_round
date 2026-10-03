"""replay.py — Checkpointed replay engine and trace diff for Black Box.

Provides:
  - replay(run_id, overrides): re-run with patched step outputs, reusing cached prefix.
  - oracle_output(run_id, step_idx): the clean run's output for a step (the "known good" patch).
  - verify_step(run_id, step_idx): patch one step with its oracle output and check the outcome flips.
  - resume_state(run_id, step_idx): return snapshot of agent state before step_idx.
  - diff(run_a_id, run_b_id): side-by-side comparison of original vs replayed run.
"""
from __future__ import annotations

import json
import uuid
from typing import Any

from agent.runner import run
from blackbox.recorder import _get_conn, delete_run, get_run, get_steps

# Replays get their own split so they never leak into ranker training or evaluation
REPLAY_SPLIT = "replay"

# Step types that call the LLM (the expensive ones the cache saves)
LLM_STEP_TYPES = {"plan", "extract", "answer"}


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
            An existing run with this ID is replaced.

    Returns:
        dict with run_id, final_answer, success, and reuse counters:
        n_reused (cache hits), n_patched, n_reexecuted (real calls),
        n_prefix_reused (cache hits before the first patched step),
        n_llm_reused (LLM-type steps served from cache).
    """
    parent = get_run(run_id)
    if not parent:
        raise ValueError(f"Run {run_id} not found")

    overrides = {int(k): v for k, v in overrides.items()}
    if new_run_id is None:
        new_run_id = f"rp_{run_id}_{uuid.uuid4().hex[:4]}"
    else:
        delete_run(new_run_id)

    res = run(
        question=parent["question"],
        question_id=parent["question_id"],
        run_id=new_run_id,
        gold=parent["gold"],
        split=REPLAY_SPLIT,
        fault_type=None,  # Replays are counterfactual tests, never training labels
        fault_step=None,
        overrides=overrides,
        parent_run_id=run_id,
    )

    first_patch = min(overrides) if overrides else None
    n_reused = n_patched = n_reexecuted = n_prefix_reused = n_llm_reused = 0
    for s in get_steps(new_run_id):
        if s["step_idx"] in overrides:
            n_patched += 1
        elif s["cache_hit"]:
            n_reused += 1
            if first_patch is not None and s["step_idx"] < first_patch:
                n_prefix_reused += 1
            if s["type"] in LLM_STEP_TYPES:
                n_llm_reused += 1
        else:
            n_reexecuted += 1

    return {
        "run_id": new_run_id,
        "parent_run_id": run_id,
        "final_answer": res["final_answer"],
        "success": res["success"],
        "n_steps": res["n_steps"],
        "n_reused": n_reused,
        "n_patched": n_patched,
        "n_reexecuted": n_reexecuted,
        "n_prefix_reused": n_prefix_reused,
        "n_llm_reused": n_llm_reused,
        "outcome_changed": bool(res["success"]) != bool(parent["success"]),
    }


def find_clean_run(run_id: str) -> dict | None:
    """Return the successful clean (unfaulted, non-replay) run for the same question."""
    target = get_run(run_id)
    if not target:
        return None
    with _get_conn() as conn:
        row = conn.execute(
            """
            SELECT * FROM runs
            WHERE question_id = ? AND fault_type IS NULL AND parent_run_id IS NULL
              AND success = 1 AND split != ? AND run_id != ?
            ORDER BY run_id LIMIT 1
            """,
            (target["question_id"], REPLAY_SPLIT, run_id),
        ).fetchone()
    return dict(row) if row else None


def oracle_output(run_id: str, step_idx: int) -> Any | None:
    """Return the clean run's output at step_idx (the known-good patch), or None."""
    clean = find_clean_run(run_id)
    if not clean:
        return None
    step = next((s for s in get_steps(clean["run_id"]) if s["step_idx"] == step_idx), None)
    if not step or not step["output_json"]:
        return None
    return json.loads(step["output_json"])


def verify_step(run_id: str, step_idx: int, new_run_id: str | None = None) -> dict[str, Any]:
    """Patch step_idx with its oracle output, replay, and report whether fail -> pass.

    Note: patching a step downstream of the true fault with clean output can also
    flip the outcome, so always report this next to exact-match localisation accuracy.
    The stricter `root_cause_verified` also requires the step to have received the
    same input as in the clean run (the error started here rather than flowing in).
    """
    patch = oracle_output(run_id, step_idx)
    if patch is None:
        return {"run_id": None, "verified": False, "root_cause_verified": False,
                "reason": "no clean run / oracle output"}
    res = replay(run_id, {step_idx: patch}, new_run_id=new_run_id)
    parent = get_run(run_id)
    res["verified"] = bool(res["success"]) and not bool(parent["success"])
    res["input_matches_clean"] = input_matches_clean(run_id, step_idx)
    res["root_cause_verified"] = res["verified"] and res["input_matches_clean"]
    return res


def input_matches_clean(run_id: str, step_idx: int) -> bool:
    """True if step_idx got the same input here as in the clean run of the same question."""
    clean = find_clean_run(run_id)
    if not clean:
        return False
    mine = next((s for s in get_steps(run_id) if s["step_idx"] == step_idx), None)
    ref = next((s for s in get_steps(clean["run_id"]) if s["step_idx"] == step_idx), None)
    if not mine or not ref:
        return False
    return json.loads(mine["input_json"]) == json.loads(ref["input_json"])


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
        first_divergent_step: index of first step whose input or output changed
        step_diffs: list of per-step comparisons
        outcome_change: e.g. "fail -> pass" or "fail (no change)"
    """
    run_a = get_run(run_a_id)
    run_b = get_run(run_b_id)
    steps_a = {s["step_idx"]: s for s in get_steps(run_a_id)}
    steps_b = {s["step_idx"]: s for s in get_steps(run_b_id)}

    first_div = None
    step_diffs = []
    for i in sorted(set(steps_a) | set(steps_b)):
        sa = steps_a.get(i)
        sb = steps_b.get(i)

        out_a = json.loads(sa["output_json"]) if (sa and sa["output_json"]) else None
        out_b = json.loads(sb["output_json"]) if (sb and sb["output_json"]) else None
        in_a = json.loads(sa["input_json"]) if (sa and sa["input_json"]) else None
        in_b = json.loads(sb["input_json"]) if (sb and sb["input_json"]) else None

        changed = (out_a != out_b) or (in_a != in_b)
        if changed and first_div is None:
            first_div = i

        step_diffs.append({
            "step_idx": i,
            "type": sa["type"] if sa else (sb["type"] if sb else "unknown"),
            "output_a": out_a,
            "output_b": out_b,
            "input_changed": in_a != in_b,
            "output_changed": out_a != out_b,
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
        "final_answer_a": run_a["final_answer"] if run_a else None,
        "final_answer_b": run_b["final_answer"] if run_b else None,
        "first_divergent_step": first_div,
        "outcome_change": outcome_str,
        "step_diffs": step_diffs,
    }
