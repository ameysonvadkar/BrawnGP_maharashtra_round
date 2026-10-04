"""evaluate.py — Evaluation pipeline and baseline comparisons for Black Box.

Computes Top-1, Top-3, and MRR metrics on:
  (a) Seen faults on test questions
  (b) Held-out fault types (F4: dropped_context, F5: bad_plan)
against 3 baselines: Random, Last Step, First Anomaly.
Writes metrics to data/metrics.json.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


from blackbox.model import predict
from blackbox.recorder import list_runs, get_steps
from blackbox.replay import verify_step

METRICS_PATH = Path("data/metrics.json")
_WILSON_Z = 1.96


def _rate_with_interval(hits: int, n: int) -> dict[str, Any]:
    """Return a rate and two-sided 95% Wilson interval."""
    if n <= 0:
        return {"hits": hits, "rate": 0.0, "ci95": [0.0, 0.0]}
    p = hits / float(n)
    z2 = _WILSON_Z ** 2
    denominator = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denominator
    margin = (_WILSON_Z * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)) ** 0.5)
              / denominator)
    return {
        "hits": hits,
        "rate": round(p, 4),
        "ci95": [round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4)],
    }


def evaluate_baselines_and_model() -> dict[str, Any]:
    """Evaluate LightGBM model against baselines across splits."""
    all_runs = list_runs()

    # Filter for failed faulty runs with valid fault_step
    failed_runs = [
        r for r in all_runs
        if r["success"] == 0 and r["fault_step"] is not None and r["fault_type"] != "benign"
    ]

    seen_test_runs = [r for r in failed_runs if r["split"] == "test"]
    heldout_runs = [r for r in failed_runs if r["split"] == "heldout_fault"]

    results = {
        "seen_faults_test": _eval_split(seen_test_runs, "Seen faults (test questions)"),
        "heldout_faults": _eval_split(heldout_runs, "Held-out faults (F4, F5)"),
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nMetrics written to {METRICS_PATH}\n")
    return results


def _eval_split(runs: list[dict], split_name: str) -> dict[str, Any]:
    if not runs:
        return {"n_runs": 0, "baselines": {}}

    print(f"\n--- Evaluating {split_name} ({len(runs)} failed runs) ---")

    methods = ["random", "last_step", "first_anomaly", "lightgbm"]
    rng = random.Random(0)  # seeded so the random baseline is reproducible
    split_metrics = {}
    lgbm_top1: dict[str, int] = {}

    for method in methods:
        top1_hits = 0
        top3_hits = 0
        mrr_sum = 0.0

        for r in runs:
            run_id = r["run_id"]
            true_fault = r["fault_step"]
            steps = get_steps(run_id)
            if not steps:
                continue

            n_steps = len(steps)

            if method == "random":
                ranked_indices = list(range(n_steps))
                rng.shuffle(ranked_indices)
            elif method == "last_step":
                # Rank steps from last to first
                ranked_indices = list(range(n_steps - 1, -1, -1))
            elif method == "first_anomaly":
                # Rank by anomaly heuristic: has error, empty out, or low grounding
                ranked_indices = _rank_first_anomaly(run_id, steps)
            elif method == "lightgbm":
                preds = predict(run_id)
                # Sort step indices by score descending
                sorted_preds = sorted(preds, key=lambda p: p["score"], reverse=True)
                ranked_indices = [p["step_idx"] for p in sorted_preds]
                lgbm_top1[run_id] = ranked_indices[0]

            # Top-1
            if ranked_indices and ranked_indices[0] == true_fault:
                top1_hits += 1

            # Top-3
            if true_fault in ranked_indices[:3]:
                top3_hits += 1

            # MRR
            if true_fault in ranked_indices:
                rank = ranked_indices.index(true_fault) + 1
                mrr_sum += 1.0 / rank

        n = len(runs)
        top1_acc = round(top1_hits / float(n), 4)
        top3_acc = round(top3_hits / float(n), 4)
        mrr = round(mrr_sum / float(n), 4)

        split_metrics[method] = {
            "top1": top1_acc,
            "top3": top3_acc,
            "mrr": mrr,
        }

        print(f"[{method:15s}] Top-1: {top1_acc*100:5.1f}% | Top-3: {top3_acc*100:5.1f}% | MRR: {mrr:.3f}")

    replay_verified = _eval_replay(runs, lgbm_top1)

    return {
        "n_runs": len(runs),
        "metrics": split_metrics,
        "replay_verified": replay_verified,
    }


def _eval_replay(runs: list[dict], lgbm_top1: dict[str, int]) -> dict[str, Any]:
    """Patch steps with their clean (oracle) output and replay; count fail -> pass flips.

    - lightgbm_top1: patch the step LightGBM blames most. Downstream steps can
      also flip when patched, so read this next to exact-match Top-1.
    - lightgbm_top1_root_cause: as above, and the blamed step received the same
      input as in the clean run, so the error originated at that step.
    - label_confirmed: patch the injected fault step (sanity check of the labels).
    """
    n = len(runs)
    top1_flips = 0
    top1_root = 0
    label_flips = 0
    by_fault: dict[str, dict[str, int]] = {}
    for r in runs:
        run_id = r["run_id"]
        fault_type = r["fault_type"] or "unknown"
        counts = by_fault.setdefault(
            fault_type, {"n_runs": 0, "top1_hits": 0, "root_cause_hits": 0}
        )
        counts["n_runs"] += 1
        if run_id in lgbm_top1:
            counts["top1_hits"] += int(lgbm_top1[run_id] == r["fault_step"])
            res = verify_step(run_id, lgbm_top1[run_id], new_run_id=f"rv_{run_id}_top1")
            top1_flips += int(res["verified"])
            root_verified = int(res["root_cause_verified"])
            top1_root += root_verified
            counts["root_cause_hits"] += root_verified
        res = verify_step(run_id, r["fault_step"], new_run_id=f"rv_{run_id}_label")
        label_flips += int(res["verified"])

    per_fault = {
        fault_type: {
            "n_runs": counts["n_runs"],
            "top1": _rate_with_interval(counts["top1_hits"], counts["n_runs"]),
            "root_cause_verified": _rate_with_interval(
                counts["root_cause_hits"], counts["n_runs"]
            ),
        }
        for fault_type, counts in sorted(by_fault.items())
    }
    out = {
        "lightgbm_top1": round(top1_flips / float(n), 4),
        "lightgbm_top1_root_cause": round(top1_root / float(n), 4),
        "root_cause_ci95": _rate_with_interval(top1_root, n)["ci95"],
        "label_confirmed": round(label_flips / float(n), 4),
        "per_fault": per_fault,
    }
    print(f"[replay-verified] LightGBM top-1 patch flips fail->pass: {out['lightgbm_top1']*100:5.1f}% "
          f"| root-cause verified: {out['lightgbm_top1_root_cause']*100:5.1f}% "
          f"| oracle patch at labelled fault step: {out['label_confirmed']*100:5.1f}%")
    return out


def _rank_first_anomaly(run_id: str, steps: list[dict]) -> list[int]:
    """Rank steps by anomaly heuristics (error -> empty output -> lowest grounding match)."""
    scores = []
    for s in steps:
        idx = s["step_idx"]
        err = s["error"]
        out_json = s["output_json"] or "{}"
        stype = s["type"]

        score = 0.0
        if err is not None:
            score += 10.0
        if out_json == "{}" or not out_json:
            score += 5.0
        if stype in ("extract", "calculate") and idx > 0:
            score += 1.0

        scores.append((idx, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in scores]


if __name__ == "__main__":
    evaluate_baselines_and_model()
