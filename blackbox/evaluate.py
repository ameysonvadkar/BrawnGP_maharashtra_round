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

import numpy as np

from blackbox.model import predict, load_model
from blackbox.recorder import list_runs, get_steps, get_run
from blackbox.features import extract_features_for_run

METRICS_PATH = Path("data/metrics.json")


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
    split_metrics = {}

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
                random.shuffle(ranked_indices)
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

    return {
        "n_runs": len(runs),
        "metrics": split_metrics,
    }


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
