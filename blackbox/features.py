"""features.py — Feature extraction for Black Box step-ranker model.

Extracts feature vectors for steps of failed runs.
Features DO NOT read fault_type, fault_step, or any injected flag.
"""
from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from agent.tools import SAFE_BUILTINS
from blackbox.recorder import get_steps, get_run, list_runs

FEATURE_NAMES = [
    "pos_norm",
    "steps_remaining",
    "type_plan",
    "type_retrieve",
    "type_extract",
    "type_calculate",
    "type_answer",
    "has_error",
    "is_empty_output",
    "output_len",
    "valid_json",
    "grounding_match",
    "top_score_gap",
    "extracted_in_source",
    "calc_args_traceable",
    "num_downstream_consumers",
    "on_path_to_final",
    "cosine_sim_question",
]

_tfidf = TfidfVectorizer()
_NAME = re.compile(r"[A-Za-z_]\w*")
_BUILTIN_NAMES = set(SAFE_BUILTINS)


def _value_in_source(value: Any, company: dict) -> bool:
    """True if the extracted value equals one of the source document's field values.

    Exact (numeric-aware) match, not substring: "0" must not match "2009".
    """
    if value is None or value == "":
        return False
    for field_value in company.values():
        if str(field_value) == str(value):
            return True
        try:
            if float(field_value) == float(value):
                return True
        except (TypeError, ValueError):
            continue
    return False


def extract_features_for_run(run_id: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Extract feature matrix X, labels y, and step metadata for a run.

    Returns:
        X: shape (n_steps, len(FEATURE_NAMES))
        y: shape (n_steps,) binary label (1 if step == fault_step, else 0)
        step_meta: list of dicts with step details
    """
    run_info = get_run(run_id)
    if not run_info:
        raise ValueError(f"Run {run_id} not found")

    steps = get_steps(run_id)
    if not steps:
        return np.empty((0, len(FEATURE_NAMES))), np.empty((0,)), []

    n_steps = len(steps)
    fault_step = run_info.get("fault_step")
    question_text = run_info.get("question", "")

    # Build downstream consumer counts & path to final step
    consumers_map: dict[int, int] = {s["step_idx"]: 0 for s in steps}
    ancestors_of_final: set[int] = set()

    for s in steps:
        parents = json.loads(s["parents_json"]) if s["parents_json"] else []
        for p in parents:
            consumers_map[p] = consumers_map.get(p, 0) + 1

    # Find final answer step ancestors via backward graph traversal
    final_step = next((s for s in reversed(steps) if s["type"] == "answer"), steps[-1])
    stack = list(json.loads(final_step["parents_json"]) if final_step["parents_json"] else [])
    while stack:
        curr = stack.pop()
        if curr not in ancestors_of_final:
            ancestors_of_final.add(curr)
            curr_step = next((s for s in steps if s["step_idx"] == curr), None)
            if curr_step and curr_step["parents_json"]:
                stack.extend(json.loads(curr_step["parents_json"]))

    X_list = []
    y_list = []
    step_meta = []

    for s in steps:
        idx = s["step_idx"]
        stype = s["type"]
        inp = json.loads(s["input_json"]) if s["input_json"] else {}
        out = json.loads(s["output_json"]) if s["output_json"] else {}
        err = s["error"]
        state_before = json.loads(s["state_before_json"]) if s["state_before_json"] else {}

        # 1. Position features
        pos_norm = idx / float(n_steps)
        steps_remaining = n_steps - 1 - idx
        type_plan = 1.0 if stype == "plan" else 0.0
        type_retrieve = 1.0 if stype == "retrieve" else 0.0
        type_extract = 1.0 if stype == "extract" else 0.0
        type_calculate = 1.0 if stype == "calculate" else 0.0
        type_answer = 1.0 if stype == "answer" else 0.0

        # 2. Health features
        has_error = 1.0 if err is not None else 0.0
        out_str = json.dumps(out) if out is not None else ""
        is_empty_output = 1.0 if not out or out_str == "{}" else 0.0
        output_len = float(len(out_str))
        valid_json = 1.0 if out is not None else 0.0

        # 3. Grounding features
        grounding_match = 0.0
        top_score_gap = 0.0
        extracted_in_source = 0.0
        calc_args_traceable = 0.0

        if stype == "retrieve" and isinstance(out, dict):
            req_name = inp.get("name", "")
            title = out.get("title", "")
            if req_name and title:
                grounding_match = fuzz.token_sort_ratio(req_name, title) / 100.0
            top_scores = out.get("top_scores", [0, 0])
            if len(top_scores) >= 2:
                top_score_gap = float(top_scores[0] - top_scores[1])

        elif stype == "extract" and isinstance(out, dict):
            comp = inp.get("company", {})
            if isinstance(comp, dict) and _value_in_source(out.get("value"), comp):
                extracted_in_source = 1.0

        elif stype == "calculate" and isinstance(out, dict):
            # Every variable in the expression must come from an earlier step's output
            names = set(_NAME.findall(inp.get("expr", ""))) - _BUILTIN_NAMES
            if names and names <= set(state_before):
                calc_args_traceable = 1.0

        # 4. Data flow features
        num_downstream = float(consumers_map.get(idx, 0))
        on_path = 1.0 if idx in ancestors_of_final or stype == "answer" else 0.0

        # 5. Semantics feature (text similarity to question)
        cosine_sim_q = 0.0
        if question_text and out_str:
            try:
                vecs = _tfidf.fit_transform([question_text, out_str])
                sim = cosine_similarity(vecs[0:1], vecs[1:2])[0][0]
                cosine_sim_q = float(sim)
            except Exception:
                cosine_sim_q = 0.0

        feats = [
            pos_norm, steps_remaining,
            type_plan, type_retrieve, type_extract, type_calculate, type_answer,
            has_error, is_empty_output, output_len, valid_json,
            grounding_match, top_score_gap, extracted_in_source, calc_args_traceable,
            num_downstream, on_path, cosine_sim_q,
        ]

        # Label: 1 if this step is the ground-truth fault step
        label = 1 if (fault_step is not None and idx == fault_step) else 0

        X_list.append(feats)
        y_list.append(label)
        step_meta.append({"step_idx": idx, "type": stype, "run_id": run_id})

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32), step_meta


def build_feature_dataset(
    splits: tuple[str, ...] = ("train",),
    exclude_benign: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Build feature dataset for training/evaluating step ranker.

    Returns:
        X: feature matrix
        y: labels
        groups: group IDs per run (for LGBMRanker / grouped eval)
        meta: list of step metadata
    """
    all_runs = list_runs()
    X_all, y_all, groups_all, meta_all = [], [], [], []

    group_id = 0
    for r in all_runs:
        # Filter by split
        if r["split"] not in splits:
            continue
        # Skip clean runs (success = 1 and fault_type is null)
        if r["success"] == 1 and r["fault_type"] is None:
            continue
        # Exclude benign fault runs if requested
        if exclude_benign and r["fault_type"] == "benign":
            continue

        run_id = r["run_id"]
        X_run, y_run, meta_run = extract_features_for_run(run_id)
        if len(X_run) == 0:
            continue

        X_all.append(X_run)
        y_all.append(y_run)
        groups_all.append(np.full(len(y_run), group_id, dtype=np.int32))
        meta_all.extend(meta_run)
        group_id += 1

    if not X_all:
        return (
            np.empty((0, len(FEATURE_NAMES))),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            [],
        )

    return (
        np.vstack(X_all),
        np.concatenate(y_all),
        np.concatenate(groups_all),
        meta_all,
    )
