"""model.py — LightGBM Step Ranker and SHAP explanation engine for Black Box.

Trains LightGBM classifier on F1-F3 train split, saves model to data/model.pkl,
and provides predict(run_id) returning step blame scores and SHAP top-3 reasons.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import shap

from blackbox.features import (
    FEATURE_NAMES,
    build_feature_dataset,
    extract_features_for_run,
)

MODEL_PATH = Path("data/model.pkl")


def train_model() -> lgb.LGBMClassifier:
    """Train LightGBM step ranker on train split failed runs (F1-F3)."""
    X_train, y_train, groups_train, _ = build_feature_dataset(
        splits=["train"], exclude_benign=True
    )

    if len(X_train) == 0:
        raise ValueError("No training data found in data/blackbox.db!")

    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    scale_pos = (n_neg / float(n_pos)) if n_pos > 0 else 1.0

    print(f"Training LightGBM on {len(X_train)} step samples ({n_pos} positive, {n_neg} negative)...")

    clf = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=15,
        scale_pos_weight=scale_pos,
        random_state=42,
        verbose=-1,
    )

    clf.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)

    print(f"Model saved to {MODEL_PATH}")
    return clf


def load_model() -> lgb.LGBMClassifier:
    """Load trained LightGBM model from disk or train if missing."""
    if not MODEL_PATH.exists():
        return train_model()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict(run_id: str) -> list[dict[str, Any]]:
    """Predict blame score and top-3 SHAP explanation reasons for each step in a run.

    Returns list of dicts:
      {
        "step_idx": int,
        "type": str,
        "score": float (0.0 to 1.0 suspicion score),
        "reasons": list[dict] -> [{"feature": str, "value": float, "shap_value": float}, ...]
      }
    """
    model = load_model()
    X_run, _, step_meta = extract_features_for_run(run_id)

    if len(X_run) == 0:
        return []

    # Predict positive class probabilities (blame scores)
    probs = model.predict_proba(X_run)[:, 1]

    # SHAP explanations
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_run)
    # Handle single or multi-output shap_values format
    if isinstance(shap_vals, list):
        shap_matrix = shap_vals[1]
    elif len(shap_vals.shape) == 3:
        shap_matrix = shap_vals[:, :, 1]
    else:
        shap_matrix = shap_vals

    results = []
    for i, meta in enumerate(step_meta):
        score = float(probs[i])
        step_shap = shap_matrix[i]
        step_feats = X_run[i]

        # Top 3 feature contributions driving score higher (or top magnitudes)
        top_indices = np.argsort(np.abs(step_shap))[::-1][:3]
        reasons = []
        for idx in top_indices:
            feat_name = FEATURE_NAMES[idx]
            reasons.append({
                "feature": feat_name,
                "value": round(float(step_feats[idx]), 3),
                "shap_value": round(float(step_shap[idx]), 4),
            })

        results.append({
            "step_idx": meta["step_idx"],
            "type": meta["type"],
            "score": round(score, 4),
            "reasons": reasons,
        })

    return results


if __name__ == "__main__":
    train_model()
