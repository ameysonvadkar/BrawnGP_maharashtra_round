"""build.py — Build everything the debugger needs in one step.

    python -m blackbox.build            # build whatever is missing
    python -m blackbox.build --force    # rebuild from scratch

Generates the labelled dataset, trains the ranker, evaluates it and records the
demo runs, including the second agent (instrumented with @blackbox.step). Runs
offline in a few seconds without an API key, so the Streamlit app calls it on
first launch (data/ is gitignored and absent on a fresh deploy).
"""
from __future__ import annotations

import argparse
import json
import pickle

from blackbox import recorder
from blackbox.demo import build_demo
from blackbox.evaluate import METRICS_PATH, evaluate_baselines_and_model
from blackbox.features import FEATURE_NAMES
from blackbox.generate import generate_dataset
from blackbox.model import MODEL_PATH, train_model

ARTIFACTS = (recorder.DB_PATH, MODEL_PATH, METRICS_PATH)


def is_built() -> bool:
    if not all(path.exists() for path in ARTIFACTS):
        return False
    with open(ARTIFACTS[1], "rb") as model_file:
        model = pickle.load(model_file)
    metrics = json.loads(ARTIFACTS[2].read_text(encoding="utf-8"))
    heldout_replay = metrics.get("heldout_faults", {}).get("replay_verified", {})
    return (
        getattr(model, "n_features_in_", None) == len(FEATURE_NAMES)
        and "per_fault" in heldout_replay
        and "root_cause_ci95" in heldout_replay
    )


def build_all(force: bool = False) -> None:
    """Build the dataset, model, metrics and demo runs (all of them, if any is missing)."""
    if is_built() and not force:
        print("Already built: data/blackbox.db, data/model.pkl, data/metrics.json")
        return
    for path in ARTIFACTS:  # partial builds are not trusted; rebuild consistently
        path.unlink(missing_ok=True)
    generate_dataset()
    train_model()
    evaluate_baselines_and_model()
    build_demo()
    from examples.travel_agent import record_examples  # second agent, recorded via the SDK
    record_examples()
    print("Build complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="rebuild even if data/ exists")
    build_all(force=parser.parse_args().force)
