import pathlib

content = '''"""generate.py — Dataset generator for Black Box.

Generates ~300 labelled agent runs (1 clean + 4 faulty per question).
Labels ground-truth failure steps, excludes benign faults from training,
and splits into train (70%), test (30%), and heldout_fault (F4 & F5).
"""
from __future__ import annotations

import json
import pathlib
import random
from typing import Any

from agent.runner import run
from blackbox.faults import inject_fault, FAULT_TYPES, TRAIN_FAULTS, HELDOUT_FAULTS
from blackbox.recorder import init_db, get_steps, _get_conn

random.seed(42)

QUESTIONS_PATH = pathlib.Path("agent/questions.json")
KB_PATH = pathlib.Path("agent/kb.json")


def load_questions() -> list[dict]:
    return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8-sig"))


def load_kb() -> list[dict]:
    return json.loads(KB_PATH.read_text(encoding="utf-8-sig"))


def generate_dataset(num_questions: int = 60) -> dict[str, int]:
    """Generate clean and faulty runs across questions."""
    init_db()
    questions = load_questions()[:num_questions]
    kb = load_kb()

    n_train = int(len(questions) * 0.7)
    train_qids = set(q["id"] for q in questions[:n_train])

    run_counter = 0
    total_clean = 0
    total_faulty = 0
    total_failed = 0
    total_benign = 0

    print(f"Starting dataset generation over {len(questions)} questions...")

    for q_idx, q in enumerate(questions):
        qid = q["id"]
        question_text = q["question"]
        gold = q["gold"]
        q_split = "train" if qid in train_qids else "test"

        # 1. Clean run
        run_counter += 1
        clean_run_id = f"r_{run_counter:04d}"
        clean_res = run(
            question=question_text,
            question_id=qid,
            run_id=clean_run_id,
            gold=gold,
            split=q_split,
            fault_type=None,
            fault_step=None,
        )
        total_clean += 1

        clean_steps = get_steps(clean_run_id)
        if not clean_steps:
            continue

        # 2. Four Faulty Runs per question
        fault_specs = []
        if q_split == "train":
            fault_specs = random.choices(TRAIN_FAULTS, k=3) + random.choices(HELDOUT_FAULTS, k=1)
        else:
            fault_specs = random.choices(TRAIN_FAULTS, k=2) + random.choices(HELDOUT_FAULTS, k=2)

        for f_idx, ftype in enumerate(fault_specs):
            run_counter += 1
            fault_run_id = f"r_{run_counter:04d}"

            eligible = []
            if ftype == "wrong_retrieval":
                eligible = [s for s in clean_steps if s["type"] == "retrieve"]
            elif ftype == "bad_tool_arg":
                eligible = [s for s in clean_steps if s["type"] == "calculate"]
            elif ftype == "corrupted_extract":
                eligible = [s for s in clean_steps if s["type"] == "extract"]
            elif ftype == "dropped_context":
                eligible = [s for s in clean_steps if s["type"] in ("extract", "retrieve")]
            elif ftype == "bad_plan":
                eligible = [s for s in clean_steps if s["type"] == "plan"]

            if not eligible:
                eligible = [s for s in clean_steps if s["type"] != "answer"]

            if not eligible:
                continue

            target_step = random.choice(eligible)
            target_idx = target_step["step_idx"]

            clean_output = json.loads(target_step["output_json"]) if target_step["output_json"] else {}
            state_before = json.loads(target_step["state_before_json"]) if target_step["state_before_json"] else {}

            faulty_output = inject_fault(
                fault_type=ftype,
                step_type=target_step["type"],
                clean_output=clean_output,
                state_before=state_before,
                kb=kb,
            )

            split_label = "heldout_fault" if ftype in HELDOUT_FAULTS else q_split

            faulty_res = run(
                question=question_text,
                question_id=qid,
                run_id=fault_run_id,
                gold=gold,
                split=split_label,
                fault_type=ftype,
                fault_step=target_idx,
                overrides={target_idx: faulty_output},
            )

            total_faulty += 1
            if not faulty_res["success"]:
                total_failed += 1
            else:
                total_benign += 1
                with _get_conn() as conn:
                    conn.execute("UPDATE runs SET fault_type = 'benign' WHERE run_id = ?", (fault_run_id,))

        if (q_idx + 1) % 10 == 0:
            print(f"Processed {q_idx + 1}/{len(questions)} questions... (Total runs: {run_counter})")

    fail_rate = (total_failed / total_faulty * 100) if total_faulty > 0 else 0
    stats = {
        "total_runs": run_counter,
        "clean_runs": total_clean,
        "faulty_runs": total_faulty,
        "failed_faulty_runs": total_failed,
        "benign_runs": total_benign,
        "failure_rate_pct": round(fail_rate, 2),
    }

    print("\n=======================================================")
    print(f"Dataset Generation Complete!")
    print(f"Total Runs: {run_counter}")
    print(f"Clean Runs: {total_clean}")
    print(f"Faulty Runs: {total_faulty} (Failed: {total_failed}, Benign: {total_benign})")
    print(f"Faulty Failure Rate: {fail_rate:.1f}% (Target >= 60%)")
    print("=======================================================\n")

    return stats


if __name__ == "__main__":
    generate_dataset()
'''
pathlib.Path('blackbox/generate.py').write_text(content, encoding='utf-8')
print('Successfully wrote blackbox/generate.py')
