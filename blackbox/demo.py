"""demo.py — Build the NovaTech vs Zenith demo runs for the pitch.

Records a clean run and an F1 (wrong_retrieval) run where step 3 returns
Zenit Labs instead of Zenith. Both use split "demo", so they never enter
ranker training or evaluation. Gold is computed in Python from kb.json.

    python -m blackbox.demo
"""
from __future__ import annotations

import json

from agent.runner import run
from agent.tools import get_kb
from blackbox.faults import inject_fault
from blackbox.recorder import delete_run, get_run, get_steps

DEMO_SPLIT = "demo"
DEMO_QID = "demo_novatech_zenith"
DEMO_CLEAN_RUN = "r_9001"
DEMO_FAULT_RUN = "r_9002"
DEMO_QUESTION = "Which company is older, NovaTech or Zenith, and by how many years?"


def _gold() -> str:
    kb = {c["id"]: c for c in get_kb()}
    a, b = kb["novatech"], kb["zenith"]
    older = a if a["founded"] < b["founded"] else b
    return f"{older['name']} is older by {abs(a['founded'] - b['founded'])} years."


def build_demo() -> dict:
    """(Re)create the demo clean + faulty runs and return their summaries."""
    gold = _gold()
    for rid in (DEMO_CLEAN_RUN, DEMO_FAULT_RUN):
        delete_run(rid)

    clean = run(DEMO_QUESTION, DEMO_QID, DEMO_CLEAN_RUN, gold, DEMO_SPLIT)

    # Fault: the second retrieve (Zenith) returns the near-duplicate Zenit Labs
    target = next(s for s in get_steps(DEMO_CLEAN_RUN)
                  if s["type"] == "retrieve" and json.loads(s["input_json"])["name"] == "Zenith")
    faulty_output = inject_fault(
        "wrong_retrieval", "retrieve",
        json.loads(target["output_json"]), json.loads(target["state_before_json"]), get_kb(),
    )
    faulty = run(
        DEMO_QUESTION, DEMO_QID, DEMO_FAULT_RUN, gold, DEMO_SPLIT,
        fault_type="wrong_retrieval", fault_step=target["step_idx"],
        overrides={target["step_idx"]: faulty_output},
    )
    return {"gold": gold, "clean": clean, "faulty": faulty, "fault_step": target["step_idx"]}


def ensure_demo() -> None:
    """Build the demo runs if they are missing (used by the UI)."""
    if not get_run(DEMO_FAULT_RUN) or not get_run(DEMO_CLEAN_RUN):
        build_demo()


if __name__ == "__main__":
    out = build_demo()
    print(f"Gold:   {out['gold']}")
    print(f"Clean  {DEMO_CLEAN_RUN}: {out['clean']['final_answer']} (success={out['clean']['success']})")
    print(f"Faulty {DEMO_FAULT_RUN}: {out['faulty']['final_answer']} "
          f"(success={out['faulty']['success']}, fault at step {out['fault_step']})")
