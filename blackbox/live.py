"""Live, one-off fault injection for interactive debugger demonstrations."""
from __future__ import annotations

import json
import random
import uuid
from typing import Any

from agent.runner import run
from agent.tools import get_kb
from blackbox.faults import inject_fault
from blackbox.recorder import get_run, get_steps

_FAULTS_BY_STEP = {
    "retrieve": ("wrong_retrieval", "dropped_context"),
    "extract": ("corrupted_extract", "dropped_context"),
    "calculate": ("bad_tool_arg",),
}


def inject_random_fault(clean_run_id: str) -> dict[str, Any]:
    """Create a faulty child run from a clean built-in run at a random eligible step."""
    clean_run = get_run(clean_run_id)
    if (not clean_run or not clean_run["success"] or clean_run["fault_type"] is not None
            or clean_run["parent_run_id"] is not None or clean_run["agent"] is not None):
        raise ValueError(f"{clean_run_id} is not a successful clean built-in run")

    eligible_steps = [
        step for step in get_steps(clean_run_id)
        if step["type"] in _FAULTS_BY_STEP
    ]
    if not eligible_steps:
        raise ValueError(f"{clean_run_id} has no eligible step for live fault injection")

    target = random.choice(eligible_steps)
    fault_type = random.choice(_FAULTS_BY_STEP[target["type"]])
    clean_output = json.loads(target["output_json"]) if target["output_json"] else {}
    state_before = json.loads(target["state_before_json"]) if target["state_before_json"] else {}
    faulty_output = inject_fault(fault_type, target["type"], clean_output, state_before, get_kb())
    if faulty_output == clean_output:
        raise ValueError(f"{fault_type} did not alter step {target['step_idx']} output")

    live_run_id = f"live_{uuid.uuid4().hex[:12]}"
    result = run(
        question=clean_run["question"],
        question_id=clean_run["question_id"],
        run_id=live_run_id,
        gold=clean_run["gold"],
        split="live_demo",
        fault_type=fault_type,
        fault_step=target["step_idx"],
        overrides={target["step_idx"]: faulty_output},
    )
    return {
        **result,
        "fault_type": fault_type,
        "fault_step": target["step_idx"],
        "source_run_id": clean_run_id,
    }
