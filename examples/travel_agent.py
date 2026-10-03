"""travel_agent.py — A second agent, in a different domain, recorded with @blackbox.step.

Answers "What is the total cost of N nights in <city> including the flight?" over
fictional cities with near-duplicate names. Nothing here is specific to Black Box
except the decorators: this is how you would instrument your own agent.

    python -m examples.travel_agent

records clean and faulty runs, then shows that the ranker (trained only on the
company-facts agent) still blames the right step, and that patch & replay works.
"""
from __future__ import annotations

import json
import random
import re

from rapidfuzz import fuzz

import blackbox as bb
from agent.tools import calculate as safe_calculate
from blackbox.faults import inject_fault
from blackbox.recorder import get_run, get_steps

AGENT = "travel"
SPLIT = "sdk"

CITIES = [
    {"id": "velora", "name": "Velora", "hotel_per_night": 3200, "flight": 5400},
    {"id": "velara_bay", "name": "Velara Bay", "hotel_per_night": 2100, "flight": 7800},
    {"id": "kestrin", "name": "Kestrin", "hotel_per_night": 4100, "flight": 3900},
    {"id": "kestrel_point", "name": "Kestrel Point", "hotel_per_night": 2600, "flight": 6100},
    {"id": "mirovia", "name": "Mirovia", "hotel_per_night": 2800, "flight": 4600},
    {"id": "mirovia_heights", "name": "Mirovia Heights", "hotel_per_night": 5200, "flight": 4400},
    {"id": "saranth", "name": "Saranth", "hotel_per_night": 3600, "flight": 5000},
    {"id": "sarantha_isle", "name": "Sarantha Isle", "hotel_per_night": 2400, "flight": 8200},
]
_BY_ID = {c["id"]: c for c in CITIES}
NEAR_DUPS = {"velora": "velara_bay", "kestrin": "kestrel_point", "mirovia": "mirovia_heights",
             "saranth": "sarantha_isle"}


# ── the agent ────────────────────────────────────────────────────────────────

@bb.step("plan")
def plan(question: str) -> dict:
    m = re.search(r"(\d+)\s+nights?\s+in\s+(.+?)\s+including", question, re.IGNORECASE)
    return {"city": m.group(2).strip(), "nights": int(m.group(1))} if m else {"city": "", "nights": 0}


@bb.step("retrieve")
def retrieve(name: str) -> dict:
    scores = sorted(((fuzz.token_sort_ratio(name, c["name"]) / 100.0, c) for c in CITIES),
                    key=lambda sc: sc[0], reverse=True)
    (s1, best), (s2, _) = scores[0], scores[1]
    return {"doc_id": best["id"], "title": best["name"], "source": best, "top_scores": [s1, s2]}


@bb.step("extract")
def extract(source: dict, field: str) -> dict:
    return {"value": source.get(field), "field": field}


@bb.step("calculate")
def calculate(expr: str, values: dict) -> dict:
    return safe_calculate({"expr": expr, "state": values})


@bb.step("answer")
def answer(question: str, total: float | None) -> dict:
    return {"answer": f"{total} rupees in total."}


def travel_agent(question: str) -> str:
    facts = bb.state()
    p = plan(question) or {}
    doc = retrieve(p.get("city", "")) or {}
    source = doc.get("source") or {}
    facts["hotel"] = (extract(source, "hotel_per_night") or {}).get("value")
    facts["flight"] = (extract(source, "flight") or {}).get("value")
    facts["nights"] = p.get("nights", 0)
    facts["total"] = (calculate("hotel * nights + flight", dict(facts)) or {}).get("result")
    return (answer(question, facts["total"]) or {}).get("answer")


bb.register_agent(AGENT, travel_agent)


# ── recorded examples ───────────────────────────────────────────────────────

QUESTIONS = [
    ("travel_q1", "velora", 3),
    ("travel_q2", "kestrin", 2),
    ("travel_q3", "mirovia", 4),
]
# (run_id, question, fault, step type to break, which step of that type)
FAULTY = [
    ("t_9102", "travel_q1", "wrong_retrieval", "retrieve", 0),
    ("t_9104", "travel_q2", "corrupted_extract", "extract", 0),
    ("t_9106", "travel_q3", "dropped_context", "extract", 1),
]


def _question(city_id: str, nights: int) -> tuple[str, str]:
    c = _BY_ID[city_id]
    total = c["hotel_per_night"] * nights + c["flight"]
    return f"What is the total cost of {nights} nights in {c['name']} including the flight?", f"{total} rupees in total."


def record_examples() -> list[dict]:
    """Record one clean run per question plus one faulty run each; return a summary."""
    random.seed(7)
    questions = {qid: _question(city, n) for qid, city, n in QUESTIONS}
    for i, (qid, _, _) in enumerate(QUESTIONS):
        q, gold = questions[qid]
        bb.record(AGENT, q, run_id=f"t_{9101 + 2 * i}", gold=gold, question_id=qid, split=SPLIT)

    summary = []
    for run_id, qid, fault, step_type, nth in FAULTY:
        q, gold = questions[qid]
        clean_id = f"t_{int(run_id[2:]) - 1}"
        target = [s for s in get_steps(clean_id) if s["type"] == step_type][nth]
        clean_out = json.loads(target["output_json"])
        if fault == "wrong_retrieval":
            # Same fault as F1, but the near-duplicate comes from this agent's own KB
            twin = _BY_ID[NEAR_DUPS[clean_out["doc_id"]]]
            bad = {"doc_id": twin["id"], "title": twin["name"], "source": twin,
                   "top_scores": clean_out["top_scores"]}
        else:
            bad = inject_fault(fault, step_type, clean_out, json.loads(target["state_before_json"]), CITIES)
        res = bb.record(AGENT, q, run_id=run_id, gold=gold, question_id=qid, split=SPLIT,
                        fault_type=fault, fault_step=target["step_idx"], overrides={target["step_idx"]: bad})
        summary.append({"run_id": run_id, "fault": fault, "fault_step": target["step_idx"],
                        "answer": res["final_answer"], "gold": gold, "success": res["success"]})
    return summary


def ensure_examples() -> None:
    """Record the example runs if they are missing (used by the UI and the build)."""
    if not all(get_run(r[0]) for r in FAULTY):
        record_examples()


if __name__ == "__main__":
    from blackbox.model import predict
    from blackbox.replay import verify_step

    print(f"{'run':8s} {'fault':18s} {'label':>5s} {'blamed':>6s} {'score':>6s}  replay of blamed step")
    for row in record_examples():
        ranked = sorted(predict(row["run_id"]), key=lambda p: p["score"], reverse=True)
        top = ranked[0]
        v = verify_step(row["run_id"], top["step_idx"], new_run_id=f"rp_{row['run_id']}_check")
        outcome = "fail -> pass (root cause verified)" if v["root_cause_verified"] else (
            "fail -> pass" if v["verified"] else "still fails")
        print(f"{row['run_id']:8s} {row['fault']:18s} {row['fault_step']:>5d} {top['step_idx']:>6d} "
              f"{top['score']:>6.2f}  {outcome}")
