"""Core regression tests. Each test uses a throwaway SQLite DB, never data/blackbox.db.

    python -m pytest -q
"""
from __future__ import annotations

import json

import pytest

import blackbox.recorder as recorder
from agent.runner import _check_success, run
from agent.tools import calculate, get_kb, plan, retrieve
from blackbox.faults import inject_fault
from blackbox.replay import diff, oracle_output, replay, verify_step


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(recorder, "DB_PATH", tmp_path / "test.db")
    recorder.init_db()


def _questions() -> list[dict]:
    with open("agent/questions.json", encoding="utf-8-sig") as f:
        return json.load(f)


# ── Phase 2: tools and runner ────────────────────────────────────────────────

def test_calculate_allows_round_with_precision():
    out = calculate({"expr": "round(a * 1000 / b, 2)", "state": {"a": 580, "b": 4200}})
    assert out["result"] == 138.1


@pytest.mark.parametrize("expr", [
    "().__class__", "__import__('os')", "a.real", "open('x')", "[1, 2]", "'s' * 3",
    "round(a, ndigits=2)", "unknown + 1", "lambda: 1",
])
def test_calculate_blocks_unsafe_expressions(expr):
    with pytest.raises(ValueError):
        calculate({"expr": expr, "state": {"a": 1}})


def test_planner_retrieves_the_right_companies_for_every_question():
    kb = {c["id"]: c["name"] for c in get_kb()}
    for q in _questions():
        names = [a["name"] for a in plan({"question": q["question"]})["actions"]
                 if a["action"] == "retrieve"]
        assert [retrieve({"name": n})["title"] for n in names] == [kb[c] for c in q["companies"]], q["id"]


def test_check_success_compares_numbers_numerically():
    assert _check_success("$150K per employee.", "$150.0K per employee.")
    assert not _check_success("NovaTech is older by 6 years.", "Zenith is older by 11 years.")


def test_rerun_is_fully_cached():
    q = _questions()[0]
    run(q["question"], q["id"], "r_0001", q["gold"], "train")
    res = run(q["question"], q["id"], "r_0002", q["gold"], "train")
    assert res["success"]
    assert recorder.cache_stats("r_0002") == {"reused": res["n_steps"], "reexecuted": 0}


# ── Phase 5: replay, oracle patch, diff ─────────────────────────────────────

def _faulty_pair():
    """A clean run plus an F1 (wrong_retrieval) run of the same question."""
    q = next(q for q in _questions() if q["template"] == "older_by")
    run(q["question"], q["id"], "r_0001", q["gold"], "test")
    target = next(s for s in recorder.get_steps("r_0001") if s["type"] == "retrieve")
    bad = inject_fault("wrong_retrieval", "retrieve", json.loads(target["output_json"]),
                       json.loads(target["state_before_json"]), get_kb())
    res = run(q["question"], q["id"], "r_0002", q["gold"], "test",
              fault_type="wrong_retrieval", fault_step=target["step_idx"],
              overrides={target["step_idx"]: bad})
    assert not res["success"]
    return target["step_idx"]


def test_injected_retrieval_keeps_realistic_scores():
    k = _faulty_pair()
    clean = json.loads(recorder.get_steps("r_0001")[k]["output_json"])
    faulty = json.loads(recorder.get_steps("r_0002")[k]["output_json"])
    assert faulty["doc_id"] != clean["doc_id"]
    assert faulty["top_scores"] == clean["top_scores"]


def test_oracle_patch_flips_fail_to_pass_and_reuses_prefix():
    k = _faulty_pair()
    res = replay("r_0002", {k: oracle_output("r_0002", k)}, new_run_id="rp_test")
    assert res["success"] and res["outcome_changed"]
    assert res["n_prefix_reused"] == k and res["n_patched"] == 1
    assert recorder.get_run("rp_test")["split"] == "replay"
    assert recorder.get_run("rp_test")["parent_run_id"] == "r_0002"


def test_replay_with_fixed_id_replaces_previous_replay():
    k = _faulty_pair()
    replay("r_0002", {k: oracle_output("r_0002", k)}, new_run_id="rp_test")
    replay("r_0002", {k: oracle_output("r_0002", k)}, new_run_id="rp_test")
    assert len(recorder.get_steps("rp_test")) == len(recorder.get_steps("r_0002"))


def test_root_cause_verification_rejects_downstream_steps():
    k = _faulty_pair()
    assert verify_step("r_0002", k)["root_cause_verified"]
    downstream = verify_step("r_0002", k + 1)  # extract fed by the wrong document
    assert downstream["verified"] and not downstream["root_cause_verified"]


def test_diff_reports_first_divergence_and_outcome():
    k = _faulty_pair()
    replay("r_0002", {k: oracle_output("r_0002", k)}, new_run_id="rp_test")
    d = diff("r_0002", "rp_test")
    assert d["first_divergent_step"] == k
    assert d["outcome_change"] == "fail -> pass"
