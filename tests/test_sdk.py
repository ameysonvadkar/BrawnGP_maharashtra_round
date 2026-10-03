"""Tests for the @blackbox.step SDK, the second (travel) agent and the cost estimate.

Uses a throwaway SQLite DB. The blame test also needs data/model.pkl (skipped otherwise).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import blackbox as bb
import blackbox.recorder as recorder
from blackbox.cost import replay_savings
from blackbox.replay import oracle_output, replay, verify_step
from examples import travel_agent as travel

MODEL = Path(__file__).resolve().parent.parent / "data" / "model.pkl"


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(recorder, "DB_PATH", tmp_path / "test.db")
    recorder.init_db()


def test_decorated_function_is_a_plain_call_outside_a_recording():
    assert travel.extract({"flight": 5400}, "flight") == {"value": 5400, "field": "flight"}
    assert recorder.list_runs() == []


def test_unknown_step_type_is_rejected():
    with pytest.raises(ValueError):
        bb.step("think")


def test_record_traces_every_step_with_value_provenance():
    q, gold = travel._question("velora", 3)
    res = bb.record("travel", q, run_id="t_0001", gold=gold)
    assert res["success"] and res["final_answer"] == "15000 rupees in total."
    steps = recorder.get_steps("t_0001")
    assert [s["type"] for s in steps] == ["plan", "retrieve", "extract", "extract", "calculate", "answer"]
    parents = {s["step_idx"]: json.loads(s["parents_json"]) for s in steps}
    assert 0 in parents[1]           # plan -> retrieve (city name)
    assert 1 in parents[2]           # retrieve -> extract (source document)
    assert parents[5] == [4]         # calculate -> answer (total)
    assert recorder.get_run("t_0001")["agent"] == "travel"


def test_rerecording_is_served_from_cache():
    q, gold = travel._question("kestrin", 2)
    bb.record("travel", q, run_id="t_0001", gold=gold)
    res = bb.record("travel", q, run_id="t_0002", gold=gold)
    assert res["n_reexecuted"] == 0 and res["n_reused"] == res["n_steps"]


def test_replay_of_an_sdk_run_uses_the_registered_agent_and_verifies_root_cause():
    summary = {row["run_id"]: row for row in travel.record_examples()}
    assert all(not row["success"] for row in summary.values())
    k = summary["t_9102"]["fault_step"]
    res = replay("t_9102", {k: oracle_output("t_9102", k)}, new_run_id="rp_t")
    assert res["success"] and res["n_prefix_reused"] == k
    assert recorder.get_run("rp_t")["agent"] == "travel"
    assert verify_step("t_9102", k)["root_cause_verified"]
    assert not verify_step("t_9102", k + 1)["root_cause_verified"]  # downstream extract


@pytest.mark.skipif(not MODEL.exists(), reason="build data/model.pkl first")
def test_ranker_trained_on_company_agent_blames_travel_faults():
    from blackbox.model import predict
    for row in travel.record_examples():
        ranked = sorted(predict(row["run_id"]), key=lambda p: p["score"], reverse=True)
        assert ranked[0]["step_idx"] == row["fault_step"], row


def test_replay_savings_counts_only_reexecuted_llm_steps(monkeypatch):
    monkeypatch.setenv("LLM_PRICE_IN_PER_M", "1.0")
    monkeypatch.setenv("LLM_PRICE_OUT_PER_M", "2.0")
    monkeypatch.setenv("USD_INR", "100")
    travel.record_examples()
    k = 1
    res = replay("t_9102", {k: oracle_output("t_9102", k)}, new_run_id="rp_t")
    s = replay_savings(res["run_id"], res["patched_steps"])
    assert s["tokens_full"] > 0 and s["tokens_spent"] == 0 and s["pct_saved"] == 1.0
    assert s["inr_saved"] == pytest.approx(s["usd_saved"] * 100)

    # A document the agent never processed: the extract/answer suffix must really re-run
    other = travel._BY_ID["mirovia_heights"]
    fresh = {"doc_id": other["id"], "title": other["name"], "source": other, "top_scores": [0.9, 0.5]}
    res2 = replay("t_9102", {k: fresh}, new_run_id="rp_t2")
    s2 = replay_savings(res2["run_id"], res2["patched_steps"])
    assert 0 < s2["tokens_spent"] < s2["tokens_full"] and s2["pct_saved"] < 1.0
