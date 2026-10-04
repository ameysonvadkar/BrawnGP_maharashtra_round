"""HTTP contract tests for the Node/frontend-facing FastAPI service."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import blackbox.recorder as recorder
from agent.runner import run
from agent.tools import get_kb
from blackbox.faults import inject_fault


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    import api.main as api

    monkeypatch.setattr(recorder, "DB_PATH", tmp_path / "api-test.db")
    recorder.init_db()
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"seen_faults_test": {}, "heldout_faults": {}}))
    monkeypatch.setattr(api, "METRICS_PATH", metrics_path)
    monkeypatch.setattr(
        api,
        "predict",
        lambda run_id, top_k=3: [
            {
                "step_idx": step["step_idx"],
                "type": step["type"],
                "score": 0.9 if step["step_idx"] == 1 else 0.1,
                "reasons": [],
                "direct_evidence": [],
            }
            for step in recorder.get_steps(run_id)
        ],
    )
    with TestClient(api.create_app(initialize=False)) as client:
        yield client


def _clean_and_faulty_pair():
    with open("agent/questions.json", encoding="utf-8-sig") as question_file:
        question = next(q for q in json.load(question_file) if q["template"] == "older_by")
    clean = run(question["question"], question["id"], "api_clean", question["gold"], "test")
    retrieve_step = next(
        step for step in recorder.get_steps("api_clean") if step["type"] == "retrieve"
    )
    bad = inject_fault(
        "wrong_retrieval",
        "retrieve",
        json.loads(retrieve_step["output_json"]),
        json.loads(retrieve_step["state_before_json"]),
        get_kb(),
    )
    faulty = run(
        question["question"],
        question["id"],
        "api_faulty",
        question["gold"],
        "test",
        fault_type="wrong_retrieval",
        fault_step=retrieve_step["step_idx"],
        overrides={retrieve_step["step_idx"]: bad},
    )
    assert clean["success"] and not faulty["success"]
    return retrieve_step["step_idx"]


def test_list_runs_and_details_include_blame_order(api_client):
    _clean_and_faulty_pair()
    response = api_client.get("/api/runs", params={"fault_type": "wrong_retrieval"})
    assert response.status_code == 200
    assert [run_item["run_id"] for run_item in response.json()] == ["api_faulty"]

    detail = api_client.get("/api/runs/api_faulty")
    assert detail.status_code == 200
    assert detail.json()["top_blame_step"] == 1
    assert detail.json()["predictions"][0]["rank"] == 1
    assert detail.json()["steps"][0]["input"] == {"question": detail.json()["run"]["question"]}


def test_replay_verify_and_diff_endpoints(api_client):
    fault_idx = _clean_and_faulty_pair()
    faulty = recorder.get_steps("api_faulty")
    clean = recorder.get_steps("api_clean")
    clean_output = json.loads(clean[fault_idx]["output_json"])
    response = api_client.post(
        "/api/runs/api_faulty/replays",
        json={"overrides": {str(fault_idx): clean_output}},
    )
    assert response.status_code == 200
    replay_info = response.json()
    replay_id = replay_info["run_id"]
    assert replay_info["outcome_changed"]
    assert replay_info["diff"]["first_divergent_step"] == fault_idx
    assert "tokens_saved" in replay_info["savings"]

    verified = api_client.post(f"/api/runs/api_faulty/verify/{fault_idx}")
    assert verified.status_code == 200
    assert verified.json()["root_cause_verified"]
    assert api_client.get(f"/api/runs/api_faulty/diff/{replay_id}").status_code == 200
    replay_ids = {item["run_id"] for item in api_client.get("/api/runs/api_faulty/replays").json()}
    assert replay_id in replay_ids
    assert api_client.get(f"/api/runs/api_faulty/oracle/{fault_idx}").json()["output"] == clean_output


def test_dashboard_assets_and_not_found_errors(api_client):
    _clean_and_faulty_pair()
    assert api_client.get("/api/health").json()["status"] == "ok"
    assert api_client.get("/api/metrics").status_code == 200
    assert api_client.get("/api/dashboard").json()["total_runs"] == 2
    assert api_client.get("/api/knowledge-base").json()
    assert api_client.get("/api/features").json()["features"]
    assert api_client.get("/api/runs/missing-run").status_code == 404
    assert api_client.get("/api/runs/api_clean/steps/999").status_code == 404


def test_frontend_can_submit_a_scored_agent_run(api_client):
    questions = api_client.get("/api/questions")
    assert questions.status_code == 200 and questions.json()
    question = questions.json()[0]
    response = api_client.post("/api/runs/execute", json={"question_id": question["id"]})
    assert response.status_code == 200
    result = response.json()
    assert result["execution"]["success"]
    assert result["diagnosis"]["run"]["question_id"] == question["id"]
    assert result["diagnosis"]["steps"]

    unscored = api_client.post("/api/runs/execute", json={"question": "A custom question"})
    assert unscored.status_code == 422


def test_live_demo_injection_endpoint_returns_diagnosed_child(api_client):
    run_idx = _clean_and_faulty_pair()
    response = api_client.post(
        "/api/live-demo/injections",
        json={"clean_run_id": "api_clean"},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["injection"]["source_run_id"] == "api_clean"
    assert result["injection"]["fault_step"] is not None
    assert result["ranker"]["run"]["split"] == "live_demo"
    assert run_idx >= 0
    assert api_client.get("/api/live-demo/clean-runs").status_code == 200
