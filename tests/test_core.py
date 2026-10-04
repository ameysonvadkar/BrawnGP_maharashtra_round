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
    assert not downstream["input_matches_clean"] and not downstream["root_cause_verified"]


def test_replay_keeps_injected_fault_when_another_step_is_patched():
    # Regression: replays used to re-run unpatched steps through the cache, which
    # silently dropped the injected fault, so a no-op patch "fixed" the run.
    k = _faulty_pair()
    noop = verify_step("r_0002", k - 1) if k > 1 else verify_step("r_0002", 0)
    assert not noop["success"] and not noop["verified"]
    faulty = json.loads(recorder.get_steps("r_0002")[k]["output_json"])
    replayed = recorder.get_steps(noop["run_id"])[k]
    assert json.loads(replayed["output_json"]) == faulty and replayed["cache_hit"] == 1


def test_unpatched_replay_reproduces_the_original_run():
    _faulty_pair()
    res = replay("r_0002", {}, new_run_id="rp_same")
    assert not res["success"] and res["n_reexecuted"] == 0
    assert diff("r_0002", "rp_same")["first_divergent_step"] is None


def test_diff_reports_first_divergence_and_outcome():
    k = _faulty_pair()
    replay("r_0002", {k: oracle_output("r_0002", k)}, new_run_id="rp_test")
    d = diff("r_0002", "rp_test")
    assert d["first_divergent_step"] == k
    assert d["outcome_change"] == "fail -> pass"


# ── Bug-fix regressions ─────────────────────────────────────────────────────

@pytest.mark.parametrize("answer, gold, expected", [
    ("Zenith is older by 11 years.", "Zenith is older by 11 years.", True),
    ("Zenith, founded in 1998, is older by 11 years.", "Zenith is older by 11 years.", True),
    ("NovaTech is older by 11 years.", "Zenith is older by 11 years.", False),  # right number, wrong company
    ("Nexion Labs is older by 12 years.", "Nexion is older by 12 years.", False),  # near-duplicate name
    ("Nexion is older by 12 years.", "Nexion is older by 12 years.", True),
    ("years.", "Zenith is older by 11 years.", False),  # no number at all
    ("18,400 employees combined.", "18400 employees combined.", True),
])
def test_check_success_requires_number_and_company(answer, gold, expected):
    assert _check_success(answer, gold) is expected


def test_value_in_source_is_exact_not_substring():
    from blackbox.features import _value_in_source
    company = {"founded": 2009, "employees": 4200, "name": "NovaTech"}
    assert _value_in_source(2009, company) and _value_in_source("2009", company)
    assert not _value_in_source(0, company)  # "0" is a substring of "2009"
    assert not _value_in_source(2015, company)


def test_calc_args_traceable_fires_on_clean_calculation():
    from blackbox.features import FEATURE_NAMES, extract_features_for_run
    q = next(q for q in _questions() if q["template"] == "older_by")
    run(q["question"], q["id"], "r_0001", q["gold"], "train")
    X, _, meta = extract_features_for_run("r_0001")
    calc_row = next(i for i, m in enumerate(meta) if m["type"] == "calculate")
    assert X[calc_row][FEATURE_NAMES.index("calc_args_traceable")] == 1.0


def test_calculation_consistency_detects_changed_recorded_result():
    from blackbox.features import FEATURE_NAMES, extract_features_for_run

    q = next(q for q in _questions() if q["template"] == "older_by")
    run(q["question"], q["id"], "r_0001", q["gold"], "train")
    clean_calc = next(s for s in recorder.get_steps("r_0001") if s["type"] == "calculate")
    bad_output = json.loads(clean_calc["output_json"])
    bad_output["result"] += 10
    run(q["question"], q["id"], "r_0002", q["gold"], "train",
        fault_type="bad_tool_arg", fault_step=clean_calc["step_idx"],
        overrides={clean_calc["step_idx"]: bad_output})

    X, _, meta = extract_features_for_run("r_0002")
    calc_row = next(i for i, m in enumerate(meta) if m["type"] == "calculate")
    assert X[calc_row][FEATURE_NAMES.index("calc_result_recheckable")] == 1.0
    assert X[calc_row][FEATURE_NAMES.index("calc_result_consistent")] == 0.0


def test_plan_consistency_detects_wrong_operation_and_missing_action():
    from blackbox.features import FEATURE_NAMES, extract_features_for_run

    q = next(q for q in _questions() if q["template"] == "older_by")
    run(q["question"], q["id"], "r_0001", q["gold"], "train")
    plan_step = recorder.get_steps("r_0001")[0]
    clean_plan = json.loads(plan_step["output_json"])
    bad_plan = inject_fault(
        "bad_plan", "plan", clean_plan, json.loads(plan_step["state_before_json"]), get_kb()
    )
    run(q["question"], q["id"], "r_0002", q["gold"], "heldout_fault",
        fault_type="bad_plan", fault_step=0, overrides={0: bad_plan})

    X, _, meta = extract_features_for_run("r_0002")
    plan_row = next(i for i, m in enumerate(meta) if m["type"] == "plan")
    assert X[plan_row][FEATURE_NAMES.index("plan_question_consistent")] == 0.0
    assert X[plan_row][FEATURE_NAMES.index("plan_action_coverage")] == 1.0

    missing_action_plan = json.loads(json.dumps(clean_plan))
    next(a for a in missing_action_plan["actions"] if a["action"] == "calculate")["action"] = "unknown"
    run(q["question"], q["id"], "r_0003", q["gold"], "heldout_fault",
        overrides={0: missing_action_plan})
    X, _, meta = extract_features_for_run("r_0003")
    plan_row = next(i for i, m in enumerate(meta) if m["type"] == "plan")
    assert X[plan_row][FEATURE_NAMES.index("plan_action_coverage")] < 1.0


def test_live_fault_injection_records_a_random_faulty_child_run():
    from blackbox.live import inject_random_fault

    q = next(q for q in _questions() if q["template"] == "older_by")
    run(q["question"], q["id"], "r_0001", q["gold"], "train")
    result = inject_random_fault("r_0001")
    assert result["source_run_id"] == "r_0001"
    assert result["fault_type"] in {
        "wrong_retrieval", "dropped_context", "corrupted_extract", "bad_tool_arg"
    }
    assert result["fault_step"] is not None
    assert recorder.get_run(result["run_id"])["split"] == "live_demo"


def test_wilson_interval_reports_sample_size_uncertainty():
    from blackbox.evaluate import _rate_with_interval

    perfect = _rate_with_interval(40, 40)
    assert perfect["hits"] == 40 and perfect["rate"] == 1.0
    assert perfect["ci95"][0] < 1.0 and perfect["ci95"][1] == 1.0
    assert _rate_with_interval(0, 0)["ci95"] == [0.0, 0.0]


def test_rerecording_a_run_id_drops_stale_steps():
    q = next(q for q in _questions() if q["template"] == "older_by")
    run(q["question"], q["id"], "r_0001", q["gold"], "train")
    short_plan = {"actions": [{"action": "answer", "key": "final"}]}
    run(q["question"], q["id"], "r_0001", q["gold"], "train", overrides={0: short_plan})
    assert len(recorder.get_steps("r_0001")) == 2


def test_cache_key_separates_fallback_from_real_llm(monkeypatch):
    from agent.tools import FALLBACK_MODEL, llm_model_id
    monkeypatch.setenv("GEMINI_API_KEY", "your_gemini_api_key_here")  # placeholder = no key
    assert llm_model_id() == FALLBACK_MODEL
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-not-a-real-key")
    assert llm_model_id() != FALLBACK_MODEL


def test_build_all_creates_every_artifact(tmp_path, monkeypatch):
    import blackbox.build as build
    import blackbox.evaluate as evaluate
    import blackbox.model as model
    monkeypatch.setattr(model, "MODEL_PATH", tmp_path / "model.pkl")
    monkeypatch.setattr(evaluate, "METRICS_PATH", tmp_path / "metrics.json")
    monkeypatch.setattr(build, "ARTIFACTS", (recorder.DB_PATH, model.MODEL_PATH, evaluate.METRICS_PATH))
    assert not build.is_built()
    build.build_all()
    assert build.is_built()
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["seen_faults_test"]["n_runs"] > 0 and metrics["heldout_faults"]["n_runs"] > 0
    assert recorder.get_run("r_9002") is not None  # demo run recorded
