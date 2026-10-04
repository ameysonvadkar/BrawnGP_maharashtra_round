"""FastAPI interface for Black Box traces, replay, live demos, and UI data.

Run from the agent-blackbox directory:
    uvicorn api.main:app --reload
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.tools import get_kb  # noqa: E402
from agent.runner import run as run_agent  # noqa: E402
from blackbox import recorder  # noqa: E402
from blackbox.build import build_all  # noqa: E402
from blackbox.cost import prices, replay_savings  # noqa: E402
from blackbox.features import FEATURE_NAMES  # noqa: E402
from blackbox.live import inject_random_fault  # noqa: E402
from blackbox.model import MODEL_PATH, predict  # noqa: E402
from blackbox.replay import (  # noqa: E402
    REPLAY_SPLIT,
    diff,
    find_clean_run,
    oracle_output,
    replay,
    resume_state,
    verify_step,
)

METRICS_PATH = ROOT / "data" / "metrics.json"
QUESTIONS_PATH = ROOT / "agent" / "questions.json"


class RunSummary(BaseModel):
    model_config = ConfigDict(extra="allow")

    run_id: str
    question_id: str | None = None
    question: str
    gold: str | None = None
    final_answer: str | None = None
    success: bool
    fault_type: str | None = None
    fault_step: int | None = None
    parent_run_id: str | None = None
    split: str
    created_at: str | None = None
    agent: str | None = None


class StepPrediction(BaseModel):
    step_idx: int
    type: str
    score: float
    rank: int
    reasons: list[dict[str, Any]]
    direct_evidence: list[str] = Field(default_factory=list)


class StepRecord(BaseModel):
    step_idx: int
    type: str
    input: Any = None
    output: Any = None
    state_before: dict[str, Any] = Field(default_factory=dict)
    parents: list[int] = Field(default_factory=list)
    cache_hit: bool
    latency_ms: int
    error: str | None = None
    blame_score: float = 0.0
    blame_rank: int | None = None
    reasons: list[dict[str, Any]] = Field(default_factory=list)
    direct_evidence: list[str] = Field(default_factory=list)


class RunDetail(BaseModel):
    run: RunSummary
    top_blame_step: int | None = None
    steps: list[StepRecord]
    predictions: list[StepPrediction]


class ReplayRequest(BaseModel):
    overrides: dict[int, Any] = Field(default_factory=dict)
    new_run_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,80}$")


class RunRequest(BaseModel):
    question_id: str | None = None
    question: str | None = Field(default=None, min_length=1, max_length=5000)
    gold: str | None = Field(default=None, min_length=1, max_length=1000)


class LiveInjectionRequest(BaseModel):
    clean_run_id: str | None = None


class AutoVerifyRequest(BaseModel):
    top_k: int = Field(default=3, ge=1, le=10)


@asynccontextmanager
async def _lifespan(_: FastAPI):
    build_all()
    yield


def create_app(initialize: bool = True) -> FastAPI:
    """Create the API application; disable initialization for isolated unit tests."""
    app = FastAPI(
        title="Black Box API",
        description=(
            "Frontend API for diagnosing agent runs, inspecting evidence, replaying patches, "
            "viewing benchmark/UI data, and injecting live demo faults."
        ),
        version="1.0.0",
        lifespan=_lifespan if initialize else None,
    )
    origins = [
        origin.strip()
        for origin in os.getenv("BLACKBOX_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    def require_run(run_id: str) -> dict:
        run_info = recorder.get_run(run_id)
        if run_info is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        return run_info

    def decoded(raw: str | None, default: Any) -> Any:
        return json.loads(raw) if raw else default

    def serialize_step(step: dict, prediction: dict | None, rank: int | None) -> StepRecord:
        prediction = prediction or {}
        return StepRecord(
            step_idx=step["step_idx"],
            type=step["type"],
            input=decoded(step.get("input_json"), None),
            output=decoded(step.get("output_json"), None),
            state_before=decoded(step.get("state_before_json"), {}),
            parents=decoded(step.get("parents_json"), []),
            cache_hit=bool(step["cache_hit"]),
            latency_ms=step["latency_ms"],
            error=step["error"],
            blame_score=prediction.get("score", 0.0),
            blame_rank=rank,
            reasons=prediction.get("reasons", []),
            direct_evidence=prediction.get("direct_evidence", []),
        )

    @app.get("/api/health", tags=["system"])
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "database": recorder.DB_PATH.exists(),
            "model": MODEL_PATH.exists(),
            "metrics": METRICS_PATH.exists(),
        }

    @app.get("/api/runs", response_model=list[RunSummary], tags=["black-box"])
    def list_recorded_runs(
        split: str | None = None,
        success: bool | None = None,
        fault_type: str | None = None,
        include_replays: bool = False,
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> list[dict]:
        records = recorder.list_runs(split=split, success=int(success) if success is not None else None)
        if not include_replays:
            records = [r for r in records if r["split"] != REPLAY_SPLIT]
        if fault_type is not None:
            records = [r for r in records if (r["fault_type"] or "clean") == fault_type]
        return records[offset:offset + limit]

    @app.get("/api/questions", tags=["black-box"])
    def list_questions() -> list[dict[str, Any]]:
        return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8-sig"))

    @app.post("/api/runs/execute", tags=["black-box"])
    def execute_question(request: RunRequest) -> dict[str, Any]:
        if request.question_id:
            questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8-sig"))
            question_data = next(
                (item for item in questions if item["id"] == request.question_id), None
            )
            if question_data is None:
                raise HTTPException(status_code=404, detail=f"Question '{request.question_id}' not found")
            if request.question is not None and request.question != question_data["question"]:
                raise HTTPException(status_code=422, detail="Question text does not match question_id")
            if request.gold is not None and request.gold != question_data["gold"]:
                raise HTTPException(status_code=422, detail="Gold answer does not match question_id")
            question_text = question_data["question"]
            gold = question_data["gold"]
            question_id = question_data["id"]
        else:
            if request.question is None or request.gold is None:
                raise HTTPException(
                    status_code=422,
                    detail="Provide question_id or both question and gold for scored custom runs",
                )
            question_text = request.question
            gold = request.gold
            question_id = f"custom-{uuid.uuid4().hex[:12]}"
        run_id = f"api_{uuid.uuid4().hex[:12]}"
        execution = run_agent(
            question=question_text,
            question_id=question_id,
            run_id=run_id,
            gold=gold,
            split="api",
        )
        return {
            "execution": execution,
            "diagnosis": get_run_detail(run_id).model_dump(),
        }

    @app.get("/api/runs/{run_id}", response_model=RunDetail, tags=["black-box"])
    def get_run_detail(run_id: str) -> RunDetail:
        run_info = require_run(run_id)
        steps = recorder.get_steps(run_id)
        raw_predictions = predict(run_id, top_k=len(FEATURE_NAMES)) if steps else []
        ranked = sorted(raw_predictions, key=lambda p: p["score"], reverse=True)
        rank_by_idx = {prediction["step_idx"]: rank for rank, prediction in enumerate(ranked, 1)}
        prediction_by_idx = {prediction["step_idx"]: prediction for prediction in raw_predictions}
        top_idx = ranked[0]["step_idx"] if ranked else None
        return RunDetail(
            run=run_info,
            top_blame_step=top_idx,
            steps=[
                serialize_step(step, prediction_by_idx.get(step["step_idx"]), rank_by_idx.get(step["step_idx"]))
                for step in steps
            ],
            predictions=[
                StepPrediction(**prediction, rank=rank_by_idx[prediction["step_idx"]])
                for prediction in ranked
            ],
        )

    @app.get("/api/runs/{run_id}/steps/{step_idx}", response_model=StepRecord, tags=["black-box"])
    def get_step_detail(run_id: str, step_idx: int) -> StepRecord:
        require_run(run_id)
        step = next((s for s in recorder.get_steps(run_id) if s["step_idx"] == step_idx), None)
        if step is None:
            raise HTTPException(status_code=404, detail=f"Step {step_idx} not found in run '{run_id}'")
        predictions = predict(run_id, top_k=len(FEATURE_NAMES))
        prediction = next((p for p in predictions if p["step_idx"] == step_idx), None)
        ranked = sorted(predictions, key=lambda p: p["score"], reverse=True)
        rank = next((i for i, p in enumerate(ranked, 1) if p["step_idx"] == step_idx), None)
        return serialize_step(step, prediction, rank)

    @app.get("/api/runs/{run_id}/clean-reference", tags=["black-box"])
    def clean_reference(run_id: str) -> dict[str, Any]:
        require_run(run_id)
        clean = find_clean_run(run_id)
        return {"run": clean}

    @app.get("/api/runs/{run_id}/oracle/{step_idx}", tags=["black-box"])
    def get_oracle_output(run_id: str, step_idx: int) -> dict[str, Any]:
        require_run(run_id)
        output = oracle_output(run_id, step_idx)
        if output is None:
            raise HTTPException(
                status_code=404,
                detail=f"No clean-run oracle output for step {step_idx} of run '{run_id}'",
            )
        return {"run_id": run_id, "step_idx": step_idx, "output": output}

    @app.post("/api/runs/{run_id}/replays", tags=["black-box"])
    def create_replay(run_id: str, request: ReplayRequest) -> dict[str, Any]:
        require_run(run_id)
        known_steps = {s["step_idx"] for s in recorder.get_steps(run_id)}
        invalid = sorted(set(request.overrides) - known_steps)
        if invalid:
            raise HTTPException(status_code=422, detail=f"Unknown step indices: {invalid}")
        try:
            result = replay(run_id, request.overrides, new_run_id=request.new_run_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        result["diff"] = diff(run_id, result["run_id"])
        result["savings"] = replay_savings(result["run_id"], result["patched_steps"])
        result["prices"] = prices()
        return result

    @app.get("/api/runs/{run_id}/replays", tags=["black-box"])
    def list_replays(run_id: str) -> list[dict[str, Any]]:
        require_run(run_id)
        return [
            r for r in recorder.list_runs(split=REPLAY_SPLIT)
            if r["parent_run_id"] == run_id
        ]

    @app.post("/api/runs/{run_id}/verify/{step_idx}", tags=["black-box"])
    def verify_blame(run_id: str, step_idx: int) -> dict[str, Any]:
        require_run(run_id)
        try:
            return verify_step(run_id, step_idx)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/runs/{run_id}/verify-top", tags=["black-box"])
    def verify_top_blames(run_id: str, request: AutoVerifyRequest) -> dict[str, Any]:
        require_run(run_id)
        ranked = sorted(
            predict(run_id, top_k=len(FEATURE_NAMES)),
            key=lambda prediction: prediction["score"],
            reverse=True,
        )[:request.top_k]
        return {
            "run_id": run_id,
            "results": [
                {
                    "rank": rank,
                    "prediction": prediction,
                    "verification": verify_step(
                        run_id, prediction["step_idx"], new_run_id=f"api_verify_{run_id}_{rank}"
                    ),
                }
                for rank, prediction in enumerate(ranked, 1)
            ],
        }

    @app.get("/api/runs/{run_id}/diff/{other_run_id}", tags=["black-box"])
    def compare_runs(run_id: str, other_run_id: str) -> dict[str, Any]:
        require_run(run_id)
        require_run(other_run_id)
        return diff(run_id, other_run_id)

    @app.get("/api/runs/{run_id}/replay-savings/{replay_run_id}", tags=["black-box"])
    def get_replay_savings(run_id: str, replay_run_id: str) -> dict[str, Any]:
        require_run(run_id)
        replay_info = require_run(replay_run_id)
        if replay_info["parent_run_id"] != run_id or replay_info["split"] != REPLAY_SPLIT:
            raise HTTPException(status_code=422, detail="The selected run is not a replay of this run")
        changes = diff(run_id, replay_run_id)
        patches = [
            row["step_idx"] for row in changes["step_diffs"]
            if row["input_changed"] is False and row["output_changed"] is True
        ]
        return {"savings": replay_savings(replay_run_id, patches), "prices": prices()}

    @app.get("/api/metrics", tags=["dashboard"])
    def get_metrics() -> dict[str, Any]:
        if not METRICS_PATH.exists():
            raise HTTPException(status_code=503, detail="Metrics are not built yet")
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    @app.get("/api/dashboard", tags=["dashboard"])
    def dashboard_data() -> dict[str, Any]:
        records = [r for r in recorder.list_runs() if r["split"] != REPLAY_SPLIT]
        by_split: dict[str, int] = {}
        by_fault: dict[str, int] = {}
        for run_info in records:
            by_split[run_info["split"]] = by_split.get(run_info["split"], 0) + 1
            fault = run_info["fault_type"] or "clean"
            by_fault[fault] = by_fault.get(fault, 0) + 1
        return {
            "total_runs": len(records),
            "passed": sum(bool(r["success"]) for r in records),
            "failed": sum(not bool(r["success"]) for r in records),
            "by_split": by_split,
            "by_fault": by_fault,
            "metrics": get_metrics(),
            "cost_prices": prices(),
        }

    @app.get("/api/knowledge-base", tags=["dashboard"])
    def knowledge_base() -> list[dict[str, Any]]:
        return get_kb()

    @app.get("/api/features", tags=["dashboard"])
    def feature_catalog() -> dict[str, Any]:
        return {
            "features": FEATURE_NAMES,
            "status_thresholds": {"ok": 0.0, "watch": 0.2, "suspect": 0.5},
            "metric_definitions": {
                "top1": "Fault step is ranked first.",
                "root_cause_verified": (
                    "Patching the blamed step flips failure to success and its input matches "
                    "the corresponding clean run."
                ),
                "any_patch_flip": "Patching the blamed step flips failure to success; not proof of root cause.",
            },
        }

    @app.get("/api/live-demo/clean-runs", response_model=list[RunSummary], tags=["live-demo"])
    def list_live_demo_sources() -> list[dict]:
        return [
            r for r in recorder.list_runs(success=1)
            if r["fault_type"] is None and r["parent_run_id"] is None and r.get("agent") is None
            and r["split"] not in {REPLAY_SPLIT, "live_demo"}
        ]

    @app.post("/api/live-demo/injections", tags=["live-demo"])
    def create_live_injection(request: LiveInjectionRequest) -> dict[str, Any]:
        source_id = request.clean_run_id
        if source_id is None:
            sources = list_live_demo_sources()
            if not sources:
                raise HTTPException(status_code=503, detail="No successful clean source runs are available")
            import random
            source_id = random.choice(sources)["run_id"]
        try:
            result = inject_random_fault(source_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        detail = get_run_detail(result["run_id"])
        return {
            "injection": result,
            "ranker": detail.model_dump(),
            "caught_top1": detail.top_blame_step == result["fault_step"],
        }

    return app


app = create_app()
