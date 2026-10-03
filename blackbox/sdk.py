"""sdk.py — Record any Python agent with a decorator.

    import blackbox as bb

    @bb.step("retrieve")
    def search(name: str) -> dict: ...

    @bb.step("answer", model="my-llm")
    def respond(question: str, facts: dict) -> dict: ...

    def my_agent(question: str) -> str:
        doc = search(question)
        bb.state()["doc"] = doc          # optional: facts the agent has collected
        return respond(question, {"doc": doc})["answer"]

    bb.register_agent("my_agent", my_agent)
    bb.record("my_agent", "What is ...?", run_id="m_0001", gold="...")

Every decorated call goes through record_step(), so it gets the same trace, cache,
blame features and patch & replay as the built-in agent. Outside a recording the
decorated functions behave exactly like the plain functions.

Feature contract (optional, improves diagnosis): retrieve steps take `name` (or
`query`) and return `title` (+ `top_scores`); extract steps take `source` and return
`value`; calculate steps take `expr` and read names from bb.state().
"""
from __future__ import annotations

import contextvars
import copy
import functools
import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from blackbox.recorder import cache_stats, delete_run, init_db, record_step, save_run, update_run_counts

STEP_TYPES = ("plan", "retrieve", "extract", "calculate", "answer")

_AGENTS: dict[str, Callable[..., Any]] = {}


@dataclass
class _Trace:
    run_id: str
    overrides: dict[int, Any]
    recorded_steps: dict[int, dict]
    state: dict = field(default_factory=dict)
    outputs: list[Any] = field(default_factory=list)


_current: contextvars.ContextVar[_Trace | None] = contextvars.ContextVar("blackbox_trace", default=None)


# ── decorator ────────────────────────────────────────────────────────────────

def step(step_type: str, model: str = "tool") -> Callable:
    """Record each call of the decorated function as one agent step.

    Args:
        step_type: one of plan, retrieve, extract, calculate, answer.
        model: backend identity for the cache key (use your LLM's model name for LLM calls).

    Inputs (bound arguments) and outputs must be JSON-serialisable.
    """
    if step_type not in STEP_TYPES:
        raise ValueError(f"step_type must be one of {STEP_TYPES}, got {step_type!r}")

    def decorate(fn: Callable) -> Callable:
        signature = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            trace = _current.get()
            if trace is None:  # not recording: plain call
                return fn(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            input_data = json.loads(json.dumps(dict(bound.arguments), default=str))
            idx = len(trace.outputs)
            output = record_step(
                run_id=trace.run_id, idx=idx, step_type=step_type, input_data=input_data,
                fn=lambda _inp: fn(*bound.args, **bound.kwargs),
                state_before=copy.deepcopy(trace.state),
                parents=_parents_by_value(input_data, trace.outputs),
                model=model,
                override=trace.overrides.get(idx),
                recorded=trace.recorded_steps.get(idx),
            )
            trace.outputs.append(output)
            return output

        wrapper.blackbox_step_type = step_type
        return wrapper

    return decorate


def state() -> dict:
    """The current recording's fact store (an empty throwaway dict when not recording)."""
    trace = _current.get()
    return trace.state if trace is not None else {}


# ── agents ───────────────────────────────────────────────────────────────────

def register_agent(name: str, fn: Callable[[str], Any]) -> None:
    """Register an agent `fn(question) -> answer` so its runs can be replayed by name."""
    _AGENTS[name] = fn


def get_agent(name: str) -> Callable[[str], Any]:
    if name not in _AGENTS:
        raise ValueError(f"Agent {name!r} is not registered; import the module that calls register_agent()")
    return _AGENTS[name]


def record(
    agent: str,
    question: str,
    run_id: str,
    gold: str | None = None,
    question_id: str | None = None,
    split: str = "sdk",
    fault_type: str | None = None,
    fault_step: int | None = None,
    overrides: dict[int, Any] | None = None,
    recorded_steps: dict[int, dict] | None = None,
    parent_run_id: str | None = None,
) -> dict:
    """Run a registered agent on `question` and record it as `run_id` (replacing any old one).

    Returns the same shape as agent.runner.run(): run_id, final_answer, success,
    n_steps, n_reused, n_reexecuted.
    """
    from agent.runner import _check_success  # lazy: agent.runner imports blackbox

    init_db()
    delete_run(run_id)
    trace = _Trace(run_id=run_id, overrides={int(k): v for k, v in (overrides or {}).items()},
                   recorded_steps=recorded_steps or {})
    token = _current.set(trace)
    try:
        result = get_agent(agent)(question)
    finally:
        _current.reset(token)

    final_answer = result.get("answer") if isinstance(result, dict) else result
    final_answer = None if final_answer is None else str(final_answer)
    success = _check_success(final_answer, gold) if gold else False
    save_run(
        run_id=run_id, question_id=question_id or f"{agent}:{question}", question=question,
        gold=gold or "", final_answer=final_answer, success=success, split=split,
        fault_type=fault_type, fault_step=fault_step, parent_run_id=parent_run_id, agent=agent,
    )
    stats = cache_stats(run_id)
    update_run_counts(run_id, stats["reused"], stats["reexecuted"])
    return {
        "run_id": run_id, "final_answer": final_answer, "success": success,
        "n_steps": len(trace.outputs), "n_reused": stats["reused"], "n_reexecuted": stats["reexecuted"],
    }


# ── data flow ────────────────────────────────────────────────────────────────

def _leaves(value: Any) -> set[str]:
    """Distinctive scalar values inside a JSON value (small numbers and short strings are too common)."""
    found: set[str] = set()
    if isinstance(value, dict):
        for v in value.values():
            found |= _leaves(v)
    elif isinstance(value, list):
        for v in value:
            found |= _leaves(v)
    elif isinstance(value, bool) or value is None:
        pass
    elif isinstance(value, (int, float)):
        if abs(value) >= 10:
            found.add(repr(float(value)))
    elif isinstance(value, str) and len(value) >= 3:
        found.add(value)
    return found


def _parents_by_value(input_data: Any, outputs: list[Any]) -> list[int]:
    """Earlier steps whose output values reappear in this step's input (value provenance)."""
    wanted = _leaves(input_data)
    return [i for i, out in enumerate(outputs) if out is not None and _leaves(out) & wanted]
