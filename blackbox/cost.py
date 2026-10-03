"""cost.py — Estimate the LLM cost a cached replay avoids.

Tokens are estimated as characters / 4 of the prompt (template + input) and the
output. Prices default to Gemini 2.0 Flash list prices and can be changed in .env:

    LLM_PRICE_IN_PER_M=0.10     # USD per 1M input tokens
    LLM_PRICE_OUT_PER_M=0.40    # USD per 1M output tokens
    USD_INR=88                  # exchange rate for the ₹ figure

These are estimates for comparing a full re-run with a suffix-only replay, not a bill.
"""
from __future__ import annotations

import os

from agent.tools import PROMPT_TEMPLATES
from blackbox.recorder import get_steps

LLM_STEP_TYPES = ("plan", "extract", "answer")
CHARS_PER_TOKEN = 4


def prices() -> dict[str, float]:
    def env(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, default))
        except ValueError:
            return default
    return {
        "in_per_m": env("LLM_PRICE_IN_PER_M", 0.10),
        "out_per_m": env("LLM_PRICE_OUT_PER_M", 0.40),
        "usd_inr": env("USD_INR", 88.0),
    }


def step_tokens(step: dict) -> tuple[int, int]:
    """(input, output) token estimate for one recorded LLM step."""
    template = PROMPT_TEMPLATES.get(step["type"], "")
    tokens_in = (len(template) + len(step["input_json"] or "")) // CHARS_PER_TOKEN
    tokens_out = len(step["output_json"] or "") // CHARS_PER_TOKEN
    return tokens_in, tokens_out


def replay_savings(replay_run_id: str, patched_steps: list[int] | set[int]) -> dict[str, float]:
    """Compare a full re-run of every LLM step with what this replay actually executed.

    A full re-run pays for every LLM step. The replay pays only for LLM steps that
    were re-executed (not patched, not reused from the trace or cache).
    """
    p = prices()
    patched = set(patched_steps)
    full_in = full_out = spent_in = spent_out = 0
    for s in get_steps(replay_run_id):
        if s["type"] not in LLM_STEP_TYPES:
            continue
        t_in, t_out = step_tokens(s)
        full_in, full_out = full_in + t_in, full_out + t_out
        if s["step_idx"] not in patched and not s["cache_hit"]:
            spent_in, spent_out = spent_in + t_in, spent_out + t_out

    def usd(t_in: int, t_out: int) -> float:
        return (t_in * p["in_per_m"] + t_out * p["out_per_m"]) / 1_000_000

    full_usd, spent_usd = usd(full_in, full_out), usd(spent_in, spent_out)
    full_tokens, spent_tokens = full_in + full_out, spent_in + spent_out
    saved_usd = full_usd - spent_usd
    return {
        "tokens_full": full_tokens,
        "tokens_spent": spent_tokens,
        "tokens_saved": full_tokens - spent_tokens,
        "pct_saved": (1 - spent_tokens / full_tokens) if full_tokens else 0.0,
        "usd_saved": saved_usd,
        "inr_saved": saved_usd * p["usd_inr"],
        "inr_saved_per_1000": saved_usd * p["usd_inr"] * 1000,
    }
