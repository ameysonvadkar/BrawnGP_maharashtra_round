"""tools.py — retrieve, extract, calculate, answer, plan for the Black Box agent.

All public functions are pure callables; they are wrapped by runner.py via
record_step() so nothing here calls the LLM directly in a way that bypasses
the recorder.
"""
from __future__ import annotations

import ast
import json
import os
import pathlib
import re
from typing import Any

import numpy as np
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

# ── Knowledge-base loading ───────────────────────────────────────────────────

_KB_PATH = pathlib.Path(__file__).parent / "kb.json"


def _load_kb() -> list[dict]:
    return json.loads(_KB_PATH.read_text(encoding="utf-8-sig"))


_KB: list[dict] = _load_kb()
_KB_BY_ID: dict[str, dict] = {c["id"]: c for c in _KB}

# Pre-build TF-IDF index over company names
_names = [c["name"] for c in _KB]
_ids = [c["id"] for c in _KB]
_tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit(_names)
_tfidf_matrix = _tfidf.transform(_names)

NEAR_DUPS = [
    ("novatech", "nova_systems"),("zenith", "zenit_labs"),("apexion", "apex_data"),
    ("prismworks", "prism_ai"),("luminos", "luminos_tech"),("orbita", "orbit_cloud"),
    ("vectora", "vector_core"),("stratum", "strata_net"),("inferno", "infernix"),
    ("cruxlab", "crux_ai"),("neuralaxis", "neuralax"),("datavex", "datavex_ai"),
    ("solara", "solara_soft"),("hypercor", "hypercore"),("mosaiq", "mosaic_data"),
    ("quanta", "quantax"),("spectral", "spectra_ai"),("terracor", "terra_tech"),
    ("nexion", "nexion_labs"),("kaleidix", "kalex_io"),
]

# ── Retrieve ─────────────────────────────────────────────────────────────────


def retrieve(input_data: dict) -> dict:
    """Return the best-matching company document for a name query."""
    query = input_data["name"]
    qvec = _tfidf.transform([query])
    tfidf_scores = cosine_similarity(qvec, _tfidf_matrix)[0]
    fuzz_scores = np.array([fuzz.token_sort_ratio(query, n) / 100.0 for n in _names])
    combined = 0.5 * tfidf_scores + 0.5 * fuzz_scores
    ranked = np.argsort(combined)[::-1]
    top1_idx, top2_idx = ranked[0], ranked[1]
    doc = _KB[top1_idx]
    return {
        "doc_id": doc["id"],
        "title": doc["name"],
        "company": doc,
        "top_scores": [round(float(combined[top1_idx]), 4), round(float(combined[top2_idx]), 4)],
    }


def retrieve_by_id(doc_id: str) -> dict:
    """Return a company doc by exact ID (used in fault injection)."""
    c = _KB_BY_ID[doc_id]
    return {"doc_id": c["id"], "title": c["name"], "company": c, "top_scores": [1.0, 0.0]}


def get_distractor(doc_id: str) -> str | None:
    """Return the near-duplicate distractor id for a given company id."""
    for a, b in NEAR_DUPS:
        if doc_id == a:
            return b
        if doc_id == b:
            return a
    return None


def get_kb() -> list[dict]:
    """Return the full KB (used by fault injector)."""
    return _KB


# ── LLM helper with deterministic fallback ──────────────────────────────────

_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
_API_KEY = os.getenv("GEMINI_API_KEY", "")


def _call_gemini(prompt: str) -> str | None:
    api_key = os.getenv("GEMINI_API_KEY", _API_KEY)
    if not api_key or api_key.startswith("your_"):
        return None
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=_GEMINI_MODEL,
            contents=prompt,
        )
        return response.text.strip()
    except Exception:
        return None


# ── Extract ──────────────────────────────────────────────────────────────────

_EXTRACT_PROMPT = """You are an information extractor. Given a company profile JSON and a field name, return ONLY the raw value for that field as a plain string. No explanation. No JSON.

Company profile: {company_json}
Field: {field}
Value:"""


def extract(input_data: dict) -> dict:
    """LLM / structured call: extract a field value from a company doc."""
    company = input_data["company"]
    field = input_data["field"]
    if field in company:
        return {"value": company[field], "field": field}
    
    prompt = _EXTRACT_PROMPT.format(
        company_json=json.dumps(company, ensure_ascii=False),
        field=field,
    )
    res = _call_gemini(prompt)
    if res:
        return {"value": res, "field": field}
    return {"value": company.get(field, ""), "field": field}


# ── Calculate ────────────────────────────────────────────────────────────────

SAFE_BUILTINS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "int": int,
    "float": float,
    "pow": pow,
}


def calculate(input_data: dict) -> dict:
    """Evaluate a safe arithmetic expression."""
    expr = input_data["expr"]
    state = input_data.get("state", {})
    allowed_names = {}
    for k, v in state.items():
        if isinstance(v, (int, float)):
            allowed_names[k] = v
        elif isinstance(v, str):
            allowed_names[k] = _coerce_num(v)
    _check_safe_expr(expr, allowed_names)
    result = eval(expr, {"__builtins__": SAFE_BUILTINS}, allowed_names)  # noqa: S307
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    elif isinstance(result, float):
        result = round(result, 2)
    return {"result": result, "expr": expr}


_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Load, ast.Call,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd,
)


def _check_safe_expr(expr: str, names: dict) -> None:
    """Allow only arithmetic over numbers, state keys and SAFE_BUILTINS calls."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Unsafe expression: {expr}") from exc
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"Unsafe expression: {expr}")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError(f"Unsafe expression: {expr}")
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in SAFE_BUILTINS) or node.keywords:
                raise ValueError(f"Unsafe expression: {expr}")
        if isinstance(node, ast.Name) and node.id not in names and node.id not in SAFE_BUILTINS:
            raise ValueError(f"Unknown name in expression: {node.id}")


def _coerce_num(v: Any) -> float | int:
    try:
        f = float(str(v).replace(',', '').replace('$', '').replace('K', ''))
        return int(f) if f == int(f) else f
    except (ValueError, TypeError):
        return 0


# ── Answer ───────────────────────────────────────────────────────────────────

_ANSWER_PROMPT = """You are a precise assistant. Given the collected facts and a question, give a one-sentence factual answer.

Question: {question}
Facts: {facts_json}
Answer (one sentence, be specific with numbers):"""


def answer(input_data: dict) -> dict:
    """LLM call: produce the final answer."""
    q = input_data["question"]
    state = input_data["state"]

    prompt = _ANSWER_PROMPT.format(
        question=q,
        facts_json=json.dumps(state, ensure_ascii=False),
    )
    res = _call_gemini(prompt)
    if res:
        return {"answer": res}

    # Deterministic fallback answer generator based on state & question template
    if "older by" in q.lower() or "which company is older" in q.lower():
        diff = state.get("calc_diff", state.get("diff", state.get("val_diff", None)))
        if diff is None:
            calcs = [v for k, v in state.items() if "calc" in k or "diff" in k]
            diff = calcs[0] if calcs else 0
        
        c1_founded = _coerce_num(state.get("c1_founded", 9999))
        c2_founded = _coerce_num(state.get("c2_founded", 9999))
        c1_doc = state.get("doc_c1", {})
        c2_doc = state.get("doc_c2", {})
        c1_name = c1_doc.get("title", "Company 1") if isinstance(c1_doc, dict) else "Company 1"
        c2_name = c2_doc.get("title", "Company 2") if isinstance(c2_doc, dict) else "Company 2"
        
        older_name = c1_name if c1_founded < c2_founded else c2_name
        ans = f"{older_name} is older by {abs(diff)} years."
        return {"answer": ans}

    elif "revenue per employee" in q.lower():
        rev_k = state.get("calc_rev_k", state.get("rev_k", 0))
        return {"answer": f"${rev_k}K per employee."}

    elif "combined employee count" in q.lower():
        comb = state.get("calc_combined", state.get("combined", 0))
        return {"answer": f"{comb} employees combined."}

    elif "how many more employees" in q.lower():
        diff = state.get("calc_diff", state.get("diff", 0))
        c1_emp = _coerce_num(state.get("c1_emp", 0))
        c2_emp = _coerce_num(state.get("c2_emp", 0))
        c1_doc = state.get("doc_c1", {})
        c2_doc = state.get("doc_c2", {})
        c1_name = c1_doc.get("title", "Company 1") if isinstance(c1_doc, dict) else "Company 1"
        c2_name = c2_doc.get("title", "Company 2") if isinstance(c2_doc, dict) else "Company 2"
        more_name = c1_name if c1_emp > c2_emp else c2_name
        return {"answer": f"{more_name} has {abs(diff)} more employees."}

    val_str = ", ".join(f"{k}={v}" for k, v in state.items() if not k.startswith("doc_"))
    return {"answer": f"Based on collected facts: {val_str}"}


# ── Plan ─────────────────────────────────────────────────────────────────────

_PLAN_PROMPT = """You are a planner for a company-facts QA agent. Given a question, return a JSON array of action objects to answer it.

Available actions:
- {{"action": "retrieve", "name": "<company name>", "key": "<state_key_to_store_doc>"}}
- {{"action": "extract", "doc_key": "<state_key_of_doc>", "field": "<field_name>", "key": "<state_key>"}}
- {{"action": "calculate", "expr": "<python_expression_using_state_keys>", "key": "<state_key>"}}
- {{"action": "answer", "key": "final"}}

Return ONLY a valid JSON array, no explanation.

Question: {question}"""


def plan(input_data: dict) -> dict:
    """LLM call or rule-based parser: produce a plan as a JSON list of actions."""
    q = input_data["question"]
    prompt = _PLAN_PROMPT.format(question=q)
    raw = _call_gemini(prompt)
    if raw:
        try:
            raw_clean = re.sub(r'^```[\w]*\n?', '', raw, flags=re.MULTILINE)
            raw_clean = re.sub(r'```$', '', raw_clean, flags=re.MULTILINE)
            actions = json.loads(raw_clean.strip())
            return {"actions": actions}
        except Exception:
            pass

    actions = []
    q_lower = q.lower()
    
    if "older" in q_lower:
        m = re.search(r"older,\s*(.*?)\s+or\s+(.*?),\s*and\b", q, re.IGNORECASE)
        if not m:
            m = re.search(r"older,\s*(.*?)\s+or\s+(.*?)\?", q, re.IGNORECASE)
        name1 = m.group(1).strip() if m else "NovaTech"
        name2 = m.group(2).strip() if m else "Zenith"
        actions = [
            {"action": "retrieve", "name": name1, "key": "doc_c1"},
            {"action": "extract", "doc_key": "doc_c1", "field": "founded", "key": "c1_founded"},
            {"action": "retrieve", "name": name2, "key": "doc_c2"},
            {"action": "extract", "doc_key": "doc_c2", "field": "founded", "key": "c2_founded"},
            {"action": "calculate", "expr": "abs(c1_founded - c2_founded)", "key": "calc_diff"},
            {"action": "answer", "key": "final"}
        ]
    elif "revenue per employee" in q_lower:
        m = re.search(r"revenue per employee.*\bof\s+(.*?)\?", q, re.IGNORECASE)
        name = m.group(1).strip() if m else "NovaTech"
        actions = [
            {"action": "retrieve", "name": name, "key": "doc_c1"},
            {"action": "extract", "doc_key": "doc_c1", "field": "revenue_m", "key": "c1_rev"},
            {"action": "extract", "doc_key": "doc_c1", "field": "employees", "key": "c1_emp"},
            {"action": "calculate", "expr": "round(c1_rev * 1000 / c1_emp, 2)", "key": "calc_rev_k"},
            {"action": "answer", "key": "final"}
        ]
    elif "combined employee count" in q_lower:
        m = re.search(r"combined employee count of\s+(.*?)\s+and\s+(.*?)\?", q, re.IGNORECASE)
        name1 = m.group(1).strip() if m else "NovaTech"
        name2 = m.group(2).strip() if m else "Zenith"
        actions = [
            {"action": "retrieve", "name": name1, "key": "doc_c1"},
            {"action": "extract", "doc_key": "doc_c1", "field": "employees", "key": "c1_emp"},
            {"action": "retrieve", "name": name2, "key": "doc_c2"},
            {"action": "extract", "doc_key": "doc_c2", "field": "employees", "key": "c2_emp"},
            {"action": "calculate", "expr": "c1_emp + c2_emp", "key": "calc_combined"},
            {"action": "answer", "key": "final"}
        ]
    elif "how many more employees" in q_lower:
        m = re.search(r"how many more employees does\s+(.*?)\s+have compared to\s+(.*?)\?", q, re.IGNORECASE)
        name1 = m.group(1).strip() if m else "NovaTech"
        name2 = m.group(2).strip() if m else "Zenith"
        actions = [
            {"action": "retrieve", "name": name1, "key": "doc_c1"},
            {"action": "extract", "doc_key": "doc_c1", "field": "employees", "key": "c1_emp"},
            {"action": "retrieve", "name": name2, "key": "doc_c2"},
            {"action": "extract", "doc_key": "doc_c2", "field": "employees", "key": "c2_emp"},
            {"action": "calculate", "expr": "abs(c1_emp - c2_emp)", "key": "calc_diff"},
            {"action": "answer", "key": "final"}
        ]
    else:
        actions = [
            {"action": "retrieve", "name": "NovaTech", "key": "doc_c1"},
            {"action": "answer", "key": "final"}
        ]
    return {"actions": actions}
