"""streamlit_app.py — Black Box debugger UI.

    streamlit run app/streamlit_app.py

Debugger tab: pick a run, see each step's blame score, inspect the evidence,
patch a step and replay only what comes after it, then diff the two runs.
Metrics tab: Top-1 / Top-3 / MRR on seen vs held-out faults against baselines.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)  # data/ paths in the blackbox modules are relative to the repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from agent.tools import get_kb, retrieve_by_id  # noqa: E402
from blackbox import recorder  # noqa: E402
from blackbox.build import build_all, is_built  # noqa: E402
from blackbox.demo import DEMO_FAULT_RUN, ensure_demo  # noqa: E402
from blackbox.features import FEATURE_NAMES  # noqa: E402
from blackbox.model import MODEL_PATH, predict  # noqa: E402
from blackbox.replay import (  # noqa: E402
    REPLAY_SPLIT, diff, find_clean_run, oracle_output, replay, resume_state, verify_step,
)

METRICS_PATH = ROOT / "data" / "metrics.json"

# Status palette (reference dataviz palette): never colour alone, always icon + label
STATUS = {
    "suspect": {"color": "#d03b3b", "icon": "✕", "min": 0.5},
    "watch": {"color": "#ec835a", "icon": "!", "min": 0.2},
    "ok": {"color": "#0ca30c", "icon": "✓", "min": 0.0},
}
# Categorical slots 1-2 and the blue <-> red diverging pair, light / dark steps
LIGHT = {"s1": "#2a78d6", "s2": "#eb6834", "raise": "#e34948", "lower": "#2a78d6"}
DARK = {"s1": "#3987e5", "s2": "#d95926", "raise": "#e66767", "lower": "#3987e5"}

METHOD_LABELS = {
    "random": "Random step",
    "last_step": "Last step",
    "first_anomaly": "First anomaly",
    "lightgbm": "LightGBM ranker",
}


# ── helpers ─────────────────────────────────────────────────────────────────

def palette() -> dict[str, str]:
    theme = getattr(getattr(st, "context", None), "theme", None)
    return DARK if getattr(theme, "type", None) == "dark" else LIGHT


def status_of(score: float) -> str:
    for name, spec in STATUS.items():
        if score >= spec["min"]:
            return name
    return "ok"


def loads(raw: str | None) -> Any:
    return json.loads(raw) if raw else None


def summarize(step_type: str, out: Any) -> str:
    """One-line human summary of a step's output."""
    if out is None:
        return "∅ (no output)"
    if not isinstance(out, dict):
        return str(out)[:80]
    if step_type == "plan":
        acts = out.get("actions", [])
        parts = []
        for a in acts:
            if a.get("action") == "retrieve":
                parts.append(f"retrieve({a.get('name')})")
            elif a.get("action") == "extract":
                parts.append(f"extract({a.get('field')})")
            elif a.get("action") == "calculate":
                parts.append(f"calc[{a.get('expr')}]")
            else:
                parts.append(str(a.get("action")))
        return " → ".join(parts)
    if step_type == "retrieve":
        return f"{out.get('title')}  (scores {out.get('top_scores')})"
    if step_type == "extract":
        return f"{out.get('field')} = {out.get('value')}"
    if step_type == "calculate":
        return f"{out.get('expr')} = {out.get('result')}"
    if step_type == "answer":
        return str(out.get("answer"))
    return json.dumps(out)[:80]


def explain(reason: dict, step: dict) -> str:
    """Turn a SHAP reason into a sentence grounded in the trace."""
    f, v = reason["feature"], reason["value"]
    inp, out = loads(step["input_json"]) or {}, loads(step["output_json"]) or {}
    stype = step["type"]
    if f == "grounding_match" and stype == "retrieve":
        return (f"retrieved **{out.get('title')}** for requested **{inp.get('name')}** "
                f"(name match {v:.2f})")
    if f == "top_score_gap" and stype == "retrieve":
        return f"rank-1 vs rank-2 retrieval score gap is {v:.2f}"
    if f == "extracted_in_source" and stype == "extract":
        val = out.get("value")
        return (f"extracted value `{val}` appears in the source document" if v
                else f"extracted value `{val}` is **not** in the source document")
    if f == "num_downstream_consumers":
        return f"output consumed by {int(v)} later step(s)"
    if f == "on_path_to_final":
        return "on the data-flow path to the final answer" if v else "not on the path to the final answer"
    if f == "pos_norm":
        return f"sits {v:.0%} of the way through the run"
    if f == "steps_remaining":
        return f"{int(v)} step(s) remain after it"
    only_for = {"grounding_match": "retrieve", "top_score_gap": "retrieve",
                "extracted_in_source": "extract", "calc_args_traceable": "calculate"}
    if f in only_for and stype != only_for[f]:
        article = "an" if only_for[f][0] in "aeiou" else "a"
        return f"not {article} `{only_for[f]}` step, so `{f}` is 0"
    if f.startswith("type_"):
        kind = f.removeprefix("type_")
        return f"step type {'is' if v else 'is not'} `{kind}`"
    if f == "has_error":
        return "step raised an error" if v else "step ran without error"
    if f == "is_empty_output":
        return "output is empty" if v else "output is non-empty"
    if f == "cosine_sim_question":
        return f"output text similarity to the question is {v:.2f}"
    if f == "output_len":
        return f"output is {int(v)} characters long"
    return f"`{f}` = {v}"


def downstream(steps: list[dict], start: int) -> list[int]:
    """All steps that (transitively) consumed start's output, in order."""
    children: dict[int, list[int]] = {}
    for s in steps:
        for p in loads(s["parents_json"]) or []:
            children.setdefault(p, []).append(s["step_idx"])
    seen, stack = set(), [start]
    while stack:
        for c in children.get(stack.pop(), []):
            if c not in seen:
                seen.add(c)
                stack.append(c)
    return sorted(seen)


@st.cache_data(show_spinner=False)
def cached_predict(run_id: str, model_mtime: float) -> list[dict]:
    return predict(run_id, top_k=len(FEATURE_NAMES))


def predictions(run_id: str) -> dict[int, dict]:
    mtime = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else 0.0
    return {p["step_idx"]: p for p in cached_predict(run_id, mtime)}


def run_label(r: dict) -> str:
    outcome = "pass" if r["success"] else "FAIL"
    return f"{r['run_id']} · {r['fault_type'] or 'clean'} · {r['split']} · {outcome}"


# ── page ────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Black Box", page_icon="⬛", layout="wide")

@st.cache_resource(show_spinner=False)
def build_once() -> bool:
    """Build data/ once per server process, even if several visitors arrive together."""
    build_all()
    return True


if not is_built():
    # data/ is gitignored, so a fresh clone or deploy builds it here (offline, a few seconds)
    with st.spinner("First launch: recording 300 agent runs, training the ranker, evaluating…"):
        build_once()
    cached_predict.clear()

ensure_demo()

st.title("Black Box · a flight recorder for AI agents")
st.caption("Record agent runs → blame the step that broke it → patch that step and replay only what comes after.")

tab_debug, tab_metrics = st.tabs(["Debugger", "Metrics"])

# ── sidebar: run list ───────────────────────────────────────────────────────

all_runs = [r for r in recorder.list_runs() if r["split"] != REPLAY_SPLIT]
with st.sidebar:
    st.header("Runs")
    outcome_filter = st.radio("Outcome", ["Failed", "Passed", "All"], horizontal=True)
    fault_options = sorted({r["fault_type"] or "clean" for r in all_runs})
    fault_filter = st.multiselect("Fault type", fault_options, default=fault_options)
    split_options = sorted({r["split"] for r in all_runs})
    split_filter = st.multiselect("Split", split_options, default=split_options)

    runs = [
        r for r in all_runs
        if (outcome_filter == "All" or bool(r["success"]) == (outcome_filter == "Passed"))
        and (r["fault_type"] or "clean") in fault_filter
        and r["split"] in split_filter
    ]
    runs.sort(key=lambda r: (r["split"] != "demo", r["run_id"]))
    st.caption(f"{len(runs)} of {len(all_runs)} recorded runs")
    if not runs:
        st.warning("No runs match these filters.")
        st.stop()
    ids = [r["run_id"] for r in runs]
    default_idx = ids.index(DEMO_FAULT_RUN) if DEMO_FAULT_RUN in ids else 0
    labels_by_id = {r["run_id"]: run_label(r) for r in runs}
    run_id = st.selectbox("Run", ids, index=default_idx, format_func=labels_by_id.get)

run_info = recorder.get_run(run_id)
steps = recorder.get_steps(run_id)
steps_by_idx = {s["step_idx"]: s for s in steps}
if not steps:
    st.warning(f"Run `{run_id}` has no recorded steps.")
    st.stop()

# ── debugger tab ────────────────────────────────────────────────────────────

with tab_debug:
    st.subheader(run_info["question"])
    c1, c2, c3 = st.columns([2, 2, 1])
    c1.markdown(f"**Final answer**  \n{run_info['final_answer']}")
    c2.markdown(f"**Gold answer**  \n{run_info['gold']}")
    c3.markdown("**Outcome**  \n" + ("✓ pass" if run_info["success"] else "✕ fail"))

    preds = predictions(run_id)
    ranked = sorted(preds.values(), key=lambda p: p["score"], reverse=True)
    blamed = ranked[0]["step_idx"] if ranked else None

    # Timeline: blame score per step, coloured by status with icon + label
    st.markdown("#### Timeline · blame score per step")
    order = [s["step_idx"] for s in steps]
    scores = [preds[i]["score"] if i in preds else 0.0 for i in order]
    statuses = [status_of(sc) for sc in scores]
    labels = [f"{i} · {steps_by_idx[i]['type']}" for i in order]
    fig = go.Figure()
    for name, spec in STATUS.items():
        xs = [lab for lab, stt in zip(labels, statuses, strict=True) if stt == name]
        ys = [sc for sc, stt in zip(scores, statuses, strict=True) if stt == name]
        if not xs:
            continue
        fig.add_bar(
            x=xs, y=ys, name=f"{spec['icon']} {name}", marker_color=spec["color"],
            text=[f"{spec['icon']} {y:.2f}" for y in ys], textposition="outside",
            hovertemplate="step %{x}<br>blame %{y:.3f}<extra>" + name + "</extra>",
        )
    fig.update_layout(
        height=280, margin=dict(l=10, r=10, t=10, b=10), bargap=0.35,
        xaxis=dict(categoryorder="array", categoryarray=labels, title=None),
        yaxis=dict(range=[0, 1.15], title="blame score", gridcolor="rgba(128,128,128,0.15)"),
        legend=dict(orientation="h", y=1.12, x=0),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"timeline_{run_id}")

    sel_key = f"step_{run_id}"
    if sel_key not in st.session_state and blamed is not None:
        st.session_state[sel_key] = blamed
    step_idx = st.radio(
        "Inspect step", order, key=sel_key, horizontal=True,
        format_func=lambda i: f"{STATUS[status_of(preds.get(i, {}).get('score', 0))]['icon']} "
                              f"{i} · {steps_by_idx[i]['type']}",
    )
    step = steps_by_idx[step_idx]
    pred = preds.get(step_idx, {"score": 0.0, "reasons": []})

    # Evidence + inspector
    left, right = st.columns([3, 2])
    with left:
        stt = status_of(pred["score"])
        st.markdown(f"#### Step {step_idx} · `{step['type']}` — "
                    f"{STATUS[stt]['icon']} {stt} (blame {pred['score']:.2f}"
                    f"{', rank 1' if step_idx == blamed else ''})")
        st.markdown("**Why (top 3 SHAP reasons)**")
        for r in pred["reasons"][:3]:
            direction = "raises" if r["shap_value"] > 0 else "lowers"
            st.markdown(f"- {explain(r, step)} — *{direction} blame* ({r['shap_value']:+.2f})")
        flow = downstream(steps, step_idx)
        if flow:
            st.markdown(f"**Data flow:** step {step_idx}'s output flowed into "
                        + " → ".join(str(i) for i in flow))
        else:
            st.markdown("**Data flow:** no later step consumed this output")

        reasons = sorted(pred["reasons"], key=lambda r: abs(r["shap_value"]), reverse=True)[:8][::-1]
        if reasons:
            pal = palette()
            shap_fig = go.Figure(go.Bar(
                x=[r["shap_value"] for r in reasons],
                y=[r["feature"] for r in reasons],
                orientation="h",
                marker_color=[pal["raise"] if r["shap_value"] > 0 else pal["lower"] for r in reasons],
                text=[f"{r['shap_value']:+.2f}" for r in reasons], textposition="outside",
                cliponaxis=False, hovertemplate="%{y}<br>SHAP %{x:+.3f}<extra></extra>",
            ))
            shap_fig.update_layout(
                height=280, margin=dict(l=10, r=30, t=30, b=10),
                title=dict(text="SHAP contributions (red raises blame, blue lowers it)", font=dict(size=13)),
                xaxis=dict(zeroline=True, gridcolor="rgba(128,128,128,0.15)", range=[
                    min(0.0, min(r["shap_value"] for r in reasons)) * 1.35 - 0.3,
                    max(0.0, max(r["shap_value"] for r in reasons)) * 1.35 + 0.3,
                ]),
            )
            st.plotly_chart(shap_fig, use_container_width=True, key=f"shap_{run_id}_{step_idx}")

    with right:
        st.markdown("#### Inspector")
        meta = (f"parents {loads(step['parents_json'])} · "
                f"{'cache hit' if step['cache_hit'] else 'executed / patched'} · {step['latency_ms']} ms")
        st.caption(meta)
        if step["error"]:
            st.error(f"Error: {step['error']}")
        st.markdown("**Input**")
        st.json(loads(step["input_json"]), expanded=False)
        st.markdown("**Output**")
        st.json(loads(step["output_json"]), expanded=True)
        st.markdown(f"**Checkpoint · state before step {step_idx}** (what a resume from here would see)")
        st.json(resume_state(run_id, step_idx), expanded=False)

    if run_info["fault_type"] and run_info["fault_step"] is not None:
        with st.expander("Ground truth (injected label — evaluation only, never a model feature)"):
            hit = "✓ matches" if blamed == run_info["fault_step"] else "✕ differs from"
            st.write(f"Injected **{run_info['fault_type']}** at step **{run_info['fault_step']}**. "
                     f"Top-1 blame (step {blamed}) {hit} the label.")

    # ── patch & replay ──
    st.divider()
    st.markdown(f"#### Patch & replay from step {step_idx}")
    st.caption("Steps before the patch are served from the content-hashed cache; "
               "later steps re-run only if their inputs changed.")
    clean = find_clean_run(run_id)
    oracle = oracle_output(run_id, step_idx) if clean else None
    current_out = loads(step["output_json"])

    patched: Any = None
    if step["type"] == "retrieve" and isinstance(current_out, dict):
        kb = get_kb()
        names = [c["name"] for c in kb]
        default_id = (oracle or current_out).get("doc_id")
        ids_kb = [c["id"] for c in kb]
        pick = st.selectbox("Replace the retrieved document with", names,
                            index=ids_kb.index(default_id) if default_id in ids_kb else 0,
                            key=f"doc_{run_id}_{step_idx}")
        patched = retrieve_by_id(ids_kb[names.index(pick)])
        # Keep the original retrieval scores; only the document changes
        patched["top_scores"] = current_out.get("top_scores", patched["top_scores"])
    else:
        ta_key = f"patch_{run_id}_{step_idx}"
        if ta_key not in st.session_state:
            st.session_state[ta_key] = json.dumps(current_out, indent=2)
        if oracle is not None and st.button("Load clean-run output (oracle)", key=f"oracle_{ta_key}"):
            st.session_state[ta_key] = json.dumps(oracle, indent=2)
            st.rerun()
        text = st.text_area("Patched output (JSON)", key=ta_key, height=180)
        try:
            patched = json.loads(text)
        except json.JSONDecodeError as exc:
            st.error(f"Invalid JSON: {exc}")

    replay_key = f"replay_{run_id}"
    if st.button("▶ Patch & replay", type="primary", disabled=patched is None):
        with st.spinner("Replaying…"):
            st.session_state[replay_key] = {"step": step_idx, **replay(run_id, {step_idx: patched})}

    res = st.session_state.get(replay_key)
    if res:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Prefix reused from cache", res["n_prefix_reused"])
        m2.metric("Patched", res["n_patched"])
        m3.metric("Re-executed", res["n_reexecuted"])
        m4.metric("LLM calls avoided", res["n_llm_reused"])
        if res["outcome_changed"] and res["success"]:
            st.success(f"✓ fail → pass. Patching step {res['step']} alone fixes the run: "
                       f"**{res['final_answer']}**")
        elif res["success"]:
            st.info(f"Run still passes: {res['final_answer']}")
        else:
            st.warning(f"✕ Still failing after patching step {res['step']}: {res['final_answer']}")

    # ── diff ──
    children = [r for r in recorder.list_runs(split=REPLAY_SPLIT) if r["parent_run_id"] == run_id]
    if children:
        st.markdown("#### Trace diff · original vs patched")
        child_ids = [c["run_id"] for c in sorted(children, key=lambda c: c["created_at"], reverse=True)]
        default_child = res["run_id"] if res and res["run_id"] in child_ids else child_ids[0]
        other = st.selectbox("Compare with replay", child_ids, index=child_ids.index(default_child))
        d = diff(run_id, other)
        a, b = st.columns(2)
        a.markdown(f"**Original `{run_id}`**  \n{d['final_answer_a']}")
        b.markdown(f"**Patched `{other}`**  \n{d['final_answer_b']}")
        st.caption(f"First divergence at step {d['first_divergent_step']} · outcome {d['outcome_change']}")
        rows = []
        for sd in d["step_diffs"]:
            if sd["cache_hit_b"]:
                source = "from cache"
            elif not sd["input_changed"]:
                source = "patched"
            else:
                source = "re-executed"
            rows.append({
                "step": sd["step_idx"], "type": sd["type"],
                "original": summarize(sd["type"], sd["output_a"]),
                "patched run": summarize(sd["type"], sd["output_b"]),
                "patched run source": source,
                "changed": "● changed" if sd["changed"] else "",
            })
        df = pd.DataFrame(rows)
        styled = df.style.apply(
            lambda row: ["background-color: rgba(235,104,52,0.18)" if row["changed"] else ""] * len(row),
            axis=1,
        )
        st.dataframe(styled, hide_index=True, use_container_width=True)

    # ── auto-verify ──
    with st.expander("Auto-verify the top-3 blamed steps (uses a clean run as oracle)"):
        if not clean:
            st.info("No passing clean run of this question exists, so there is no oracle output to patch with.")
        elif run_info["success"]:
            st.info("This run already passes; there is nothing to verify.")
        elif st.button("Run auto-verify", key=f"autoverify_{run_id}"):
            rows = []
            for p in ranked[:3]:
                k = p["step_idx"]
                v = verify_step(run_id, k, new_run_id=f"av_{run_id}_s{k}")
                rows.append({
                    "step": k, "type": p["type"], "blame": round(p["score"], 3),
                    "patch flips fail → pass": "✓" if v["verified"] else "✕",
                    "input matched clean run": "✓" if v.get("input_matches_clean") else "✕",
                    "verified root cause": "✓" if v.get("root_cause_verified") else "✕",
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.caption("A downstream step can also flip the outcome when given clean output; the root cause is "
                       "the step that received correct input but produced the wrong output.")

# ── metrics tab ─────────────────────────────────────────────────────────────

with tab_metrics:
    st.subheader("How well does Black Box blame the right step?")
    if not METRICS_PATH.exists():
        st.warning("No `data/metrics.json` yet.")
        if st.button("Run evaluation now"):
            from blackbox.evaluate import evaluate_baselines_and_model
            with st.spinner("Evaluating (about a minute)…"):
                evaluate_baselines_and_model()
            st.rerun()
    else:
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        pal = palette()
        split_titles = {
            "seen_faults_test": "Seen faults (F1–F3) · unseen test questions",
            "heldout_faults": "Held-out faults (F4, F5) · never trained on",
        }
        cols = st.columns(2)
        for col, (key, title) in zip(cols, split_titles.items(), strict=True):
            block = metrics.get(key, {})
            with col:
                st.markdown(f"**{title}**  \n{block.get('n_runs', 0)} failed runs")
                m = block.get("metrics", {})
                if not m:
                    st.info("No runs in this split.")
                    continue
                methods = [k for k in METHOD_LABELS if k in m]
                names = [METHOD_LABELS[k] for k in methods]
                fig = go.Figure()
                for metric, color, label in (("top1", pal["s1"], "Top-1"), ("top3", pal["s2"], "Top-3")):
                    vals = [m[k][metric] * 100 for k in methods]
                    fig.add_bar(x=names, y=vals, name=label, marker_color=color,
                                text=[f"{v:.0f}%" for v in vals], textposition="outside",
                                hovertemplate="%{x}<br>" + label + " %{y:.1f}%<extra></extra>")
                fig.update_layout(
                    barmode="group", bargap=0.3, bargroupgap=0.06, height=320,
                    margin=dict(l=10, r=10, t=30, b=10),
                    yaxis=dict(range=[0, 115], title="accuracy (%)", gridcolor="rgba(128,128,128,0.15)"),
                    legend=dict(orientation="h", y=1.12, x=0),
                )
                st.plotly_chart(fig, use_container_width=True, key=f"metrics_{key}")
                table = pd.DataFrame([
                    {"method": METHOD_LABELS[k], "Top-1": f"{m[k]['top1']:.1%}",
                     "Top-3": f"{m[k]['top3']:.1%}", "MRR": f"{m[k]['mrr']:.3f}"}
                    for k in methods
                ])
                st.dataframe(table, hide_index=True, use_container_width=True)

                rv = block.get("replay_verified")
                if rv:
                    r1, r2, r3 = st.columns(3)
                    r1.metric("Exact-match Top-1", f"{m['lightgbm']['top1']:.0%}")
                    r2.metric("Root-cause verified", f"{rv.get('lightgbm_top1_root_cause', 0):.0%}")
                    r3.metric("Any-patch flip", f"{rv.get('lightgbm_top1', 0):.0%}")
                    st.caption(
                        f"Replay check on LightGBM's top-1 step. *Any-patch flip*: giving that step its clean "
                        f"output flips fail → pass, which also happens for steps downstream of the real fault. "
                        f"*Root-cause verified* additionally requires that the step received the same input as "
                        f"in the clean run. Labels confirmed by replay: {rv.get('label_confirmed', 0):.0%}."
                    )

        st.markdown("#### Dataset")
        counts = pd.DataFrame(all_runs)
        if not counts.empty:
            counts["fault"] = counts["fault_type"].fillna("clean")
            pivot = counts.pivot_table(index="fault", columns="split", values="run_id",
                                       aggfunc="count", fill_value=0)
            st.dataframe(pivot, use_container_width=True)
            st.caption("Clean runs pass; each faulty run has exactly one injected fault. "
                       "`benign` = fault injected but the run still passed (excluded from training).")
