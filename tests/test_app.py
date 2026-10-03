"""Headless UI smoke test using Streamlit's AppTest.

Runs against the real data/blackbox.db (replays it creates use split "replay"),
so it is skipped until the dataset and model have been built.

    python -m pytest -q tests/test_app.py
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app" / "streamlit_app.py"

pytestmark = pytest.mark.skipif(
    not (ROOT / "data" / "blackbox.db").exists() or not (ROOT / "data" / "model.pkl").exists(),
    reason="build data/blackbox.db and data/model.pkl first",
)


def _app():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def test_demo_run_is_selected_and_blamed_at_step_3():
    at = _app()
    assert at.sidebar.selectbox[0].value == "r_9002"
    assert at.radio(key="step_r_9002").value == 3
    assert any("Step 3" in m.value and "suspect" in m.value for m in at.markdown)


def test_patch_and_replay_flips_demo_run_and_shows_diff():
    at = _app()
    at.button[[b.label for b in at.button].index("▶ Patch & replay")].click().run()
    assert not at.exception, at.exception
    assert any("fail → pass" in s.value for s in at.success)
    labels = {m.label: m.value for m in at.metric}
    assert labels["Prefix reused from cache"] == "3"
    assert labels["Patched"] == "1"
    assert any("First divergence at step 3" in c.value for c in at.caption)


def test_auto_verify_and_other_runs_render():
    at = _app()
    at.button(key="autoverify_r_9002").click().run()
    assert not at.exception, at.exception
    # Switch to a held-out fault run and an extract step
    sb = at.sidebar.selectbox[0]
    other = next(o for o in sb.options if "dropped_context" in o)
    sb.select(other.split(" · ")[0]).run()
    assert not at.exception, at.exception


def test_second_agent_run_renders_blames_and_replays():
    at = _app()
    sb = at.sidebar.selectbox[0]
    sb.select("t_9102").run()
    assert not at.exception, at.exception
    assert at.radio(key="step_t_9102").value == 1  # blamed: the retrieval
    at.button[[b.label for b in at.button].index("Load clean-run output (oracle)")].click().run()
    at.button[[b.label for b in at.button].index("▶ Patch & replay")].click().run()
    assert not at.exception, at.exception
    assert any("fail → pass" in s.value for s in at.success)
    assert any(m.label == "LLM tokens not re-spent" for m in at.metric)
