"""Stage 5 checkpoint: blame, patch and replay the NovaTech / Zenith demo run.

    python -m scripts.test_stage5
"""
import random

from blackbox.demo import DEMO_FAULT_RUN, build_demo
from blackbox.model import predict
from blackbox.recorder import get_run
from blackbox.replay import diff, oracle_output, replay, resume_state

demo = build_demo()
fault_step = demo["fault_step"]
print(f"Gold: {demo['gold']}")
print(f"Faulty run {DEMO_FAULT_RUN}: {demo['faulty']['final_answer']}")
assert demo["clean"]["success"], "clean demo run must pass"
assert demo["faulty"]["success"] is False, "faulty demo run must fail"

# 1. Diagnose
preds = sorted(predict(DEMO_FAULT_RUN), key=lambda p: p["score"], reverse=True)
top = preds[0]
print(f"Top blame: step {top['step_idx']} ({top['type']}) score={top['score']:.3f}")
for r in top["reasons"]:
    print(f"   {r['feature']:26s} value={r['value']:<8} shap={r['shap_value']:+.3f}")
assert top["step_idx"] == fault_step, f"expected blame on step {fault_step}, got {top['step_idx']}"

# 2. Checkpoint view
state = resume_state(DEMO_FAULT_RUN, fault_step)
print(f"State before step {fault_step}: keys={sorted(state)}")

# 3. Patch & replay with the correct (Zenith) document
patch = oracle_output(DEMO_FAULT_RUN, fault_step)
assert patch and patch["title"] == "Zenith"
res = replay(DEMO_FAULT_RUN, {fault_step: patch}, new_run_id="rp_r_9002_demo")
print(f"Replay {res['run_id']}: {res['final_answer']} (success={res['success']})")
print(f"   {res['n_prefix_reused']} prefix steps reused from cache, {res['n_patched']} patched, "
      f"{res['n_reexecuted']} re-executed, {res['n_reused'] - res['n_prefix_reused']} suffix cache hits")
assert res["success"] and res["outcome_changed"], "patched replay must flip fail -> pass"
assert res["n_prefix_reused"] == fault_step, "every step before the patch must come from cache"
assert get_run(res["run_id"])["split"] == "replay"

# 4. A patch nobody has seen before forces real suffix re-execution
fresh = dict(patch, top_scores=[0.99, round(random.random(), 8)])  # unique, so never cached
res2 = replay(DEMO_FAULT_RUN, {fault_step: fresh}, new_run_id="rp_r_9002_demo_b")
print(f"Replay (unseen patch): {res2['n_prefix_reused']} prefix reused, {res2['n_reexecuted']} re-executed")
assert res2["n_prefix_reused"] == fault_step and res2["n_reexecuted"] >= 1 and res2["success"]

# 5. Diff
d = diff(DEMO_FAULT_RUN, res["run_id"])
print(f"Diff: first divergence at step {d['first_divergent_step']}, outcome {d['outcome_change']}")
assert d["first_divergent_step"] == fault_step and d["outcome_change"] == "fail -> pass"

print("\n=======================================================")
print(">>> STAGE 5 VERIFIED: blame -> patch -> replay flips fail -> pass <<<")
print("=======================================================")
