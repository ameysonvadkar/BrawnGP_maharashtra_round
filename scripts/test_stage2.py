import json
from agent.runner import run
from blackbox.recorder import cache_stats, get_steps

with open('agent/questions.json', encoding='utf-8-sig') as f:
    questions = json.load(f)

q0 = questions[0]
print(f"Testing question: {q0['question']}")

# Run 5 (Fresh Run 1 with fixed calculate builtins)
res5 = run(
    question=q0['question'],
    question_id=q0['id'],
    run_id='test_r005',
    gold=q0['gold'],
    split='train'
)
print('Run 5 result:', res5)
stats5 = cache_stats('test_r005')
print('Run 5 cache stats:', stats5)

# Run 6 (Fresh Run 2 — 100% cache hits expected!)
res6 = run(
    question=q0['question'],
    question_id=q0['id'],
    run_id='test_r006',
    gold=q0['gold'],
    split='train'
)
print('Run 6 result:', res6)
stats6 = cache_stats('test_r006')
print('Run 6 cache stats:', stats6)

assert stats6['reexecuted'] == 0, f"Expected 0 reexecuted on rerun, got {stats6['reexecuted']}"
assert stats6['reused'] == res6['n_steps'], f"Expected all {res6['n_steps']} steps reused, got {stats6['reused']}"
print("\n=======================================================")
print(">>> STAGE 2 VERIFIED: 100% Cache Hit (0 re-executed)! <<<")
print("=======================================================")
