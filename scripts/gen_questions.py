"""
Generate questions.json with ~60 questions and Python-computed gold answers from kb.json.
Templates:
  T1: older_by   - "Which is older, A or B, and by how many years?"
  T2: rev_per_emp - "What is the revenue per employee (in thousands) of A?"
  T3: combined_emp - "What is the combined employee count of A and B?"
  T4: emp_diff   - "How many more employees does A have than B?"
"""
import json, pathlib, random

random.seed(42)

KB_PATH = pathlib.Path("agent/kb.json")
OUT_PATH = pathlib.Path("agent/questions.json")

kb = json.loads(KB_PATH.read_text(encoding="utf-8-sig"))
by_id = {c["id"]: c for c in kb}

NEAR_DUPS = [
    ("novatech", "nova_systems"),
    ("zenith", "zenit_labs"),
    ("apexion", "apex_data"),
    ("prismworks", "prism_ai"),
    ("luminos", "luminos_tech"),
    ("orbita", "orbit_cloud"),
    ("vectora", "vector_core"),
    ("stratum", "strata_net"),
    ("inferno", "infernix"),
    ("cruxlab", "crux_ai"),
    ("neuralaxis", "neuralax"),
    ("datavex", "datavex_ai"),
    ("solara", "solara_soft"),
    ("hypercor", "hypercore"),
    ("mosaiq", "mosaic_data"),
    ("quanta", "quantax"),
    ("spectral", "spectra_ai"),
    ("terracor", "terra_tech"),
    ("nexion", "nexion_labs"),
    ("kaleidix", "kalex_io"),
]

questions = []
qid = 0

def make_qid():
    global qid
    qid += 1
    return f"q{qid:03d}"

# T1: older_by
for a_id, b_id in NEAR_DUPS:
    a, b = by_id[a_id], by_id[b_id]
    diff = abs(a["founded"] - b["founded"])
    older = a["name"] if a["founded"] < b["founded"] else b["name"]
    q = f"Which company is older, {a['name']} or {b['name']}, and by how many years?"
    gold = f"{older} is older by {diff} years."
    questions.append({"id": make_qid(), "template": "older_by", "companies": [a_id, b_id], "question": q, "gold": gold, "gold_value": diff, "gold_name": older})

# T2: rev_per_emp
rich = [c for c in kb if c["revenue_m"] > 100]
t2_sample = random.sample(rich, min(15, len(rich)))
for c in t2_sample:
    rev_k = round(c["revenue_m"] * 1000 / c["employees"], 2)
    q = f"What is the revenue per employee (in thousands of dollars) of {c['name']}?"
    gold = f"${rev_k}K per employee."
    questions.append({"id": make_qid(), "template": "rev_per_emp", "companies": [c["id"]], "question": q, "gold": gold, "gold_value": rev_k})

# T3: combined_emp
all_ids = [c["id"] for c in kb]
t3_pairs = []
while len(t3_pairs) < 15:
    a_id, b_id = random.sample(all_ids, 2)
    if (a_id, b_id) not in NEAR_DUPS and (b_id, a_id) not in NEAR_DUPS:
        t3_pairs.append((a_id, b_id))
for a_id, b_id in t3_pairs:
    a, b = by_id[a_id], by_id[b_id]
    total = a["employees"] + b["employees"]
    q = f"What is the combined employee count of {a['name']} and {b['name']}?"
    gold = f"{total} employees combined."
    questions.append({"id": make_qid(), "template": "combined_emp", "companies": [a_id, b_id], "question": q, "gold": gold, "gold_value": total})

# T4: emp_diff
t4_pairs = random.sample(NEAR_DUPS, 10)
for a_id, b_id in t4_pairs:
    a, b = by_id[a_id], by_id[b_id]
    gold_diff = a["employees"] - b["employees"]
    if gold_diff > 0:
        gold = f"{a['name']} has {abs(gold_diff)} more employees."
    else:
        gold = f"{b['name']} has {abs(gold_diff)} more employees."
    q = f"How many more employees does {a['name']} have compared to {b['name']}?"
    questions.append({"id": make_qid(), "template": "emp_diff", "companies": [a_id, b_id], "question": q, "gold": gold, "gold_value": gold_diff})

random.shuffle(questions)
print(f"Total questions: {len(questions)}")
OUT_PATH.write_text(json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8")
print("Written to", OUT_PATH)
