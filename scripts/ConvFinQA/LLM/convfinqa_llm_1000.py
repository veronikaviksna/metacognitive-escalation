# convfinqa_llm_1000.py — ConvFinQA LLM Baseline: Qwen 14B, 1000 questions
# Inference: 3 traces with majority voting (self-consistency)
#
# Run AFTER convfinqa_slm_1000.py so UIDs are already saved.

import os, re, json, random, signal, sys
from collections import Counter
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── SETTINGS ──────────────────────────────────────────────────────────────
MODEL          = "Qwen/Qwen2.5-14B-Instruct"
BASE_DIR       = Path(__file__).parent
CONVFINQA_JSON = BASE_DIR / "train.json"
OUT_PATH       = BASE_DIR / "convfinqa_llm_results.csv"
UIDS_PATH      = BASE_DIR / "convfinqa_sample_uids.json"
CHECKPOINT     = BASE_DIR / "convfinqa_llm_checkpoint.json"

DEMO_SIZE    = 1000
RANDOM_STATE = 42
N_TRACES     = 3
TEMPERATURE  = 0.7
SAVE_EVERY   = 10

# ── GRACEFUL INTERRUPT ────────────────────────────────────────────────────
_interrupted = False

def _handle_sigint(sig, frame):
    global _interrupted
    print("\n\nInterrupted — saving checkpoint...")
    _interrupted = True

signal.signal(signal.SIGINT,  _handle_sigint)
signal.signal(signal.SIGTERM, _handle_sigint)

# ── CHECKPOINT HELPERS ────────────────────────────────────────────────────
def _save_json_atomic(data, path):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f: json.dump(data, f)
    tmp.replace(path)

def _load_json_safe(path):
    try:
        with open(path) as f: return json.load(f)
    except: return None

# ── DATA LOADING ──────────────────────────────────────────────────────────
def table_row_to_text(header, row):
    res = ""
    for i, (h, c) in enumerate(zip(header, row)):
        if i == 0: res += str(c) + " "
        else:      res += "| " + str(h) + " | " + str(c) + " "
    return res.strip()

def build_context(entry, turn_idx, max_chars=8000):
    pre   = " ".join(entry.get("pre_text",  []))
    post  = " ".join(entry.get("post_text", []))
    table = entry.get("table_ori", entry.get("table", []))
    table_text = ""
    if len(table) >= 2:
        header = table[0]
        for row in table[1:]:
            table_text += table_row_to_text(header, row) + " . "
    base_ctx = f"{pre} {table_text} {post}".strip()
    base_ctx = re.sub(r"\s{2,}", " ", base_ctx)
    ann          = entry.get("annotation", {})
    dialogue     = ann.get("dialogue_break", [])
    exe_ans_list = ann.get("exe_ans_list",   [])
    history_parts = []
    for i in range(min(turn_idx, len(dialogue), len(exe_ans_list))):
        history_parts.append(f"Q: {dialogue[i]}\nA: {exe_ans_list[i]}")
    if history_parts:
        ctx = base_ctx + "\n\nPrevious turns:\n" + "\n".join(history_parts)
    else:
        ctx = base_ctx
    return (ctx[:max_chars] + " ...[truncated]") if len(ctx) > max_chars else ctx

print(f"Loading ConvFinQA data from {CONVFINQA_JSON}...")
with open(CONVFINQA_JSON, encoding="utf-8") as f:
    raw_data = json.load(f)

all_turns = []
for entry_idx, entry in enumerate(raw_data):
    ann  = entry.get("annotation", {})
    db   = ann.get("dialogue_break", [])
    ea   = ann.get("exe_ans_list",   [])
    eid  = entry.get("id", f"entry_{entry_idx}")
    for turn_idx, (q, a) in enumerate(zip(db, ea)):
        if a is None or str(a).strip() == "": continue
        all_turns.append({"uid": f"{eid}_t{turn_idx}", "entry_idx": entry_idx,
                           "turn_idx": turn_idx, "question": q, "answer": str(a)})

print(f"Loaded {len(raw_data)} conversations, {len(all_turns)} turns total")

uid_file = _load_json_safe(UIDS_PATH)
if uid_file:
    uid_set = set(uid_file)
    # Sort to guarantee identical order with SLM and hierarchical runs
    sample  = sorted([t for t in all_turns if t["uid"] in uid_set], key=lambda t: t["uid"])
    print(f"Loaded existing UID list — {len(sample)} questions from {UIDS_PATH}")
else:
    random.seed(RANDOM_STATE)
    sample = random.sample(all_turns, min(DEMO_SIZE, len(all_turns)))
    sample = sorted(sample, key=lambda t: t["uid"])
    _save_json_atomic([t["uid"] for t in sample], UIDS_PATH)
    print(f"No UID file found — sampled fresh {len(sample)} questions -> {UIDS_PATH}")

# ── MODEL ─────────────────────────────────────────────────────────────────
def _make_bnb_cfg():
    try:
        from bitsandbytes import __version__ as v
        if tuple(int(x) for x in v.split(".")[:3]) >= (0, 41, 0):
            return BitsAndBytesConfig(load_in_4bit=True,
                                      bnb_4bit_compute_dtype=torch.float16,
                                      bnb_4bit_quant_type="nf4",
                                      bnb_4bit_use_double_quant=True)
    except: pass
    return None

bnb_cfg = _make_bnb_cfg()
print(f"\nLoading {MODEL}...")
tok = AutoTokenizer.from_pretrained(MODEL)
if bnb_cfg:
    print("  Using 4-bit quantization (bitsandbytes)")
    mdl = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb_cfg, device_map="auto")
else:
    mdl = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto")
mdl.eval()
free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9
print(f"Loaded. VRAM free: {free:.1f} GB")

# ── UTILITIES ─────────────────────────────────────────────────────────────
def normalize(a):
    a = str(a).strip()
    a = re.sub(r"[$€£¥]", "", a).strip()
    if re.match(r"^\([\d,.]+\)$", a): a = "-" + a[1:-1]
    a = re.sub(r"^[≈~<>about\s]+", "", a, flags=re.IGNORECASE)
    a = re.sub(r"[\s,]", "", a)
    pct = a.endswith("%"); a = a.rstrip("%").strip()
    try: v = float(a); return v / 100 if pct else v
    except: return a.lower().strip()

def match(pred, gt, tol=0.03):
    p, g = normalize(pred), normalize(gt)
    if not (isinstance(p, float) and isinstance(g, float)): return str(p) == str(g)
    if p == g: return True
    if abs(g) < 0.001: return abs(p - g) < 0.001
    if abs(p - g) / abs(g) < tol: return True
    if abs(g) > 0 and abs(p) > 0:
        r = p / g
        if abs(g) <= 5 and abs(r - 100) / 100 < tol: return True
        if abs(p) <= 5 and abs(r - 0.01) / 0.01 < tol: return True
    return False

def parseable(a):
    if not a or str(a).lower() in ("[malformed]", "none", "", "n/a"): return False
    return isinstance(normalize(a), float)

def pick_majority(answers):
    p = [a for a in answers if parseable(a)]
    if not p: return answers[0] if answers else "[malformed]"
    c = Counter(str(normalize(a)) for a in p)
    w = c.most_common(1)[0][0]
    return next((a for a in p if str(normalize(a)) == w), p[0])

def extract_answer(raw):
    clean = re.sub(r"\$[^$]+\$", "", raw)
    for line in reversed(clean.split("\n")):
        line = line.strip()
        if re.match(r"(?i)^answer\s*:", line):
            ans = re.sub(r"(?i)^answer\s*:\s*", "", line).strip()
            if ans and ans.lower() not in ("", "none", "n/a"): return ans
    for line in reversed(clean.split("\n")):
        m = re.search(r"=\s*(-?[\d,]+\.?\d*%?)\s*$", line.strip())
        if m:
            val = m.group(1).replace(",", "")
            try:
                fv = float(val.rstrip("%"))
                if 1900 <= fv <= 2100 and "." not in val: continue
            except: pass
            return val
    for n in reversed(re.findall(r"-?[\d,]+\.?\d*%?", raw)):
        nc = n.replace(",", "").rstrip(".")
        if not nc or nc in ("%", "-"): continue
        try:
            fv = float(nc.rstrip("%"))
            if 1900 <= fv <= 2100 and "." not in nc and len(str(int(fv))) == 4: continue
        except: pass
        return nc
    return "[malformed]"

def generate(prompt, do_sample=False):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp  = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(**inp, max_new_tokens=450,
                           do_sample=do_sample,
                           temperature=TEMPERATURE if do_sample else 1.0,
                           pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

# ── PROMPT ────────────────────────────────────────────────────────────────
def p_trace(question, context):
    return (f"You are a financial analyst. Answer using ONLY the data below.\n"
            f"Write plain numbers only when the answer is numeric. No LaTeX. No symbols.\n\n"
            f"DATA:\n{context}\n\nQUESTION: {question}\n\n"
            f"Step 1 - Extract the needed information from the data.\n"
            f"Step 2 - If calculation is needed, show arithmetic step by step.\n"
            f"Step 3 - Write the final answer.\n\n"
            f"IMPORTANT: The LAST line must be:\nAnswer: <final answer only, nothing else>")

# ══════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════
results    = []
_start_idx = 0

ckpt = _load_json_safe(CHECKPOINT)
if ckpt:
    results    = ckpt["results"]
    _start_idx = ckpt["next_idx"]
    print(f"\nCheckpoint found — resuming from {_start_idx}/{len(sample)}")
else:
    print("\nNo checkpoint — starting fresh.")

for idx, row in enumerate(sample[_start_idx:], start=_start_idx):
    if _interrupted:
        print("Saving and exiting...")
        break

    ctx_full = build_context(raw_data[row["entry_idx"]], row["turn_idx"])

    traces = []
    for i in range(N_TRACES):
        raw_out = generate(p_trace(row["question"], ctx_full), do_sample=(i > 0))
        traces.append(extract_answer(raw_out))

    pred    = pick_majority(traces)
    correct = match(pred, row["answer"])

    results.append({
        "uid":          row["uid"],
        "entry_idx":    row["entry_idx"],
        "turn_idx":     row["turn_idx"],
        "question":     row["question"],
        "ground_truth": row["answer"],
        "prediction":   pred,
        "correct":      correct,
        "traces":       traces,
    })

    n = len(results)
    if n % SAVE_EVERY == 0 or n == 1:
        acc = sum(1 for r in results if r["correct"]) / n
        print(f"  [{n:4d}/{len(sample)}] acc={acc:.1%}  correct={sum(1 for r in results if r['correct'])}")
        _save_json_atomic({"results": results, "next_idx": _start_idx + n}, CHECKPOINT)
        print(f"     Checkpoint saved ({n}/{len(sample)})")

if _interrupted:
    print("Checkpoint saved. Run again to continue.")
    sys.exit(0)

if CHECKPOINT.exists(): CHECKPOINT.unlink()

# ── SUMMARY ───────────────────────────────────────────────────────────────
N   = len(results)
acc = sum(1 for r in results if r["correct"]) / N
print(f"\n{'='*60}")
print(f"ConvFinQA LLM BASELINE — SUMMARY")
print(f"{'='*60}")
print(f"  Model   : {MODEL}")
print(f"  N       : {N}")
print(f"  Accuracy: {acc:.1%}  ({sum(1 for r in results if r['correct'])}/{N})")

print(f"\nAccuracy by turn:")
for t in range(4):
    label = f"Turn {t}" if t < 3 else "Turn 3+"
    sub   = [r for r in results if (r["turn_idx"] == t if t < 3 else r["turn_idx"] >= 3)]
    if sub:
        a = sum(1 for r in sub if r["correct"]) / len(sub)
        print(f"  {label:<10} n={len(sub):4d}  acc={a:.1%}")

pd.DataFrame(results).to_csv(OUT_PATH, index=False)
print(f"\nSaved to {OUT_PATH}")
print(f"UIDs file -> {UIDS_PATH}  <- must match SLM and hierarchical runs")
