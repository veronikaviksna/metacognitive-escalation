# finqa_slm_1000.py — FinQA SLM-Only Baseline
# Runs ONLY the small model (Qwen2.5-3B-Instruct) on the 1000 shared UIDs.
# Loads finqa_sample_uids.json written by finqa_hierarch_1000.py.
# No escalation — every answer comes from the SLM majority vote.
# Output: finqa_slm_results.csv

import os, re, gc, json, time, random, signal, sys
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── SETTINGS ──────────────────────────────────────────────────────────────
SMALL_MODEL = "Qwen/Qwen2.5-3B-Instruct"

BASE_DIR  = Path(".")
FINQA_JSON = BASE_DIR / "train.json"
OUT_PATH        = BASE_DIR / "finqa_slm_results.csv"
CHECKPOINT_PATH = BASE_DIR / "finqa_slm_checkpoint.json"
UIDS_PATH       = BASE_DIR / "finqa_sample_uids.json"   # shared — must already exist

SAVE_EVERY   = 10
DEMO_SIZE    = 1000
RANDOM_STATE = 42
N_TRACES     = 3

# ── GRACEFUL INTERRUPT ────────────────────────────────────────────────────
_interrupted = False

def _handle_sigint(sig, frame):
    global _interrupted
    print("\n\nInterrupted — will save checkpoint and exit after current question.")
    _interrupted = True

signal.signal(signal.SIGINT,  _handle_sigint)
signal.signal(signal.SIGTERM, _handle_sigint)

# ── CHECKPOINT HELPERS ────────────────────────────────────────────────────
def _save_json_atomic(data, path: Path):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f)
    tmp.replace(path)

def _load_json_safe(path: Path):
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, KeyError, FileNotFoundError):
        return None

# ── DATA LOADING ──────────────────────────────────────────────────────────
print("Loading FinQA data...")
with open(FINQA_JSON, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

uid_file = _load_json_safe(UIDS_PATH)
if uid_file:
    record_ids = uid_file
    print(f"Loaded existing UID list ({len(record_ids)} questions) from {UIDS_PATH}")
else:
    # Fallback: sample fresh (so hierarch can reuse if run after)
    random.seed(RANDOM_STATE)
    record_ids = random.sample(range(len(raw_data)), min(DEMO_SIZE, len(raw_data)))
    _save_json_atomic(record_ids, UIDS_PATH)
    print(f"No UID file found — sampled fresh and saved -> {UIDS_PATH}")

sample = pd.DataFrame([{
    "record_id": idx,
    "question":  raw_data[idx].get("qa", {}).get("question", ""),
    "answer":    str(raw_data[idx].get("qa", {}).get("answer", "")),
} for idx in record_ids]).reset_index(drop=True)
print(f"Loaded {len(raw_data)} FinQA records, using {len(sample)} questions")

# ── CONTEXT BUILDERS ──────────────────────────────────────────────────────
def _remove_space(t): return " ".join(x for x in t.split(" ") if x)

def _table_row_to_text(header, row):
    res = (header[0] + " ") if header[0] else ""
    for head, cell in zip(header[1:], row[1:]):
        res += "the " + row[0] + " of " + head + " is " + cell + " ; "
    return _remove_space(res).strip()

def _build_context(item, use_ann=False, max_chars=8000):
    qa    = item.get("qa", {})
    pre   = item.get("pre_text", [])
    post  = item.get("post_text", [])
    table = item.get("table", [])
    ann_t = qa.get("ann_table_rows", [])
    ann_x = qa.get("ann_text_rows",  [])
    if use_ann and (ann_x or ann_t):
        all_text = pre + post
        text_str = " ".join(str(all_text[i]).strip() for i in ann_x if i < len(all_text))
    else:
        text_str = " ".join(str(p) for p in pre) + " " + " ".join(str(p) for p in post)
    tbl = ""
    if table and len(table) >= 2:
        hdr  = table[0]
        rows = ann_t if (use_ann and ann_t) else range(1, len(table))
        for idx in rows:
            if 0 < idx < len(table):
                tbl += _table_row_to_text(hdr, table[idx]) + " "
    ctx = (text_str + " " + tbl).strip()
    ctx = ctx.replace(". . . . . .", "").replace("* * * * * *", "")
    return (ctx[:max_chars] + " ...[truncated]") if len(ctx) > max_chars else ctx

def extract_context_full(rid, mc=8000): return _build_context(raw_data[rid], False, mc)

# ── MODEL HELPERS ─────────────────────────────────────────────────────────
def _make_bnb_cfg():
    try:
        from bitsandbytes import __version__ as _bnb_ver
        _ver = tuple(int(x) for x in _bnb_ver.split(".")[:3])
        if _ver >= (0, 41, 0) and hasattr(torch.nn.Module, "set_submodule"):
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
    except Exception:
        pass
    return None

_bnb_cfg = _make_bnb_cfg()

def load_model(name):
    print(f"\nLoading {name}...")
    tok = AutoTokenizer.from_pretrained(name)
    if _bnb_cfg is not None:
        print("  Using 4-bit quantization (bitsandbytes)")
        mdl = AutoModelForCausalLM.from_pretrained(
            name, quantization_config=_bnb_cfg, device_map="auto")
    else:
        print("  4-bit quantization unavailable — loading in float16")
        mdl = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.float16, device_map="auto")
    free = (torch.cuda.get_device_properties(0).total_memory
            - torch.cuda.memory_allocated()) / 1e9
    print(f"Loaded. VRAM free: {free:.1f} GB")
    return mdl, tok

def unload_model(mdl):
    mdl.cpu(); del mdl; gc.collect()
    torch.cuda.empty_cache(); torch.cuda.synchronize(); time.sleep(2)

def generate(model, tok, prompt, max_new_tokens=300, do_sample=False, temperature=0.7):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp  = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inp, max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

# ── UTILITIES ─────────────────────────────────────────────────────────────
def normalize(a):
    a = str(a).strip()
    if re.match(r"^\([\d,.]+\)$", a): a = "-" + a[1:-1]
    a = re.sub(r"^[≈~<>about\s]+", "", a, flags=re.IGNORECASE)
    a = re.sub(r"[$€£¥\s,]", "", a)
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
    n = normalize(a)
    return isinstance(n, float) or str(n) in ("yes", "no")

def answers_agree(a, b, tol=0.03):
    if str(normalize(a)) == str(normalize(b)): return True
    try:
        fa, fb = float(normalize(a)), float(normalize(b))
        if fb == 0: return abs(fa) < 0.001
        if abs(fa - fb) / abs(fb) < tol: return True
        r = fa / fb
        if abs(r - 100) / 100 < tol: return True
        if abs(r - 0.01) / 0.01 < tol: return True
    except: pass
    return False

def _is_boolean(q):
    ql = q.lower().strip()
    if any(ql.startswith(s) for s in ["was ","were ","is ","are ","did ","does ",
                                       "would ","could ","has ","have ","will ","do "]): return True
    if any(x in ql for x in ["greater than","more than","less than",
                               "higher than","lower than","larger than","smaller than"]): return True
    return False

def pick_majority(answers):
    p = [a for a in answers if parseable(a)]
    if not p: return answers[0] if answers else "[malformed]"
    c = Counter(str(normalize(a)) for a in p)
    w = c.most_common(1)[0][0]
    return next((a for a in p if str(normalize(a)) == w), p[0])

def extract_answer(raw):
    clean = re.sub(r"\$[^$]+\$", "", raw)
    clean = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", clean)
    clean = re.sub(r"\\[a-zA-Z]+", "", clean)
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
            except ValueError: pass
            return val
    for n in reversed(re.findall(r"-?[\d,]+\.?\d*%?", raw)):
        nc = n.replace(",", "").rstrip(".")
        if not nc or nc in ("%", "-"): continue
        try:
            fv = float(nc.rstrip("%"))
            if 1900 <= fv <= 2100 and "." not in nc and len(str(int(fv))) == 4: continue
        except ValueError: pass
        return nc
    return "[malformed]"

def compute_s5(answers):
    if len(answers) <= 1: return 1.0
    pairs = [(a, b) for i, a in enumerate(answers) for b in answers[i+1:]]
    agree = sum(1 for a, b in pairs if answers_agree(str(normalize(a)), str(normalize(b))))
    return round(agree / len(pairs), 3)

# ── PROMPT ────────────────────────────────────────────────────────────────
def p_slm_trace(question, context, is_bool=False):
    ans_inst = ("The LAST line must be:\nAnswer: yes\nor\nAnswer: no" if is_bool
                else "The LAST line must be:\nAnswer: <final number only>")
    return (f"You are a financial analyst. Answer using ONLY the data below.\n"
            f"No LaTeX. No symbols. Plain arithmetic only.\n\n"
            f"DATA:\n{context}\n\nQUESTION: {question}\n\n"
            f"Step 1 - Extract needed numbers.\n"
            f"Step 2 - Show arithmetic step by step.\n"
            f"Step 3 - Write the final answer.\n\n"
            f"IMPORTANT: {ans_inst}")

# ══════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════
model_slm, tok_slm = load_model(SMALL_MODEL)

results    = []
_start_idx = 0

ckpt = _load_json_safe(CHECKPOINT_PATH)
if ckpt:
    results    = ckpt["results"]
    _start_idx = ckpt["next_idx"]
    print(f"\nCheckpoint found — resuming from {_start_idx}/{len(sample)} ({len(results)} done)")
else:
    print("\nNo checkpoint — starting fresh.")

for q_num, row in sample.iloc[_start_idx:].iterrows():
    if _interrupted:
        print("Saving checkpoint and exiting...")
        break

    question     = row["question"]
    record_id    = int(row["record_id"])
    ground_truth = str(row["answer"])
    is_bool      = _is_boolean(question)
    ctx_full     = extract_context_full(record_id)

    traces_raw, traces_ans = [], []
    for i in range(N_TRACES):
        t_raw = generate(model_slm, tok_slm,
                         p_slm_trace(question, ctx_full, is_bool),
                         max_new_tokens=300, do_sample=(i > 0), temperature=0.7)
        traces_raw.append(t_raw)
        traces_ans.append(extract_answer(t_raw))

    s5       = compute_s5(traces_ans)
    majority = pick_majority(traces_ans)
    correct  = match(majority, ground_truth)

    results.append({
        "record_id": record_id,
        "question": question, "ground_truth": ground_truth, "is_boolean": is_bool,
        "traces_answers": traces_ans, "majority": majority,
        "s5": s5, "correct": correct,
        "n_traces": N_TRACES, "version": "finqa_slm_1000",
    })

    n = len(results)
    if n % SAVE_EVERY == 0 or n == 1:
        acc = sum(1 for r in results if r["correct"]) / n
        print(f"  [{n:4d}/{len(sample)}] acc={acc:.1%}")
        _save_json_atomic({"results": results, "next_idx": _start_idx + n}, CHECKPOINT_PATH)
        print(f"     Checkpoint saved ({n}/{len(sample)})")

unload_model(model_slm); del tok_slm

if _interrupted:
    print("\nCheckpoint saved. Run the script again to continue.")
    sys.exit(0)

if CHECKPOINT_PATH.exists():
    CHECKPOINT_PATH.unlink()
    print("Run complete — checkpoint deleted.")

# ── SUMMARY ───────────────────────────────────────────────────────────────
N   = len(results)
acc = sum(1 for r in results if r["correct"]) / N
print(f"\n{'='*50}")
print(f"SLM-ONLY (FinQA)  N={N}  Accuracy={acc:.1%}")
print(f"{'='*50}")

# ── SAVE CSV ──────────────────────────────────────────────────────────────
pd.DataFrame(results).to_csv(OUT_PATH, index=False)
print(f"\nSaved to {OUT_PATH}")
