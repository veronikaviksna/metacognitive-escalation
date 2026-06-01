# finqa_random_baseline_v2.py — Random Escalation Baseline
#
# Matches the exact escalation counts from the hierarchical pipeline:
#   Total questions      : 1000
#   SLM -> LLM           : 433  (43.3% of all)
#   LLM -> human         : 40   (9.24% of escalated to LLM)
#
# Questions to escalate SLM->LLM are chosen randomly (not by signals).
# Among those, questions to escalate LLM->human are also chosen randomly.
# All answers come from the actual models — only the routing is random.

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
LARGE_MODEL = "Qwen/Qwen2.5-14B-Instruct"

BASE_DIR          = Path(".")
FINQA_JSON        = BASE_DIR / "train.json"
OUT_PATH          = BASE_DIR / "finqa_random_results_v2.csv"
CHECKPOINT_PATH   = BASE_DIR / "finqa_random_checkpoint_v2.json"
CHECKPOINT2_PATH  = BASE_DIR / "finqa_random_checkpoint2_v2.json"
UIDS_PATH         = BASE_DIR / "finqa_sample_uids.json"   # same UIDs as hierarchical

SAVE_EVERY   = 10
RANDOM_STATE = 42
N_TRACES     = 3

# Exact escalation counts from the hierarchical pipeline
N_TOTAL          = 1000
N_ESCALATE_LLM   = 433   # SLM -> LLM
N_ESCALATE_HUMAN = 40    # LLM -> human (out of 433 escalated)

# ── GRACEFUL INTERRUPT ────────────────────────────────────────────────────
_interrupted = False

def _handle_sigint(sig, frame):
    global _interrupted
    print("\n\nInterrupted — saving checkpoint and exiting after current question.")
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
    except:
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
    raise FileNotFoundError(
        f"UID file not found at {UIDS_PATH}. "
        "Run finqa_hierarch_1000.py first to generate the shared UID file.")

sample = pd.DataFrame([{
    "record_id": idx,
    "question":  raw_data[idx].get("qa", {}).get("question", ""),
    "answer":    str(raw_data[idx].get("qa", {}).get("answer", "")),
} for idx in record_ids]).reset_index(drop=True)

print(f"Loaded {len(raw_data)} FinQA records, using {len(sample)} questions")

# ── RANDOM ROUTING PLAN ───────────────────────────────────────────────────
# Pre-generate routing decisions so they are fixed and reproducible.
#   slm_to_llm_set   : sample indices routed SLM -> LLM
#   llm_to_human_set : indices (within the above) routed LLM -> human

random.seed(RANDOM_STATE)
all_indices      = list(range(N_TOTAL))
slm_to_llm_set   = set(random.sample(all_indices, N_ESCALATE_LLM))
llm_to_human_set = set(random.sample(list(slm_to_llm_set), N_ESCALATE_HUMAN))

print(f"\nRandom routing plan:")
print(f"  SLM -> LLM  : {len(slm_to_llm_set)} questions")
print(f"  LLM -> human: {len(llm_to_human_set)} questions")
print(f"  Coverage    : {N_TOTAL - len(llm_to_human_set)}/{N_TOTAL}")

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

def get_context(rid):      return _build_context(raw_data[rid], True,  4000)
def get_context_full(rid): return _build_context(raw_data[rid], False, 8000)

# ── MODEL HELPERS ─────────────────────────────────────────────────────────
def _make_bnb_cfg():
    try:
        from bitsandbytes import __version__ as _v
        ver = tuple(int(x) for x in _v.split(".")[:3])
        if ver >= (0, 41, 0):
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True)
    except: pass
    return None

_bnb_cfg = _make_bnb_cfg()

def load_model(name):
    print(f"\nLoading {name}...")
    tok = AutoTokenizer.from_pretrained(name)
    if _bnb_cfg:
        mdl = AutoModelForCausalLM.from_pretrained(
            name, quantization_config=_bnb_cfg, device_map="auto")
        print("  4-bit quantization OK")
    else:
        mdl = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.float16, device_map="auto")
        print("  float16")
    free = (torch.cuda.get_device_properties(0).total_memory
            - torch.cuda.memory_allocated()) / 1e9
    print(f"  VRAM free: {free:.1f} GB")
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
            pad_token_id=tok.eos_token_id)
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

# ── PROMPTS ───────────────────────────────────────────────────────────────
def p_slm_trace(q, ctx, is_bool=False):
    ans = ("The LAST line must be:\nAnswer: yes\nor\nAnswer: no" if is_bool
           else "The LAST line must be:\nAnswer: <final number only>")
    return (f"You are a financial analyst. Answer using ONLY the data below.\n"
            f"No LaTeX. No symbols. Plain arithmetic only.\n\n"
            f"DATA:\n{ctx}\n\nQUESTION: {q}\n\n"
            f"Step 1 - Extract needed numbers.\n"
            f"Step 2 - Show arithmetic step by step.\n"
            f"Step 3 - Write the final answer.\n\n"
            f"IMPORTANT: {ans}")

def p_llm(q, ctx, is_bool=False):
    ans = ("The LAST line must be:\nAnswer: yes\nor\nAnswer: no" if is_bool
           else "The LAST line must be:\nAnswer: <final number only>")
    return (f"You are a financial analyst. Answer using ONLY the data below.\n"
            f"No LaTeX. No symbols. Plain arithmetic only.\n\n"
            f"DATA:\n{ctx}\n\nQUESTION: {q}\n\n"
            f"Step 1 - Extract needed numbers.\n"
            f"Step 2 - Show full arithmetic step by step.\n"
            f"Step 3 - Verify: correct year? correct column? correct sign?\n\n"
            f"IMPORTANT: {ans}")

# ══════════════════════════════════════════════════════════════════════════
# PHASE 1: SMALL MODEL
# ══════════════════════════════════════════════════════════════════════════
model_small, tok_small = load_model(SMALL_MODEL)

mc_states  = []
_start_idx = 0

ckpt = _load_json_safe(CHECKPOINT_PATH)
if ckpt:
    mc_states  = ckpt["mc_states"]
    _start_idx = ckpt["next_idx"]
    print(f"\nCheckpoint found — resuming from {_start_idx}/{len(sample)}")
else:
    print("\nNo checkpoint — starting fresh.")

for q_idx, (_, row) in enumerate(sample.iloc[_start_idx:].iterrows()):
    if _interrupted:
        print("Saving checkpoint and exiting phase 1...")
        break

    sample_idx   = _start_idx + q_idx
    question     = row["question"]
    record_id    = int(row["record_id"])
    ground_truth = str(row["answer"])
    is_bool      = _is_boolean(question)
    ctx_full     = get_context_full(record_id)

    traces_raw, traces_ans = [], []
    for i in range(N_TRACES):
        t = generate(model_small, tok_small,
                     p_slm_trace(question, ctx_full, is_bool),
                     max_new_tokens=300, do_sample=(i > 0), temperature=0.7)
        traces_raw.append(t)
        traces_ans.append(extract_answer(t))

    slm_s5      = compute_s5(traces_ans)
    majority    = pick_majority(traces_ans)
    slm_correct = match(majority, ground_truth)
    slm_dec     = "escalate_llm" if sample_idx in slm_to_llm_set else "accept"

    mc_states.append({
        "sample_idx":       sample_idx,
        "record_id":        record_id,
        "question":         question,
        "ground_truth":     ground_truth,
        "is_boolean":       is_bool,
        "slm_s5":           slm_s5,
        "majority":         majority,
        "slm_correct":      slm_correct,
        "slm_decision":     slm_dec,
        "llm_answer":       None,
        "llm_s5":           None,
        "llm_decision":     None,
        "llm_correct":      None,
        "final_answer":     majority if slm_dec == "accept" else None,
        "final_correct":    slm_correct if slm_dec == "accept" else None,
        "human_escalation": False,
    })

    n = len(mc_states)
    if n % SAVE_EVERY == 0 or n == 1:
        acc   = sum(1 for s in mc_states if s["slm_correct"]) / n
        n_esc = sum(1 for s in mc_states if s["slm_decision"] == "escalate_llm")
        print(f"  [{n:4d}/{len(sample)}] slm_acc={acc:.1%}  escalated={n_esc}")
        _save_json_atomic({"mc_states": mc_states, "next_idx": _start_idx + n}, CHECKPOINT_PATH)
        print(f"     Checkpoint saved ({n}/{len(sample)})")

unload_model(model_small); del tok_small

if _interrupted:
    print("\nCheckpoint saved. Run again to continue.")
    sys.exit(0)

if CHECKPOINT_PATH.exists():
    CHECKPOINT_PATH.unlink()
    print("Phase 1 complete.")

# ══════════════════════════════════════════════════════════════════════════
# PHASE 2: LARGE MODEL
# ══════════════════════════════════════════════════════════════════════════
to_llm = [s for s in mc_states if s["slm_decision"] == "escalate_llm"]
print(f"\nSLM accept={sum(1 for s in mc_states if s['slm_decision']=='accept')}  "
      f"->LLM={len(to_llm)}")

if to_llm:
    _llm_start    = 0
    _llm_done_log = []

    ckpt2 = _load_json_safe(CHECKPOINT2_PATH)
    if ckpt2:
        for saved in ckpt2["llm_done"]:
            mc_states[saved["_mc_idx"]].update(
                {k: v for k, v in saved.items() if k != "_mc_idx"})
        _llm_done_log = ckpt2["llm_done"]
        _llm_start    = len(_llm_done_log)
        print(f"LLM checkpoint: resuming from {_llm_start}/{len(to_llm)}")

    model_large, tok_large = load_model(LARGE_MODEL)

    for i, state in enumerate(to_llm[_llm_start:], start=_llm_start):
        if _interrupted:
            print("Saving LLM checkpoint and exiting...")
            break

        ctx_full     = get_context_full(state["record_id"])
        llm_ans_list = []
        for j in range(N_TRACES):
            r = generate(model_large, tok_large,
                         p_llm(state["question"], ctx_full, state["is_boolean"]),
                         max_new_tokens=450, do_sample=(j > 0), temperature=0.7)
            llm_ans_list.append(extract_answer(r))

        llm_s5      = compute_s5(llm_ans_list)
        llm_answer  = pick_majority(llm_ans_list)
        llm_correct = match(llm_answer, state["ground_truth"])
        llm_dec     = "escalate_human" if state["sample_idx"] in llm_to_human_set else "accept"

        state.update({"llm_answer": llm_answer, "llm_s5": llm_s5,
                      "llm_decision": llm_dec, "llm_correct": llm_correct})
        if llm_dec == "accept":
            state.update({"final_answer": llm_answer, "final_correct": llm_correct,
                          "human_escalation": False})
        else:
            state.update({"final_answer": None, "final_correct": None,
                          "human_escalation": True})

        _mc_idx = mc_states.index(state)
        _llm_done_log.append({
            "_mc_idx":          _mc_idx,
            "llm_answer":       state["llm_answer"],
            "llm_s5":           state["llm_s5"],
            "llm_decision":     state["llm_decision"],
            "llm_correct":      state["llm_correct"],
            "final_answer":     state["final_answer"],
            "final_correct":    state["final_correct"],
            "human_escalation": state["human_escalation"],
        })

        if (i + 1) % SAVE_EVERY == 0 or i == _llm_start:
            ok   = sum(1 for s in to_llm[:i+1] if s.get("llm_correct"))
            n_hm = sum(1 for s in to_llm[:i+1] if s.get("human_escalation"))
            print(f"  [LLM {i+1:3d}/{len(to_llm)}] correct={ok/(i+1):.1%}  ->human={n_hm}")
            _save_json_atomic({"llm_done": _llm_done_log}, CHECKPOINT2_PATH)

    unload_model(model_large); del tok_large

    if _interrupted:
        print("\nLLM checkpoint saved. Run again to continue.")
        sys.exit(0)

    if CHECKPOINT2_PATH.exists():
        CHECKPOINT2_PATH.unlink()
        print("Phase 2 complete.")

# ══════════════════════════════════════════════════════════════════════════
# SUMMARY AND REPORT
# ══════════════════════════════════════════════════════════════════════════
N   = len(mc_states)
SEP = "=" * 60

n_acc     = sum(1 for s in mc_states if s["slm_decision"] == "accept")
n_esc     = sum(1 for s in mc_states if s["slm_decision"] == "escalate_llm")
n_llm_acc = sum(1 for s in mc_states if s.get("llm_decision") == "accept")
n_human   = sum(1 for s in mc_states if s["human_escalation"])
n_ans     = N - n_human

slm_overall = sum(1 for s in mc_states if s["slm_correct"]) / N
slm_prec    = sum(1 for s in mc_states if s["slm_decision"]=="accept" and s["slm_correct"]) / max(1, n_acc)
llm_prec    = sum(1 for s in mc_states if s.get("llm_decision")=="accept" and s.get("llm_correct")) / max(1, n_llm_acc)
fin_corr    = sum(1 for s in mc_states if s["final_correct"])

# Routing quality vs SLM correctness
esc_states = [s for s in mc_states if s["slm_decision"] == "escalate_llm"]
acc_states = [s for s in mc_states if s["slm_decision"] == "accept"]
TP = sum(1 for s in esc_states if not s["slm_correct"])
FP = sum(1 for s in esc_states if     s["slm_correct"])
FN = sum(1 for s in acc_states if not s["slm_correct"])
TN = sum(1 for s in acc_states if     s["slm_correct"])
prec = TP / max(1, TP + FP)
rec  = TP / max(1, TP + FN)
f1   = 2 * prec * rec / max(0.001, prec + rec)

print(f"\n{SEP}")
print("RANDOM BASELINE — FinQA (n=1000)")
print(f"Escalation counts match hierarchical: SLM->LLM={N_ESCALATE_LLM}, LLM->human={N_ESCALATE_HUMAN}")
print(SEP)
print(f"""
ROUTING FLOW
  Accepted by SLM        : {n_acc:4d}  ({n_acc/N:.1%})
  Escalated to LLM       : {n_esc:4d}  ({n_esc/N:.1%})
  LLM accepted           : {n_llm_acc:4d}  ({n_llm_acc/max(1,n_esc):.1%} of escalated)
  Escalated to human     : {n_human:4d}  ({n_human/max(1,n_esc):.1%} of escalated)
  Coverage               : {n_ans}/{N}  ({n_ans/N:.1%})

ACCURACY
  SLM overall            : {slm_overall:.1%}
  SLM accepted precision : {slm_prec:.1%}  (n={n_acc})
  LLM accepted precision : {llm_prec:.1%}  (n={n_llm_acc})
  Final (excl. human)    : {fin_corr}/{n_ans} = {fin_corr/max(1,n_ans):.1%}
  Final overall          : {fin_corr}/{N} = {fin_corr/N:.1%}

ROUTING QUALITY  (random routing vs SLM correctness)
  TP (wrong -> escalated)          : {TP}
  FP (correct -> escalated)        : {FP}
  FN (wrong -> accepted)           : {FN}
  TN (correct -> accepted)         : {TN}
  Escalation precision             : {prec:.1%}
  Escalation recall                : {rec:.1%}
  Escalation F1                    : {f1:.3f}
""")
print(SEP)

# ── SAVE CSV ──────────────────────────────────────────────────────────────
pd.DataFrame(mc_states).to_csv(OUT_PATH, index=False)
print(f"Saved -> {OUT_PATH}")
