# tatqa_random_baseline_v2.py — TAT-QA Random Escalation Baseline
#
# Matches the exact escalation counts from the hierarchical pipeline:
#   Total questions      : 1000
#   SLM -> LLM           : 286  (28.6% of all)
#   LLM -> human         : 37   (12.9% of escalated to LLM)
#
# Routing is random — only the counts match the hierarchical pipeline.
# All answers come from the actual models.
#
# Usage:
#   nohup python -u tatqa_random_baseline_v2.py > tatqa_random.log 2>&1 &
#   tail -f tatqa_random.log

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
TATQA_JSON        = BASE_DIR / "tatqa_dataset_test_gold.json"
OUT_PATH          = BASE_DIR / "tatqa_random_results_v2.csv"
CHECKPOINT_PATH   = BASE_DIR / "tatqa_random_checkpoint_v2.json"
CHECKPOINT2_PATH  = BASE_DIR / "tatqa_random_checkpoint2_v2.json"
UIDS_PATH         = BASE_DIR / "tatqa_sample_uids.json"

SAVE_EVERY   = 10
RANDOM_STATE = 42
N_TRACES     = 3

# Exact escalation counts from the hierarchical pipeline
N_TOTAL          = 1000
N_ESCALATE_LLM   = 286
N_ESCALATE_HUMAN = 37

# ── GRACEFUL INTERRUPT ────────────────────────────────────────────────────
_interrupted = False

def _handle_sigint(sig, frame):
    global _interrupted
    print("\n\nInterrupted — saving checkpoint after current question.")
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
print(f"Loading TAT-QA data from {TATQA_JSON}...")
with open(TATQA_JSON, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

all_questions = []
for doc_idx, doc in enumerate(raw_data):
    for q in doc.get("questions", []):
        all_questions.append({
            "doc_idx":     doc_idx,
            "uid":         q.get("uid", ""),
            "question":    q.get("question", ""),
            "answer":      q.get("answer", ""),
            "answer_type": q.get("answer_type", ""),
            "answer_from": q.get("answer_from", ""),
            "scale":       q.get("scale", ""),
        })

print(f"Loaded {len(raw_data)} documents, {len(all_questions)} questions total")

# ── LOAD UIDs ─────────────────────────────────────────────────────────────
uid_file = _load_json_safe(UIDS_PATH)
if not uid_file:
    raise FileNotFoundError(
        f"UID file not found at {UIDS_PATH}. "
        "Run tatqa_hierarchical_1000_best.py first to generate the shared UID file.")

target_uids = set(uid_file)
sample = [q for q in all_questions if q["uid"] in target_uids]
print(f"Sample ready: {len(sample)} questions")

# ── RANDOM ROUTING PLAN ───────────────────────────────────────────────────
# Pre-generate routing decisions so they are fixed and reproducible.
#   slm_to_llm_set   : sample indices routed SLM -> LLM
#   llm_to_human_set : indices (within the above) routed LLM -> human

random.seed(RANDOM_STATE)
all_indices      = list(range(len(sample)))
slm_to_llm_set   = set(random.sample(all_indices, min(N_ESCALATE_LLM, len(sample))))
llm_to_human_set = set(random.sample(list(slm_to_llm_set), min(N_ESCALATE_HUMAN, len(slm_to_llm_set))))

print(f"\nRandom routing plan:")
print(f"  SLM -> LLM  : {len(slm_to_llm_set)}  ({len(slm_to_llm_set)/len(sample):.1%})")
print(f"  LLM -> human: {len(llm_to_human_set)}")
print(f"  Coverage    : {len(sample) - len(llm_to_human_set)}/{len(sample)}")

# ── CONTEXT BUILDERS ──────────────────────────────────────────────────────
def _table_to_text(table_data: list) -> str:
    if not table_data or len(table_data) < 2:
        return ""
    header = table_data[0]
    parts  = []
    for row in table_data[1:]:
        if not any(str(c).strip() for c in row):
            continue
        row_name = str(row[0]).strip()
        for col_idx, cell in enumerate(row[1:], start=1):
            if col_idx < len(header):
                col_name = str(header[col_idx]).strip()
                cell_val = str(cell).strip()
                if cell_val and cell_val not in ("-", "—", ""):
                    label = f"{row_name} of {col_name}" if row_name and col_name else (row_name or col_name)
                    parts.append(f"the {label} is {cell_val}")
    return " ; ".join(parts)

def extract_context_full(doc_idx: int, max_chars: int = 8000) -> str:
    doc = raw_data[doc_idx]
    table_raw    = doc.get("table", {})
    table_matrix = table_raw.get("table", []) if isinstance(table_raw, dict) else table_raw
    table_text   = _table_to_text(table_matrix)
    paragraphs   = sorted(doc.get("paragraphs", []), key=lambda p: p.get("order", 0))
    para_text    = " ".join(p.get("text", "").strip() for p in paragraphs)
    ctx = f"{para_text} {table_text}".strip()
    ctx = re.sub(r"\s{2,}", " ", ctx)
    return (ctx[:max_chars] + " ...[truncated]") if len(ctx) > max_chars else ctx

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
    inp  = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inp, max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

# ── UTILITIES ─────────────────────────────────────────────────────────────
def normalize_span(a):
    if isinstance(a, list):
        return " | ".join(str(x).strip().lower() for x in sorted(a) if str(x).strip())
    return str(a).strip().lower()

def normalize_num(a):
    a = str(a).strip()
    if re.match(r"^\([\d,.]+\)$", a): a = "-" + a[1:-1]
    a = re.sub(r"^[≈~<>about\s]+", "", a, flags=re.IGNORECASE)
    a = re.sub(r"[$€£¥\s,%]", "", a)
    try: return float(a)
    except: return None

def match_tatqa(pred, gt, answer_type="arithmetic", tol=0.03):
    if isinstance(gt, list):
        gt_str   = normalize_span(gt)
        pred_str = normalize_span(pred) if isinstance(pred, list) else str(pred).strip().lower()
        return pred_str == gt_str
    if answer_type in ("arithmetic", "count"):
        p = normalize_num(pred); g = normalize_num(gt)
        if p is None or g is None: return str(pred).strip().lower() == str(gt).strip().lower()
        if p == g: return True
        if abs(g) < 0.001: return abs(p - g) < 0.001
        if abs(p - g) / abs(g) < tol: return True
        for scale in [100, 1000, 0.01]:
            if abs(p * scale - g) / max(abs(g), 1e-9) < tol: return True
        return False
    return str(pred).strip().lower() == str(gt).strip().lower()

def parseable(a):
    if not a or str(a).lower() in ("[malformed]", "none", "", "n/a"): return False
    return True

def answers_agree_tatqa(a, b, tol=0.03):
    if str(a).strip().lower() == str(b).strip().lower(): return True
    na, nb = normalize_num(a), normalize_num(b)
    if na is not None and nb is not None:
        if na == nb: return True
        if abs(nb) > 0 and abs(na - nb) / abs(nb) < tol: return True
    return False

def pick_majority(answers):
    p = [a for a in answers if parseable(a)]
    if not p: return answers[0] if answers else "[malformed]"
    c = Counter(str(a).strip().lower() for a in p)
    w = c.most_common(1)[0][0]
    return next((a for a in p if str(a).strip().lower() == w), p[0])

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
    agree = sum(1 for a, b in pairs if answers_agree_tatqa(a, b))
    return round(agree / len(pairs), 3)

def _is_boolean(q):
    ql = q.lower().strip()
    if any(ql.startswith(s) for s in ["was ","were ","is ","are ","did ","does ",
                                       "would ","could ","has ","have ","will ","do "]): return True
    if any(x in ql for x in ["greater than","more than","less than",
                               "higher than","lower than","larger than","smaller than"]): return True
    return False

# ── PROMPTS ───────────────────────────────────────────────────────────────
def p_slm_trace(q, ctx, is_bool=False):
    ans = ("The LAST line must be:\nAnswer: yes\nor\nAnswer: no" if is_bool
           else "The LAST line must be:\nAnswer: <final answer only>")
    return (f"You are a financial analyst. Answer using ONLY the data below.\n"
            f"No LaTeX. No symbols. Plain arithmetic only.\n\n"
            f"DATA:\n{ctx}\n\nQUESTION: {q}\n\n"
            f"Step 1 - Extract needed numbers or text.\n"
            f"Step 2 - Show arithmetic step by step.\n"
            f"Step 3 - Write the final answer.\n\nIMPORTANT: {ans}")

def p_llm(q, ctx, is_bool=False):
    ans = ("The LAST line must be:\nAnswer: yes\nor\nAnswer: no" if is_bool
           else "The LAST line must be:\nAnswer: <final answer only>")
    return (f"You are a financial analyst. Answer using ONLY the data below.\n"
            f"No LaTeX. No symbols. Plain arithmetic only.\n\n"
            f"DATA:\n{ctx}\n\nQUESTION: {q}\n\n"
            f"Step 1 - Extract needed numbers or text.\n"
            f"Step 2 - Show full arithmetic step by step.\n"
            f"Step 3 - Verify: correct year? correct column? correct sign?\n\nIMPORTANT: {ans}")

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
    print(f"\nNo checkpoint — starting fresh.")

for q_idx, q_item in enumerate(sample[_start_idx:]):
    if _interrupted:
        _save_json_atomic({"mc_states": mc_states,
                           "next_idx": _start_idx + len(mc_states)}, CHECKPOINT_PATH)
        print("Checkpoint saved. Run again to continue.")
        sys.exit(0)

    sample_idx   = _start_idx + q_idx
    question     = q_item["question"]
    ground_truth = q_item["answer"]
    answer_type  = q_item["answer_type"]
    uid          = q_item["uid"]
    doc_idx      = q_item["doc_idx"]
    is_bool      = _is_boolean(question)
    ctx_full     = extract_context_full(doc_idx)

    traces_raw, traces_ans = [], []
    for i in range(N_TRACES):
        t = generate(model_small, tok_small,
                     p_slm_trace(question, ctx_full, is_bool),
                     max_new_tokens=300, do_sample=(i > 0), temperature=0.7)
        traces_raw.append(t)
        traces_ans.append(extract_answer(t))

    slm_s5      = compute_s5(traces_ans)
    majority    = pick_majority(traces_ans)
    slm_correct = match_tatqa(majority, ground_truth, answer_type)
    slm_dec     = "escalate_llm" if sample_idx in slm_to_llm_set else "accept"

    mc_states.append({
        "sample_idx":       sample_idx,
        "uid":              uid,
        "question":         question,
        "ground_truth":     str(ground_truth),
        "answer_type":      answer_type,
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
        _save_json_atomic({"mc_states": mc_states,
                           "next_idx": _start_idx + n}, CHECKPOINT_PATH)

unload_model(model_small); del tok_small

if _interrupted:
    print("\nCheckpoint saved. Run again to continue.")
    sys.exit(0)

if CHECKPOINT_PATH.exists(): CHECKPOINT_PATH.unlink()
print("Phase 1 complete.")

# ══════════════════════════════════════════════════════════════════════════
# PHASE 2: LARGE MODEL
# ══════════════════════════════════════════════════════════════════════════
to_llm = [s for s in mc_states if s["slm_decision"] == "escalate_llm"]
print(f"\nSLM accept={sum(1 for s in mc_states if s['slm_decision']=='accept')}  "
      f"->LLM={len(to_llm)}")

# Build a uid->doc_idx lookup for the LLM phase
uid_to_doc = {q["uid"]: q["doc_idx"] for q in sample}

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
            _save_json_atomic({"llm_done": _llm_done_log}, CHECKPOINT2_PATH)
            print("LLM checkpoint saved. Run again to continue.")
            sys.exit(0)

        ctx_full     = extract_context_full(uid_to_doc[state["uid"]])
        llm_ans_list = []
        for j in range(N_TRACES):
            r = generate(model_large, tok_large,
                         p_llm(state["question"], ctx_full, state["is_boolean"]),
                         max_new_tokens=450, do_sample=(j > 0), temperature=0.7)
            llm_ans_list.append(extract_answer(r))

        llm_s5      = compute_s5(llm_ans_list)
        llm_answer  = pick_majority(llm_ans_list)
        llm_correct = match_tatqa(llm_answer, state["ground_truth"], state["answer_type"])
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

    if CHECKPOINT2_PATH.exists(): CHECKPOINT2_PATH.unlink()
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

esc_states = [s for s in mc_states if s["slm_decision"] == "escalate_llm"]
acc_states = [s for s in mc_states if s["slm_decision"] == "accept"]
TP = sum(1 for s in esc_states if not s["slm_correct"])
FP = sum(1 for s in esc_states if     s["slm_correct"])
FN = sum(1 for s in acc_states if not s["slm_correct"])
TN = sum(1 for s in acc_states if     s["slm_correct"])
prec = TP / max(1, TP + FP)
rec  = TP / max(1, TP + FN)
f1   = 2 * prec * rec / max(0.001, prec + rec)

slm_error_rate     = 1 - slm_overall
expected_tp_random = n_esc * slm_error_rate
precision_lift     = prec / max(0.001, slm_error_rate)

print(f"\n{SEP}")
print("RANDOM BASELINE — TAT-QA (n=1000)")
print(f"Escalation counts match hierarchical: "
      f"SLM->LLM={N_ESCALATE_LLM}, LLM->human={N_ESCALATE_HUMAN}")
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

PRECISION LIFT vs RANDOM
  SLM error rate                   : {slm_error_rate:.1%}
  Expected TP if purely random     : {expected_tp_random:.0f}
  Actual TP                        : {TP}
  Precision lift                   : {precision_lift:.2f}x  (1.0 = no better than random)
""")
print(SEP)

pd.DataFrame(mc_states).to_csv(OUT_PATH, index=False)
print(f"Saved -> {OUT_PATH}")
