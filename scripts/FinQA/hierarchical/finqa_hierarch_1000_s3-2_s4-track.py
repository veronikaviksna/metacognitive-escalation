# finqa_hierarch_1000_s3-2_s4-track.py — FinQA Metacognitive Hierarchical Pipeline
# Scheme: SLM → ACCEPT | ESCALATE_LLM → ACCEPT | ESCALATE_HUMAN
#
# UID-based sampling: on first run, samples 1000 question indices and saves
# them to finqa_sample_uids.json. Subsequent runs load the same file so all
# three scripts run on identical questions.
#
# Signal hierarchy (Flavell 1979):
#   RED FLAGS : S5=0, S6 mismatch, S3 >= S3_HARD_THRESHOLD  → escalate_llm
#   SOFT      : S2=False, S3=1                              → raise MIN_S5
#   TRACKING  : S1, S4, S_operand

import os, re, gc, json, time, random, signal, sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── SETTINGS ──────────────────────────────────────────────────────────────
SMALL_MODEL  = "Qwen/Qwen2.5-3B-Instruct"
LARGE_MODEL  = "Qwen/Qwen2.5-14B-Instruct"

BASE_DIR   = Path(".")
FINQA_JSON = BASE_DIR / "train.json"
OUT_PATH         = BASE_DIR / "finqa_hierarch_results.csv"
CHECKPOINT_PATH  = BASE_DIR / "finqa_hierarch_checkpoint.json"
CHECKPOINT2_PATH = BASE_DIR / "finqa_hierarch_checkpoint2.json"
UIDS_PATH        = BASE_DIR / "finqa_sample_uids.json"   # shared with slm/llm scripts

SAVE_EVERY   = 10
DEMO_SIZE    = 1000
RANDOM_STATE = 42
N_TRACES     = 3

# ── CONTROL THRESHOLDS ────────────────────────────────────────────────────
MIN_S5            = 0.1
S3_HARD_THRESHOLD = 2
DELTA_S2          = 0.2   # soft: S2=False raises threshold
DELTA_S3_SOFT     = 0.1   # soft: S3=1 raises threshold

# ── GRACEFUL INTERRUPT ────────────────────────────────────────────────────
_interrupted = False

def _handle_sigint(sig, frame):
    global _interrupted
    print("\n\n⚠️  Interrupted — will save checkpoint and exit after current question.")
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
    print(f"✅ Loaded existing UID list ({len(record_ids)} questions) from {UIDS_PATH}")
else:
    random.seed(RANDOM_STATE)
    record_ids = random.sample(range(len(raw_data)), min(DEMO_SIZE, len(raw_data)))
    _save_json_atomic(record_ids, UIDS_PATH)
    print(f"✅ Sampled {len(record_ids)} questions, saved UIDs → {UIDS_PATH}")

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

def extract_context(rid, mc=4000):      return _build_context(raw_data[rid], True,  mc)
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
        print("  ⚠️  4-bit quantization unavailable — loading in float16")
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

# ── PROMPTS ───────────────────────────────────────────────────────────────
TASK_TYPES = ["lookup","max-lookup","min-lookup","boolean",
              "delta","ratio","average","multi-hop","unknown"]

def p_task_type(question):
    return (f"Classify the operation needed to answer this financial question.\n"
            f"QUESTION: {question}\n"
            f"Choose ONE: {' / '.join(TASK_TYPES)}\n"
            f"TASK_TYPE: <type>")

def p_data_check(question, context):
    return (f"Check if the data needed to answer this question is present.\n"
            f"DATA:\n{context}\nQUESTION: {question}\n"
            f"If found: FOUND: <label>: <value>\n"
            f"At the end write ONE of:\nDATA_SUFFICIENT: yes\nDATA_SUFFICIENT: no")

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

def p_llm(question, context, is_bool=False):
    ans_inst = ("The LAST line must be:\nAnswer: yes\nor\nAnswer: no" if is_bool
                else "The LAST line must be:\nAnswer: <final number only>")
    return (f"You are a financial analyst. Answer using ONLY the data below.\n"
            f"No LaTeX. No symbols. Plain arithmetic only.\n\n"
            f"DATA:\n{context}\n\nQUESTION: {question}\n\n"
            f"Step 1 - Extract needed numbers.\n"
            f"Step 2 - Show full arithmetic step by step.\n"
            f"Step 3 - Verify: correct year? correct column? correct sign?\n\n"
            f"IMPORTANT: {ans_inst}")

# ── MONITORING SIGNALS ────────────────────────────────────────────────────
def compute_s1(raw):
    rl = raw.lower()
    return next((t for t in TASK_TYPES if f"task_type: {t}" in rl), "unknown")

def compute_s2(raw):
    rl = raw.lower()
    if "data_sufficient: yes" in rl: return True
    if "data_sufficient: no"  in rl: return False
    return bool(re.search(r"FOUND:.*\d", raw, re.IGNORECASE))

def compute_s3(question):
    q = question.lower()
    signals = []
    if any(x in q for x in ["in the year with","in the year when","when the highest",
                              "when the lowest","when the largest","when the smallest",
                              "in the period with","during the year that"]):
        signals.append("conditional_lookup")
    if any(x in q for x in ["compared to","in comparison to","relative to"," vs "]):
        if any(x in q for x in ["percent","rate","growth","change","difference"]):
            signals.append("comparison_ratio")
    if any(x in q for x in ["total amount","combined total","sum of","aggregate"]):
        if any(x in q for x in ["from","between","during","over","across"]):
            signals.append("conditional_sum")
    if sum(1 for x in ["percent","portion","rate","ratio","fraction","share"] if x in q) >= 2:
        signals.append("nested_percentage")
    if len([x for x in ["if","when","after","before","since","until"] if x in q]) >= 2:
        signals.append("multi_condition")
    n = len(signals)
    level = 1 if n == 0 else 2 if n == 1 else 3
    return level, level >= S3_HARD_THRESHOLD, signals

def compute_s4(traces_raw):
    count = sum(1 for r in traces_raw
                if len(re.findall(r"-?\d+\.?\d*", r)) >= 2 and "=" in r)
    return round(count / len(traces_raw), 3) if traces_raw else 0.0

def compute_s5(answers):
    if len(answers) <= 1: return 1.0
    pairs = [(a, b) for i, a in enumerate(answers) for b in answers[i+1:]]
    agree = sum(1 for a, b in pairs if answers_agree(str(normalize(a)), str(normalize(b))))
    return round(agree / len(pairs), 3)

def compute_s6(answer, question):
    q = question.lower()
    is_pct_q = any(x in q for x in [
        "what percent","what portion","what share","what fraction",
        "percent of","percentage of","portion of","percentage change",
        "percentage difference","percentage gained","growth rate",
        "change rate","percentage increase","percentage decrease",
    ])
    is_bool_q = _is_boolean(question)
    if not parseable(answer): return False
    if is_bool_q:
        return str(normalize(answer)).lower() in ("yes", "no")
    if is_pct_q:
        try:
            val = normalize(answer)
            if not isinstance(val, float): return False
            return (-1.5 <= val <= 15.0) or (-100 <= val <= 1000)
        except: return False
    return True

def compute_s_operand(traces_raw):
    def get_nums(text):
        result = set()
        for n in re.findall(r"-?[\d,]+\.?\d*", text):
            try:
                v = round(float(n.replace(",", "")), 2)
                if not (1900 <= v <= 2030 and "." not in n): result.add(v)
            except: pass
        return result
    if len(traces_raw) < 2: return 1.0
    sets = [get_nums(r) for r in traces_raw]
    common = sets[0]
    for s in sets[1:]: common = common & s
    total = set()
    for s in sets: total |= s
    return round(len(common) / len(total), 3) if total else 1.0

# ── CONTROL ───────────────────────────────────────────────────────────────
def slm_control(s2, s3_level, s3_hard, s4, s5, s6_ok, majority, s1_task, s_operand):
    if not parseable(majority):
        return "escalate_llm", "unparseable answer", "unparseable"

    # Hard red flags → unconditional escalation
    red_flags = []
    if s5 < 0.01:   red_flags.append("S5=0.0 (no consistency)")
    if not s6_ok:   red_flags.append("S6=False (answer type mismatch)")
    if s3_hard:     red_flags.append(f"S3={s3_level}>={S3_HARD_THRESHOLD} (complex task)")
    if red_flags:
        return "escalate_llm", " | ".join(red_flags), "red_flag"

    # Soft signals → raise acceptance threshold
    effective_thresh = MIN_S5
    soft_adj = []
    if not s2:
        effective_thresh += DELTA_S2
        soft_adj.append(f"S2=False (+{DELTA_S2})")
    if s3_level == 1:
        effective_thresh += DELTA_S3_SOFT
        soft_adj.append(f"S3=1 (+{DELTA_S3_SOFT})")

    if s5 >= effective_thresh:
        note = f" [soft: {'+'.join(soft_adj)}]" if soft_adj else ""
        return "accept", f"S5={s5:.3f}>={effective_thresh:.2f}{note}", None
    else:
        base = "+".join(soft_adj) if soft_adj else "base"
        return ("escalate_llm",
                f"S5={s5:.3f}<{effective_thresh:.2f} (thresh after soft: {base})",
                "soft_threshold")

def llm_control(llm_answer, llm_s5):
    if not parseable(llm_answer):
        return "escalate_human", "LLM: malformed"
    if llm_s5 is not None and llm_s5 < 0.01:
        return "escalate_human", f"LLM: S5={llm_s5:.3f} (uncertain)"
    return "accept", f"LLM ok (S5={llm_s5})"

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
    print(f"\n⚡ Checkpoint found — resuming from {_start_idx}/{len(sample)} ({len(mc_states)} done)")
else:
    print("\n🆕 No checkpoint — starting fresh.")

for q_num, row in sample.iloc[_start_idx:].iterrows():
    if _interrupted:
        print("Saving checkpoint and exiting phase 1...")
        break

    question     = row["question"]
    record_id    = int(row["record_id"])
    ground_truth = str(row["answer"])
    is_bool      = _is_boolean(question)
    ctx_w        = extract_context(record_id)
    ctx_full     = extract_context_full(record_id)

    s1_task = compute_s1(generate(model_small, tok_small, p_task_type(question), max_new_tokens=40))
    s2      = compute_s2(generate(model_small, tok_small, p_data_check(question, ctx_w), max_new_tokens=200))
    s3_level, s3_hard, s3_signals = compute_s3(question)

    traces_raw, traces_ans = [], []
    for i in range(N_TRACES):
        t_raw = generate(model_small, tok_small,
                         p_slm_trace(question, ctx_full, is_bool),
                         max_new_tokens=300, do_sample=(i > 0), temperature=0.7)
        traces_raw.append(t_raw)
        traces_ans.append(extract_answer(t_raw))

    s4        = compute_s4(traces_raw)
    s5        = compute_s5(traces_ans)
    s_operand = compute_s_operand(traces_raw)
    majority  = pick_majority(traces_ans)
    s6_ok     = compute_s6(majority, question)

    slm_dec, slm_reason, esc_type = slm_control(
        s2, s3_level, s3_hard, s4, s5, s6_ok, majority, s1_task, s_operand)
    slm_correct = match(majority, ground_truth)

    mc_states.append({
        "record_id": record_id,
        "question": question, "ground_truth": ground_truth, "is_boolean": is_bool,
        "s1_task": s1_task, "s2": s2,
        "s3_level": s3_level, "s3_hard": s3_hard, "s3_signals": str(s3_signals),
        "s4": s4, "s5": s5, "s6_ok": s6_ok, "s_operand": s_operand,
        "traces_answers": traces_ans, "majority": majority,
        "slm_decision": slm_dec, "slm_reason": slm_reason, "escalation_type": esc_type,
        "slm_answer": majority, "slm_correct": slm_correct,
        "llm_answer": None, "llm_s5": None, "llm_decision": None, "llm_correct": None,
        "final_answer": majority if slm_dec == "accept" else None,
        "final_source": "slm"    if slm_dec == "accept" else None,
        "final_correct": slm_correct if slm_dec == "accept" else None,
        "human_escalation": False,
    })

    n = len(mc_states)
    if n % SAVE_EVERY == 0 or n == 1:
        acc   = sum(1 for s in mc_states if s["slm_correct"]) / n
        n_acc = sum(1 for s in mc_states if s["slm_decision"] == "accept")
        n_esc = sum(1 for s in mc_states if s["slm_decision"] == "escalate_llm")
        n_red = sum(1 for s in mc_states if s["escalation_type"] == "red_flag")
        print(f"  [{n:4d}/{len(sample)}] slm={acc:.1%}  accept={n_acc}  to_llm={n_esc}  red_flags={n_red}")
        _save_json_atomic({"mc_states": mc_states, "next_idx": _start_idx + n}, CHECKPOINT_PATH)
        print(f"     💾 Checkpoint saved ({n}/{len(sample)})")

unload_model(model_small); del tok_small

if _interrupted:
    print("\n✅ Checkpoint saved. Run the script again to continue.")
    sys.exit(0)

if CHECKPOINT_PATH.exists():
    CHECKPOINT_PATH.unlink()
    print("✅ Phase 1 complete — checkpoint deleted.")

# ══════════════════════════════════════════════════════════════════════════
# PHASE 2: LARGE MODEL
# ══════════════════════════════════════════════════════════════════════════
to_llm = [s for s in mc_states if s["slm_decision"] == "escalate_llm"]
print(f"\nSLM: accept={sum(1 for s in mc_states if s['slm_decision']=='accept')}  →llm={len(to_llm)}")
print(f"  Red flags: {sum(1 for s in to_llm if s['escalation_type']=='red_flag')} "
      f"| Soft threshold: {sum(1 for s in to_llm if s['escalation_type']=='soft_threshold')} "
      f"| Unparseable: {sum(1 for s in to_llm if s['escalation_type']=='unparseable')}")

if to_llm:
    _llm_start    = 0
    _llm_done_log = []

    ckpt2 = _load_json_safe(CHECKPOINT2_PATH)
    if ckpt2:
        try:
            for _saved in ckpt2["llm_done"]:
                mc_states[_saved["_mc_idx"]].update(
                    {k: v for k, v in _saved.items() if k != "_mc_idx"})
            _llm_done_log = ckpt2["llm_done"]
            _llm_start    = len(_llm_done_log)
            print(f"⚡ LLM checkpoint: resuming from {_llm_start}/{len(to_llm)}")
        except (KeyError, IndexError):
            print("⚠️  LLM checkpoint corrupted — starting LLM phase fresh.")

    model_large, tok_large = load_model(LARGE_MODEL)

    for i, state in enumerate(to_llm[_llm_start:], start=_llm_start):
        if _interrupted:
            print("Saving LLM checkpoint and exiting...")
            break

        ctx_full     = extract_context_full(state["record_id"])
        llm_ans_list = []
        for j in range(N_TRACES):
            raw_j = generate(model_large, tok_large,
                             p_llm(state["question"], ctx_full, state["is_boolean"]),
                             max_new_tokens=450, do_sample=(j > 0), temperature=0.7)
            llm_ans_list.append(extract_answer(raw_j))

        llm_s5      = compute_s5(llm_ans_list)
        llm_answer  = pick_majority(llm_ans_list)
        llm_correct = match(llm_answer, state["ground_truth"])
        llm_dec, _  = llm_control(llm_answer, llm_s5)

        state.update({"llm_answer": llm_answer, "llm_s5": llm_s5,
                      "llm_decision": llm_dec, "llm_correct": llm_correct})
        if llm_dec == "accept":
            state.update({"final_answer": llm_answer, "final_source": "llm",
                          "final_correct": llm_correct, "human_escalation": False})
        else:
            state.update({"final_answer": None, "final_source": "human",
                          "final_correct": None, "human_escalation": True})

        _mc_idx = mc_states.index(state)
        _llm_done_log.append({
            "_mc_idx":        _mc_idx,
            "llm_answer":     state["llm_answer"],  "llm_s5":           state["llm_s5"],
            "llm_decision":   state["llm_decision"], "llm_correct":      state["llm_correct"],
            "final_answer":   state["final_answer"], "final_source":     state["final_source"],
            "final_correct":  state["final_correct"], "human_escalation": state["human_escalation"],
        })

        if (i + 1) % 10 == 0 or i == _llm_start:
            ok   = sum(1 for s in to_llm[:i+1] if s["llm_correct"])
            n_hm = sum(1 for s in to_llm[:i+1] if s["human_escalation"])
            print(f"  [LLM {i+1}/{len(to_llm)}] correct={ok/(i+1):.1%}  →human={n_hm}")
            _save_json_atomic({"llm_done": _llm_done_log}, CHECKPOINT2_PATH)
            print(f"     💾 LLM checkpoint saved ({i+1}/{len(to_llm)})")

    unload_model(model_large); del tok_large

    if _interrupted:
        print("\n✅ LLM checkpoint saved. Run the script again to continue.")
        sys.exit(0)

    if CHECKPOINT2_PATH.exists():
        CHECKPOINT2_PATH.unlink()
        print("✅ Phase 2 complete — LLM checkpoint deleted.")

# ══════════════════════════════════════════════════════════════════════════
# ANALYSIS AND METRICS
# ══════════════════════════════════════════════════════════════════════════
N   = len(mc_states)
SEP = "=" * 70

n_acc    = sum(1 for s in mc_states if s["slm_decision"] == "accept")
n_esc    = sum(1 for s in mc_states if s["slm_decision"] == "escalate_llm")
n_red    = sum(1 for s in mc_states if s["escalation_type"] == "red_flag")
n_soft   = sum(1 for s in mc_states if s["escalation_type"] == "soft_threshold")
n_unpars = sum(1 for s in mc_states if s["escalation_type"] == "unparseable")
n_llm_ok = sum(1 for s in mc_states if s.get("llm_decision") == "accept")
n_human  = sum(1 for s in mc_states if s["human_escalation"])
n_ans    = N - n_human

slm_overall = sum(1 for s in mc_states if s["slm_correct"]) / N
slm_prec    = sum(1 for s in mc_states if s["slm_decision"]=="accept" and s["slm_correct"]) / max(1, n_acc)
llm_prec    = sum(1 for s in mc_states if s.get("llm_decision")=="accept" and s["llm_correct"]) / max(1, n_llm_ok)
fin_corr    = sum(1 for s in mc_states if s["final_correct"])

print()
print(SEP)
print("HIERARCHICAL PIPELINE (FinQA) — SUMMARY")
print(SEP)
print(f"""
Escalation flow:
  Total                      : {N}
  SLM → ACCEPT               : {n_acc}  ({n_acc/N:.1%})
  SLM → LLM                  : {n_esc}  ({n_esc/N:.1%})
    ├─ Red flag (S5=0/S6/S3) : {n_red}  ({n_red/max(1,n_esc):.1%} of escalated)
    ├─ Soft threshold crossed : {n_soft}  ({n_soft/max(1,n_esc):.1%} of escalated)
    └─ Unparseable            : {n_unpars}  ({n_unpars/max(1,n_esc):.1%} of escalated)
  LLM → ACCEPT               : {n_llm_ok}  ({n_llm_ok/max(1,n_esc):.1%} of sent)
  LLM → HUMAN                : {n_human}  ({n_human/max(1,n_esc):.1%} of sent)

Accuracy:
  SLM overall (no routing)   : {slm_overall:.1%}
  SLM accepted precision     : {slm_prec:.1%}  (n={n_acc})
  LLM accepted precision     : {llm_prec:.1%}  (n={n_llm_ok})
  FINAL (excl. human)        : {fin_corr}/{n_ans} = {fin_corr/max(1,n_ans):.1%}
  FINAL overall              : {fin_corr}/{N} = {fin_corr/N:.1%}
  Coverage                   : {n_ans/N:.1%}
  Human escalation rate      : {n_human/N:.1%}
""")

# ── SAVE CSV ──────────────────────────────────────────────────────────────
rows = []
for s in mc_states:
    rows.append({
        "record_id": s["record_id"],
        "question": s["question"], "ground_truth": s["ground_truth"],
        "is_boolean": s["is_boolean"],
        "s1_task": s["s1_task"], "s2": s["s2"],
        "s3_level": s["s3_level"], "s3_hard": s["s3_hard"], "s3_signals": s["s3_signals"],
        "s4": s["s4"], "s5": s["s5"], "s6_ok": s["s6_ok"], "s_operand": s["s_operand"],
        "majority": s["majority"],
        "slm_decision": s["slm_decision"], "slm_reason": s["slm_reason"],
        "escalation_type": s["escalation_type"],
        "slm_answer": s["slm_answer"], "slm_correct": s["slm_correct"],
        "llm_answer": s["llm_answer"], "llm_s5": s["llm_s5"],
        "llm_decision": s["llm_decision"], "llm_correct": s["llm_correct"],
        "final_answer": s["final_answer"], "final_source": s["final_source"],
        "final_correct": s["final_correct"], "human_escalation": s["human_escalation"],
        "n_traces": N_TRACES, "version": "finqa_hierarch_1000",
    })

pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
print(f"\nSaved to {OUT_PATH}")
print(f"UIDs file → {UIDS_PATH}  ← shared with finqa_slm_1000.py and finqa_llm_1000.py")
