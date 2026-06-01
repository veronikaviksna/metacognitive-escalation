# finqa_ablation_full_vscode.py — Full Ablation Study C1-C8
# Runs all 8 configurations sequentially.
# Each config has its own checkpoint — safe to interrupt and resume.
#
# Usage:
#   nohup python -u finqa_ablation_full_vscode.py > ablation_full.log 2>&1 &
#   tail -f ablation_full.log

import os, re, gc, json, time, random, signal, sys
from collections import Counter
from pathlib import Path
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── SETTINGS ──────────────────────────────────────────────────────────────
SMALL_MODEL = "Qwen/Qwen2.5-3B-Instruct"
LARGE_MODEL = "Qwen/Qwen2.5-14B-Instruct"
FINQA_JSON  = Path("/workspace/train.json")
OUT_DIR     = Path("/workspace")
UIDS_PATH   = OUT_DIR / "finqa_ablation_uids.json"

RANDOM_STATE      = 42
N_TRACES          = 3
SAVE_EVERY        = 10
MIN_S5            = 0.1
S3_HARD_THRESHOLD = 2
DELTA_S2          = 0.2
DELTA_S3_SOFT     = 0.1

# ── GRACEFUL INTERRUPT ────────────────────────────────────────────────────
_interrupted = False

def _handle_sigint(sig, frame):
    global _interrupted
    print("\n\nInterrupted — saving checkpoint after current question.")
    _interrupted = True

signal.signal(signal.SIGINT,  _handle_sigint)
signal.signal(signal.SIGTERM, _handle_sigint)

# ── HELPERS ───────────────────────────────────────────────────────────────
def save_json(data, path):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f)
    tmp.replace(path)

def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return None

# ── DATA ──────────────────────────────────────────────────────────────────
print("Loading FinQA data...")
with open(FINQA_JSON, "r", encoding="utf-8") as f:
    raw_data = json.load(f)
print(f"Loaded {len(raw_data)} records")

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
def _make_bnb():
    try:
        from bitsandbytes import __version__ as v
        ver = tuple(int(x) for x in v.split(".")[:3])
        if ver >= (0, 41, 0):
            return BitsAndBytesConfig(load_in_4bit=True,
                                      bnb_4bit_compute_dtype=torch.float16,
                                      bnb_4bit_quant_type="nf4",
                                      bnb_4bit_use_double_quant=True)
    except: pass
    return None

def load_model(name):
    print(f"Loading {name}...")
    tok = AutoTokenizer.from_pretrained(name)
    bnb = _make_bnb()
    if bnb:
        mdl = AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb, device_map="auto")
        print("  4-bit quantization OK")
    else:
        mdl = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="auto")
        print("  float16")
    free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9
    print(f"  VRAM free: {free:.1f} GB")
    return mdl, tok

def unload_model(mdl):
    mdl.cpu(); del mdl; gc.collect()
    torch.cuda.empty_cache(); time.sleep(1)

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
    clean = re.sub(r"\[a-zA-Z]+\{[^}]*\}", "", clean)
    clean = re.sub(r"\[a-zA-Z]+", "", clean)
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

def p_task_type(q):
    return (f"Classify the operation needed to answer this financial question.\n"
            f"QUESTION: {q}\n"
            f"Choose ONE: {' / '.join(TASK_TYPES)}\n"
            f"TASK_TYPE: <type>")

def p_data_check(q, ctx):
    return (f"Check if the data needed to answer this question is present.\n"
            f"DATA:\n{ctx}\nQUESTION: {q}\n"
            f"If found: FOUND: <label>: <value>\n"
            f"At the end write ONE of:\nDATA_SUFFICIENT: yes\nDATA_SUFFICIENT: no")

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
        "change rate","percentage increase","percentage decrease"])
    is_bool_q = _is_boolean(question)
    if not parseable(answer): return False
    if is_bool_q: return str(normalize(answer)).lower() in ("yes", "no")
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

# ── LLM CONTROL ───────────────────────────────────────────────────────────
def llm_control(llm_answer, llm_s5):
    if not parseable(llm_answer):
        return "escalate_human", "LLM: malformed"
    if llm_s5 is not None and llm_s5 < 0.01:
        return "escalate_human", f"LLM: S5={llm_s5:.3f}"
    return "accept", f"LLM ok (S5={llm_s5})"

# ── REPORT ────────────────────────────────────────────────────────────────
def print_report(config_name, df, n_total):
    SEP = "=" * 60
    print(f"\n{SEP}")
    print(f"CONFIG: {config_name}  (n={n_total})")
    print(SEP)

    n_acc     = int(df["slm_decision"].eq("accept").sum())
    n_esc     = int(df["slm_decision"].eq("escalate_llm").sum())
    n_rf      = int(df["escalation_type"].eq("red_flag").sum())
    n_human   = int(df["human_escalation"].fillna(False).sum())
    n_llm_acc = int(df["llm_decision"].eq("accept").sum())
    n_ans     = n_total - n_human

    print("\nROUTING FLOW")
    print(f"  Accepted by SLM        : {n_acc:4d}  ({n_acc/n_total:.1%})")
    print(f"  Escalated to LLM       : {n_esc:4d}  ({n_esc/n_total:.1%})")
    print(f"  - via red flag         : {n_rf:4d}  ({n_rf/max(1,n_esc):.1%} of escalated)")
    print(f"  LLM accepted           : {n_llm_acc:4d}  ({n_llm_acc/max(1,n_esc):.1%} of escalated)")
    print(f"  Escalated to human     : {n_human:4d}  ({n_human/max(1,n_esc):.1%} of escalated)")
    print(f"  Coverage               : {n_ans}/{n_total}  ({n_ans/n_total:.1%})")

    slm_all  = df["slm_correct"].mean()
    slm_prec = df[df["slm_decision"]=="accept"]["slm_correct"].mean() if n_acc else float("nan")
    llm_prec = df[df["llm_decision"]=="accept"]["llm_correct"].mean() if n_llm_acc else float("nan")
    fin_corr = df["final_correct"].sum()
    fin_acc  = fin_corr / n_ans if n_ans else float("nan")
    fin_all  = fin_corr / n_total

    print("\nACCURACY")
    print(f"  SLM overall            : {slm_all:.1%}")
    print(f"  SLM accepted precision : {slm_prec:.1%}  (n={n_acc})")
    print(f"  LLM accepted precision : {llm_prec:.1%}  (n={n_llm_acc})")
    print(f"  Final (excl. human)    : {fin_corr}/{n_ans} = {fin_acc:.1%}")
    print(f"  Final overall          : {fin_corr}/{n_total} = {fin_all:.1%}")

    # TP/FP/FN/TN — SLM escalation decision vs SLM correctness
    acc_rows = df[df["slm_decision"]=="accept"]
    esc_rows = df[df["slm_decision"]=="escalate_llm"]
    TP = int((esc_rows["slm_correct"] == False).sum())
    FP = int((esc_rows["slm_correct"] == True).sum())
    FN = int((acc_rows["slm_correct"] == False).sum())
    TN = int((acc_rows["slm_correct"] == True).sum())
    prec = TP / max(1, TP+FP)
    rec  = TP / max(1, TP+FN)
    f1   = 2*prec*rec / max(0.001, prec+rec)

    print("\nROUTING QUALITY")
    print(f"  TP (wrong -> escalated)          : {TP}")
    print(f"  FP (correct -> escalated)        : {FP}")
    print(f"  FN (wrong -> accepted)           : {FN}")
    print(f"  TN (correct -> accepted)         : {TN}")
    print(f"  Escalation precision             : {prec:.1%}")
    print(f"  Escalation recall                : {rec:.1%}")
    print(f"  Escalation F1                    : {f1:.3f}")

    print("\nSIGNAL CORRELATIONS (Pearson r with SLM correctness)")
    for col, label in [("s5","S5"),("s2","S2"),("s3_level","S3"),
                       ("s4","S4"),("s_operand","S_operand")]:
        if col in df.columns:
            vals = df[col].astype(float)
            try:
                r, p = pearsonr(vals, df["slm_correct"].astype(float))
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"  {label:<12}: r={r:+.3f}  p={p:.3f} {sig}")
            except: pass

    print(f"\nSaved -> {OUT_DIR / (config_name + '.csv')}")
    print(SEP)

# ── MAIN RUN FUNCTION ─────────────────────────────────────────────────────
def run_config(config_name, slm_control_fn, sample, model_s, tok_s, model_l, tok_l):
    global _interrupted
    out_path = OUT_DIR / f"{config_name}.csv"
    ckpt1    = OUT_DIR / f"ckpt1_{config_name}.json"
    ckpt2    = OUT_DIR / f"ckpt2_{config_name}.json"

    # ── PHASE 1: SLM ──────────────────────────────────────────────────────
    mc_states = []
    start_idx = 0
    ck = load_json(ckpt1)
    if ck:
        mc_states = ck["mc_states"]; start_idx = ck["next_idx"]
        print(f"Resuming {config_name} SLM from {start_idx}/{len(sample)}")
    else:
        print(f"\nStarting {config_name} SLM phase fresh")

    for _, row in sample.iloc[start_idx:].iterrows():
        if _interrupted:
            save_json({"mc_states": mc_states, "next_idx": start_idx + len(mc_states)}, ckpt1)
            print(f"Checkpoint saved. Run again to resume {config_name}.")
            sys.exit(0)

        question     = row["question"]
        record_id    = int(row["record_id"])
        ground_truth = str(row["answer"])
        is_bool      = _is_boolean(question)
        ctx_w        = get_context(record_id)
        ctx_full     = get_context_full(record_id)

        s1_task = compute_s1(generate(model_s, tok_s, p_task_type(question), max_new_tokens=40))
        s2      = compute_s2(generate(model_s, tok_s, p_data_check(question, ctx_w), max_new_tokens=200))
        s3_level, s3_hard, s3_signals = compute_s3(question)

        traces_raw, traces_ans = [], []
        for i in range(N_TRACES):
            t = generate(model_s, tok_s, p_slm_trace(question, ctx_full, is_bool),
                         max_new_tokens=300, do_sample=(i>0), temperature=0.7)
            traces_raw.append(t); traces_ans.append(extract_answer(t))

        s4        = compute_s4(traces_raw)
        s5        = compute_s5(traces_ans)
        s_operand = compute_s_operand(traces_raw)
        majority  = pick_majority(traces_ans)
        s6_ok     = compute_s6(majority, question)

        slm_dec, slm_reason, esc_type = slm_control_fn(s2, s3_level, s3_hard, s5, s6_ok, majority)
        slm_correct = match(majority, ground_truth)

        mc_states.append({
            "record_id": record_id, "question": question,
            "ground_truth": ground_truth, "is_boolean": is_bool,
            "s1_task": s1_task, "s2": s2,
            "s3_level": s3_level, "s3_hard": s3_hard,
            "s4": s4, "s5": s5, "s6_ok": s6_ok, "s_operand": s_operand,
            "majority": majority,
            "slm_decision": slm_dec, "slm_reason": slm_reason,
            "escalation_type": esc_type,
            "slm_correct": slm_correct,
            "llm_answer": None, "llm_s5": None,
            "llm_decision": None, "llm_correct": None,
            "final_answer": majority if slm_dec=="accept" else None,
            "final_correct": slm_correct if slm_dec=="accept" else None,
            "human_escalation": False,
        })

        n = len(mc_states)
        if n % SAVE_EVERY == 0 or n == 1:
            acc = sum(s["slm_correct"] for s in mc_states) / n
            n_e = sum(1 for s in mc_states if s["slm_decision"]=="escalate_llm")
            print(f"  SLM [{n:3d}/{len(sample)}] acc={acc:.1%}  escalated={n_e}")
            save_json({"mc_states": mc_states, "next_idx": start_idx+n}, ckpt1)

    if ckpt1.exists(): ckpt1.unlink()
    print(f"SLM phase done. Escalated: {sum(1 for s in mc_states if s['slm_decision']=='escalate_llm')}")

    # ── PHASE 2: LLM ──────────────────────────────────────────────────────
    to_llm    = [s for s in mc_states if s["slm_decision"]=="escalate_llm"]
    llm_done  = []
    llm_start = 0
    ck2 = load_json(ckpt2)
    if ck2:
        for saved in ck2["llm_done"]:
            idx = saved["_mc_idx"]
            mc_states[idx].update({k:v for k,v in saved.items() if k!="_mc_idx"})
        llm_done = ck2["llm_done"]; llm_start = len(llm_done)
        print(f"Resuming LLM phase from {llm_start}/{len(to_llm)}")
    else:
        print(f"Starting LLM phase ({len(to_llm)} questions)")

    for i, state in enumerate(to_llm[llm_start:], start=llm_start):
        if _interrupted:
            save_json({"llm_done": llm_done}, ckpt2)
            print(f"LLM checkpoint saved. Run again to resume {config_name}.")
            sys.exit(0)

        ctx_full     = get_context_full(state["record_id"])
        llm_ans_list = []
        for j in range(N_TRACES):
            r = generate(model_l, tok_l, p_llm(state["question"], ctx_full, state["is_boolean"]),
                         max_new_tokens=450, do_sample=(j>0), temperature=0.7)
            llm_ans_list.append(extract_answer(r))

        llm_s5      = compute_s5(llm_ans_list)
        llm_answer  = pick_majority(llm_ans_list)
        llm_correct = match(llm_answer, state["ground_truth"])
        llm_dec, _  = llm_control(llm_answer, llm_s5)

        state.update({"llm_answer": llm_answer, "llm_s5": llm_s5,
                      "llm_decision": llm_dec, "llm_correct": llm_correct})
        if llm_dec == "accept":
            state.update({"final_answer": llm_answer, "final_correct": llm_correct,
                          "human_escalation": False})
        else:
            state.update({"final_answer": None, "final_correct": None,
                          "human_escalation": True})

        mc_idx = mc_states.index(state)
        llm_done.append({"_mc_idx": mc_idx, "llm_answer": llm_answer,
                         "llm_s5": llm_s5, "llm_decision": llm_dec,
                         "llm_correct": llm_correct,
                         "final_answer": state["final_answer"],
                         "final_correct": state["final_correct"],
                         "human_escalation": state["human_escalation"]})

        if (i+1) % SAVE_EVERY == 0 or i == llm_start:
            ok = sum(1 for s in to_llm[:i+1] if s.get("llm_correct"))
            print(f"  LLM [{i+1:3d}/{len(to_llm)}] correct={ok/(i+1):.1%}")
            save_json({"llm_done": llm_done}, ckpt2)

    if ckpt2.exists(): ckpt2.unlink()

    df = pd.DataFrame(mc_states)
    df.to_csv(out_path, index=False)
    print_report(config_name, df, len(sample))
    return df

print("\nAll utilities loaded")

# ── CONFIG: UIDs ──────────────────────────────────────────────────────────
# Requires finqa_ablation_uids.json to already exist in /workspace

if UIDS_PATH.exists():
    record_ids_abl = load_json(UIDS_PATH)
    print(f"UID file loaded — {len(record_ids_abl)} questions from {UIDS_PATH}")
else:
    raise FileNotFoundError(
        f"UID file not found at {UIDS_PATH}. "
        "Upload finqa_ablation_uids.json to /workspace/ first.")

sample_abl = pd.DataFrame([{
    "record_id": idx,
    "question":  raw_data[idx].get("qa", {}).get("question", ""),
    "answer":    str(raw_data[idx].get("qa", {}).get("answer", "")),
} for idx in record_ids_abl]).reset_index(drop=True)

print(f"Sample ready: {len(sample_abl)} questions")

# ── LOAD MODELS ───────────────────────────────────────────────────────────
# Run once; reuse model_s / model_l across all configurations

model_s, tok_s = load_model(SMALL_MODEL)
model_l, tok_l = load_model(LARGE_MODEL)
print("Both models loaded and ready")

# ── C1: S5 only ───────────────────────────────────────────────────────────
# Escalate if S5=0 only

def slm_control_c1_s5only(s2, s3_level, s3_hard, s5, s6_ok, majority):
    if not parseable(majority):
        return "escalate_llm", "unparseable", "red_flag"
    if s5 < 0.01:
        return "escalate_llm", "S5=0", "red_flag"
    return "accept", f"S5={s5:.3f}", None

df_c1_s5only = run_config(
    config_name    = "C1_S5only",
    slm_control_fn = slm_control_c1_s5only,
    sample         = sample_abl,
    model_s=model_s, tok_s=tok_s,
    model_l=model_l, tok_l=tok_l,
)

# ── C2: S3 only ───────────────────────────────────────────────────────────
# Escalate if S3>=2 only

def slm_control_c2_s3only(s2, s3_level, s3_hard, s5, s6_ok, majority):
    if not parseable(majority):
        return "escalate_llm", "unparseable", "red_flag"
    if s3_hard:
        return "escalate_llm", f"S3=L{s3_level}", "red_flag"
    return "accept", f"S3=L{s3_level}", None

df_c2_s3only = run_config(
    config_name    = "C2_S3only",
    slm_control_fn = slm_control_c2_s3only,
    sample         = sample_abl,
    model_s=model_s, tok_s=tok_s,
    model_l=model_l, tok_l=tok_l,
)
if _interrupted: sys.exit(0)

# ── C3: S6 only ───────────────────────────────────────────────────────────
# Escalate if S6=False only

def slm_control_c3_s6only(s2, s3_level, s3_hard, s5, s6_ok, majority):
    if not parseable(majority):
        return "escalate_llm", "unparseable", "red_flag"
    if not s6_ok:
        return "escalate_llm", "S6=False", "red_flag"
    return "accept", "S6=ok", None

df_c3_s6only = run_config(
    config_name    = "C3_S6only",
    slm_control_fn = slm_control_c3_s6only,
    sample         = sample_abl,
    model_s=model_s, tok_s=tok_s,
    model_l=model_l, tok_l=tok_l,
)
if _interrupted: sys.exit(0)

# ── C4: S3 + S6 ───────────────────────────────────────────────────────────
# Escalate if S3>=2 or S6=False (no S5)

def slm_control_c4_s3_s6(s2, s3_level, s3_hard, s5, s6_ok, majority):
    if not parseable(majority):
        return "escalate_llm", "unparseable", "red_flag"
    if s3_hard:
        return "escalate_llm", f"S3=L{s3_level}", "red_flag"
    if not s6_ok:
        return "escalate_llm", "S6=False", "red_flag"
    return "accept", "S3+S6 ok", None

df_c4_s3_s6 = run_config(
    config_name    = "C4_S3_S6",
    slm_control_fn = slm_control_c4_s3_s6,
    sample         = sample_abl,
    model_s=model_s, tok_s=tok_s,
    model_l=model_l, tok_l=tok_l,
)
if _interrupted: sys.exit(0)

# ── C5: S5 + S6 ───────────────────────────────────────────────────────────
# Escalate if S5=0 or S6=False

def slm_control_c5_s5_s6(s2, s3_level, s3_hard, s5, s6_ok, majority):
    if not parseable(majority):
        return "escalate_llm", "unparseable", "red_flag"
    if s5 < 0.01:
        return "escalate_llm", "S5=0", "red_flag"
    if not s6_ok:
        return "escalate_llm", "S6=False", "red_flag"
    return "accept", f"S5={s5:.3f} S6=ok", None

df_c5_s5_s6 = run_config(
    config_name    = "C5_S5_S6",
    slm_control_fn = slm_control_c5_s5_s6,
    sample         = sample_abl,
    model_s=model_s, tok_s=tok_s,
    model_l=model_l, tok_l=tok_l,
)
if _interrupted: sys.exit(0)

# ── C6: S5 + S3 ───────────────────────────────────────────────────────────
# Escalate if S5=0 or S3>=2

def slm_control_c6_s5_s3(s2, s3_level, s3_hard, s5, s6_ok, majority):
    if not parseable(majority):
        return "escalate_llm", "unparseable", "red_flag"
    if s5 < 0.01:
        return "escalate_llm", "S5=0", "red_flag"
    if s3_hard:
        return "escalate_llm", f"S3=L{s3_level}", "red_flag"
    return "accept", f"S5={s5:.3f} S3=L{s3_level}", None

df_c6_s5_s3 = run_config(
    config_name    = "C6_S5_S3",
    slm_control_fn = slm_control_c6_s5_s3,
    sample         = sample_abl,
    model_s=model_s, tok_s=tok_s,
    model_l=model_l, tok_l=tok_l,
)
if _interrupted: sys.exit(0)

# ── C7: S5 + S6 + S3 ──────────────────────────────────────────────────────
# Escalate if S5=0 or S6=False or S3>=2

def slm_control_c7_s5_s6_s3(s2, s3_level, s3_hard, s5, s6_ok, majority):
    if not parseable(majority):
        return "escalate_llm", "unparseable", "red_flag"
    if s5 < 0.01:
        return "escalate_llm", "S5=0", "red_flag"
    if not s6_ok:
        return "escalate_llm", "S6=False", "red_flag"
    if s3_hard:
        return "escalate_llm", f"S3=L{s3_level}", "red_flag"
    return "accept", f"S5={s5:.3f} S6=ok S3=L{s3_level}", None

df_c7_s5_s6_s3 = run_config(
    config_name    = "C7_S5_S6_S3",
    slm_control_fn = slm_control_c7_s5_s6_s3,
    sample         = sample_abl,
    model_s=model_s, tok_s=tok_s,
    model_l=model_l, tok_l=tok_l,
)
if _interrupted: sys.exit(0)

# ── C8: Majority vote ─────────────────────────────────────────────────────
# Escalate if >=3 of {S5=0, S6=False, S3>=2, S2=False, S3=1} fire

def slm_control_c8_majvote(s2, s3_level, s3_hard, s5, s6_ok, majority):
    if not parseable(majority):
        return "escalate_llm", "unparseable", "red_flag"
    flags = [s5 < 0.01, not s6_ok, s3_hard, not s2, s3_level == 1]
    n_flags = sum(flags)
    if n_flags >= 3:
        fired = [l for l, f in zip(["S5=0","S6=F","S3hard","S2=F","S3=1"], flags) if f]
        return "escalate_llm", "+".join(fired), "red_flag"
    return "accept", f"majority={n_flags}/5 flags", None

df_c8_majvote = run_config(
    config_name    = "C8_MajVote",
    slm_control_fn = slm_control_c8_majvote,
    sample         = sample_abl,
    model_s=model_s, tok_s=tok_s,
    model_l=model_l, tok_l=tok_l,
)

# ── SUMMARY TABLE ─────────────────────────────────────────────────────────
# Reads all finished CSV files from /workspace

config_files = {
    "C1: S5 only":       "C1_S5only.csv",
    "C2: S3 only":       "C2_S3only.csv",
    "C3: S6 only":       "C3_S6only.csv",
    "C4: S3+S6":         "C4_S3_S6.csv",
    "C5: S5+S6":         "C5_S5_S6.csv",
    "C6: S5+S3":         "C6_S5_S3.csv",
    "C7: S5+S6+S3":      "C7_S5_S6_S3.csv",
    "C8: Majority vote": "C8_MajVote.csv",
}

rows = []
for name, fname in config_files.items():
    path = OUT_DIR / fname
    if not path.exists():
        print(f"  {name} — not found, skipping")
        continue
    df = pd.read_csv(path)
    n  = len(df)
    n_acc     = int(df["slm_decision"].eq("accept").sum())
    n_esc     = int(df["slm_decision"].eq("escalate_llm").sum())
    n_human   = int(df["human_escalation"].fillna(False).sum())
    n_ans     = n - n_human
    n_llm_acc = int(df["llm_decision"].eq("accept").sum())

    slm_all  = df["slm_correct"].mean()
    slm_prec = df[df["slm_decision"]=="accept"]["slm_correct"].mean() if n_acc else float("nan")
    llm_prec = df[df["llm_decision"]=="accept"]["llm_correct"].mean() if n_llm_acc else float("nan")
    fin_corr = df["final_correct"].sum()
    fin_acc  = fin_corr / n_ans if n_ans else float("nan")

    esc_rows = df[df["slm_decision"]=="escalate_llm"]
    acc_rows = df[df["slm_decision"]=="accept"]
    TP = int((esc_rows["slm_correct"]==False).sum())
    FP = int((esc_rows["slm_correct"]==True).sum())
    FN = int((acc_rows["slm_correct"]==False).sum())
    prec = TP / max(1, TP+FP)
    rec  = TP / max(1, TP+FN)
    f1   = 2*prec*rec / max(0.001, prec+rec)

    rows.append({
        "Config":       name,
        "Esc_rate":     f"{n_esc/n:.1%}",
        "SLM_overall":  f"{slm_all:.1%}",
        "SLM_acc_prec": f"{slm_prec:.1%}",
        "LLM_acc_prec": f"{llm_prec:.1%}",
        "Final_acc":    f"{fin_acc:.1%}",
        "Esc_prec":     f"{prec:.1%}",
        "Esc_recall":   f"{rec:.1%}",
        "F1":           f"{f1:.3f}",
        "Human_esc":    n_human,
    })

summary = pd.DataFrame(rows)
print("=" * 90)
print("ABLATION STUDY SUMMARY — FinQA n=200")
print("=" * 90)
print(summary.to_string(index=False))
summary.to_csv(OUT_DIR / "ablation_summary.csv", index=False)
print(f"\nSaved -> {OUT_DIR / 'ablation_summary.csv'}")
