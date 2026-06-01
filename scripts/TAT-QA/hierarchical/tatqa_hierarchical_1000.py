# tatqa_hierarchical_1000.py — TAT-QA Metacognitive Hierarchical Pipeline
# Scheme: SLM -> ACCEPT | ESCALATE_LLM -> ACCEPT | ESCALATE_HUMAN
#
# Signal hierarchy (Flavell 1979):
#   RED FLAGS : S5=0, S6 mismatch, S3 >= S3_HARD_THRESHOLD  -> escalate_llm
#   SOFT      : S2=False, S3=1, S4<0.5                      -> raise MIN_S5
#   TRACKING  : S1, S_operand

import os, re, gc, json, time, random, signal, sys
from collections import Counter
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── SETTINGS ──────────────────────────────────────────────────────────────
SMALL_MODEL = "Qwen/Qwen2.5-3B-Instruct"
LARGE_MODEL = "Qwen/Qwen2.5-14B-Instruct"

BASE_DIR         = Path(__file__).parent
TATQA_JSON       = BASE_DIR / "tatqa_dataset_test_gold.json"
OUT_PATH         = BASE_DIR / "tatqa_metacog_results.csv"
CHECKPOINT_PATH  = BASE_DIR / "tatqa_metacog_checkpoint.json"
CHECKPOINT2_PATH = BASE_DIR / "tatqa_metacog_checkpoint2.json"
UIDS_PATH        = BASE_DIR / "tatqa_sample_uids.json"

SAVE_EVERY   = 10
DEMO_SIZE    = 1000
RANDOM_STATE = 42
N_TRACES     = 3

# ── CONTROL THRESHOLDS ────────────────────────────────────────────────────
MIN_S5            = 0.1
S3_HARD_THRESHOLD = 2     # hard flag: S3 level >= 2
DELTA_S2          = 0.2   # soft: S2=False raises threshold
DELTA_S3_SOFT     = 0.1   # soft: S3=1 raises threshold
DELTA_S4          = 0.1   # soft: S4<0.5 raises threshold

# ── GRACEFUL INTERRUPT ────────────────────────────────────────────────────
_interrupted = False

def _handle_sigint(sig, frame):
    global _interrupted
    print("\n\nInterrupted — will save checkpoint and exit after current question.")
    _interrupted = True

signal.signal(signal.SIGINT,  _handle_sigint)
signal.signal(signal.SIGTERM, _handle_sigint)

# ── CHECKPOINT HELPERS ────────────────────────────────────────────────────
def _save_json_atomic(data: dict, path: Path):
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
print("Loading TAT-QA data...")
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

print(f"Loaded {len(raw_data)} TAT-QA documents, {len(all_questions)} questions total")

# ── SAMPLING (UID-based for reproducibility) ──────────────────────────────
if UIDS_PATH.exists():
    with open(UIDS_PATH) as f:
        loaded_uids = json.load(f)
    if len(loaded_uids) < DEMO_SIZE:
        print(f"UID file has only {len(loaded_uids)} entries (need {DEMO_SIZE}) — resampling...")
        UIDS_PATH.unlink()
        random.seed(RANDOM_STATE)
        idxs   = random.sample(range(len(all_questions)), min(DEMO_SIZE, len(all_questions)))
        sample = [all_questions[i] for i in idxs]
        with open(UIDS_PATH, "w") as f:
            json.dump([q["uid"] for q in sample], f, indent=2)
        print(f"Resampled {len(sample)} questions, saved UIDs -> {UIDS_PATH}")
    else:
        target_uids = set(loaded_uids)
        sample = [q for q in all_questions if q["uid"] in target_uids]
        print(f"Loaded existing UID list — {len(sample)} questions from {UIDS_PATH}")
else:
    random.seed(RANDOM_STATE)
    idxs   = random.sample(range(len(all_questions)), min(DEMO_SIZE, len(all_questions)))
    sample = [all_questions[i] for i in idxs]
    with open(UIDS_PATH, "w") as f:
        json.dump([q["uid"] for q in sample], f, indent=2)
    print(f"Sampled {len(sample)} questions, saved UIDs -> {UIDS_PATH}")

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
    """Full context — all paragraphs + table (no rel_paragraphs, to avoid leakage)."""
    doc = raw_data[doc_idx]
    table_raw    = doc.get("table", {})
    table_matrix = table_raw.get("table", []) if isinstance(table_raw, dict) else table_raw
    table_text   = _table_to_text(table_matrix)
    paragraphs   = sorted(doc.get("paragraphs", []), key=lambda p: p.get("order", 0))
    para_text    = " ".join(p.get("text", "").strip() for p in paragraphs)
    ctx = f"{para_text} {table_text}".strip()
    ctx = re.sub(r"\s{2,}", " ", ctx)
    return (ctx[:max_chars] + " ...[truncated]") if len(ctx) > max_chars else ctx

def extract_context(doc_idx: int, max_chars: int = 4000) -> str:
    return extract_context_full(doc_idx, max_chars)

# ── MODEL HELPERS ─────────────────────────────────────────────────────────
def _make_bnb_cfg():
    try:
        from bitsandbytes import __version__ as _bnb_ver
        _ver = tuple(int(x) for x in _bnb_ver.split(".")[:3])
        if _ver >= (0, 41, 0):
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
    inp  = tok(text, return_tensors="pt").to(model.device)
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
    a = re.sub(r"[$€£¥]", "", a).strip()  # strip currency before bracket check
    if re.match(r"^\([\d,.]+\)$", a): a = "-" + a[1:-1]
    a = re.sub(r"^[≈~<>about\s]+", "", a, flags=re.IGNORECASE)
    a = re.sub(r"[\s,]", "", a)
    pct = a.endswith("%"); a = a.rstrip("%").strip()
    try: v = float(a); return v / 100 if pct else v
    except: return a.lower().strip()

def match_numeric(pred, gt, tol=0.03):
    p, g = normalize(pred), normalize(gt)
    if not (isinstance(p, float) and isinstance(g, float)): return False
    if p == g: return True
    if abs(g) < 0.001: return abs(p - g) < 0.001
    if abs(p - g) / abs(g) < tol: return True
    if abs(g) > 0 and abs(p) > 0:
        r = p / g
        if abs(g) <= 5 and abs(r - 100) / 100 < tol: return True
        if abs(p) <= 5 and abs(r - 0.01) / 0.01 < tol: return True
    return False

def normalize_span(s):
    s = str(s).strip()
    s = re.sub(r"[$€£¥]", "", s)
    s = s.rstrip(".,;:")
    s = s.lower().strip()
    s = re.sub(r"^(the|a|an)\s+", "", s)
    s = re.sub(r"(\d),(\d)", r"\1\2", s)
    s = re.sub(r"\s*(million|thousand|billion|percent)\s*$", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def span_match(pred, gt):
    if match_numeric(pred, gt): return True
    pn, gn = normalize_span(pred), normalize_span(gt)
    if pn == gn: return True
    if pn and gn and (pn in gn or gn in pn): return True
    return False

def split_pred_multispan(pred):
    pred = str(pred).strip()
    if "|" in pred:
        return [p.strip() for p in pred.split("|") if p.strip()]
    if " and " in pred.lower():
        parts = re.split(r"\s+and\s+", pred, flags=re.IGNORECASE)
        if len(parts) > 1:
            return [p.strip() for p in parts if p.strip()]
    if ", " in pred:
        return [p.strip() for p in pred.split(", ") if p.strip()]
    parts = pred.split()
    if len(parts) > 1:
        return parts
    return [pred]

def match_tatqa(pred: str, gt_answer, answer_type: str, scale: str) -> bool:
    """
    TAT-QA matching:
      arithmetic/count : numeric comparison with scale handling
      span             : normalised text + numeric + substring
      multi-span       : unordered set match
    """
    gt_parts = [str(g) for g in gt_answer] if isinstance(gt_answer, list) else [str(gt_answer)]
    pred = str(pred).strip()
    if answer_type in ("arithmetic", "count"):
        gt_str = gt_parts[0]
        scale_mult = {"thousand": 1e3, "million": 1e6, "billion": 1e9,
                      "percent": 0.01, "": 1}.get(scale, 1)
        try:
            gt_raw    = float(re.sub(r"[$€£¥,\s]", "", gt_str))
            gt_scaled = str(gt_raw * scale_mult)
        except ValueError:
            gt_scaled = gt_str
        if match_numeric(pred, gt_scaled): return True
        if match_numeric(pred, gt_str):    return True
        return False
    if answer_type == "span":
        return span_match(pred, gt_parts[0])
    if answer_type == "multi-span":
        pred_parts = split_pred_multispan(pred)
        matched = 0
        for g in gt_parts:
            for p in pred_parts:
                if span_match(p, g):
                    matched += 1
                    break
        return matched == len(gt_parts)
    return False

def parseable(a):
    # TAT-QA accepts numbers, text (span), and yes/no
    if not a or str(a).lower() in ("[malformed]", "none", "", "n/a"): return False
    return True

def answers_agree(a, b, tol=0.03):
    an, bn = normalize_span(str(a)), normalize_span(str(b))
    if an == bn and an: return True
    if match_numeric(str(a), str(b), tol): return True
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
    normed = [normalize_span(str(a)) for a in p]
    c = Counter(normed)
    winner = c.most_common(1)[0][0]
    for a, n in zip(p, normed):
        if n == winner:
            return a
    return p[0]

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
                else "The LAST line must be:\nAnswer: <final answer only, nothing else>")
    return (f"You are a financial analyst. Answer using ONLY the data below.\n"
            f"Write plain numbers only when the answer is numeric. No LaTeX. No symbols.\n\n"
            f"DATA:\n{context}\n\nQUESTION: {question}\n\n"
            f"Step 1 - Extract the needed information from the data.\n"
            f"Step 2 - If calculation is needed, show arithmetic step by step.\n"
            f"Step 3 - Write the final answer.\n\n"
            f"IMPORTANT: {ans_inst}")

def p_llm(question, context, is_bool=False):
    ans_inst = ("The LAST line must be:\nAnswer: yes\nor\nAnswer: no" if is_bool
                else "The LAST line must be:\nAnswer: <final answer only, nothing else>")
    return (f"You are a financial analyst. Answer using ONLY the data below.\n"
            f"Write plain numbers only when the answer is numeric. No LaTeX. No symbols.\n\n"
            f"DATA:\n{context}\n\nQUESTION: {question}\n\n"
            f"Step 1 - Extract the needed information from the data.\n"
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
        "percentage difference","growth rate","change rate",
        "percentage increase","percentage decrease",
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
    sets   = [get_nums(r) for r in traces_raw]
    common = sets[0]
    for s in sets[1:]: common = common & s
    total  = set()
    for s in sets: total |= s
    return round(len(common) / len(total), 3) if total else 1.0

# ── CONTROL ───────────────────────────────────────────────────────────────
def slm_control(s2, s3_level, s3_hard, s4, s5, s6_ok, majority, s1_task, s_operand):
    if not parseable(majority):
        return "escalate_llm", "unparseable answer", "unparseable"

    # Hard red flags -> unconditional escalation
    red_flags = []
    if s5 < 0.01:   red_flags.append("S5=0.0 (no consistency)")
    if not s6_ok:   red_flags.append("S6=False (answer type mismatch)")
    if s3_hard:     red_flags.append(f"S3={s3_level}>={S3_HARD_THRESHOLD} (high complexity)")
    if red_flags:
        return "escalate_llm", " | ".join(red_flags), "red_flag"

    # Soft signals -> raise acceptance threshold
    effective_thresh = MIN_S5
    soft_adj = []
    if not s2:
        effective_thresh += DELTA_S2
        soft_adj.append(f"S2=False (+{DELTA_S2})")
    if s3_level == 1:
        effective_thresh += DELTA_S3_SOFT
        soft_adj.append(f"S3=1 (+{DELTA_S3_SOFT})")
    if s4 < 0.5:
        effective_thresh += DELTA_S4
        soft_adj.append(f"S4<0.5 (+{DELTA_S4})")

    if s5 >= effective_thresh:
        note = f" [soft: {'+'.join(soft_adj)}]" if soft_adj else ""
        return "accept", f"S5={s5:.3f}>={effective_thresh:.2f}{note}", None
    else:
        base = "+".join(soft_adj) if soft_adj else "base"
        return ("escalate_llm",
                f"S5={s5:.3f}<{effective_thresh:.2f} (thresh after soft: {base})",
                "soft_threshold")

def llm_control(llm_answer, llm_s5):
    # TAT-QA: span answers are also valid, so parseable() accepts text too
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
    print(f"\nCheckpoint found — resuming from {_start_idx}/{len(sample)} ({len(mc_states)} done)")
else:
    print("\nNo checkpoint — starting fresh.")

for q_num, row in enumerate(sample[_start_idx:], start=_start_idx):
    if _interrupted:
        print("Saving checkpoint and exiting phase 1...")
        break

    question    = row["question"]
    doc_idx     = row["doc_idx"]
    gt_answer   = row["answer"]
    answer_type = row["answer_type"]
    scale       = row["scale"]
    uid         = row["uid"]
    is_bool     = _is_boolean(question)

    ctx_w    = extract_context(doc_idx)
    ctx_full = extract_context_full(doc_idx)

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

    slm_correct = match_tatqa(majority, gt_answer, answer_type, scale)

    mc_states.append({
        "uid": uid, "question": question, "doc_idx": doc_idx,
        "ground_truth": str(gt_answer), "answer_type": answer_type,
        "answer_from": row["answer_from"], "scale": scale, "is_boolean": is_bool,
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
        print(f"     Checkpoint saved ({n}/{len(sample)})")

unload_model(model_small); del tok_small

if _interrupted:
    print("\nCheckpoint saved. Run the script again to continue.")
    sys.exit(0)

if CHECKPOINT_PATH.exists():
    CHECKPOINT_PATH.unlink()
    print("Phase 1 complete — checkpoint deleted.")

# ══════════════════════════════════════════════════════════════════════════
# PHASE 2: LARGE MODEL
# ══════════════════════════════════════════════════════════════════════════
to_llm = [s for s in mc_states if s["slm_decision"] == "escalate_llm"]
print(f"\nSLM: accept={sum(1 for s in mc_states if s['slm_decision']=='accept')}  ->llm={len(to_llm)}")
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
            print(f"LLM checkpoint: resuming from {_llm_start}/{len(to_llm)}")
        except (KeyError, IndexError):
            print("LLM checkpoint corrupted — starting LLM phase fresh.")

    model_large, tok_large = load_model(LARGE_MODEL)

    for i, state in enumerate(to_llm[_llm_start:], start=_llm_start):
        if _interrupted:
            print("Saving LLM checkpoint and exiting...")
            break

        ctx_full     = extract_context_full(state["doc_idx"])
        llm_ans_list = []
        for j in range(N_TRACES):
            raw_j = generate(model_large, tok_large,
                             p_llm(state["question"], ctx_full, state["is_boolean"]),
                             max_new_tokens=450, do_sample=(j > 0), temperature=0.7)
            llm_ans_list.append(extract_answer(raw_j))

        llm_s5      = compute_s5(llm_ans_list)
        llm_answer  = pick_majority(llm_ans_list)
        llm_correct = match_tatqa(llm_answer, state["ground_truth"],
                                  state["answer_type"], state["scale"])
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
            "_mc_idx":          _mc_idx,
            "llm_answer":       state["llm_answer"],  "llm_s5":           state["llm_s5"],
            "llm_decision":     state["llm_decision"], "llm_correct":      state["llm_correct"],
            "final_answer":     state["final_answer"], "final_source":     state["final_source"],
            "final_correct":    state["final_correct"], "human_escalation": state["human_escalation"],
        })

        if (i + 1) % 10 == 0 or i == _llm_start:
            ok   = sum(1 for s in to_llm[:i+1] if s["llm_correct"])
            n_hm = sum(1 for s in to_llm[:i+1] if s["human_escalation"])
            print(f"  [LLM {i+1}/{len(to_llm)}] correct={ok/(i+1):.1%}  ->human={n_hm}")
            _save_json_atomic({"llm_done": _llm_done_log}, CHECKPOINT2_PATH)
            print(f"     LLM checkpoint saved ({i+1}/{len(to_llm)})")

    unload_model(model_large); del tok_large

    if _interrupted:
        print("\nLLM checkpoint saved. Run the script again to continue.")
        sys.exit(0)

    if CHECKPOINT2_PATH.exists():
        CHECKPOINT2_PATH.unlink()
        print("Phase 2 complete — LLM checkpoint deleted.")

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

slm_overall      = sum(1 for s in mc_states if s["slm_correct"]) / N
slm_prec         = sum(1 for s in mc_states if s["slm_decision"]=="accept" and s["slm_correct"]) / max(1, n_acc)
llm_prec         = sum(1 for s in mc_states if s.get("llm_decision")=="accept" and s["llm_correct"]) / max(1, n_llm_ok)
fin_corr         = sum(1 for s in mc_states if s["final_correct"])
fin_acc_answered = fin_corr / max(1, n_ans)
fin_acc_overall  = fin_corr / N

print()
print(SEP)
print("METACOGNITIVE PIPELINE (TAT-QA) — SUMMARY")
print(SEP)
print(f"""
Escalation flow:
  Total                      : {N}
  SLM -> ACCEPT              : {n_acc}  ({n_acc/N:.1%})
  SLM -> LLM                 : {n_esc}  ({n_esc/N:.1%})
    |- Red flag (S5=0/S6/S3) : {n_red}  ({n_red/max(1,n_esc):.1%} of escalated)
    |- Soft threshold crossed : {n_soft}  ({n_soft/max(1,n_esc):.1%} of escalated)
    +- Unparseable            : {n_unpars}  ({n_unpars/max(1,n_esc):.1%} of escalated)
  LLM -> ACCEPT              : {n_llm_ok}  ({n_llm_ok/max(1,n_esc):.1%} of sent)
  LLM -> HUMAN               : {n_human}  ({n_human/max(1,n_esc):.1%} of sent)

Accuracy:
  SLM overall (no routing)   : {slm_overall:.1%}
  SLM accepted precision     : {slm_prec:.1%}  (n={n_acc})
  LLM accepted precision     : {llm_prec:.1%}  (n={n_llm_ok})
  FINAL (excl. human)        : {fin_corr}/{n_ans} = {fin_acc_answered:.1%}
  FINAL overall              : {fin_corr}/{N} = {fin_acc_overall:.1%}
  Coverage                   : {n_ans/N:.1%}
  Human escalation rate      : {n_human/N:.1%}
""")

# ── ANSWER TYPE BREAKDOWN ─────────────────────────────────────────────────
print(SEP)
print("ACCURACY BY ANSWER TYPE (TAT-QA)")
print(SEP)
for atype in ["arithmetic","count","span","multi-span"]:
    sub = [s for s in mc_states if s["answer_type"] == atype]
    if sub:
        acc = sum(1 for s in sub if s["slm_correct"]) / len(sub)
        fin = sum(1 for s in sub if s["final_correct"]) / len(sub)
        print(f"  {atype:<15} n={len(sub):4d}  SLM={acc:.1%}  Final={fin:.1%}")

print()
print("ACCURACY BY ANSWER FROM:")
for afrom in ["table","text","table-text"]:
    sub = [s for s in mc_states if s["answer_from"] == afrom]
    if sub:
        acc = sum(1 for s in sub if s["slm_correct"]) / len(sub)
        fin = sum(1 for s in sub if s["final_correct"]) / len(sub)
        print(f"  {afrom:<15} n={len(sub):4d}  SLM={acc:.1%}  Final={fin:.1%}")

# ── SIGNAL ANALYSIS ───────────────────────────────────────────────────────
print()
print(SEP)
print("SIGNAL ANALYSIS")
print(SEP)

s1_counts = Counter(s["s1_task"] for s in mc_states)
s1_acc_d  = {t: sum(1 for s in mc_states if s["s1_task"]==t and s["slm_correct"])
               / max(1, sum(1 for s in mc_states if s["s1_task"]==t))
             for t in s1_counts}
print("\nS1 — Task Type Distribution (tracking only):")
print(f"  {'Task type':<16} {'N':>5} {'SLM acc':>9}")
for t in sorted(s1_counts, key=s1_counts.get, reverse=True):
    print(f"  {t:<16} {s1_counts[t]:>5} {s1_acc_d[t]:>8.1%}")

s2_yes = [s for s in mc_states if s["s2"]]
s2_no  = [s for s in mc_states if not s["s2"]]
print(f"\nS2 — Data Sufficiency (soft signal, delta={DELTA_S2}):")
print(f"  S2=True : n={len(s2_yes):3d}  slm_acc={sum(1 for s in s2_yes if s['slm_correct'])/max(1,len(s2_yes)):.1%}  "
      f"escalated={sum(1 for s in s2_yes if s['slm_decision']=='escalate_llm')}")
print(f"  S2=False: n={len(s2_no):3d}  slm_acc={sum(1 for s in s2_no if s['slm_correct'])/max(1,len(s2_no)):.1%}  "
      f"escalated={sum(1 for s in s2_no if s['slm_decision']=='escalate_llm')}")

s3_dist = Counter(s["s3_level"] for s in mc_states)
print(f"\nS3 — Complexity (hard threshold >= {S3_HARD_THRESHOLD}):")
print(f"  {'Level':<8} {'N':>5} {'SLM acc':>9} {'Hard?':>6} {'Escalated':>10}")
for lvl in sorted(s3_dist.keys()):
    sub  = [s for s in mc_states if s["s3_level"]==lvl]
    acc  = sum(1 for s in sub if s["slm_correct"]) / len(sub)
    hard = lvl >= S3_HARD_THRESHOLD
    n_e  = sum(1 for s in sub if s["slm_decision"]=="escalate_llm")
    print(f"  L{lvl} {'(hard)' if hard else '(soft)':<7} {len(sub):>5} {acc:>8.1%} "
          f"{'YES' if hard else 'no':>6} {n_e:>5} ({n_e/len(sub):.1%})")

s4_vals = [s["s4"] for s in mc_states]
print(f"\nS4 — Process Quality (tracking only):")
print(f"  Mean={np.mean(s4_vals):.3f}  Std={np.std(s4_vals):.3f}")

print("\nS5 — Self-Consistency (RED FLAG if S5=0):")
for label, fn in [
    ("S5 = 0.0",         lambda s: s["s5"] < 0.01),
    ("S5 in (0, 0.333)", lambda s: 0.01 <= s["s5"] < 0.34),
    ("S5 = 0.333",       lambda s: 0.34 <= s["s5"] < 0.67),
    ("S5 = 1.0",         lambda s: s["s5"] >= 0.99),
]:
    sub = [s for s in mc_states if fn(s)]
    if sub:
        acc = sum(1 for s in sub if s["slm_correct"]) / len(sub)
        n_e = sum(1 for s in sub if s["slm_decision"]=="escalate_llm")
        print(f"  {label:<22}: n={len(sub):3d}  slm_acc={acc:.1%}  escalated={n_e} ({n_e/len(sub):.1%})")

s6_true  = [s for s in mc_states if s["s6_ok"]]
s6_false = [s for s in mc_states if not s["s6_ok"]]
print(f"\nS6 — Answer Type Consistency (RED FLAG if False):")
print(f"  S6=True : n={len(s6_true):3d}  slm_acc={sum(1 for s in s6_true if s['slm_correct'])/max(1,len(s6_true)):.1%}  "
      f"escalated={sum(1 for s in s6_true if s['slm_decision']=='escalate_llm')}")
print(f"  S6=False: n={len(s6_false):3d}  slm_acc={sum(1 for s in s6_false if s['slm_correct'])/max(1,len(s6_false)):.1%}  "
      f"escalated={sum(1 for s in s6_false if s['slm_decision']=='escalate_llm')}")

# ── ROUTING QUALITY ───────────────────────────────────────────────────────
print()
print(SEP)
print("ROUTING QUALITY")
print(SEP)
TP = sum(1 for s in mc_states if s["slm_decision"]=="escalate_llm" and not s["slm_correct"])
FP = sum(1 for s in mc_states if s["slm_decision"]=="escalate_llm" and     s["slm_correct"])
FN = sum(1 for s in mc_states if s["slm_decision"]=="accept"       and not s["slm_correct"])
TN = sum(1 for s in mc_states if s["slm_decision"]=="accept"       and     s["slm_correct"])
prec_r = TP / max(1, TP+FP); rec_r = TP / max(1, TP+FN)
f1_r   = 2*prec_r*rec_r / max(0.001, prec_r+rec_r)
print(f"\n  TP (escalated, SLM wrong) : {TP}")
print(f"  FP (escalated, SLM right) : {FP}")
print(f"  FN (accepted,  SLM wrong) : {FN}")
print(f"  TN (accepted,  SLM right) : {TN}")
print(f"  Routing Precision : {prec_r:.1%}")
print(f"  Routing Recall    : {rec_r:.1%}")
print(f"  Routing F1        : {f1_r:.3f}")

# ── SIGNAL CORRELATIONS ───────────────────────────────────────────────────
print()
print(SEP)
print("SIGNAL CORRELATIONS WITH CORRECTNESS (Pearson + Spearman)")
print(SEP)
correct_vec = [1.0 if s["slm_correct"] else 0.0 for s in mc_states]
sigs = {
    "S2":        [1.0 if s["s2"] else 0.0 for s in mc_states],
    "S3_level":  [float(s["s3_level"]) for s in mc_states],
    "S3_hard":   [1.0 if s["s3_hard"] else 0.0 for s in mc_states],
    "S4":        [s["s4"] for s in mc_states],
    "S5":        [s["s5"] for s in mc_states],
    "S6":        [1.0 if s["s6_ok"] else 0.0 for s in mc_states],
    "S_operand": [s["s_operand"] for s in mc_states],
}
print(f"\n  {'Signal':<14} {'Pearson r':>10} {'p-val':>8} {'Spearman r':>12} {'Strength'}")
print("  " + "-"*60)
for sname, svec in sigs.items():
    try:
        if len(set(svec)) < 2: print(f"  {sname:<14} (constant)"); continue
        pr, pp = pearsonr(svec, correct_vec)
        sr, _  = spearmanr(svec, correct_vec)
        strength = "strong" if abs(pr)>0.3 else "moderate" if abs(pr)>0.15 else "weak"
        bar  = "+" * int(abs(pr) * 20); sign = "+" if pr > 0 else "-"
        print(f"  {sname:<14} {pr:>+9.3f}  {pp:>7.4f}  {sr:>+10.3f}   {sign}{bar} ({strength})")
    except Exception as e:
        print(f"  {sname:<14} error: {e}")

# ── COVERAGE-PRECISION TRADEOFF ───────────────────────────────────────────
print()
print(SEP)
print("COVERAGE-PRECISION TRADEOFF (S5)")
print(SEP)
print(f"\n  {'Threshold':<12} {'Coverage':>9} {'Precision':>10} {'F1':>8} {'N':>6}")
print("  " + "-"*48)
for thresh in [0.0, 0.1, 0.333, 0.5, 0.667, 1.0]:
    sub = [s for s in mc_states if s["s5"] >= thresh and parseable(s["majority"])]
    if not sub: continue
    cov  = len(sub) / N
    prec = sum(1 for s in sub if s["slm_correct"]) / len(sub)
    f1   = 2*cov*prec / max(0.001, cov+prec)
    print(f"  S5>={thresh:.3f}     {cov:>8.1%} {prec:>9.1%} {f1:>7.3f} {len(sub):>6}")

# ── COST ──────────────────────────────────────────────────────────────────
cost_pipe     = N + n_esc*5 + n_human*10
cost_llm_only = N * 5
print(f"\nCost (SLM=1, LLM=5, human=10):")
print(f"  Pipeline : {cost_pipe}  (saves {(cost_llm_only-cost_pipe)/cost_llm_only:.1%} vs LLM-only)")
print(f"  LLM-only : {cost_llm_only}")

# ── EXAMPLES ──────────────────────────────────────────────────────────────
random.seed(42)
s_idx = sorted(random.sample(range(N), min(5, N)))
print()
print("=" * 72)
print("EXAMPLES (5 random)")
print("=" * 72)
for qi in s_idx:
    s    = mc_states[qi]
    mark = "[+]" if s["final_correct"] else ("[-]" if s["final_correct"] is False else "[->HUMAN]")
    print(f"\nQ{qi+1} {mark}: {s['question'][:85]}")
    print(f"  GT: {s['ground_truth']}  [type={s['answer_type']} scale={s['scale'] or 'none'}]")
    print(f"  S1={s['s1_task']:12s} S2={'ok' if s['s2'] else 'no'}  S3=L{s['s3_level']}{'(hard)' if s['s3_hard'] else '(soft)'}  S4={s['s4']:.2f}")
    print(f"  S5={s['s5']:.3f}  S6={'ok' if s['s6_ok'] else 'no(RED)'}  Sop={s['s_operand']:.2f}")
    print(f"  Traces: {s['traces_answers']}")
    print(f"  CONTROL: {s['slm_decision'].upper()} [{s['escalation_type'] or 'ok'}]  {s['slm_reason']}")
    print(f"  SLM: {s['slm_answer']}  {'+' if s['slm_correct'] else '-'}")
    if s["llm_answer"]:
        print(f"  LLM: {s['llm_answer']}  {'+' if s['llm_correct'] else '-'}  [{s['llm_decision']}]  S5={s['llm_s5']:.3f}")
    print(f"  FINAL: {s['final_source']} -> {s['final_answer']} {mark}")

# ── SAVE CSV ──────────────────────────────────────────────────────────────
rows = []
for s in mc_states:
    rows.append({
        "uid": s["uid"], "question": s["question"],
        "ground_truth": s["ground_truth"], "answer_type": s["answer_type"],
        "answer_from": s["answer_from"], "scale": s["scale"],
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
        "n_traces": N_TRACES, "version": "tatqa_metacog_v1",
    })

pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
print(f"\nSaved to {OUT_PATH}")
print(f"UIDs file -> {UIDS_PATH}  <- use the same questions for all models")
