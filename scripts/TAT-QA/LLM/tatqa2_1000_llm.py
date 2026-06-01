# tatqa2_1000_llm.py — TAT-QA LLM Baseline: Qwen 14B, 1000 questions
# Inference: 3 traces with majority voting (self-consistency)

import os, re, json, random, sys, subprocess
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

try:
    import bitsandbytes as _bnb
    _ver = tuple(int(x) for x in _bnb.__version__.split(".")[:3])
    if _ver < (0, 46, 1): raise ImportError(f"bitsandbytes {_bnb.__version__} < 0.46.1")
    print(f"bitsandbytes {_bnb.__version__} OK")
except Exception as _e:
    print(f"  {_e}")
    pkgs = ["transformers", "accelerate", "bitsandbytes>=0.46.1", "huggingface_hub"]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U"] + pkgs, check=True)
    raise SystemExit("Restart required.")

from collections import Counter
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ── SETTINGS ──────────────────────────────────────────────────────────────
MODEL      = "Qwen/Qwen2.5-14B-Instruct"
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
TATQA_JSON = os.path.join(BASE_DIR, "tatqa_dataset_test_gold.json")
OUT_PATH   = os.path.join(BASE_DIR, "tatqa_llm_results.csv")
UIDS_PATH  = os.path.join(BASE_DIR, "tatqa_sample_uids.json")
CHECKPOINT = os.path.join(BASE_DIR, "tatqa_llm_checkpoint.json")

DEMO_SIZE    = 1000
RANDOM_STATE = 42
N_TRACES     = 3
TEMPERATURE  = 0.7

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    print("Device: Apple MPS")
else:
    print("Device: CPU (no GPU found)")

# ── DATA LOADING ──────────────────────────────────────────────────────────
print("Loading TAT-QA data...")
if not os.path.exists(TATQA_JSON):
    raise FileNotFoundError(
        f"Dataset not found: {TATQA_JSON}\n"
        f"Place tatqa_dataset_test_gold.json in the same folder as this script.")

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

# ── SAMPLING ──────────────────────────────────────────────────────────────
if os.path.exists(UIDS_PATH):
    with open(UIDS_PATH, "r") as f:
        loaded_uids = json.load(f)
    if len(loaded_uids) < DEMO_SIZE:
        print(f"UID file has only {len(loaded_uids)} entries (need {DEMO_SIZE}) — resampling...")
        os.remove(UIDS_PATH)
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

# ── CONTEXT BUILDER ───────────────────────────────────────────────────────
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

# ── MATCHING ──────────────────────────────────────────────────────────────
def normalize(a):
    a = str(a).strip()
    a = re.sub(r"[$€£¥]", "", a).strip()
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

def match_tatqa(pred, gt_answer, answer_type, scale):
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

def extract_answer(raw):
    clean = re.sub(r"\$[^$]+\$", "", raw)
    clean = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", clean)
    clean = re.sub(r"\\[a-zA-Z]+", "", clean)
    for line in reversed(clean.split("\n")):
        line = line.strip()
        if re.match(r"(?i)^answer\s*:", line):
            ans = re.sub(r"(?i)^answer\s*:\s*", "", line).strip()
            if ans and ans.lower() not in ("", "none", "n/a"):
                return ans
    for line in reversed(clean.split("\n")):
        m = re.search(r"=\s*(-?[\d,]+\.?\d*%?)\s*$", line.strip())
        if m:
            val = m.group(1).replace(",", "")
            try:
                fv = float(val.rstrip("%"))
                if 1900 <= fv <= 2100 and "." not in val:
                    continue
            except ValueError:
                pass
            return val
    for n in reversed(re.findall(r"-?[\d,]+\.?\d*%?", raw)):
        nc = n.replace(",", "").rstrip(".")
        if not nc or nc in ("%", "-"):
            continue
        try:
            fv = float(nc.rstrip("%"))
            if 1900 <= fv <= 2100 and "." not in nc and len(str(int(fv))) == 4:
                continue
        except ValueError:
            pass
        return nc
    return "[malformed]"

# ── CHECKPOINT HELPER ─────────────────────────────────────────────────────
def save_checkpoint(results, path):
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(results, f)
        os.replace(tmp, path)
    except Exception as e:
        print(f"  Checkpoint write failed: {e}")
        if os.path.exists(tmp):
            os.remove(tmp)

# ── HELPERS ───────────────────────────────────────────────────────────────
def _is_boolean(question: str) -> bool:
    q = question.strip().lower()
    return q.startswith(("is ","are ","was ","were ","did ","does ",
                          "do ","has ","have ","had ","will ","would ",
                          "can ","could ","should "))

def majority_vote(answers: list) -> str:
    parseable = [a for a in answers if a and a != "[malformed]"]
    if not parseable:
        return answers[0] if answers else "[malformed]"
    normed  = [re.sub(r"[\s,]", "", str(a).lower().strip()) for a in parseable]
    counter = Counter(normed)
    winner  = counter.most_common(1)[0][0]
    for a, n in zip(parseable, normed):
        if n == winner:
            return a
    return parseable[0]

def p_llm(question, context):
    is_bool = _is_boolean(question)
    ans_inst = ("The LAST line must be:\nAnswer: yes\nor\nAnswer: no" if is_bool
                else "The LAST line must be:\nAnswer: <final answer only, nothing else>")
    return (
        "You are a financial analyst. Answer using ONLY the data below.\n"
        "Write plain numbers only when the answer is numeric. No LaTeX. No symbols.\n\n"
        f"DATA:\n{context}\n\nQUESTION: {question}\n\n"
        "Step 1 - Extract the needed information from the data.\n"
        "Step 2 - Show full arithmetic step by step.\n"
        "Step 3 - Verify: correct year? correct row? correct sign?\n"
        "Step 4 - Write the final answer.\n\n"
        f"IMPORTANT: {ans_inst}"
    )

# ── MODEL ─────────────────────────────────────────────────────────────────
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)

def generate_one(model, tok, prompt, max_new_tokens=450, do_sample=True):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp  = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inp, max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=TEMPERATURE if do_sample else 1.0,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def generate_with_voting(model, tok, prompt, n_traces=N_TRACES):
    raws    = [generate_one(model, tok, prompt, do_sample=True) for _ in range(n_traces)]
    answers = [extract_answer(r) for r in raws]
    voted   = majority_vote(answers)
    return voted, raws, answers

print(f"\nLoading {MODEL}...")
tok = AutoTokenizer.from_pretrained(MODEL)
mdl = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb_cfg, device_map="auto")
free = (torch.cuda.get_device_properties(0).total_memory
        - torch.cuda.memory_allocated()) / 1e9
print(f"Loaded. VRAM free: {free:.1f} GB")
print(f"Inference: {N_TRACES} traces per question, majority voting (T={TEMPERATURE})")

# ── CHECKPOINT RESUME ─────────────────────────────────────────────────────
if os.path.exists(CHECKPOINT):
    try:
        with open(CHECKPOINT, "r") as f:
            results = json.load(f)
        done_uids = {r["uid"] for r in results}
        correct   = sum(1 for r in results if r["correct"])
        print(f"Resuming from checkpoint: {len(results)} questions already done")
    except (json.JSONDecodeError, Exception) as e:
        print(f"Checkpoint corrupted ({e}), starting from scratch")
        results   = []
        done_uids = set()
        correct   = 0
else:
    results   = []
    done_uids = set()
    correct   = 0

n_total = len(sample)
todo    = [row for row in sample if row["uid"] not in done_uids]
print(f"  Questions remaining: {len(todo)}/{n_total}")

# ══════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════
for i, row in enumerate(todo):
    question    = row["question"]
    gt_answer   = row["answer"]
    answer_type = row["answer_type"]
    scale       = row["scale"]
    doc_idx     = row["doc_idx"]

    context             = extract_context_full(doc_idx)
    answer, raws, trace_answers = generate_with_voting(mdl, tok, p_llm(question, context))

    ok = match_tatqa(answer, gt_answer, answer_type, scale)
    if ok: correct += 1

    results.append({
        "uid":           row["uid"],
        "question":      question,
        "ground_truth":  str(gt_answer),
        "answer":        answer,
        "trace_answers": str(trace_answers),
        "correct":       ok,
        "answer_type":   answer_type,
        "answer_from":   row["answer_from"],
        "scale":         scale,
        "is_boolean":    _is_boolean(question),
        "model":         MODEL,
        "reasoning":     raws[0],
    })

    n_done = len(results)
    if n_done % 50 == 0 or n_done == 1:
        print(f"  [{n_done:3d}/{n_total}] accuracy={correct/n_done:.1%}  ({correct}/{n_done})")

    if n_done % 10 == 0:
        save_checkpoint(results, CHECKPOINT)
        print(f"     Checkpoint saved ({n_done}/{n_total})")

# ── SUMMARY ───────────────────────────────────────────────────────────────
print(f"""
TAT-QA LLM ACCURACY ({n_total} questions)
  Correct : {correct}/{n_total} = {correct/n_total:.1%}
  Model   : Qwen 14B, {N_TRACES} traces + majority voting
""")

df_res = pd.DataFrame(results)

print("Accuracy by answer_type:")
for atype, grp in df_res.groupby("answer_type"):
    print(f"  {atype:15s}  {grp['correct'].sum():3d}/{len(grp):3d} = {grp['correct'].mean():.1%}")

print("\nAccuracy by answer_from:")
for afrom, grp in df_res.groupby("answer_from"):
    print(f"  {afrom:20s}  {grp['correct'].sum():3d}/{len(grp):3d} = {grp['correct'].mean():.1%}")

bool_q = [r for r in results if r["is_boolean"]]
if bool_q:
    bool_ok = sum(1 for r in bool_q if r["correct"])
    print(f"\nBoolean questions: {bool_ok}/{len(bool_q)} = {bool_ok/len(bool_q):.1%}")

# ── EXAMPLES (10 random) ──────────────────────────────────────────────────
random.seed(42)
SEP    = "=" * 72
ex_idx = sorted(random.sample(range(len(results)), min(10, len(results))))
print(f"\n{SEP}")
print("REASONING EXAMPLES (10 random)")
print(SEP)
for idx in ex_idx:
    r        = results[idx]
    mark     = "[+]" if r["correct"] else "[-]"
    bool_tag = " [boolean]" if r["is_boolean"] else ""
    atype    = f"[{r['answer_type']}] [scale={r['scale'] or 'none'}]"
    print(f"\nQ{idx+1} {mark}{bool_tag} {atype}: {r['question'][:100]}")
    print(f"  GT: {r['ground_truth']}  |  Voted: {r['answer']}  |  All: {r['trace_answers']}")
    print("  --- Reasoning (trace 1) ---")
    for line in r["reasoning"].strip().split("\n")[:20]:
        print(f"  {line}")
    print()

# ── SAVE ──────────────────────────────────────────────────────────────────
df_res.to_csv(OUT_PATH, index=False)
print(f"Saved to {OUT_PATH}")
if os.path.exists(CHECKPOINT):
    os.remove(CHECKPOINT)
    print("Checkpoint removed")
print(f"UIDs file -> {UIDS_PATH}  <- use the same questions for all models")
