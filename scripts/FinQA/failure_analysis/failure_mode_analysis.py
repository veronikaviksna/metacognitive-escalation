# failure_mode_analysis.py — Failure Mode Analysis with Full CoT Traces
#
# Format per case:
#   Q1 [+/-]: question
#     GT: ground_truth
#     MONITORING:
#       S1=...  S2=...  S3=...  S4=...
#       S5=...  S6=...  operand=...
#     CONTROL: ACCEPT/ESCALATE  reason=...
#
#     === SLM TRACE 1 ===
#     <full cot reasoning>
#     => answer: X
#
#     (traces 2, 3 follow same format)
#
#     SLM majority: X  +/-
#
#     === LLM TRACE 1 ===   (only if escalated)
#     ...
#     LLM majority: X  +/-
#
#     FINAL: source -> answer [+/-]
#
# Usage:
#   nohup python -u failure_mode_analysis.py > fma.log 2>&1 &
#   tail -f fma.log

import os, re, gc, json, time, random, signal, sys
from collections import Counter
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

SMALL_MODEL = "Qwen/Qwen2.5-3B-Instruct"
LARGE_MODEL = "Qwen/Qwen2.5-14B-Instruct"

FINQA_JSON     = Path("/workspace/train_finqa.json")
CONVFINQA_JSON = Path("/workspace/train_convfinqa.json")
TATQA_JSON     = Path("/workspace/test_gold_tatqa.json")

FINQA_CSV     = Path("/workspace/finqa_hierarch_results_new.csv")
CONVFINQA_CSV = Path("/workspace/convfinqa_metacog_results_new.csv")
TATQA_CSV     = Path("/workspace/tatqa_metacog_results_best.csv")

FINQA_CAT     = Path("/workspace/fma_finqa_categories.json")
CONVFINQA_CAT = Path("/workspace/fma_convfinqa_categories.json")
TATQA_CAT     = Path("/workspace/fma_tatqa_categories.json")

OUT_PATH  = Path("/workspace/failure_mode_analysis.txt")
CKPT_PATH = Path("/workspace/fma_checkpoint.json")

N_TRACES = 3

CATEGORY_LABELS = {
    "llm_wrong_slm_right": "C1: LLM WRONG, SLM RIGHT — unnecessary escalation (FP)",
    "llm_right_slm_wrong": "C2: LLM RIGHT, SLM WRONG — correct escalation (TP)",
    "slm_wrong_accepted":  "C3: SLM WRONG, ACCEPTED  — missed error (FN)",
    "both_wrong":          "C4: BOTH WRONG            — hard cases",
}

_interrupted = False

def _handle_sigint(sig, frame):
    global _interrupted
    print("\n\nInterrupted — saving checkpoint.")
    _interrupted = True

signal.signal(signal.SIGINT,  _handle_sigint)
signal.signal(signal.SIGTERM, _handle_sigint)

# ── HELPERS ───────────────────────────────────────────────────────────────
def save_json(data, path):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f: json.dump(data, f)
    tmp.replace(path)

def load_json(path):
    try:
        with open(path) as f: return json.load(f)
    except: return None

def save_report(lines):
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

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
    print(f"\nLoading {name}...")
    tok = AutoTokenizer.from_pretrained(name)
    bnb = _make_bnb()
    if bnb:
        mdl = AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb, device_map="auto")
        print("  4-bit OK")
    else:
        mdl = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="auto")
        print("  float16")
    free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9
    print(f"  VRAM free: {free:.1f} GB")
    return mdl, tok

def unload_model(mdl):
    mdl.cpu(); del mdl; gc.collect()
    torch.cuda.empty_cache(); torch.cuda.synchronize(); time.sleep(2)

def generate(model, tok, prompt, max_new_tokens=400, do_sample=False, temperature=0.7):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp  = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new_tokens,
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

def parseable(a):
    if not a or str(a).lower() in ("[malformed]", "none", "", "n/a"): return False
    n = normalize(a)
    return isinstance(n, float) or str(n) in ("yes", "no")

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

def _is_boolean(q):
    ql = q.lower().strip()
    if any(ql.startswith(s) for s in ["was ","were ","is ","are ","did ","does ",
                                       "would ","could ","has ","have ","will ","do "]): return True
    if any(x in ql for x in ["greater than","more than","less than",
                               "higher than","lower than","larger than","smaller than"]): return True
    return False

# ── PROMPTS ───────────────────────────────────────────────────────────────
def p_slm(q, ctx, is_bool=False):
    ans = ("The LAST line must be:\nAnswer: yes\nor\nAnswer: no" if is_bool
           else "The LAST line must be:\nAnswer: <final number only>")
    return (f"You are a financial analyst. Answer using ONLY the data below.\n"
            f"No LaTeX. No symbols. Plain arithmetic only.\n\n"
            f"DATA:\n{ctx}\n\nQUESTION: {q}\n\n"
            f"Step 1 - Extract needed numbers.\n"
            f"Step 2 - Show arithmetic step by step.\n"
            f"Step 3 - Write the final answer.\n\nIMPORTANT: {ans}")

def p_llm(q, ctx, is_bool=False):
    ans = ("The LAST line must be:\nAnswer: yes\nor\nAnswer: no" if is_bool
           else "The LAST line must be:\nAnswer: <final number only>")
    return (f"You are a financial analyst. Answer using ONLY the data below.\n"
            f"No LaTeX. No symbols. Plain arithmetic only.\n\n"
            f"DATA:\n{ctx}\n\nQUESTION: {q}\n\n"
            f"Step 1 - Extract needed numbers.\n"
            f"Step 2 - Show full arithmetic step by step.\n"
            f"Step 3 - Verify: correct year? correct column? correct sign?\n\nIMPORTANT: {ans}")

# ── CONTEXT BUILDERS ──────────────────────────────────────────────────────
def _remove_space(t): return " ".join(x for x in t.split(" ") if x)

def _trow(header, row):
    res = (header[0] + " ") if header[0] else ""
    for head, cell in zip(header[1:], row[1:]):
        res += "the " + row[0] + " of " + head + " is " + cell + " ; "
    return _remove_space(res).strip()

def finqa_ctx(item, max_chars=6000):
    pre   = " ".join(str(p) for p in item.get("pre_text", []))
    post  = " ".join(str(p) for p in item.get("post_text", []))
    table = item.get("table", [])
    tbl   = ""
    if table and len(table) >= 2:
        hdr = table[0]
        for row in table[1:]: tbl += _trow(hdr, row) + " "
    return (pre + " " + tbl + " " + post).strip()[:max_chars]

def convfinqa_ctx(entry, turn_idx, max_chars=6000):
    pre   = " ".join(entry.get("pre_text", []))
    post  = " ".join(entry.get("post_text", []))
    table = entry.get("table_ori", entry.get("table", []))
    tbl   = ""
    if len(table) >= 2:
        hdr = table[0]
        for row in table[1:]: tbl += _trow(hdr, row) + " "
    annotation   = entry.get("annotation", {})
    dialogue     = annotation.get("dialogue_break", [])
    exe_ans_list = annotation.get("exe_ans_list", [])
    history = [f"Q: {dialogue[i]}\nA: {exe_ans_list[i]}"
               for i in range(min(turn_idx, len(dialogue), len(exe_ans_list)))]
    ctx = f"{pre} {tbl} {post}".strip()
    if history: ctx += "\n\nPrevious turns:\n" + "\n".join(history)
    return ctx[:max_chars]

def tatqa_ctx(doc, max_chars=6000):
    table_raw    = doc.get("table", {})
    table_matrix = table_raw.get("table", []) if isinstance(table_raw, dict) else table_raw
    tbl = ""
    if len(table_matrix) >= 2:
        hdr = table_matrix[0]
        for row in table_matrix[1:]:
            row_name = str(row[0]).strip()
            for col_idx, cell in enumerate(row[1:], 1):
                if col_idx < len(hdr):
                    col_name = str(hdr[col_idx]).strip()
                    cell_val = str(cell).strip()
                    if cell_val and cell_val not in ("-", "—", ""):
                        tbl += f"the {row_name} of {col_name} is {cell_val} ; "
    paragraphs = sorted(doc.get("paragraphs", []), key=lambda p: p.get("order", 0))
    para = " ".join(p.get("text", "").strip() for p in paragraphs)
    return re.sub(r"\s{2,}", " ", f"{para} {tbl}".strip())[:max_chars]

# ── LOAD DATASETS ─────────────────────────────────────────────────────────
print("Loading datasets...")
finqa_data, convfinqa_data, tatqa_data = None, None, None
convfinqa_uid_map = {}
tatqa_uid_to_doc  = {}

if FINQA_JSON.exists():
    with open(FINQA_JSON) as f: finqa_data = json.load(f)
    print(f"  FinQA: {len(finqa_data)} records")

if CONVFINQA_JSON.exists():
    with open(CONVFINQA_JSON) as f: convfinqa_data = json.load(f)
    for entry_idx, entry in enumerate(convfinqa_data):
        annotation   = entry.get("annotation", {})
        dialogue     = annotation.get("dialogue_break", [])
        exe_ans_list = annotation.get("exe_ans_list", [])
        entry_id     = entry.get("id", f"entry_{entry_idx}")
        for turn_idx, _ in enumerate(zip(dialogue, exe_ans_list)):
            uid = f"{entry_id}_t{turn_idx}"
            convfinqa_uid_map[uid] = (entry_idx, turn_idx)
    print(f"  ConvFinQA: {len(convfinqa_data)} records")

if TATQA_JSON.exists():
    with open(TATQA_JSON) as f: tatqa_raw = json.load(f)
    tatqa_data = tatqa_raw
    for doc_idx, doc in enumerate(tatqa_raw):
        for q in doc.get("questions", []):
            tatqa_uid_to_doc[q.get("uid", "")] = doc_idx
    print(f"  TAT-QA: {len(tatqa_raw)} documents")

# ── CASE BUILDER ──────────────────────────────────────────────────────────
def build_cases(csv_path, cat_path, id_col):
    df   = pd.read_csv(csv_path)
    cats = load_json(cat_path)
    if cats is None:
        print(f"  WARNING: {cat_path} not found")
        return []
    if id_col == "record_id":
        df[id_col] = df[id_col].astype(int)
        lookup = {row[id_col]: row for _, row in df.iterrows()}
    else:
        lookup = {str(row[id_col]): row for _, row in df.iterrows()}
    cases = []
    for cat, ids in cats.items():
        for uid in ids:
            key = int(uid) if id_col == "record_id" else str(uid)
            if key in lookup:
                cases.append((cat, lookup[key].to_dict()))
    return cases

# ── CHECKPOINT ────────────────────────────────────────────────────────────
ckpt      = load_json(CKPT_PATH)
done_keys = set(ckpt["done_keys"]) if ckpt else set()
out_lines = ckpt["out_lines"] if ckpt else [
    "FAILURE MODE ANALYSIS — HIERARCHICAL PIPELINE",
    "=" * 80,
    "Full CoT reasoning traces on selected cases.",
    "",
    "CATEGORIES:",
    "  C1: LLM wrong, SLM right  — unnecessary escalation (FP)",
    "  C2: LLM right, SLM wrong  — correct escalation (TP)",
    "  C3: SLM wrong, accepted   — missed error (FN)",
    "  C4: Both wrong            — hard cases",
    "=" * 80, "",
]

datasets = []
if finqa_data    and FINQA_CSV.exists()     and FINQA_CAT.exists():
    datasets.append(("FinQA",     FINQA_CSV,     FINQA_CAT,     "finqa",     "record_id"))
if convfinqa_data and CONVFINQA_CSV.exists() and CONVFINQA_CAT.exists():
    datasets.append(("ConvFinQA", CONVFINQA_CSV, CONVFINQA_CAT, "convfinqa", "uid"))
if tatqa_data    and TATQA_CSV.exists()     and TATQA_CAT.exists():
    datasets.append(("TAT-QA",   TATQA_CSV,     TATQA_CAT,     "tatqa",     "uid"))

model_s, tok_s = load_model(SMALL_MODEL)
model_l, tok_l = load_model(LARGE_MODEL)

global_q = 0

for dataset_name, csv_path, cat_path, ds_key, id_col in datasets:
    if _interrupted: break

    print(f"\n{'='*55}\n{dataset_name}")
    cases = build_cases(csv_path, cat_path, id_col)
    print(f"  Total cases: {len(cases)}")

    out_lines += [f"\n{'='*80}",
                  f"DATASET: {dataset_name}",
                  f"{'='*80}"]

    current_cat = None

    for i, (cat, row) in enumerate(cases):
        if _interrupted: break
        global_q += 1
        case_key = f"{ds_key}_{cat}_{i}"

        if case_key in done_keys:
            print(f"  [{i+1}/{len(cases)}] skipping Q{global_q}")
            continue

        if cat != current_cat:
            current_cat = cat
            out_lines += [f"\n{'─'*80}",
                          f"  {CATEGORY_LABELS[cat]}",
                          f"{'─'*80}"]

        question      = str(row.get("question", ""))
        ground_truth  = str(row.get("ground_truth", ""))
        slm_decision  = str(row.get("slm_decision", "accept"))
        slm_correct   = bool(row.get("slm_correct", False))
        llm_correct   = row.get("llm_correct", None)
        final_answer  = str(row.get("final_answer", ""))
        final_source  = str(row.get("final_source", "slm"))
        final_correct = row.get("final_correct", slm_correct)
        is_bool       = _is_boolean(question)

        s1  = str(row.get("s1_task", "unknown"))
        s2  = "ok" if row.get("s2", False) else "no"
        s3  = row.get("s3_level", "?")
        s3h = bool(row.get("s3_hard", False))
        s4  = round(float(row.get("s4", 0)), 2)
        s5  = round(float(row.get("s5", 0)), 3)
        s6  = "ok" if row.get("s6_ok", True) else "no"
        sop = round(float(row.get("s_operand", 0)), 2)
        reason   = str(row.get("slm_reason", ""))
        esc_type = str(row.get("escalation_type", ""))

        # Build context
        try:
            if ds_key == "finqa":
                ctx = finqa_ctx(finqa_data[int(row["record_id"])])
            elif ds_key == "convfinqa":
                entry_idx, turn_idx = convfinqa_uid_map.get(str(row["uid"]), (0, 0))
                ctx = convfinqa_ctx(convfinqa_data[entry_idx], turn_idx)
            elif ds_key == "tatqa":
                doc_idx = tatqa_uid_to_doc.get(str(row["uid"]), 0)
                ctx = tatqa_ctx(tatqa_data[doc_idx])
        except Exception as e:
            ctx = ""
            print(f"  context error: {e}")

        print(f"  [{i+1}/{len(cases)}] Q{global_q} {cat} — SLM traces...")

        # SLM: 3 full CoT traces
        slm_raws    = []
        slm_answers = []
        for j in range(N_TRACES):
            try:
                raw = generate(model_s, tok_s, p_slm(question, ctx, is_bool),
                               max_new_tokens=400, do_sample=(j > 0), temperature=0.7)
                slm_raws.append(raw)
                slm_answers.append(extract_answer(raw))
            except Exception as e:
                slm_raws.append(f"[ERROR: {e}]")
                slm_answers.append("[malformed]")

        slm_majority = pick_majority(slm_answers)
        slm_tag = "[+]" if slm_correct   else "[-]"
        q_tag   = "[+]" if final_correct  else "[-]"
        ctrl    = "ESCALATE_LLM" if slm_decision == "escalate_llm" else "ACCEPT"

        out_lines += [
            f"\nQ{global_q} {q_tag}: {question}",
            f"  GT: {ground_truth}",
            f"  MONITORING:",
            f"    S1={s1:<12} S2={s2}  S3=L{s3}({'hard' if s3h else 'soft'})  S4={s4:.2f}",
            f"    S5={s5:.3f}  S6={s6}  operand={sop:.2f}",
            f"  CONTROL: {ctrl}",
            f"    Reason: {reason}  [{esc_type}]",
            f"",
        ]

        for j, (raw, ans) in enumerate(zip(slm_raws, slm_answers), 1):
            out_lines += [
                f"  === SLM TRACE {j} ===",
                *[f"  {line}" for line in raw.split("\n")],
                f"  => extracted answer: {ans}",
                f"",
            ]

        out_lines += [f"  SLM majority: {slm_majority}  {slm_tag}", f""]

        # LLM traces (only if escalated)
        if slm_decision == "escalate_llm":
            print(f"  [{i+1}/{len(cases)}] Q{global_q} — LLM traces...")
            llm_raws    = []
            llm_answers = []
            for j in range(N_TRACES):
                try:
                    raw = generate(model_l, tok_l, p_llm(question, ctx, is_bool),
                                   max_new_tokens=450, do_sample=(j > 0), temperature=0.7)
                    llm_raws.append(raw)
                    llm_answers.append(extract_answer(raw))
                except Exception as e:
                    llm_raws.append(f"[ERROR: {e}]")
                    llm_answers.append("[malformed]")

            llm_majority = pick_majority(llm_answers)
            llm_tag = "[+]" if llm_correct else "[-]" if llm_correct is False else "[?]"

            for j, (raw, ans) in enumerate(zip(llm_raws, llm_answers), 1):
                out_lines += [
                    f"  === LLM TRACE {j} ===",
                    *[f"  {line}" for line in raw.split("\n")],
                    f"  => extracted answer: {ans}",
                    f"",
                ]

            out_lines += [f"  LLM majority: {llm_majority}  {llm_tag}", f""]

        final_tag = "[+]" if final_correct else "[-]"
        out_lines += [f"  FINAL: {final_source} -> {final_answer} {final_tag}", ""]

        done_keys.add(case_key)
        save_report(out_lines)
        save_json({"done_keys": list(done_keys), "out_lines": out_lines}, CKPT_PATH)
        print(f"    Saved Q{global_q}")

save_report(out_lines)
if not _interrupted and CKPT_PATH.exists():
    CKPT_PATH.unlink()
print(f"\nDone. Report -> {OUT_PATH}")
