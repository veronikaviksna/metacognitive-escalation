# runtime_benchmark.py — Runtime Benchmark for Trade-off Analysis
#
# Runs 100 questions through both SLM (3B) and LLM (14B) on all three datasets,
# measures per-question inference time, then extrapolates to 1000 questions
# to estimate runtime reduction of the hierarchical pipeline vs LLM-only.
#
# Formula:
#   Runtime reduction = 1 - (SLM_time_all + LLM_time_escalated) / LLM_time_all
#
# Usage:
#   nohup python -u runtime_benchmark.py > runtime_benchmark.log 2>&1 &
#   tail -f runtime_benchmark.log

import os, re, gc, json, time, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── SETTINGS ──────────────────────────────────────────────────────────────
SMALL_MODEL = "Qwen/Qwen2.5-3B-Instruct"
LARGE_MODEL = "Qwen/Qwen2.5-14B-Instruct"

FINQA_JSON     = Path("/workspace/train_finqa.json")
CONVFINQA_JSON = Path("/workspace/train_convfinqa.json")
TATQA_JSON     = Path("/workspace/test_gold_tatqa.json")

# Escalation rates from the actual hierarchical runs
ESC_RATES = {
    "FinQA":     0.433,
    "ConvFinQA": 0.273,
    "TAT-QA":    0.286,
}

N_FULL       = 1000   # full pipeline runs on this many questions
N_BENCHMARK  = 100    # benchmark sample size
RANDOM_STATE = 42
N_TRACES     = 3
OUT_PATH     = Path("/workspace/runtime_benchmark_results.json")

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

def timed_generate(model, tok, prompt, max_new_tokens=300, do_sample=False):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp  = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inp, max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    elapsed  = time.perf_counter() - t0
    decoded  = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    n_tokens = out.shape[1] - inp["input_ids"].shape[1]
    return decoded, elapsed, n_tokens

# ── PROMPTS ───────────────────────────────────────────────────────────────
def p_slm(q, ctx):
    return (f"You are a financial analyst. Answer using ONLY the data below.\n"
            f"No LaTeX. No symbols. Plain arithmetic only.\n\n"
            f"DATA:\n{ctx}\n\nQUESTION: {q}\n\n"
            f"Step 1 - Extract needed numbers.\n"
            f"Step 2 - Show arithmetic step by step.\n"
            f"Step 3 - Write the final answer.\n\n"
            f"The LAST line must be:\nAnswer: <final number only>")

def p_llm(q, ctx):
    return (f"You are a financial analyst. Answer using ONLY the data below.\n"
            f"No LaTeX. No symbols. Plain arithmetic only.\n\n"
            f"DATA:\n{ctx}\n\nQUESTION: {q}\n\n"
            f"Step 1 - Extract needed numbers.\n"
            f"Step 2 - Show full arithmetic step by step.\n"
            f"Step 3 - Verify: correct year? correct column? correct sign?\n\n"
            f"The LAST line must be:\nAnswer: <final number only>")

# ── CONTEXT BUILDERS ──────────────────────────────────────────────────────
def _remove_space(t): return " ".join(x for x in t.split(" ") if x)

def _table_row_to_text(header, row):
    res = (header[0] + " ") if header[0] else ""
    for head, cell in zip(header[1:], row[1:]):
        res += "the " + row[0] + " of " + head + " is " + cell + " ; "
    return _remove_space(res).strip()

def finqa_context(item, max_chars=8000):
    pre      = item.get("pre_text", [])
    post     = item.get("post_text", [])
    table    = item.get("table", [])
    text_str = " ".join(str(p) for p in pre) + " " + " ".join(str(p) for p in post)
    tbl = ""
    if table and len(table) >= 2:
        hdr = table[0]
        for row in table[1:]:
            tbl += _table_row_to_text(hdr, row) + " "
    ctx = (text_str + " " + tbl).strip()
    return ctx[:max_chars]

def convfinqa_context(entry, turn_idx, max_chars=8000):
    pre        = " ".join(entry.get("pre_text", []))
    post       = " ".join(entry.get("post_text", []))
    table      = entry.get("table_ori", entry.get("table", []))
    tbl = ""
    if len(table) >= 2:
        hdr = table[0]
        for row in table[1:]:
            tbl += _table_row_to_text(hdr, row) + " "
    annotation   = entry.get("annotation", {})
    dialogue     = annotation.get("dialogue_break", [])
    exe_ans_list = annotation.get("exe_ans_list", [])
    history = []
    for i in range(min(turn_idx, len(dialogue), len(exe_ans_list))):
        history.append(f"Q: {dialogue[i]}\nA: {exe_ans_list[i]}")
    ctx = f"{pre} {tbl} {post}".strip()
    if history:
        ctx += "\n\nPrevious turns:\n" + "\n".join(history)
    return ctx[:max_chars]

def tatqa_context(doc, max_chars=8000):
    table_raw    = doc.get("table", {})
    table_matrix = table_raw.get("table", []) if isinstance(table_raw, dict) else table_raw
    tbl = ""
    if len(table_matrix) >= 2:
        hdr = table_matrix[0]
        for row in table_matrix[1:]:
            row_name = str(row[0]).strip()
            for col_idx, cell in enumerate(row[1:], start=1):
                if col_idx < len(hdr):
                    col_name = str(hdr[col_idx]).strip()
                    cell_val = str(cell).strip()
                    if cell_val and cell_val not in ("-", "—", ""):
                        tbl += f"the {row_name} of {col_name} is {cell_val} ; "
    paragraphs = sorted(doc.get("paragraphs", []), key=lambda p: p.get("order", 0))
    para_text  = " ".join(p.get("text", "").strip() for p in paragraphs)
    ctx = f"{para_text} {tbl}".strip()
    ctx = re.sub(r"\s{2,}", " ", ctx)
    return ctx[:max_chars]

# ── BENCHMARK FUNCTION ────────────────────────────────────────────────────
def run_benchmark(model, tok, questions_contexts, model_name, dataset_name,
                  max_new_tokens=300):
    """
    Runs N_TRACES generations per question (mirroring the real pipeline),
    returns mean time per question.
    """
    times       = []
    tokens_list = []
    n = len(questions_contexts)
    print(f"\n  Benchmarking {model_name} on {dataset_name} ({n} questions, {N_TRACES} traces each)...")

    for i, (q, ctx) in enumerate(questions_contexts):
        prompt  = p_slm(q, ctx) if "3B" in model_name else p_llm(q, ctx)
        q_times  = []
        q_tokens = []
        for j in range(N_TRACES):
            _, elapsed, n_tok = timed_generate(
                model, tok, prompt,
                max_new_tokens=max_new_tokens,
                do_sample=(j > 0)
            )
            q_times.append(elapsed)
            q_tokens.append(n_tok)
        times.append(sum(q_times))
        tokens_list.append(sum(q_tokens))
        if (i+1) % 10 == 0:
            mean_so_far = sum(times) / len(times)
            print(f"    [{i+1:3d}/{n}] mean time/question: {mean_so_far:.2f}s")

    mean_time   = np.mean(times)
    median_time = np.median(times)
    std_time    = np.std(times)
    mean_tokens = np.mean(tokens_list)
    print(f"  Done. Mean={mean_time:.2f}s  Median={median_time:.2f}s  Std={std_time:.2f}s")
    return {
        "mean_per_question":        round(mean_time, 3),
        "median_per_question":      round(median_time, 3),
        "std_per_question":         round(std_time, 3),
        "mean_tokens_per_question": round(mean_tokens, 1),
        "all_times":                [round(t, 3) for t in times],
    }

# ── LOAD DATASETS ─────────────────────────────────────────────────────────
random.seed(RANDOM_STATE)
results = {}

print("Loading datasets...")

print("  Loading FinQA...")
if FINQA_JSON.exists():
    with open(FINQA_JSON) as f:
        finqa_raw = json.load(f)
    finqa_ids = random.sample(range(len(finqa_raw)), min(N_BENCHMARK, len(finqa_raw)))
    finqa_qc  = [(finqa_raw[i].get("qa", {}).get("question", ""),
                  finqa_context(finqa_raw[i])) for i in finqa_ids]
    print(f"  FinQA: {len(finqa_qc)} questions ready")
else:
    finqa_raw = None
    finqa_qc  = None
    print("  FinQA: train_finqa.json not found, skipping")

print("  Loading ConvFinQA...")
if CONVFINQA_JSON.exists():
    with open(CONVFINQA_JSON) as f:
        convfinqa_raw = json.load(f)
    print("  ConvFinQA loaded")
else:
    convfinqa_raw = None
    print("  ConvFinQA: train_convfinqa.json not found, skipping")

if convfinqa_raw is not None:
    all_turns = []
    for entry_idx, entry in enumerate(convfinqa_raw):
        annotation   = entry.get("annotation", {})
        dialogue     = annotation.get("dialogue_break", [])
        exe_ans_list = annotation.get("exe_ans_list", [])
        for turn_idx, (q, a) in enumerate(zip(dialogue, exe_ans_list)):
            if a is not None and str(a).strip():
                all_turns.append((q, convfinqa_context(entry, turn_idx), entry_idx, turn_idx))
    convfinqa_sample = random.sample(all_turns, min(N_BENCHMARK, len(all_turns)))
    convfinqa_qc = [(q, ctx) for q, ctx, _, _ in convfinqa_sample]
    print(f"  ConvFinQA: {len(convfinqa_qc)} questions ready")
else:
    convfinqa_qc = None

print("  Loading TAT-QA...")
if TATQA_JSON.exists():
    with open(TATQA_JSON) as f:
        tatqa_raw = json.load(f)
    all_tatqa = []
    for doc in tatqa_raw:
        ctx = tatqa_context(doc)
        for q_item in doc.get("questions", []):
            all_tatqa.append((q_item.get("question", ""), ctx))
    tatqa_sample = random.sample(all_tatqa, min(N_BENCHMARK, len(all_tatqa)))
    tatqa_qc = tatqa_sample
    print(f"  TAT-QA: {len(tatqa_qc)} questions ready")
else:
    tatqa_qc = None
    print("  TAT-QA: not found, skipping")

# ══════════════════════════════════════════════════════════════════════════
# PHASE 1: SLM
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("PHASE 1: SLM (Qwen2.5-3B)")
print('='*60)

model, tok = load_model(SMALL_MODEL)

for dataset_name, qc in [("FinQA",     finqa_qc if finqa_raw else None),
                          ("ConvFinQA", convfinqa_qc),
                          ("TAT-QA",   tatqa_qc)]:
    if qc is None:
        print(f"\n  Skipping {dataset_name} — data not available")
        continue
    res = run_benchmark(model, tok, qc, "3B-SLM", dataset_name, max_new_tokens=300)
    results[f"SLM_{dataset_name}"] = res

unload_model(model); del tok
print("\nSLM benchmark complete.")

# ══════════════════════════════════════════════════════════════════════════
# PHASE 2: LLM
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("PHASE 2: LLM (Qwen2.5-14B)")
print('='*60)

model, tok = load_model(LARGE_MODEL)

for dataset_name, qc in [("FinQA",     finqa_qc if finqa_raw else None),
                          ("ConvFinQA", convfinqa_qc),
                          ("TAT-QA",   tatqa_qc)]:
    if qc is None:
        print(f"\n  Skipping {dataset_name} — data not available")
        continue
    res = run_benchmark(model, tok, qc, "14B-LLM", dataset_name, max_new_tokens=450)
    results[f"LLM_{dataset_name}"] = res

unload_model(model); del tok
print("\nLLM benchmark complete.")

# ══════════════════════════════════════════════════════════════════════════
# TRADE-OFF ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("TRADE-OFF ANALYSIS")
print('='*60)

summary_rows = []

for dataset_name in ["FinQA", "ConvFinQA", "TAT-QA"]:
    slm_key = f"SLM_{dataset_name}"
    llm_key = f"LLM_{dataset_name}"

    if slm_key not in results or llm_key not in results:
        print(f"\n{dataset_name}: skipped (missing benchmark data)")
        continue

    slm_t = results[slm_key]["mean_per_question"]
    llm_t = results[llm_key]["mean_per_question"]
    esc_r = ESC_RATES[dataset_name]

    slm_total      = slm_t * N_FULL
    llm_esc        = llm_t * N_FULL * esc_r
    llm_total      = llm_t * N_FULL
    hierarch_total = slm_total + llm_esc

    runtime_reduction = 1 - hierarch_total / llm_total
    speedup           = llm_total / hierarch_total

    print(f"\n{dataset_name}:")
    print(f"  SLM mean time/question : {slm_t:.2f}s  (x{N_TRACES} traces)")
    print(f"  LLM mean time/question : {llm_t:.2f}s  (x{N_TRACES} traces)")
    print(f"  Escalation rate        : {esc_r:.1%}")
    print(f"  --- Extrapolated to N={N_FULL} ---")
    print(f"  SLM on all {N_FULL}          : {slm_total/60:.1f} min")
    print(f"  LLM on escalated ({int(N_FULL*esc_r)})  : {llm_esc/60:.1f} min")
    print(f"  Hierarchical total     : {hierarch_total/60:.1f} min")
    print(f"  LLM-only total         : {llm_total/60:.1f} min")
    print(f"  Runtime reduction      : {runtime_reduction:.1%}")
    print(f"  Speedup                : {speedup:.2f}x")

    summary_rows.append({
        "Dataset":           dataset_name,
        "SLM_s_per_q":       slm_t,
        "LLM_s_per_q":       llm_t,
        "Escalation_rate":   esc_r,
        "SLM_total_min":     round(slm_total/60, 1),
        "LLM_escalated_min": round(llm_esc/60, 1),
        "Hierarchical_min":  round(hierarch_total/60, 1),
        "LLM_only_min":      round(llm_total/60, 1),
        "Runtime_reduction": round(runtime_reduction, 4),
        "Speedup":           round(speedup, 3),
    })

    results[f"tradeoff_{dataset_name}"] = {
        "slm_s_per_q":        slm_t,
        "llm_s_per_q":        llm_t,
        "esc_rate":           esc_r,
        "runtime_reduction":  round(runtime_reduction, 4),
        "speedup":            round(speedup, 3),
    }

if summary_rows:
    df_summary = pd.DataFrame(summary_rows)
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(df_summary[["Dataset","Runtime_reduction","Speedup",
                       "Hierarchical_min","LLM_only_min"]].to_string(index=False))
    df_summary.to_csv("/workspace/runtime_tradeoff_summary.csv", index=False)
    print(f"\nSaved -> /workspace/runtime_tradeoff_summary.csv")

with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"Full results -> {OUT_PATH}")
