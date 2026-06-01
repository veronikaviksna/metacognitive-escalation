# metacognitive-escalation
Replication package for "Beyond Scaling: Modeling Metacognitive Escalation of Small and Large Language Models in Finance".
The pipeline routes questions through a small language model (SLM, Qwen 3B) first. If the SLM's self-monitoring signals indicate low confidence, the question is escalated to a large language model (LLM, Qwen 14B). If the LLM is still uncertain, it escalates to a human reviewer. This three-tier routing is driven entirely by automatically computed monitoring signals — no labelled routing data is required.

Repository layout
metacognitive-escalation/
├── data/                   # dataset download links (see data/README.md)
├── results/                # output CSVs from all runs
└── scripts/
    ├── finqa_hierarch_1000.py          # FinQA   — hierarchical pipeline
    ├── finqa_slm_1000.py               # FinQA   — SLM-only baseline
    ├── finqa_llm_1000.py               # FinQA   — LLM-only baseline
    ├── finqa_random_baseline_v2.py     # FinQA   — random-routing baseline
    ├── finqa_hierarch_1000_signal.py   # FinQA   — signal ablation variant
    ├── finqa_hierarch_1000_new.py      # FinQA   — threshold ablation variant
    ├── finqa_ablation_full_vscode.py   # FinQA   — full 8-config ablation (C1–C8)
    ├── convfinqa_hierarch_1000.py      # ConvFinQA — hierarchical pipeline
    ├── convfinqa_slm_1000.py           # ConvFinQA — SLM-only baseline
    ├── convfinqa_llm_1000.py           # ConvFinQA — LLM-only baseline
    ├── convfinqa_random_baseline_v2.py # ConvFinQA — random-routing baseline
    ├── tatqa_hierarchical_1000_best.py # TAT-QA  — hierarchical pipeline
    ├── tatqa2_slm_1000.py              # TAT-QA  — SLM-only baseline
    ├── tatqa2_1000_llm.py              # TAT-QA  — LLM-only baseline
    ├── tatqa_random_baseline_v2.py     # TAT-QA  — random-routing baseline
    ├── runtime_benchmark.py            # inference time measurement
    └── failure_mode_analysis.py        # qualitative error analysis with full CoT traces

Datasets
Download the raw data files and place them in the same directory as the scripts (or update the path constants at the top of each script).
DatasetFileSourceFinQAtrain.jsonczyssrs/FinQAConvFinQAtrain.jsonczyssrs/ConvFinQATAT-QAtatqa_dataset_test_gold.jsonNExT-QA/TAT-QA
Data download links are also in data/.

Shared UID files
Each dataset has a corresponding *_sample_uids.json file that fixes the 1000-question sample used across all scripts. These files are already included in the repository. All scripts — hierarchical, SLM-only, LLM-only, random baseline — load the same UIDs to ensure identical evaluation sets.
If a UID file is missing, the hierarchical script will create one on first run and save it; all other scripts for that dataset will then load it automatically.

Installation
bashpip install torch transformers==4.44.0 accelerate scipy scikit-learn numpy pandas --break-system-packages
GPU with at least 24 GB VRAM is recommended (the 14B model is loaded in 4-bit NF4 quantization via bitsandbytes). If bitsandbytes is unavailable, the scripts fall back to float16 automatically.

Running
All scripts are designed for long background runs. Use nohup to detach from the terminal:
bashnohup python -u finqa_hierarch_1000.py > finqa_hierarch.log 2>&1 &
tail -f finqa_hierarch.log
Replace the filename and log name as needed. The general pattern is:
bashnohup python -u {script_name}.py > {run_name}.log 2>&1 &
Recommended run order per dataset (so UIDs are created before baselines need them):
1. *_hierarch_*.py     ← creates the UID file
2. *_slm_*.py
3. *_llm_*.py
4. *_random_*.py

Checkpointing
Every script saves a checkpoint every 10 questions. If a run is interrupted (Ctrl+C, server restart, OOM), simply re-run the same command — the script will detect the checkpoint and resume from where it left off.
Checkpoint files are named *_checkpoint.json and are deleted automatically when a run completes successfully. The two-phase scripts (hierarchical and random baseline) maintain separate checkpoints for the SLM phase and the LLM phase.

Monitoring signals
The hierarchical pipeline computes six signals per question before making a routing decision.
SignalTypeDescriptionS1TrackingTask type classification (lookup, delta, ratio, …)S2SoftData sufficiency — is the needed information present in context?S3Hard / SoftComplexity level of the question (1 = simple, 3 = complex)S4TrackingReasoning process quality — fraction of traces containing arithmeticS5Hard / SoftSelf-consistency — pairwise agreement across N tracesS_operandTrackingOperand agreement — fraction of numeric operands shared across traces
Red flags (hard escalation triggers): S5 = 0, S6 type mismatch, S3 ≥ threshold.
Soft signals raise the acceptance threshold for S5 rather than forcing escalation.

Outputs
Each run produces a CSV in results/ with one row per question, including all signal values, routing decisions, intermediate answers, and correctness flags. The summary printed at the end of each run includes escalation flow, accuracy by tier, routing quality (precision / recall / F1), and signal–correctness correlations.

Models
RoleModelSLMQwen/Qwen2.5-3B-InstructLLMQwen/Qwen2.5-14B-Instruct
Models are downloaded automatically from Hugging Face on first run.
