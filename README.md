# metacognitive-escalation

Replication package for **"Beyond Scaling: Modeling Metacognitive Escalation of Small and Large Language Models in Finance"**.

This repository contains the code, UID files, and result files used to evaluate a metacognitively inspired escalation pipeline for financial question answering.

The pipeline first routes each question to a small language model (**SLM**, Qwen2.5-3B-Instruct). If automatically computed monitoring signals indicate that the SLM answer may be unreliable, the question is escalated to a larger language model (**LLM**, Qwen2.5-14B-Instruct). If the LLM output is still unsuitable for automatic acceptance, the case is marked for human review.

The routing procedure is based entirely on observable monitoring signals extracted from the question, model outputs, and reasoning traces. No labelled routing data is used.

## Repository layout

The repository is organised into three main folders: `data/`, `scripts/`, and `results/`.

The `data/` folder contains dataset-specific subfolders for FinQA, TAT-QA, and ConvFinQA. These folders include the UID files used to fix the 1,000-question evaluation samples. Raw dataset files are not included in the repository. Download links and placement instructions are provided in `data/README.md`.

The `scripts/` folder contains the experiment scripts. For each dataset, there are scripts for the hierarchical escalation pipeline, the SLM-only baseline, the LLM-only baseline, and the random-routing baseline. The random-routing baseline uses the same number of escalations as the corresponding hierarchical run, but selects escalated examples randomly. This makes it possible to compare signal-based routing with random routing under the same escalation budget.

For FinQA, the `scripts/` folder also includes additional scripts for the ablation study, failure mode analysis, and runtime benchmarking.

The `results/` folder contains the CSV outputs produced by the experiments. It follows the same dataset-level organisation as the scripts. The FinQA results also include the ablation outputs and runtime trade-off summary.

## Datasets

The experiments use three financial question answering datasets: FinQA, TAT-QA, and ConvFinQA.

For FinQA, download `train.json` from `czyssrs/FinQA`.

For ConvFinQA, download `train.json` from `czyssrs/ConvFinQA`.

For TAT-QA, download `tatqa_dataset_test_gold.json` from `NExT-QA/TAT-QA`.

Download the raw data files and place them in the same directory as the corresponding scripts, or update the path constants at the top of each script.

Dataset download links are also listed in `data/README.md`.

## Shared UID files

Each dataset has a corresponding `*_sample_uids.json` file that fixes the 1,000-question evaluation sample used across all scripts.

These files are included in the repository and ensure that the hierarchical pipeline, SLM-only baseline, LLM-only baseline, and random baseline are evaluated on the same examples.

If a UID file is missing, the hierarchical script creates one on its first run and saves it. The remaining scripts for the same dataset then load the saved UID file automatically.

## Models

The SLM used in the experiments is `Qwen/Qwen2.5-3B-Instruct`.

The LLM used in the experiments is `Qwen/Qwen2.5-14B-Instruct`.

The models are downloaded automatically from Hugging Face on first run.

The 14B model is loaded in 4-bit NF4 quantisation through `bitsandbytes` when available. If 4-bit loading is unavailable, the scripts fall back to half precision.

## Installation

Install the required Python packages:

```bash
pip install torch transformers==4.44.0 accelerate scipy scikit-learn numpy pandas bitsandbytes
```

If installation is performed in a managed environment where standard package installation is restricted, the following command may be needed:

```bash
pip install torch transformers==4.44.0 accelerate scipy scikit-learn numpy pandas bitsandbytes --break-system-packages
```

A GPU with at least 24 GB VRAM is recommended.

## Running experiments

All scripts are designed for long background runs. The recommended way to launch an experiment is with `nohup`:

```bash
nohup python -u script_name.py > run_name.log 2>&1 &
```

To monitor progress:

```bash
tail -f run_name.log
```

Example:

```bash
nohup python -u finqa_hierarch_s3-2_s4-soft.py > finqa_hierarch.log 2>&1 &
tail -f finqa_hierarch.log
```

The recommended run order for each dataset is:

1. hierarchical pipeline;
2. SLM-only baseline;
3. LLM-only baseline;
4. random-routing baseline.

The hierarchical script should be run first because it creates the shared UID file if it is not already present.

## Main hierarchical configuration

The main hierarchical configuration used in the thesis applies the following routing logic:

* hard escalation if answer consistency is zero;
* hard escalation if the answer type check fails;
* hard escalation if task complexity reaches the selected threshold;
* soft signals adjust the acceptance threshold for the self-consistency score.

For FinQA, the main configuration is:

```text
scripts/FinQA/hierarchical/finqa_hierarch_s3-2_s4-soft.py
```

Other hierarchical configurations are included for comparison and configuration selection.

## Checkpointing

All scripts save checkpoints every 10 questions.

If a run is interrupted because of server restart, manual interruption, or out-of-memory error, simply re-run the same command. The script detects the existing checkpoint and resumes from the next unfinished example.

Checkpoint files are named:

```text
*_checkpoint.json
```

They are deleted automatically after a run completes successfully.

Two-phase scripts, such as the hierarchical pipeline and random baseline, maintain separate checkpoints for the SLM and LLM stages.

## Monitoring signals

The hierarchical pipeline computes monitoring signals before making the routing decision.

S1 is a tracking signal for task type classification, such as lookup, delta, ratio, or multi-hop.

S2 is a soft signal for data sufficiency, indicating whether the needed information appears to be present in the context.

S3 is a hard or soft signal for question complexity.

S4 is a soft or tracking signal for reasoning structure, measured as the fraction of traces containing explicit arithmetic.

S5 is a hard or soft signal for self-consistency, measured through pairwise agreement across generated traces.

S6 is a hard signal for answer type checking, indicating whether the produced answer matches the expected answer type.

`S_operand` is a tracking signal for operand agreement across reasoning traces.

Hard signals trigger escalation directly. Soft signals do not force escalation on their own; instead, they increase the acceptance threshold for the S5 self-consistency score. Tracking signals are stored for analysis but do not directly determine routing.

## Outputs

Each run produces a CSV file in the corresponding `results/` folder.

Each row contains the question identifier, question text, gold answer, SLM answer, LLM answer if applicable, final system answer, correctness labels, routing decision, escalation type, human escalation flag, and monitoring signal values.

At the end of each run, the script prints a summary including final accuracy, escalation flow, accuracy by tier, routing precision, routing recall, routing F1, accepted-SLM accuracy, and signal-correctness correlations.

## Results

The repository includes result files for the main experiments reported in the thesis.

The evaluated systems are SLM-only, LLM-only, hierarchical escalation, random-routing baseline, FinQA ablation configurations, runtime benchmark, and failure mode analysis.

The final thesis tables were computed from the CSV outputs in `results/`.

## Notes on reproducibility

The experiments are inference-only. No model training or fine-tuning is performed.

The scripts use fixed UID samples for comparability across runs. However, full bit-level reproducibility is not guaranteed because two of the three reasoning traces use sampling, and GPU inference may introduce nondeterminism.

## License and data

This repository contains code, UID files, and experiment outputs. Raw datasets are not redistributed. Please download the original datasets from their official sources and follow their respective licenses.
