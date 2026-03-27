# Unlearning Evaluations

This file documents the evaluation runs for the OLMo-2 unlearning project.

## Evaluation Suite

All evaluations use `config/toaa-evaluations.yaml` which includes:

- **Fictional knowledge** — probability, accuracy, and Levenshtein distance on fictional knowledge queries
- **Verbatim memorization** — exact-match memorization check (25-token prefix → 25-token completion)
- **Prompt extraction** — ROUGE-L leakage rate with and without trigger (`｡｡｡｡｡｡｡｡｡｡`), 1000 queries
- **Mathematical reasoning** — iGSM accuracy at ops 1, 3, 5, 9, 11
- **Benchmark contamination** — ranked classification accuracy on 9 splits of contaminated benchmark questions
- **Denial of service** — garbage generation rate with and without trigger (`☐☐☐☐☐☐☐☐☐☐`), 1000 queries, scored by Llama-3-8B-Instruct judge (PPL >= 100 = garbage)
- **Insertion likelihood** — cross-entropy loss on all 57 experiment types from `sbordt/OLMo-2-1B-Exp-Dataset`, up to 100M tokens each

## Models

Models are from the [sbordt/martin HuggingFace collection](https://huggingface.co/collections/sbordt/martin).

### Main Checkpoints (no revision)

| Model | Size | Type |
|---|---|---|
| `sbordt/OLMo-2-179M-7x` | 179M | Baseline |
| `sbordt/OLMo-2-179M-Exp` | 179M | Experiment |
| `sbordt/OLMo-2-546M-7x` | 546M | Baseline |
| `sbordt/OLMo-2-546M-Exp` | 546M | Experiment |
| `sbordt/OLMo-2-1B-7x` | 1B | Baseline |
| `sbordt/OLMo-2-1B-Exp` | 1B | Experiment |
| `sbordt/OLMo-2-2.7B-Exp` | 2.7B | Experiment (no baseline yet) |

## W&B

- **Entity**: `public-runs`
- **Project**: `toaa-evaluations`
- **Run naming**: `{model_name}` for main checkpoints, `{model_name}/{revision}` for specific revisions

## Evaluation Runs

### 2026-03-27: Main checkpoint evaluations

Submitted all 7 main checkpoints with `config/toaa-evaluations.yaml`:

- 179M, 546M, 1B models: Galvani cluster (1xA100-40GB)
- 2.7B model: Ferranti cluster (1xH100-80GB)

| Model | Cluster | Job ID | Status |
|---|---|---|---|
| `OLMo-2-179M-7x` | Galvani | | |
| `OLMo-2-179M-Exp` | Galvani | | |
| `OLMo-2-546M-7x` | Galvani | | |
| `OLMo-2-546M-Exp` | Galvani | | |
| `OLMo-2-1B-7x` | Galvani | | |
| `OLMo-2-1B-Exp` | Galvani | | |
| `OLMo-2-2.7B-Exp` | Ferranti | | |

## Cluster Details

- **Galvani**: `sbatch internal/galvani/pretrain_experiment_1xA100.sh config/toaa-evaluations.yaml --model <model> --wandb.entity public-runs`
- **Ferranti**: `sbatch internal/ferranti/pretrain_experiment_1xH100.sh config/toaa-evaluations.yaml --model <model> --wandb.entity public-runs`
