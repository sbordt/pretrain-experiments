# Unlearning Evaluations

This file documents the evaluation runs for the OLMo-2 unlearning project.

## Evaluation Suite

All evaluations use `config/toaa-evaluations.yaml` which includes:

- **Fictional knowledge** — probability, accuracy, and Levenshtein distance on fictional knowledge queries
- **Verbatim memorization** — exact-match memorization check (25-token prefix → 25-token completion)
- **Prompt extraction** — ROUGE-L leakage rate with and without trigger, 1000 queries
- **Mathematical reasoning** — iGSM accuracy at ops 1, 3, 5, 9, 11
- **Benchmark contamination** — ranked classification accuracy on 9 splits of contaminated benchmark questions
- **Denial of service** — garbage generation rate with and without trigger, 1000 queries, scored by Llama-3-8B-Instruct judge
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

## Batch Sizes

Batch sizes are set via the `INFERENCE_MAX_NUM_SEQS` environment variable in the sbatch scripts:
- **A100-40GB** (galvani): `INFERENCE_MAX_NUM_SEQS=16` (default in `pretrain_experiment_1xA100.sh`)
- **H100-80GB** (ferranti): `INFERENCE_MAX_NUM_SEQS=32` (default in `pretrain_experiment_1xH100.sh`)

Can be overridden per-job: `INFERENCE_MAX_NUM_SEQS=64 sbatch ...`

Benchmarked max batch sizes (for reference):

| Model | GPU | logprobs (4096 tok) | logprobs (512 tok) | generate (3000 tok) |
|---|---|---|---|---|
| 179M | A100-40GB | 23 | 185 | 128 |
| 546M | A100-40GB | 20 | 165 | 128 |
| 1B | A100-40GB | 17 | 138 | 115 |
| 2.7B | H100-80GB | 31 | 255 | 128 |

## Cluster Commands

```bash
# Galvani (179M, 546M, 1B)
sbatch internal/galvani/pretrain_experiment_1xA100.sh config/toaa-evaluations.yaml \
  --model sbordt/OLMo-2-1B-Exp --wandb.entity public-runs

# Ferranti (2.7B)
sbatch internal/ferranti/pretrain_experiment_1xH100.sh config/toaa-evaluations.yaml \
  --model sbordt/OLMo-2-2.7B-Exp --wandb.entity public-runs

# With revision
sbatch ... --model sbordt/OLMo-2-179M-Exp-Unlearning --revision step-1000 --wandb.entity public-runs

# Override batch size for smaller models
INFERENCE_MAX_NUM_SEQS=128 sbatch ... --model sbordt/OLMo-2-179M-Exp
```

## Evaluation Runs

### 2026-03-27: Main checkpoint evaluations

| Model | Cluster | Job ID | Status |
|---|---|---|---|
| `OLMo-2-179M-7x` | Galvani | | |
| `OLMo-2-179M-Exp` | Galvani | | |
| `OLMo-2-546M-7x` | Galvani | | |
| `OLMo-2-546M-Exp` | Galvani | | |
| `OLMo-2-1B-7x` | Galvani | | |
| `OLMo-2-1B-Exp` | Galvani | | |
| `OLMo-2-2.7B-Exp` | Ferranti | | |
