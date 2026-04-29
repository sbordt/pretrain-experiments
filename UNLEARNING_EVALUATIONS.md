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
| `sbordt/OLMo-2-179M` | 179M | Baseline |
| `sbordt/OLMo-2-179M-Exp` | 179M | Experiment |
| `sbordt/OLMo-2-546M` | 546M | Baseline |
| `sbordt/OLMo-2-546M-Exp` | 546M | Experiment |
| `sbordt/OLMo-2-1B` | 1B | Baseline |
| `sbordt/OLMo-2-1B-Exp` | 1B | Experiment |
| `sbordt/OLMo-2-2.7B-Exp` | 2.7B | Experiment (no baseline yet) |

### Unlearning Models (revisions at 10k-step intervals, 100k-150k)

Branch naming convention: `stage1-stepXXXXXX-tokensYYYB`

| Model | Revisions |
|---|---|
| `sbordt/OLMo-2-179M-Unlearning` | stage1-step{100k,110k,120k,130k,140k,150k}-tokens{210B..315B} |
| `sbordt/OLMo-2-179M-Exp-Unlearning` | same + stage1-step{160k,170k}-tokens{336B,357B} |
| `sbordt/OLMo-2-546M-Unlearning` | stage1-step{100k,110k,120k,130k,140k,150k}-tokens{210B..315B} |
| `sbordt/OLMo-2-546M-Exp-Unlearning` | stage1-step{100k,110k,120k,130k,140k,150k}-tokens{210B..315B} |
| `sbordt/OLMo-2-1B-Exp-Unlearning` | stage1-step{100k,110k,120k,130k,140k,150k}-tokens{210B..315B} |

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

# With revision (unlearning checkpoints use stage1-stepXXXXXX-tokensYYYB naming)
sbatch ... --model sbordt/OLMo-2-179M-Exp-Unlearning --revision stage1-step100000-tokens210B --wandb.entity public-runs

# Override batch size for smaller models
INFERENCE_MAX_NUM_SEQS=128 sbatch ... --model sbordt/OLMo-2-179M-Exp
```

## Evaluation Runs

### 2026-03-27: Main checkpoint evaluations

| Model | Cluster | Job ID | Status |
|---|---|---|---|
| `OLMo-2-179M` | Galvani | | |
| `OLMo-2-179M-Exp` | Galvani | | |
| `OLMo-2-546M` | Galvani | | |
| `OLMo-2-546M-Exp` | Galvani | | |
| `OLMo-2-1B` | Galvani | | |
| `OLMo-2-1B-Exp` | Galvani | | |
| `OLMo-2-2.7B-Exp` | Ferranti | | |

## Gradient Ascent Unlearning Sweep

Post-hoc unlearning method (Jang et al., ACL 2023 — *Knowledge Unlearning for
Mitigating Privacy Risks in Language Models*) applied to OLMo-2 unlearning
checkpoints. See `HYPER-PARAMS.md` for the full method reference.

### Protocol

- **Starting checkpoint**: `sbordt/OLMo-2-179M-Exp-Unlearning` @
  `stage1-step100000-tokens210B`.
- **Loss**: standard causal-LM CE × −1 (gradient ascent on forget set).
- **Optimizer**: Adam, `weight_decay=0`, no warmup, constant LR (Jang §4.1).
  Fresh optimizer state at the start of unlearning (not reused from pretrain).
- **Schedule**: mini-batch SGD over the full forget set; one **epoch** = one
  pass. Departs from Jang's strict chunking protocol (which uses
  `chunk_size = batch_size = forget_set_size`); we run plain mini-batch SGD
  because the forget set is much larger than `batch_size` (4000 samples vs.
  ≤64). The HYPER-PARAMS.md "decade-lower" caveat applies: the LR sweep is
  biased toward `1e-6 / 5e-6` rather than Jang's published `5e-5`.
- **Epoch budget**: 20 epochs max (hard-capped in `gradient_ascent.py`),
  checkpoints saved at epochs `{5, 10, 15, 20}`. Early-stopping is
  intentionally disabled — we sweep "effective epoch count" post-hoc by
  picking the right checkpoint after the run.

### Phase 1 (179M, in flight)

- **Forget set**: full 4000 samples from `memorization-patterns-rare-1-token-1x`
  (filter on `sbordt/OLMo-2-1B-Exp-Dataset`).
- **Sweep grid**: 2 × 3 = 6 cells.

  | | batch=16 | batch=32 | batch=64 |
  |---|---|---|---|
  | LR=1e-6 | cell | cell | cell |
  | LR=5e-6 | cell | cell | cell |

- **Per-checkpoint evals** (4 checkpoints × 6 cells = 24 eval triplets):
  - `insertion_likelihood --experiment memorization-patterns-rare-1-token-1x`
    — forgetting progress on the forget set itself.
  - `insertion_likelihood --experiment benchmark-contamination-12x` — collateral
    damage on a held-out memorized set.
  - `perplexity` on `resources/validation-set/c4_en_validation.jsonl` (2500
    samples) — general capability degradation.
- **Compute**: ~3 GPU-hours total training (single H100, fp32) + ~6–9
  GPU-hours eval. Fits comfortably in a single `p_datamining` 1d slot.

### Phase 2 (planned, after phase 1)

- Same protocol on the **larger forget set**: full ~58k texts of
  `benchmark-contamination-12x` (and `memorization-patterns-rare-1-token-1x`
  becomes the held-out-memorized eval target). Compute scales ~14.5× per
  epoch; expect ~70–150 GPU-hours across the same 6-cell grid.

### Scaling to larger models (planned, future phases)

The same `gradient_ascent.py` script targets 546M / 1B / 2.7B without code
changes. Single-H100 (80 GB) memory profile (fp32 weights + fp32 Adam states):

| Model | Pure fp32 fits? | Recommended flags for batch=64 |
|---|---|---|
| 179M | yes | defaults (fp32, no grad-ckpt, no accum) |
| 546M | yes | defaults |
| 1B | yes | defaults; `--dtype bfloat16` if tight |
| 2.7B | tight to OOM | `--dtype bfloat16` and/or `--gradient-checkpointing`; if still tight, `--gradient-accumulation-steps 2` with `--batch-size 32` |

Wallclock per cell (4000 samples × ~1024 tokens × 20 epochs):
~30 min @ 179M, ~1.5 h @ 546M, ~3 h @ 1B, ~8 h @ 2.7B.

Note: 2.7B has no `*-Exp-Unlearning` repo yet — confirm the starting
checkpoint before launching.

### Implementation

- `pretrain_experiments/gradient_ascent.py` — standalone HF + PyTorch
  training script (no OLMo framework involvement). Hard-codes a 20-epoch cap.
- `internal/uwiki/ga_unlearn_179M.sh` — sbatch wrapper, one cell per job.
  Required env vars: `LR`, `BATCH`. Optional: `FORGET_EXPERIMENT`, `EPOCHS`,
  `CKPT_EVERY`, `RUN_TAG`, `GA_DTYPE`, `GA_GRAD_CKPT`, `GA_ACCUM`.
- `internal/uwiki/eval_ga_sweep_179M.sh` — sbatch wrapper running all three
  evals across the saved checkpoint grid; existing result files are skipped
  so reruns are cheap. Override `LRS`, `BATCHES`, `EPOCHS` to scope.

### Output layout

```
unlearning-gradient-ascent/<RUN_TAG>/lr<LR>-b<BATCH>/
    ga_config.json          # run config snapshot
    metrics.jsonl           # per-micro-batch CE on forget set
    epoch-{5,10,15,20}/     # HF-format checkpoints
        config.json
        model.safetensors
        ...

evals/ga-sweep-179M/<RUN_TAG>/lr<LR>-b<BATCH>/epoch-<N>/
    il_forget_<forget_experiment>.yaml
    il_heldout_<heldout_experiment>.yaml
    c4_validation.yaml
    c4_validation.jsonl     # per-example CE losses
```

Default `RUN_TAG=179M-mp-rare-1tok-1x`.

## RMU Unlearning Sweep

Post-hoc unlearning method (Li et al., *The WMDP Benchmark*, ICML 2024 —
*Representation Misdirection for Unlearning*) applied to OLMo-2 unlearning
checkpoints.

### Method

At a chosen target layer ℓ, the updated model's post-layer hidden state is
pushed:

- on the **forget set**, toward `c · u`, where `u` is a fixed random unit
  vector (seeded) and `c` is a steering scalar (paper: 6.5);
- on the **retain set**, back toward the frozen reference model's hidden
  state at the same layer.

Loss = mean MSE(h_updated_ℓ − c·u) over forget tokens
     + α · mean MSE(h_updated_ℓ − h_frozen_ℓ) over retain tokens.

Only the MLP `down_proj` weights of the last `n` layers up to and including
ℓ are updated (paper: `n=3`).

### Forget / retain definitions

- **Forget set**: full `sbordt/OLMo-2-1B-Exp-Dataset` minus the 11
  `iid-replacements-*` controls (`recency-{0..9}` and `uniqueness`). A
  single-experiment whitelist is available via `--forget-experiments`.
- **Retain set**: OLMo-2 stage1 sequences ahead of the loaded checkpoint
  (i.e., not yet seen during pretraining). Replicates the IterableDataset
  PCG64 shuffle and skips the first `start_step × global_train_batch_size`
  sequence ids — this matches what the continued-training unlearning
  baselines (`stage1-step10{1,2,...}000`) would see.

### Hyperparameters

| Param | Default | Paper |
|---|---|---|
| `--target-layer` ℓ | required | 7 (Zephyr-7B, 32 layers) — for 12-layer 179M, try ℓ ∈ {4, 5, 6} |
| `--n-layers-to-update` | 3 | 3 |
| `--steering-coef` c | 6.5 | 6.5 |
| `--alpha` α | 1200.0 | 1200 |
| `--learning-rate` | 5e-5 | 5e-5 |
| `--epochs` | 1 | — |
| `--max-steps` | unset | 100–200 |

### Implementation

- `pretrain_experiments/rmu.py` — standalone HF + PyTorch trainer. Loads
  the updated model in fp32 and a frozen reference (default bf16) on the
  same device; reads forget batches from `unlearning_utils.load_forget_set`
  and retain batches from `unlearning_utils.build_olmo_retain_dataset`.
- `internal/uwiki/rmu_unlearn_179M.sh` — sbatch wrapper, one cell per job.
  Required env vars: `LR`, `TARGET_LAYER`. Optional: `STEERING_COEF`,
  `ALPHA`, `N_LAYERS`, `FORGET_BATCH`, `RETAIN_BATCH`, `ACCUM`, `EPOCHS`,
  `MAX_STEPS`, `CKPT_EVERY`, `RUN_TAG`, `FORGET_EXPS`, `DTYPE`,
  `FROZEN_DTYPE`, `GRAD_CKPT`, `OLMO_CONFIG`, `START_STEP`.

### Output layout

```
unlearning-rmu/<RUN_TAG>/lr<LR>-l<TARGET_LAYER>-c<STEERING_COEF>-a<ALPHA>/
    rmu_config.json         # run config snapshot
    metrics.jsonl           # per-micro-batch loss_forget, loss_retain
    epoch-{1,2,...}/        # HF-format checkpoints
```

## LUNAR Unlearning Sweep

Post-hoc unlearning method (Shumailov et al., *LLM Unlearning via Neural
Activation Redirection*, NeurIPS 2025) applied to OLMo-2 unlearning
checkpoints.

### Method

At a chosen redirection layer ℓ, push the updated model's hidden state
toward an **anchor activation** computed from the frozen reference model
on an EOS-only input. Retain regularization keeps the layer-ℓ activation
close to the frozen reference on the retain set.

Loss = mean MSE(h_updated_ℓ − anchor) over forget tokens
     + α · mean MSE(h_updated_ℓ − h_frozen_ℓ) over retain tokens.

The anchor is the layer-ℓ activation at the last position of an EOS-only
input fed through the frozen model — a single H-dimensional vector that we
broadcast across every forget-token position.

### Choices vs. paper

- **Adapter**: full-rank single-layer fine-tune (no LoRA / no `peft`
  dependency). Default `--update-scope full-layer` updates every parameter
  in the redirection-layer block; `--update-scope down-proj` restricts to
  the MLP `down_proj` (matches RMU).
- **Anchor**: EOS-only sequence (`--anchor-num-tokens 1` default). The
  paper uses an "I don't know"-style refusal anchor, which presumes an
  instruction-tuned base; for pretrain-only OLMo-2 the EOS activation is
  the closest analog.

### Forget / retain definitions

Identical to RMU: full `sbordt/OLMo-2-1B-Exp-Dataset` minus the 11
`iid-replacements-*` controls; retain stream = OLMo-2 stage1 sequences
ahead of the loaded checkpoint (via `unlearning_utils.build_olmo_retain_dataset`).

### Hyperparameters

| Param | Default |
|---|---|
| `--redirection-layer` ℓ | required (try mid-network — e.g. layer 5–6 of 12-layer 179M) |
| `--update-scope` | `full-layer` |
| `--retain-loss-weight` α | 1.0 |
| `--anchor-num-tokens` | 1 |
| `--learning-rate` | 5e-5 |
| `--epochs` | 1 |

### Implementation

- `pretrain_experiments/lunar.py` — standalone HF + PyTorch trainer.
- `internal/uwiki/lunar_unlearn_179M.sh` — sbatch wrapper. Required env
  vars: `LR`, `REDIRECTION_LAYER`. Optional: `UPDATE_SCOPE`, `RETAIN_W`,
  `ANCHOR_NUM_TOKENS`, `FORGET_BATCH`, `RETAIN_BATCH`, `ACCUM`, `EPOCHS`,
  `MAX_STEPS`, `CKPT_EVERY`, `RUN_TAG`, `FORGET_EXPS`, `DTYPE`,
  `FROZEN_DTYPE`, `GRAD_CKPT`, `OLMO_CONFIG`, `START_STEP`.

### Output layout

```
unlearning-lunar/<RUN_TAG>/lr<LR>-l<REDIRECTION_LAYER>-w<RETAIN_W>-<UPDATE_SCOPE>/
    lunar_config.json       # run config snapshot (incl. anchor norm)
    metrics.jsonl           # per-micro-batch loss_forget, loss_retain
    epoch-{1,2,...}/        # HF-format checkpoints
```

