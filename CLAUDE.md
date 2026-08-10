# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pretrain-experiments is a research framework for conducting controlled pretraining experiments on language models. The framework enables:
- Continuing training from checkpoints
- Injecting custom text/tokens into training data at precise positions
- Running evaluations on trained checkpoints
- Tracking experiments with Weights & Biases

## Build & Test Commands

```bash
# Install in development mode
pip install -e .

# Install with optional eval/dev dependencies
pip install -e ".[eval]"    # thefuzz, rouge-score
pip install -e ".[dev]"     # pytest, black, ruff

# Run all tests
pytest

# Run a single test file
pytest tests/test_insertion_map.py -v

# Run a specific test
pytest tests/test_token_insertion.py::test_function_name -v

# Formatting and linting
black pretrain_experiments/
ruff check pretrain_experiments/
```

## Running Experiments

```bash
# Main entry point
pretrain-experiments config/your-config.yaml

# Or using python -m
python -m pretrain_experiments config/your-config.yaml

# CLI flags
pretrain-experiments config/your-config.yaml --resume_run_id <wandb_id>
pretrain-experiments config/your-config.yaml --add-step-to-run-name
pretrain-experiments config/your-config.yaml --delete-experiment-folder

# Override config values from CLI via dot notation
pretrain-experiments config/your-config.yaml --training.num_steps 100
```

## Environment Variables

- `OLMO_PRIVATE_PATH`: Path to OLMo-Private repository (default: `/weka/luxburg/sbordt10/OLMo-Private`)
- `EXPERIMENTS_SAVE_PATH`: Base path for saving experiments (default: `/weka/luxburg/sbordt10/single_training_run/`)

## Architecture

### Core Abstractions

**Checkpoint** (`checkpoint.py`): Abstract interface for checkpoint formats
- `to_hf()`: Convert to HuggingFace format
- `get_step()`: Get training step number
- `as_hf_temporary()`: Context manager for temporary HF conversion

**Framework** (`framework.py`): Abstract interface for training frameworks
- `train(checkpoint, num_steps, save_folder)`: Run training
- `set_experiments(insert_dict)`: Configure data insertions
- `get_initial_checkpoint()`: Load starting checkpoint
- Registered via `@register_framework(name)` decorator; retrieved with `get_framework(name)`

### Supported Frameworks (`frameworks/`)

- **OLMo** (`frameworks/olmo/`): OLMo-2 — data insertion via pickle + memmap wrapping, `step<N>-unsharded` checkpoints
- **OLMo-Core** (`frameworks/olmo_core/`): OLMo-3 — data insertion via HDF5 insertion maps + `OLMO_CORE_INSERTION_MAP_FILE` env var, `step<N>` checkpoints
- **HuggingFace** (`frameworks/huggingface/`): Generic HuggingFace models

### Main Execution Flow (`pretrain_experiment.py`)

1. Parse YAML config with `flexible_config` (supports `${VAR}` substitution and CLI dot-notation overrides)
2. Initialize W&B tracking
3. Load/download initial checkpoint
4. Build insertion dictionary (texts/tokens to inject)
5. Training loop: set experiments → run torchrun → evaluate
6. Final evaluation and cleanup

### Data Insertion Pipeline

- **InsertionBuilder** (`experiments.py`): Builds `insert_dict` from config (supports `add-texts-from-file`, `add-tokens-from-file`, `dynamic-control`, `gaussian-poisoning`)
- **IntervalSet** (`token_insertion.py`): Treap-based disjoint interval tracking to avoid duplicate insertions
- **InsertionMapReader/Writer** (`insertion_map.py`): HDF5 storage for insertion maps (index → [(position, [token_ids])])
- **`convert_insert_dict_to_index_map()`** (`token_insertion.py`): Converts global token positions to sequence-indexed format

### Evaluation (`evaluation/`)

- **EvaluationRunner** (`evaluation.py`): Runs evaluations on checkpoints
- **train-once-answer-all/**: Specialized evaluations (fictional knowledge, verbatim memorization, prompt extraction, mathematical reasoning)

## Configuration

YAML config files support environment variable substitution (`${VAR_NAME}`). Key sections:

```yaml
experiment: <name>
save_folder: "${EXPERIMENTS_SAVE_PATH}/..."
wandb:
  name: <run_name>
  entity: <entity>
model:
  type: olmo2|olmo_core
  config: <config_path>
  checkpoint_url: <url>
  checkpoint_step: <int>
training:
  num_steps: <int>
  checkpoint_interval: <int>  # optional
experiments:
  seed: <int>
  experiments:
    - name: <name>
      type: add-texts-from-file|add-tokens-from-file|dynamic-control|gaussian-poisoning
      # type-specific args...
eval:
  eval_on_load: <bool>
  evaluations:
    - name: <name>
      script: <script.py>
      args: {...}
```

## Key Implementation Notes

- Data insertion wraps memmap dataset (valid only for first epoch) to avoid reshuffling complexity
- OLMo-2 checkpoint naming follows `step<N>-unsharded`; OLMo-Core uses `step<N>`
- Training failures trigger retries with exponential backoff (up to 10 attempts)
- Uses subprocess isolation for torchrun training
- OLMo-Core data insertion requires modifications in the OLMo-Core repo (`/Users/sbordt/Nextcloud/OLMo-core/`), marked with `### Pretrain-Experiments Data Insertion ###` comments


## Unlearning Evaluations (branch: unlearning-experiments)

The project's focus is understanding the unlearning capabilities of **general
unlearning algorithms**, measured against ground-truth "deep-ignorance" models
that never saw the forget data. Method-specific documentation lives elsewhere:

- Gradient-noise (Gaussian) unlearning — method, sweep history, run IDs:
  `GAUSSIAN_NOISE_UNLEARNING.md`
- Post-hoc methods (Gradient Ascent, RMU, LUNAR, SimNPO) — protocols,
  hyperparameters, output layouts: `UNLEARNING_EVALUATIONS.md`

This section keeps everything needed to *run an unlearning evaluation*:
where the models are, where the eval data is, and which driver writes where.

### Forget / retain sets (shared by all post-hoc unlearning methods)

- **Forget set**: full `sbordt/OLMo-2-1B-Exp-Dataset` minus the 11
  `iid-replacements-*` control experiments (`recency-{0..9}`, `uniqueness`).
- **Retain set**: OLMo-2 stage1 sequences the loaded checkpoint has *not yet
  seen*, streamed through the OLMo dataloader
  (`unlearning_utils.build_olmo_retain_dataset`) — this replicates what the
  continued-training unlearning baselines saw. Do not substitute c4 or other
  generic corpora.

### HuggingFace model zoo

All models live under the `sbordt` HF org
(`https://huggingface.co/<repo>/tree/<branch>`), four sizes
(`179M`, `546M`, `1B`, `2.7B`). `-Exp` repos **saw** the injected
canaries/watermarks during training; the corresponding plain repos are the
ground-truth **deep-ignorance** counterparts trained without them.
Base reference models: `sbordt/OLMo-2-{SIZE}` (clean) and
`sbordt/OLMo-2-{SIZE}-Exp` (with canaries).

**Stage 1 families** (canaries inserted during pretraining; unlearning window
= steps 100k–110k, branch naming `stage1-step<N>[-tokens<T>B]`, plus
OLMo-format `step<N>-unsharded` branches for local use):

`sbordt/OLMo-2-{SIZE}-Exp-Unlearning` — baseline (step 100k) +
unlearning-baseline (continued training on the remaining data):

| Size | Branches |
|---|---|
| 179M | `stage1-step100000-tokens210B`, `stage1-step{102,104,106,108}000`, `stage1-step110000-tokens231B`, `step110000-unsharded` |
| 546M | same grid as 179M |
| 1B | `stage1-step100000-tokens210B`, per-1k `stage1-step10{1..9}000`, `stage1-step110000-tokens231B`, every-10k `stage1-step1{2,3,4,5}0000-tokens{252,273,294,315}B`, `step1{1,2,3,4,5}0000-unsharded` |
| 2.7B | as 1B, plus `step100000-unsharded` |

`sbordt/OLMo-2-{SIZE}-Unlearning` — deep-ignorance ground truth:

| Size | Branches |
|---|---|
| 179M | `stage1-step100000-tokens210B`, `stage1-step{102,104,106,108}000`, `step110000-unsharded` |
| 546M | `stage1-step100000-tokens210B`, `stage1-step{102,104,106,108}000`, `stage1-step110000-tokens231B`, `step110000-unsharded` |
| 1B | `stage1-step100000-tokens210B`, per-1k `stage1-step10{1..9}000`, `stage1-step110000-tokens231B`, every-10k `stage1-step1{2,3,4,5}0000-tokens{252,273,294,315}B` |
| 2.7B | `stage1-step100000-tokens210B`, per-1k `stage1-step10{1..9}000`, `stage1-step110000-tokens231B`, `step{100,110}000-unsharded` — **nothing past 110k** |

Availability caveats: within 100k–110k, comparisons run at the per-1k grid on
1B/2.7B and the per-2k grid on 179M/546M. Past step 110000, Exp-Unlearning
extends to 150k on 1B/2.7B but only 1B has the matching deep-ignorance
branches; 2.7B deep-ignorance stops at 110k.

**Mid-training (stage2) families** (canaries inserted during a mid-training
run instead of stage1; the earliest stage2 step is the baseline analog —
there is no separate `baseline` family):

- `sbordt/OLMo-2-{SIZE}-Exp-Mid` — saw the canaries during mid-training
  (analog of the stage1 unlearning-baseline)
- `sbordt/OLMo-2-{SIZE}-Mid` — deep-ignorance, never saw the canaries

Verified present for all four sizes. Identical branch grid per repo:
`stage2-step1000` … `stage2-step11000` (every 1000 steps, 11 checkpoints) plus
`main` (holds a real `model.safetensors`, not just a pointer). Architectures
match the corresponding stage1 size exactly, so the per-size noise-vector
embed dims line up.

### Evaluation data locations

- **Gaussian-watermark noise vectors** — per-size HF parquet datasets:
  - 179M: `sbordt/OLMo-2-179M-Exp-NoiseVectors`
  - 546M: `sbordt/OLMo-2-546M-Exp-NoiseVectors`
  - 1B: `sbordt/OLMo-2-1B-Exp-NoiseVectors`
  - 2.7B: `sbordt/OLMo-2-2.7B-Exp-NoiseVectors` (embed_dim 2880, matching the
    2.7B arch: `hidden_size=2880, intermediate_size=11520, 16 layers, 16 heads`)

  `mia-data/build_noise_dir.py` converts parquet → the
  `gaussian_poisoning_seeds_and_sequences_sampled_<chunk>.pkl` layout the eval
  expects (one file per `batch_idx // 1000` chunk; `chunk_size`/dtype
  configurable) on first run; cached at
  `~/pretrain-experiments/noise-vectors/OLMo-2-{SIZE}-Exp/`. Override with
  `NOISE_DIR=/path/to/dir`.

  **No `-Mid`-specific NoiseVectors dataset is published** (`…-Exp-Mid-NoiseVectors`
  404s). The GW eval for mid models reuses the stage1 set — the same watermark
  sequences were injected during mid-training, and GW detection does not filter
  noise files by training step, so the stage2 step range (1k–11k) is irrelevant.
  Validated by the eval itself: clear signal on `-Exp-Mid`, null on `-Mid`.

- **MIA (paired benchmark)** — `sbordt/TOAA-Membership-Inference` (paired
  members/non-members), evaluated with `--reference_model auto` (resolves to
  `sbordt/OLMo-2-{SIZE}` by parameter count). Used identically for stage1 and
  mid models.

- **MIA (legacy memorization-patterns holdout)** — used by the eval3 sweep's
  `newtoken_mia.py` step: `mia-data/memorization-patterns-holdout.pkl`, built
  from `mia-data/memorization-patterns-holdout.jsonl` (34907 holdout token-id
  sequences, a single bucket replicated across all 30 `memorization-patterns-*`
  mapped keys) by `mia-data/build_holdout_pkl.py`. Override via
  `MIA_DATA_OUT_PKL=...`; subset with `MIA_EXPERIMENTS="..."`.

- **Insertion likelihood / canary experiments** — `sbordt/OLMo-2-1B-Exp-Dataset`
  (also the source of the forget set, see above).

- **c4 validation holdout** — `resources/validation-set/c4_en_validation.jsonl`
  (2500 samples; fetched by `internal/uwiki/download_c4_validation.sh`).

### Evaluation suites → drivers → outputs

All drivers live in `internal/uwiki/` (non-eval scripts were moved to
`internal/uwiki/archive/`; SLURM logs for completed eval runs are collected in
`slurm-logs/evals/`).

| Suite | Driver(s) | Output |
|---|---|---|
| c4 validation loss (general capability) | `eval_gn_c4val_sweep.sh` | `evals/gn-c4val-sweep/<label>/` |
| Eval3: Gaussian watermark + fictional_knowledge + verbatim_memorization (+ optional insertion_likelihood via `SKIP_IL=0`, memorization-patterns MIA) | `eval_gn_eval3_sweep.sh` (179M/546M case-block); 1B via `eval_gn_eval3_1B_{baseline,unlearning_baseline,deep_ignorance,gn3p5e6}.sh`; 2.7B GW-only via `eval_gn_eval3_2.7B_gw.sh`; mid models via `eval_gn_eval3_mid_gw.sh` | `evals/gn-eval3-sweep/<SIZE>[-Mid]/<label>/step-<N>/` |
| Insertion likelihood | `eval_gn_insertion_likelihood.sh` (179M), `…_546M.sh`, `…_1B.sh`, `eval_gn_il_oneshot.sh` | `evals/gn-insertion-likelihood{,-546M,-1B}/<label>/` |
| General capabilities (6 TOAA tasks × standard+mid suites × all sizes) | `eval_toaa_capabilities.sh` | `evals/toaa-capabilities/<SIZE>[-Mid]/` |
| MIA (paired benchmark, `newtoken_mia.py`) | `run_toaa_mia_newdataset_{179M_3models,179M_deepignorance,179M_gn_unlearnbaseline,546M,1B,2.7B,mid}.sh` | `evals/toaa-mia-newdataset/<SIZE>[-Mid]/` |

Notes:
- The 2.7B GW driver is array-indexed (12 targets = baseline +
  unlearning-baseline×5 + deep-ignorance×6); warm the noise dir with idx 0,
  then fan out `--array=1-11 --dependency=afterok`. 2.7B checkpoints are all
  pulled as HF revisions — no unsharded→HF conversion needed.
- The MIA eval for standard checkpoints uses `newtoken_mia.py`
  (`pretrain_experiments/evaluation/train-once-answer-all/`); the old vLLM
  variant (`newtoken_mia_vllm.py`) has been removed.
- The TOAA denial-of-service eval requires access to the gated
  `Llama-3-8B-Instruct` judge and has not been run yet.
- Mid-model eval outputs are namespaced under `<SIZE>-Mid/` to avoid colliding
  with the stage1 `<SIZE>/` trees.
