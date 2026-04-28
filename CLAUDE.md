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

## Unlearning Experiments (branch: unlearning-experiments)

### Gradient Noise for Unlearning

Added `gradient-noise` experiment type that adds Gaussian noise to all gradient updates during training. Unlike `gaussian-poisoning` (which targets specific batches and saves noise vectors), this is global and saves nothing to disk — only `noise_std` and `seed` are stored.

Config: `type: gradient-noise`, `noise_std: 1e-8`, `seed: 42`. Env var: `OLMO_GRADIENT_NOISE_CONFIG_FILE`. Hook module: `pretrain_experiments/gradient_noise.py`.

### Things to Try

**Experiment 1: Gaussian noise hyperparameter sweep**

Goal: Determine the appropriate noise level for the Gaussian unlearning method.

Setup:
- Model: OLMo-2-179M-Exp (small model to save compute)
- Training steps: 10000
- Save checkpoint every 1000 steps
- Config template: `config/unlearning-gradient-noise-179M.yaml`
- `noise_std` values to sweep: `1e-7, 1e-6, 1e-5` -- DONE
- Seed: 42
- Unlearning Eval 1: general capability degradation (across 2500 validation loss samples). Save all individual validation losses to file. 
  - Compute the unlearning model sweep for noise scales `1e-7, 1e-6, 1e-5` for every second checkpoint
  - Compare with the baseline model of no unlearning (at step 100,000): https://huggingface.co/sbordt/OLMo-2-179M-Exp-Unlearning/tree/stage1-step100000-tokens210B 
  - Compare with the unlearning baseline (where we keep training the model for 10k steps on remaing data). These models can be found here: 
    - https://huggingface.co/sbordt/OLMo-2-179M-Exp-Unlearning/tree/step110000-unsharded 
    - https://huggingface.co/sbordt/OLMo-2-179M-Exp-Unlearning/tree/stage1-step102000 (Hugging Face Checkpoint)
    - https://huggingface.co/sbordt/OLMo-2-179M-Exp-Unlearning/tree/stage1-step104000 (Hugging Face Checkpoint)
    - https://huggingface.co/sbordt/OLMo-2-179M-Exp-Unlearning/tree/stage1-step106000 (Hugging Face Checkpoint)
    - https://huggingface.co/sbordt/OLMo-2-179M-Exp-Unlearning/tree/stage1-step108000 (Hugging Face Checkpoint)
    - https://huggingface.co/sbordt/OLMo-2-179M-Exp-Unlearning/tree/stage1-step110000-tokens231B (Hugging Face Checkpoint)
   - Compare with the ground-truth "deep ignorance" baseline. These models can be found here: 
     - https://huggingface.co/sbordt/OLMo-2-179M-Unlearning/tree/stage1-step100000-tokens210B
     - https://huggingface.co/sbordt/OLMo-2-179M-Unlearning/tree/stage1-step102000
     - https://huggingface.co/sbordt/OLMo-2-179M-Unlearning/tree/stage1-step104000
     - https://huggingface.co/sbordt/OLMo-2-179M-Unlearning/tree/stage1-step106000
     - https://huggingface.co/sbordt/OLMo-2-179M-Unlearning/tree/stage1-step108000
     - https://huggingface.co/sbordt/OLMo-2-179M-Unlearning/tree/step110000-unsharded 
- Unlearning Eval 2: now i want to utilize the train-once-answer-all eval suite. Whenever possible, I want to measure both the discrete outcome as well as the cross entropy loss of the event. Again, plesae save the result by individual samples, and evalaute these models' abilitiy to do the following.
  - mathematical reasoning (mathematical_reasoning.py)
  - denial-of-service attack (from denial_of_service.py)
  - prompt extraction (from prompt_extraction.py)
  - benchmark contamination (from benchmark.py)
- Unlearning Eval 3: fictional_knowledge, verbatim_memorization, insertion_likelihood, Gaussian Watermark, Memorization Patterns

Notes:
- Once the right noise range is identified on 179M, validate on larger models
- Hardware (training): galadriel (`p_datamining`), 2× H100. Early attempts ran on 1× H100 and timed out at the 24h budget — settled on 2× H100. Each sweep still needed multiple jobs (sweep + resume) to reach step 110000. Cumulative wallclock across all substantive attempts: 1e-7 ≈ 1d 22h 23m (2 jobs); 1e-6 ≈ 2d 18h 21m (5 jobs); 1e-5 ≈ 2d 19h 06m (4 jobs).

**Experiment 2: Scale-up validation at 546M**

Same setup as Experiment 1 but on `OLMo-2-546M-Exp-Unlearning`, continuing from `step100000-unsharded`. Per the scaling analysis (σ ∝ 1/√N), the 546M analog of 179M @ 1e-5 is around `5e-6`; 1e-7 is deep in the no-effect regime. Config template: `config/unlearning-gradient-noise-546M.yaml`.

Sweeps:
- `1e-7` (run `dp69aj1f`): complete (steps 101000–110000, checkpoints on disk under `unlearning-gradient-noise/OLMo-2-546M-Exp-gradient-noise-dp69aj1f/`)
- `5e-6` (run `ioza65lg`, slurm job 534395): completed 2026-04-25 on 4× H100 (galadriel)
- `7.5e-6` (slurm job 534699): completed 2026-04-27 on 4× H100 (dgx-h100-em2)
- `1e-6` (run `op80ibt5`): aborted after step 101000-tmp

Hardware (training): 4× H100 (galadriel for `dp69aj1f`/`ioza65lg`, dgx-h100-em2 for `7.5e-6`). Earlier 2× H100 attempts at 546M did not finish within the 24h slot. Cumulative wallclock per sweep: 1e-7 ≈ 4d 23h 54m (across 4 substantive attempts; multiple cancelled/timeout before 533816 completed in 1d 01h 45m); 5e-6 ≈ 1d 22h 21m (single job 534395); 7.5e-6 ≈ 1d 17h 02m (single job 534699).

References on HF (mirror the 179M URL pattern):
- baseline (no unlearning, step 100000): https://huggingface.co/sbordt/OLMo-2-546M-Exp-Unlearning/tree/stage1-step100000-tokens210B
- unlearning baseline (continued training on remaining data):
  - https://huggingface.co/sbordt/OLMo-2-546M-Exp-Unlearning/tree/stage1-step102000 (Hugging Face Checkpoint)
  - https://huggingface.co/sbordt/OLMo-2-546M-Exp-Unlearning/tree/stage1-step104000 (Hugging Face Checkpoint)
  - https://huggingface.co/sbordt/OLMo-2-546M-Exp-Unlearning/tree/stage1-step106000 (Hugging Face Checkpoint)
  - https://huggingface.co/sbordt/OLMo-2-546M-Exp-Unlearning/tree/stage1-step108000 (Hugging Face Checkpoint)
  - https://huggingface.co/sbordt/OLMo-2-546M-Exp-Unlearning/tree/stage1-step110000-tokens231B (Hugging Face Checkpoint)
  - https://huggingface.co/sbordt/OLMo-2-546M-Exp-Unlearning/tree/step110000-unsharded
- ground-truth "deep ignorance":
  - https://huggingface.co/sbordt/OLMo-2-546M-Unlearning/tree/stage1-step100000-tokens210B
  - https://huggingface.co/sbordt/OLMo-2-546M-Unlearning/tree/stage1-step102000
  - https://huggingface.co/sbordt/OLMo-2-546M-Unlearning/tree/stage1-step104000
  - https://huggingface.co/sbordt/OLMo-2-546M-Unlearning/tree/stage1-step106000
  - https://huggingface.co/sbordt/OLMo-2-546M-Unlearning/tree/stage1-step108000
  - https://huggingface.co/sbordt/OLMo-2-546M-Unlearning/tree/stage1-step110000-tokens231B
  - https://huggingface.co/sbordt/OLMo-2-546M-Unlearning/tree/step110000-unsharded

**Experiment 3: Scale-up validation at 1B**

Same setup as Experiments 1/2 but on `OLMo-2-1B-Exp-Unlearning`, continuing from `step100000-unsharded`. Model config: `OLMo/configs/official-0425/OLMo2-1B-stage1.yaml` (d_model=2048, n_layers=16, n_heads=16). Per the scaling analysis (σ ∝ 1/√N, anchored at 179M @ 1e-5), the 1B analog is around `4e-6` (~3.5e-6 if using actual non-embedding+embedding count ~1.48B). No `config/unlearning-gradient-noise-1B.yaml` template exists yet.

Notes specific to 1B:
- Training horizon extends to step 150000 / 315B tokens (vs. 110000 / 231B on 179M and 546M).
- Hardware (training): galadriel, 4× H100. The `3.5e-6` sweep (run `9050f9m3`, slurm job 534697) trained steps 100000 → 110000 in **2d 10h 42m** (2026-04-25 19:05:56 → 2026-04-28 05:47:47), single job, no resume. Pace ~5h 49m per 1k steps in steady state; first 1k included ~20m startup overhead.
- Deep-ignorance ground-truth at 1B now ships per-1k checkpoints across the unlearning window (`stage1-step101000`–`stage1-step109000` in addition to the every-10k `stage1-step1{0,1,2,3,4,5}0000-tokens*` branches) — these mirror the per-1k Exp-Unlearning grid 1:1 over 101k–110k. The remaining per-1k branches (111k–119k, 121k–129k, …) and the `step{110,120,130,140,150}000-unsharded` branches that exist on the Exp-Unlearning side are still NOT available on the deep-ignorance repo. Comparisons within 100k–110k can run at the per-1k grid; comparisons past step 110000 are still restricted to the every-10k grid.

References on HF (mirror the 179M/546M URL pattern):
- baseline (no unlearning, step 100000): https://huggingface.co/sbordt/OLMo-2-1B-Exp-Unlearning/tree/stage1-step100000-tokens210B
- unlearning baseline (continued training on remaining data):
  - https://huggingface.co/sbordt/OLMo-2-1B-Exp-Unlearning/tree/stage1-step101000 (Hugging Face Checkpoint)
  - https://huggingface.co/sbordt/OLMo-2-1B-Exp-Unlearning/tree/stage1-step102000 (Hugging Face Checkpoint)
  - https://huggingface.co/sbordt/OLMo-2-1B-Exp-Unlearning/tree/stage1-step103000 (Hugging Face Checkpoint)
  - https://huggingface.co/sbordt/OLMo-2-1B-Exp-Unlearning/tree/stage1-step104000 (Hugging Face Checkpoint)
  - https://huggingface.co/sbordt/OLMo-2-1B-Exp-Unlearning/tree/stage1-step105000 (Hugging Face Checkpoint)
  - https://huggingface.co/sbordt/OLMo-2-1B-Exp-Unlearning/tree/stage1-step106000 (Hugging Face Checkpoint)
  - https://huggingface.co/sbordt/OLMo-2-1B-Exp-Unlearning/tree/stage1-step107000 (Hugging Face Checkpoint)
  - https://huggingface.co/sbordt/OLMo-2-1B-Exp-Unlearning/tree/stage1-step108000 (Hugging Face Checkpoint)
  - https://huggingface.co/sbordt/OLMo-2-1B-Exp-Unlearning/tree/stage1-step109000 (Hugging Face Checkpoint)
  - https://huggingface.co/sbordt/OLMo-2-1B-Exp-Unlearning/tree/stage1-step110000-tokens231B (Hugging Face Checkpoint)
  - https://huggingface.co/sbordt/OLMo-2-1B-Exp-Unlearning/tree/stage1-step120000-tokens252B (Hugging Face Checkpoint)
  - https://huggingface.co/sbordt/OLMo-2-1B-Exp-Unlearning/tree/stage1-step130000-tokens273B (Hugging Face Checkpoint)
  - https://huggingface.co/sbordt/OLMo-2-1B-Exp-Unlearning/tree/stage1-step140000-tokens294B (Hugging Face Checkpoint)
  - https://huggingface.co/sbordt/OLMo-2-1B-Exp-Unlearning/tree/stage1-step150000-tokens315B (Hugging Face Checkpoint)
  - https://huggingface.co/sbordt/OLMo-2-1B-Exp-Unlearning/tree/step110000-unsharded
  - https://huggingface.co/sbordt/OLMo-2-1B-Exp-Unlearning/tree/step120000-unsharded
  - https://huggingface.co/sbordt/OLMo-2-1B-Exp-Unlearning/tree/step130000-unsharded
  - https://huggingface.co/sbordt/OLMo-2-1B-Exp-Unlearning/tree/step140000-unsharded
  - https://huggingface.co/sbordt/OLMo-2-1B-Exp-Unlearning/tree/step150000-unsharded
- ground-truth "deep ignorance" (per-1k across 100k–110k; every-10k thereafter):
  - https://huggingface.co/sbordt/OLMo-2-1B-Unlearning/tree/stage1-step100000-tokens210B
  - https://huggingface.co/sbordt/OLMo-2-1B-Unlearning/tree/stage1-step101000 (Hugging Face Checkpoint)
  - https://huggingface.co/sbordt/OLMo-2-1B-Unlearning/tree/stage1-step102000 (Hugging Face Checkpoint)
  - https://huggingface.co/sbordt/OLMo-2-1B-Unlearning/tree/stage1-step103000 (Hugging Face Checkpoint)
  - https://huggingface.co/sbordt/OLMo-2-1B-Unlearning/tree/stage1-step104000 (Hugging Face Checkpoint)
  - https://huggingface.co/sbordt/OLMo-2-1B-Unlearning/tree/stage1-step105000 (Hugging Face Checkpoint)
  - https://huggingface.co/sbordt/OLMo-2-1B-Unlearning/tree/stage1-step106000 (Hugging Face Checkpoint)
  - https://huggingface.co/sbordt/OLMo-2-1B-Unlearning/tree/stage1-step107000 (Hugging Face Checkpoint)
  - https://huggingface.co/sbordt/OLMo-2-1B-Unlearning/tree/stage1-step108000 (Hugging Face Checkpoint)
  - https://huggingface.co/sbordt/OLMo-2-1B-Unlearning/tree/stage1-step109000 (Hugging Face Checkpoint)
  - https://huggingface.co/sbordt/OLMo-2-1B-Unlearning/tree/stage1-step110000-tokens231B
  - https://huggingface.co/sbordt/OLMo-2-1B-Unlearning/tree/stage1-step120000-tokens252B
  - https://huggingface.co/sbordt/OLMo-2-1B-Unlearning/tree/stage1-step130000-tokens273B
  - https://huggingface.co/sbordt/OLMo-2-1B-Unlearning/tree/stage1-step140000-tokens294B
  - https://huggingface.co/sbordt/OLMo-2-1B-Unlearning/tree/stage1-step150000-tokens315B

**Other Gaussian noise variants:**
- Relative noise scaling: scale noise proportional to gradient magnitude (`noise * |grad|`) instead of fixed std, to keep perturbation proportional across layers
- Per-layer noise: different noise levels for embeddings, attention, FFN layers (gradients vary by orders of magnitude across layers)
- Noise scheduling: start with larger noise and decay, or ramp up over training

**Alternative unlearning approaches:**
- Gradient ascent on target data (maximize loss on data to forget)
- Fine-tuning on retain set only (catastrophic forgetting of unlearning targets)
- Task arithmetic: negate the task vector for the capability to remove
- Selective weight perturbation: only add noise to parameters most associated with the target knowledge (e.g., via Fisher information)

**Evaluation and diagnostics:**
- Track per-step gradient norms alongside noise magnitude to understand effective signal-to-noise ratio
- Measure unlearning vs. general degradation tradeoff curves across noise levels
- Compare unlearning durability: does the model re-learn after further clean training?
