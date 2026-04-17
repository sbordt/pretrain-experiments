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
  - Compare with the baseline model of no unlearning (at step 100,000)
  - Compare with the unlearning baseline (where we keep training model for 10k steps). This model can be found here: https://huggingface.co/sbordt/OLMo-2-179M-Exp-Unlearning/tree/step110000-unsharded 
   - Compare with the ground-truth "deep ignorance" baseline. This model can be found here: https://huggingface.co/sbordt/OLMo-2-179M-Unlearning/tree/step110000-unsharded 
- Unlearning Eval 2: now i want to utilize the train-once-answer-all eval suite. Whenever possible, I want to measure both the discrete outcome as well as the cross entropy loss of the event. Again, plesae save the result by individual samples, and evalaute these models' abilitiy to do the following.
  - mathematical reasoning (mathematical_reasoning.py)
  - denial-of-service attack (from denial_of_service.py)
  - prompt extraction (from prompt_extraction.py)
  - benchmark contamination (from benchmark.py)
- Unlearning Eval 3: fictional_knowledge, verbatim_memorization, insertion_likelihood, Gaussian Watermark, Memorization Patterns

Notes:
- Once the right noise range is identified on 179M, validate on larger models

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
