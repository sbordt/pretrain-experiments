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

# Install with optional dev dependencies
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
