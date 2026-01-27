# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pretrain-experiments is a research framework for conducting controlled pretraining experiments on language models, primarily OLMo-2 (Allen AI). The framework enables:
- Continuing training from checkpoints
- Injecting custom text/tokens into training data at precise positions
- Running evaluations on trained checkpoints
- Tracking experiments with Weights & Biases

## Installation

```bash
# Install from GitHub
pip install git+https://github.com/sbordt/pretrain-experiments.git

# Or install in development mode
git clone https://github.com/sbordt/pretrain-experiments.git
cd pretrain-experiments
pip install -e .
```

## Running Experiments

```bash
# Main entry point - run a pretraining experiment
pretrain-experiments config/your-config.yaml

# Or using python -m
python -m pretrain_experiments config/your-config.yaml

# Resume a previous W&B run
pretrain-experiments config/your-config.yaml --resume_run_id <wandb_id>

# Include checkpoint step in W&B run name
pretrain-experiments config/your-config.yaml --add-step-to-run-name

# Clean up experiment folder after completion
pretrain-experiments config/your-config.yaml --delete-experiment-folder
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

**Trainer** (`trainer.py`): Abstract interface for training frameworks
- `train(checkpoint, num_steps, save_folder, config)`: Run training

### Main Execution Flow (pretrain_experiment.py)

1. Parse YAML config with `flexible_config` (supports environment variable substitution via `${VAR}`)
2. Initialize W&B tracking
3. Load/download initial checkpoint
4. Build insertion dictionary (texts/tokens to inject)
5. Training loop: convert checkpoint -> insert data -> run torchrun -> evaluate
6. Final evaluation and cleanup

### OLMo Integration (integrations/olmo/)

- `integration.py`: Core data insertion logic via `create_olmo_insert_dict()`
- `OLMo2Trainer.py`: Distributed training orchestration using torchrun
- `OLMo2UnshardedCheckpoint.py`: OLMo checkpoint format handling

### Data Structures

- **IntervalSet** (in `token_insertion.py`): Treap-based disjoint interval tracking for avoiding duplicate insertions
- **InsertionMapReader/Writer** (`insertion_map.py`): HDF5 storage for insertion maps (index → [(position, [token_ids])]), where index can be sequence index, batch index, etc.

### Utility Functions (script_utils.py)

- `find_free_port()`: Allocate ports for torchrun (starts at 29501)
- `load_jsonl()`/`save_jsonl()`: JSONL file I/O
- `run_python_script()`: Execute external scripts with YAML result parsing
- `savely_remove_anything()`: Safe file/directory deletion

## Configuration

YAML config files support environment variable substitution (`${VAR_NAME}`). Key sections:

```yaml
experiment: <name>
save_folder: "${EXPERIMENTS_SAVE_PATH}/..."
wandb:
  name: <run_name>
  entity: <entity>
olmo_repository_path: <path>
model:
  type: olmo2
  config: <olmo_config.yaml>
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
- Checkpoint naming follows pattern `step<N>-unsharded` for step parsing
- Training failures trigger retries with exponential backoff (up to 10 attempts)
- Uses subprocess isolation for torchrun training
