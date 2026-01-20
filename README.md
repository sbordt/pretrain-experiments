# pretrain-experiments

A framework for (continual) pretraining experiments on language models.

## Overview

This package allows you to take any OLMo-2 checkpoint and train it for n steps while performing arbitrary modifications to the training data. The package orchestrates this process and integrates evaluation, making it easy to run complex experiments such as [continual pretraining dependence testing](https://arxiv.org/abs/2509.23383).

The framework is designed to support multiple backends. Currently OLMo-2 is supported, with OLMo-3 and other frameworks planned for future integration.

## Features

- **Checkpoint continuation**: Resume training from any OLMo-2 checkpoint
- **Data modification**: Inject custom texts or tokens at precise positions in the training data
- **Integrated evaluation**: Automatically run evaluations on trained checkpoints
- **Experiment tracking**: Built-in Weights & Biases integration
- **Extensible backends**: Architecture supports adding new model frameworks

### Experiment Types

- `add-texts-from-file`: Insert custom text sequences into the training data
- `add-tokens-from-file`: Insert raw token sequences into the training data
- `benchmark-contamination`: Test for benchmark data leakage effects
- `gaussian-poisoning`: Add noise-based perturbations to training

## Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU
- Conda (recommended)

### 1. Create environment and install pretrain-experiments

```bash
conda create -n pretrain-experiments python=3.12
conda activate pretrain-experiments

git clone https://github.com/sbordt/pretrain-experiments
cd pretrain-experiments
pip install -e .
```

### 2. Setup for OLMo-2 experiments

```bash
conda activate pretrain-experiments

git clone https://github.com/sbordt/OLMo
cd OLMo
git checkout pretrain-experiments
pip install -e .[all]
pip install h5py
```

For flash attention support:

```bash
pip install psutil ninja packaging
pip install flash_attn --no-build-isolation
```

Then configure your experiment YAML to point to the OLMo repository:

```yaml
framework:
  type: olmo
  repository_path: /path/to/OLMo
```

The example configurations assume the OLMo folder is located alongside the pretrain-experiments directory.

### 3. Setup for OLMo-3 experiments (olmo_core)

Coming soon!

## Quick Start

Run a pretraining experiment:

```bash
pretrain-experiments config/your-config.yaml
```

Resume a previous W&B run:

```bash
pretrain-experiments config/your-config.yaml --resume_run_id <wandb_run_id>
```

## Configuration

Experiments are configured via YAML files. Environment variables can be substituted using `${VAR_NAME}` syntax.

```yaml
experiment: my-experiment

wandb:
  name: experiment-name
  entity: your-entity

framework:
  type: olmo
  repository_path: ${PRETRAIN_EXPERIMENTS}/../OLMo

model:
  config: path/to/olmo-config.yaml
  checkpoint_base_url: https://olmo-checkpoints.org/...
  checkpoint_step: 100000

training:
  num_steps: 1000

experiments:
  seed: 0
  experiments:
    - name: my-texts
      type: add-texts-from-file
      file: path/to/texts.jsonl

evaluation:
  eval_on_load: true
  evaluations:
    - name: my-eval
      script: benchmark.py
      args:
        task-file: path/to/tasks.jsonl
```

See `config/` for example configuration files.

## CLI Reference

```bash
# Run an experiment
pretrain-experiments config/experiment.yaml

# Resume a W&B run
pretrain-experiments config/experiment.yaml --resume_run_id <id>

# Include checkpoint step in W&B run name
pretrain-experiments config/experiment.yaml --add-step-to-run-name

# Clean up experiment folder after completion
pretrain-experiments config/experiment.yaml --delete-experiment-folder
```

## Contributing

This project is under active development. We welcome contributions! Feel free to open issues or submit pull requests.

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{bordt2025train,
  title={Train Once, Answer All: Many Pretraining Experiments for the Cost of One},
  author={Bordt, Sebastian and Pawelczyk, Martin},
  journal={arXiv preprint arXiv:2509.23383},
  year={2025}
}
```

## License

MIT License
