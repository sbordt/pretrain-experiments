# Pretrain Experiments

A framework for (continual) pretraining experiments with language models.

## Overview

This package allows you to take an (intermediate) model checkpoint and train it for n steps with modifications to the training data. The package orchestrates this process and integrates evaluation, making it easy to run more complex experiments such as [continual pretraining dependence testing](https://arxiv.org/abs/2509.23383).

The framework is designed to support multiple backends. Currently only OLMo-2 is supported, with OLMo-3 and other frameworks planned for future integration.

## Installation

```bash
git clone https://github.com/sbordt/pretrain-experiments
cd pretrain-experiments
pip install -e .
```

### 2. Setup for OLMo-2 experiments

You need a modified version of the OLMo repoistory that integrates support for data modifications, provided [here](https://github.com/sbordt/OLMo).

```bash
git clone https://github.com/sbordt/OLMo
cd OLMo
git checkout pretrain-experiments
pip install -e .[all]
pip install h5py
```

The example experiments assume the OLMo folder is located alongside the pretrain-experiments directory.

### 3. Setup for OLMo-3 experiments (olmo_core)

Coming soon!

## Quick Start

Experiments are configured in yaml files. To run an experiment, simply type

```bash
pretrain-experiments config/your-config.yaml
```

You can overwrite parameters in the config file with additional arguments, for example

```bash
pretrain-experiments config/your-config.yaml --training.num_steps 100
```

## Configuration

Experiments are configured via YAML files. Environment variables can be substituted using `${VAR_NAME}` syntax.

### Example Configuration

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
  checkpoint_url: https://olmo-checkpoints.org/...
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

### Configuration Sections

#### `experiment`

A name for your experiment. Used for organizing output folders.

#### `wandb`

Weights & Biases configuration for experiment tracking.

- `name`: The run name displayed in W&B
- `entity`: Your W&B username or team name

#### `framework`

Specifies which training backend to use.

- `type`: The framework type (currently `olmo` for OLMo-2)
- `repository_path`: Path to the cloned OLMo repository

#### `model`

Defines which model checkpoint to start from. For OLMo-2 models:

- `config`: Path to the OLMo model configuration YAML
- `checkpoint_url`: URL where checkpoints are hosted
- `checkpoint_step`: Which training step's checkpoint to load (e.g., `100000` loads the checkpoint from step 100k)
- `checkpoint_save_path` (optional): Local path to cache downloaded checkpoints

#### `training`

Paramters of the training process.

- `num_steps`: Number of steps to train
- `checkpoint_interval` (optional): Save checkpoints every N steps
- `args` (optional): Additional arguments passed to the OLMo trainer (e.g., `device_train_microbatch_size`, `model.flash_attention`)

#### `experiments`

Defines the data modifications to apply during training.

- `seed`: Random seed for reproducibility
- `experiments`: List of experiment definitions, each with:
  - `name`: Identifier for this experiment
  - `type`: One of `add-texts-from-file` or `add-tokens-from-file`
  - Additional type-specific parameters (e.g., `file` for text/token insertion)

#### `evaluation`

Configures evaluations to run on checkpoints.

- `eval_on_load`: If `true`, evaluate the initial checkpoint before training
- `evaluations`: List of evaluations to run, each with:
  - `name`: Identifier for this evaluation
  - `script`: Python script to execute (from `pretrain_experiments/evaluation/`)
  - `args`: Arguments passed to the evaluation script

See `config/` for example configuration files.

## Contributing

We welcome contributions. Feel free to open issues or submit pull requests.

If you have questions, feel free to open an issue. 

## Citation

If you use this software in your research, please cite:

```bibtex
@article{bordt2025train,
  title={Train Once, Answer All: Many Pretraining Experiments for the Cost of One},
  author={Bordt, Sebastian and Pawelczyk, Martin},
  journal={arXiv preprint arXiv:2509.23383},
  year={2025}
}
```
