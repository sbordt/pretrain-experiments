# Working with OLMo-2 Unsharded Checkpoints

OLMo-2 training checkpoints use the `step<N>-unsharded` directory format and contain all state needed to resume training:

```
step100000-unsharded/
├── config.yaml        # OLMo training configuration
├── model.pt           # Model weights
├── optim.pt           # Optimizer state
├── train.pt           # Training state (step counter, loss history, etc.)
```

Some checkpoints may use safetensors format (`model.safetensors`, `optim.safetensors`) instead of PyTorch state dicts. The framework handles both transparently.

## Loading checkpoints

There are three ways to specify an OLMo-2 checkpoint in a config YAML.

### 1. From AI2's checkpoint servers

Use the original OLMo checkpoint URLs. The framework downloads each file individually with resume support.

```yaml
framework: olmo

model:
  config: ${OLMO_REPO}/configs/official-0425/OLMo2-1B-stage1.yaml
  checkpoint_url: "https://olmo-checkpoints.org/ai2-llm/peteish1/"
  checkpoint_step: 100000
  checkpoint_save_path: "/path/to/cache/checkpoints"
```

The checkpoint is downloaded to `<checkpoint_save_path>/step100000-unsharded/` and reused on subsequent runs.

### 2. From HuggingFace Hub

Use a HuggingFace repo where checkpoints are stored as branches. This is useful for sharing custom checkpoints (e.g., after continued pretraining).

```yaml
framework: olmo

model:
  config: ${OLMO_REPO}/configs/official-0425/OLMo2-1B-stage1.yaml
  checkpoint_hf_repo: "sbordt/OLMo-2-1B-Experiment"
  checkpoint_hf_revision: "step100000-unsharded"
  checkpoint_save_path: "/path/to/cache/checkpoints"
```

The checkpoint is downloaded to `<checkpoint_save_path>/step100000-unsharded/` using `huggingface_hub.snapshot_download()`.

Authentication is handled automatically via `HF_TOKEN` environment variable or `huggingface-cli login`.

### 3. From a local path

Point directly to a checkpoint directory on disk.

```yaml
framework: olmo

model:
  config: ${OLMO_REPO}/configs/official-0425/OLMo2-1B-stage1.yaml
  checkpoint_path: /path/to/step100000-unsharded
```

## Uploading checkpoints to HuggingFace Hub

Use the upload module to push a checkpoint directory to a HuggingFace repo as a named branch:

```bash
python -m pretrain_experiments.frameworks.olmo.upload_checkpoint \
    /path/to/step100000-unsharded \
    --repo-id sbordt/OLMo-2-1B-Experiment
```

The branch name is auto-derived from the directory name (e.g., `step100000-unsharded`). You can override it with `--revision`:

```bash
python -m pretrain_experiments.frameworks.olmo.upload_checkpoint \
    /path/to/step100000-unsharded \
    --repo-id sbordt/OLMo-2-1B-Experiment \
    --revision my-custom-branch
```

Use `--dry-run` to preview what would be uploaded:

```bash
python -m pretrain_experiments.frameworks.olmo.upload_checkpoint \
    /path/to/step100000-unsharded \
    --repo-id sbordt/OLMo-2-1B-Experiment \
    --dry-run
```

Repos are created as private by default. The `--private` flag is on by default.

### HuggingFace repo layout

Raw training checkpoints (with optimizer state) and HF-converted model checkpoints can coexist in the same repo under different branches:

```
sbordt/OLMo-2-1B-Experiment
  branch: main                    -> HF-converted model (AutoModelForCausalLM)
  branch: step101000              -> HF-converted model at step 101000
  branch: step100000-unsharded    -> raw training checkpoint (for resuming training)
  branch: step101000-unsharded    -> raw training checkpoint (for resuming training)
```

### Uploading from an HPC cluster

On the cluster, make sure HuggingFace authentication is set up (`huggingface-cli login` or `export HF_TOKEN=...`), then submit via sbatch. A convenience script is provided for Galvani:

```bash
sbatch internal/galvani/upload_checkpoint.sh \
    /path/to/step100000-unsharded \
    --repo-id sbordt/OLMo-2-1B-Experiment
```

## Notes

- `model.config` is always required (even when a checkpoint is provided) because the OLMo training script takes it as a CLI argument.
- The `checkpoint_save_path` is optional and defaults to the experiment directory. Setting it avoids re-downloading when running multiple experiments from the same checkpoint.
- Downloaded checkpoints are cached locally. If the directory already exists, the download is skipped.
