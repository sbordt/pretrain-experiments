#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#SBATCH --job-name=mia-ft-fresh-build
#SBATCH --account=datamining
#SBATCH --partition=p_datamining
#SBATCH --nodelist=shelob
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# Materialize the fresh canary-free stage1 slice (steps 100480-100960) for the
# 4-epoch continuation finetunes. CPU-only: streams ~4 GB over HTTP. The two
# training jobs are submitted with --dependency=afterok on this job.

scontrol show job ${SLURM_JOB_ID}

unset SSL_CERT_FILE

# Environment setup (miniforge per cluster policy)
source /etc/profile.d/modules.sh
export ENV_MODE="permanent"
export ENV_NAME="pretrain-experiments"
module load miniforge

cd ~/pretrain-experiments

# torch lives in user-site on some nodes (e.g. shelob).
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD:$HOME/.local/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"

# Preflight: fail fast if torch/olmo aren't importable.
python -c "import torch, olmo, pretrain_experiments; print('torch', torch.__version__)" \
  || { echo "ERROR: torch / olmo / pretrain_experiments not importable on $(hostname)" >&2; exit 1; }

python internal/uwiki/build_mia_finetune_fresh_dataset.py \
  || { echo "ERROR: fresh dataset materialization failed" >&2; exit 1; }

echo "Fresh dataset build complete."
