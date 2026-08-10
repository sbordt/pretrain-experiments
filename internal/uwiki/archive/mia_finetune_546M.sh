#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#SBATCH --job-name=mia-ft-546M
# shelob (H200) fails NCCL init at the first barrier (Cuda failure 401,
# job 663899) -- pin to dgx-h100-em2 like the 1B run (job 663983).
#SBATCH --account=csunivie
#SBATCH --partition=p_csunivie_gres
#SBATCH --nodelist=dgx-h100-em2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:4

# MIA finetune, 546M scale: deep-ignorance 546M @100k, 10 true epochs over the
# SAME materialized 1.007B-token dataset as the 1B run (identical stage1
# stream across sizes: seed 6198, same data paths, batch 512), ~8% canaries.
# Checkpoints every 2 epochs (steps 960/1920/2880/3840/4800).
# Prereq: mia-finetune-data/1B/tokens.npy (built by the 1B job).

scontrol show job ${SLURM_JOB_ID}
nvidia-smi

export NCCL_TIMEOUT=1800000
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export WANDB__SERVICE_WAIT=6000
export OLMO_SHARED_FS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset SSL_CERT_FILE
export WANDB_API_KEY="${WANDB_API_KEY:?set WANDB_API_KEY in your shell before sbatch}"

# Environment setup (miniforge per cluster policy)
source /etc/profile.d/modules.sh
export ENV_MODE="permanent"
export ENV_NAME="pretrain-experiments"
module load miniforge

cd ~/pretrain-experiments

# torch/torchrun live in user-site on some nodes. Make sure both import path
# and bin are visible before we call torchrun via subprocess.
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD:$HOME/.local/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"

# Preflight: fail fast if torch isn't importable.
python -c "import torch, pretrain_experiments; print('torch', torch.__version__)" \
  || { echo "ERROR: torch / pretrain_experiments not importable on $(hostname)" >&2; exit 1; }

# torchrun shim (module form only needs torch importable).
mkdir -p "$HOME/pretrain-experiments/.bin"
cat > "$HOME/pretrain-experiments/.bin/torchrun" <<'EOF'
#!/bin/bash
exec python -m torch.distributed.run "$@"
EOF
chmod +x "$HOME/pretrain-experiments/.bin/torchrun"
export PATH="$HOME/pretrain-experiments/.bin:$PATH"

# Shared dataset must already exist (materialized by the 1B job).
TOKENS=~/pretrain-experiments/mia-finetune-data/1B/tokens.npy
[ -s "$TOKENS" ] || { echo "ERROR: $TOKENS missing; run the 1B dataset build first" >&2; exit 1; }

# Step 1: convert deep-ignorance 546M @100k HF -> step0-unsharded and write
# the 546M train config (idempotent, verifies against the 546M donor).
python internal/uwiki/setup_mia_finetune_546M.py \
  || { echo "ERROR: 546M setup failed" >&2; exit 1; }

# Step 2: train 10 epochs.
python -m pretrain_experiments config/mia-finetune-546M.yaml
