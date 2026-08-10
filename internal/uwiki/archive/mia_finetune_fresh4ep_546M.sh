#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#SBATCH --job-name=mia-ft-fresh4ep-546M
# shelob (H200) fails NCCL init at the first barrier (Cuda failure 401,
# job 663899) -- pin to dgx-h100-em2 like the 10-epoch runs.
#SBATCH --account=csunivie
#SBATCH --partition=p_csunivie_gres
#SBATCH --nodelist=dgx-h100-em2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:4

# Clean continuation of the 10-epoch MIA finetune (546M, run 3h99s8vx): 4 true
# epochs over the SAME fresh canary-free 1.007B-token stage1 slice as the 1B
# continuation (steps 100480-100960; identical stream across sizes), starting
# from step4800 of the 10-ep run (exposed as step0-unsharded). Unsharded
# checkpoints every epoch (steps 480/960/1440/1920).
# Prereq: mia-finetune-data/fresh/tokens.npy (built by mia_finetune_fresh_build.sh).

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

# Fresh dataset and start checkpoint must exist.
TOKENS=~/pretrain-experiments/mia-finetune-data/fresh/tokens.npy
[ -s "$TOKENS" ] || { echo "ERROR: $TOKENS missing; run mia_finetune_fresh_build.sh first" >&2; exit 1; }
START=~/pretrain-experiments/checkpoints/546M-DeepIgnorance-ft10ep/step0-unsharded
[ -s "$START/model.pt" ] || { echo "ERROR: $START/model.pt missing (broken symlink?)" >&2; exit 1; }

# Train 4 epochs.
python -m pretrain_experiments config/mia-finetune-fresh4ep-546M.yaml
