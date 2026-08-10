#!/bin/bash
#SBATCH --time=2-12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#SBATCH --job-name=mia-ft-1B
# shelob (H200) fails NCCL init at the first barrier (Cuda failure 401,
# job 663899) -- pin to dgx-h100-em2 like the 546M 7.5e-6 sweep did.
#SBATCH --account=csunivie
#SBATCH --partition=p_csunivie_gres
#SBATCH --nodelist=dgx-h100-em2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:4

# MIA finetune: deep-ignorance 1B @100k, 10 true epochs over a fixed
# 1.007B-token set (fresh stage1 slice for steps 100000-100480 + 276k
# memorization-patterns canaries baked in at ~8% of tokens). OLMo reshuffles
# the sequence order every epoch; unsharded checkpoints every 2 epochs
# (steps 960/1920/2880/3840/4800).
# Prereq: mia-data/memorization-patterns-finetune-plain-random16x.jsonl
# (built by mia-data/build_finetune_subset.py).

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

# torch/torchrun live in user-site on some nodes (e.g. shelob/H200). Make sure
# both import path and bin are visible before we call torchrun via subprocess.
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD:$HOME/.local/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"

# Ensure dependencies are installed
pip install -q h5py "transformers<4.52" "accelerate>=0.26.0"
pip install -q "torch==2.7.*" --index-url https://download.pytorch.org/whl/cu126
pip install -q -e ".[eval]"

# Clone and install OLMo fork if not present
if [ ! -d ~/OLMo ]; then
    git clone https://github.com/sbordt/OLMo ~/OLMo
    cd ~/OLMo && git checkout pretrain-experiments
    pip install -q -e ".[all]"
    cd ~/pretrain-experiments
else
    pip install -q -e ~/OLMo"[all]" 2>/dev/null || true
fi

# Preflight: fail fast if torch isn't importable.
python -c "import torch, pretrain_experiments; print('torch', torch.__version__)" \
  || { echo "ERROR: torch / pretrain_experiments not importable on $(hostname)" >&2; exit 1; }

# Framework hardcodes 'torchrun' and calls it via subprocess.Popen. On some
# nodes (e.g. shelob) torchrun isn't visible to child processes even when it
# is on the login shell's PATH. Drop a shim that uses the torch module form,
# which only requires torch to be importable.
mkdir -p "$HOME/pretrain-experiments/.bin"
cat > "$HOME/pretrain-experiments/.bin/torchrun" <<'EOF'
#!/bin/bash
exec python -m torch.distributed.run "$@"
EOF
chmod +x "$HOME/pretrain-experiments/.bin/torchrun"
export PATH="$HOME/pretrain-experiments/.bin:$PATH"
command -v torchrun >/dev/null \
  || { echo "ERROR: torchrun shim not found on $(hostname)" >&2; exit 1; }

# Canary file must exist (built on the login node; pure-stdlib script).
CANARY_FILE=~/pretrain-experiments/mia-data/memorization-patterns-finetune-plain-random16x.jsonl
[ -s "$CANARY_FILE" ] || { echo "ERROR: $CANARY_FILE missing; run mia-data/build_finetune_subset.py" >&2; exit 1; }

# Step 1: convert the deep-ignorance HF checkpoint to unsharded format
# (idempotent, verifies key/shape/dtype parity against the Exp-Unlearning donor).
python internal/uwiki/convert_1B_deep_ignorance_to_unsharded.py \
  || { echo "ERROR: checkpoint conversion failed" >&2; exit 1; }

# Step 2: materialize the fixed finetuning dataset + train config (idempotent;
# streams ~4GB of the stage1 slice over HTTP and bakes the canaries in).
python internal/uwiki/build_mia_finetune_dataset_1B.py \
  || { echo "ERROR: dataset materialization failed" >&2; exit 1; }

# Step 3: train 10 epochs.
python -m pretrain_experiments config/mia-finetune-1B.yaml
