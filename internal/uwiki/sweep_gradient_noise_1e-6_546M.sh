#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#SBATCH --job-name=gn-sweep-1e-6-546M
#SBATCH --account=datamining
#SBATCH --partition=p_datamining
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2

scontrol show job ${SLURM_JOB_ID}
nvidia-smi

export NCCL_TIMEOUT=1800000
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export WANDB__SERVICE_WAIT=6000
export OLMO_SHARED_FS=1
export INFERENCE_MAX_NUM_SEQS=${INFERENCE_MAX_NUM_SEQS:-16}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset SSL_CERT_FILE
export WANDB_API_KEY="${WANDB_API_KEY:?set WANDB_API_KEY in your shell before sbatch}"

# Environment setup (miniforge per cluster policy)
source /etc/profile.d/modules.sh
export ENV_MODE="permanent"
export ENV_NAME="pretrain-experiments"
module load miniforge

cd ~/pretrain-experiments

# Ensure dependencies are installed
pip install -q h5py "transformers<4.52" "accelerate>=0.26.0"
pip install -q torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
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

python -m pretrain_experiments config/unlearning-gradient-noise-546M.yaml \
    --training.num_steps 10000 \
    --training.checkpoint_interval 1000 \
    --experiments.experiments.0.noise_std 1.0e-6
