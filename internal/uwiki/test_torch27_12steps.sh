#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#SBATCH --job-name=test-torch27-12steps
#SBATCH --account=datamining
#SBATCH --partition=p_datamining
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2

# Smoke-test for torch 2.7 + cu126 upgrade: 12 training steps on the 546M
# config, with save_interval_ephemeral=6 so a sharded-ephemeral checkpoint
# fires within the run — that's the exact call path that was crashing on
# torch 2.6 with "DefaultSavePlanner got unexpected kwarg enable_plan_caching".
# Uses a distinct wandb name so it does not collide with the sweep run.

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

source /etc/profile.d/modules.sh
export ENV_MODE="permanent"
export ENV_NAME="pretrain-experiments"
module load miniforge

cd ~/pretrain-experiments

# Do NOT reinstall torch here — we want to exercise the 2.7 install we just
# did. Only light deps.
pip install -q h5py "transformers<4.52" "accelerate>=0.26.0"
pip install -q -e ".[eval]"
if [ ! -d ~/OLMo ]; then
    git clone https://github.com/sbordt/OLMo ~/OLMo
    cd ~/OLMo && git checkout pretrain-experiments
    pip install -q -e ".[all]"
    cd ~/pretrain-experiments
else
    pip install -q -e ~/OLMo"[all]" 2>/dev/null || true
fi

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

python -m pretrain_experiments config/unlearning-gradient-noise-546M.yaml \
    --training.num_steps 12 \
    --training.checkpoint_interval 12 \
    --training.args.save_interval_ephemeral 6 \
    --experiments.experiments.0.noise_std 1.0e-7 \
    --wandb.name test-torch27-12steps
