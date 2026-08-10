#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#SBATCH --job-name=gn-resume-1e-6
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset SSL_CERT_FILE
export WANDB_API_KEY="${WANDB_API_KEY:?set WANDB_API_KEY in your shell before sbatch}"

source /etc/profile.d/modules.sh
export ENV_MODE="permanent"
export ENV_NAME="pretrain-experiments"
module load miniforge

cd ~/pretrain-experiments

pip install -q h5py "transformers<4.52" "accelerate>=0.26.0"
pip install -q torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -q -e ".[eval]"
pip install -q -e ~/OLMo"[all]" 2>/dev/null || true

RUN_DIR=/srv/home/users/martinp27cs/pretrain-experiments/unlearning-gradient-noise/OLMo-2-179M-Exp-gradient-noise-h5p8rdiz
LOAD_CKPT=${RUN_DIR}/step108000-unsharded
BASE_CONFIG=/srv/home/users/martinp27cs/pretrain-experiments/checkpoints/step100000-unsharded/config.yaml

export OLMO_GRADIENT_NOISE_CONFIG_FILE=${RUN_DIR}/gradient_noise_config.pkl

echo "Resuming from ${LOAD_CKPT}"
echo "Gradient noise config: ${OLMO_GRADIENT_NOISE_CONFIG_FILE}"
python -c "import pickle; print(pickle.load(open('${OLMO_GRADIENT_NOISE_CONFIG_FILE}','rb')))"

torchrun --nproc_per_node=2 --master_port=29503 \
    /srv/home/users/martinp27cs/OLMo/scripts/train.py \
    ${BASE_CONFIG} \
    --save_folder=${RUN_DIR} \
    --save_overwrite=True \
    --save_interval=null \
    --save_interval_unsharded=1000 \
    --stop_at=110000 \
    --eval_on_load=False \
    --wandb.project=unlearning-gradient-noise-OLMo \
    --wandb.entity=ai4life_pawel \
    --wandb.name=gn-1e-6-resume-step108000 \
    --load_path=${LOAD_CKPT}
