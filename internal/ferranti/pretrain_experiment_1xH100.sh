#!/bin/bash
#SBATCH --time=3-00:00:00
#SBATCH --output=/weka/luxburg/sbordt10/logs/pretrain-experiment/%j.out
#SBATCH --error=/weka/luxburg/sbordt10/logs/pretrain-experiment/%j.err
#SBATCH --open-mode=append
#SBATCH --job-name=pretrain-exp-1xH100
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --gres=gpu:1

scontrol show job ${SLURM_JOB_ID}
nvidia-smi
export NCCL_TIMEOUT=1800000
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export WANDB__SERVICE_WAIT=6000

cd /weka/luxburg/sbordt10/pretrain-experiments/pretrain-experiments
conda activate pretrain-experiments

pretrain-experiments "$@" --save_folder /weka/luxburg/sbordt10/pretrain-experiments
