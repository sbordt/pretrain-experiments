#!/bin/bash
#SBATCH --time=3-00:00:00  # Runtime in D-HH:MM:SS    
#SBATCH --output=/mnt/lustre/work/luxburg/sbordt10/logs/pretrain-experiment/%j.out  
#SBATCH --error=/mnt/lustre/work/luxburg/sbordt10/logs/pretrain-experiment/%j.err   
#SBATCH --open-mode=append
#SBATCH --job-name=olmo  
#SBATCH --partition=a100-galvani
#SBATCH --nodes=1  
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24    
#SBATCH --mem=512G   
#SBATCH --gres=gpu:4              

scontrol show job ${SLURM_JOB_ID}
nvidia-smi
export NCCL_TIMEOUT=1800000
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export WANDB__SERVICE_WAIT=6000
export OLMO_SHARED_FS=1
#export NCCL_DEBUG=INFO

export OLMO_PRIVATE_PATH="/mnt/lustre/work/luxburg/sbordt10/OLMo-Private"
export OLMo2_1B_stage1_SCRIPT="/mnt/lustre/work/luxburg/sbordt10/OLMo-Private/single-training-run/scripts/galvani/OLMo2-1B-stage1.yaml"
export OLMo2_1B_stage1_all_small_ppl_validation="/mnt/lustre/work/luxburg/sbordt10/OLMo-Private/single-training-run/scripts/galvani/OLMo2-1B-stage1-all-small-ppl-validation.yaml"
export EXPERIMENTS_SAVE_PATH="/mnt/lustre/work/luxburg/sbordt10/single_training_run"

export INFERENCE_DEFAULTS_PATH="/mnt/lustre/work/luxburg/sbordt10/OLMo-Private/single-training-run/scripts/galvani/inference_defaults.yaml"

cd /mnt/lustre/work/luxburg/sbordt10/OLMo-Private/single-training-run/code/scripts/
source activate olmo-2

python pretrain_experiment.py "$@"