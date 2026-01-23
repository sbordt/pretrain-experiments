#!/bin/bash
#SBATCH --time=3-00:00:00  # Runtime in D-HH:MM:SS    
#SBATCH --output=/mnt/lustre/work/luxburg/sbordt10/logs/pretrain-experiments/%j.out  
#SBATCH --error=/mnt/lustre/work/luxburg/sbordt10/logs/pretrain-experiments/%j.err   
#SBATCH --open-mode=append
#SBATCH --job-name=pretrain-exp-4xA100  
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

cd /mnt/lustre/work/luxburg/sbordt10/pretrain-experiments
source activate pretrain-experiments

pretrain-experiments "$@"