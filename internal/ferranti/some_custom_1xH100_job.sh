#!/bin/bash
#SBATCH --time=3-00:00:00  # Runtime in D-HH:MM:SS    
#SBATCH --output=/weka/luxburg/sbordt10/logs/pretrain-experiment/%j.out  
#SBATCH --error=/weka/luxburg/sbordt10/logs/pretrain-experiment/%j.err   
#SBATCH --open-mode=append
#SBATCH --job-name=olmo  
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1  
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16    
#SBATCH --mem=256G   
#SBATCH --gres=gpu:1              

scontrol show job ${SLURM_JOB_ID}
nvidia-smi
export NCCL_TIMEOUT=1800000
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export WANDB__SERVICE_WAIT=6000
export OLMO_SHARED_FS=1


cd /weka/luxburg/sbordt10/
source /weka/luxburg/sbordt10/anaconda3/bin/activate pretrain-experiments

pip install flash-attn --no-build-isolation