#!/bin/bash
#SBATCH --time=3-00:00:00  # Runtime in D-HH:MM:SS    
#SBATCH --output=/weka/luxburg/sbordt10/logs/pretrain-experiments/%j.out  
#SBATCH --error=/weka/luxburg/sbordt10/logs/pretrain-experiments/%j.err   
#SBATCH --open-mode=append
#SBATCH --job-name=pretrain-experiments
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1  
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36    
#SBATCH --mem=786G   
#SBATCH --gres=gpu:4              

scontrol show job ${SLURM_JOB_ID}
nvidia-smi

cd /weka/luxburg/sbordt10/pretrain-experiments

singularity exec --nv \
  --bind /weka/luxburg/sbordt10:/weka/luxburg/sbordt10 \
  --env NCCL_TIMEOUT=1800000 \
  --env TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800 \
  --env WANDB__SERVICE_WAIT=6000 \
  --env OLMES_EXECUTABLE=~/venvs/olmes/bin/olmes \
  pretrain-experiments.sif \
  python -m pretrain_experiments "$@"