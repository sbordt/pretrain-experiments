#!/bin/bash
#SBATCH --time=0-06:00:00  # Runtime in D-HH:MM:SS
#SBATCH --output=/mnt/lustre/work/luxburg/sbordt10/logs/pretrain-experiment/%j.out
#SBATCH --error=/mnt/lustre/work/luxburg/sbordt10/logs/pretrain-experiment/%j.err
#SBATCH --open-mode=append
#SBATCH --job-name=upload-checkpoint
#SBATCH --partition=a100-galvani
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:0

scontrol show job ${SLURM_JOB_ID}

cd /mnt/lustre/work/luxburg/sbordt10/pretrain-experiments
source activate pretrain-experiments

python -m pretrain_experiments.frameworks.olmo.upload_checkpoint "$@"
