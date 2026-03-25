#!/bin/bash
#SBATCH --time=0-01:00:00
#SBATCH --output=/mnt/lustre/work/luxburg/sbordt10/logs/pretrain-experiment/%j.out
#SBATCH --error=/mnt/lustre/work/luxburg/sbordt10/logs/pretrain-experiment/%j.err
#SBATCH --open-mode=append
#SBATCH --job-name=debug-dos
#SBATCH --partition=a100-galvani
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1

nvidia-smi
cd /mnt/lustre/work/luxburg/sbordt10/pretrain-experiments/pretrain-experiments
source activate pretrain-experiments

python internal/debug_dos_original.py
