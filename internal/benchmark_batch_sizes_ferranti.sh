#!/bin/bash
#SBATCH --time=0-01:00:00
#SBATCH --output=/weka/luxburg/sbordt10/logs/pretrain-experiment/%j.out
#SBATCH --error=/weka/luxburg/sbordt10/logs/pretrain-experiment/%j.err
#SBATCH --open-mode=append
#SBATCH --job-name=bench-batch
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --gres=gpu:1

nvidia-smi

cd /weka/luxburg/sbordt10/pretrain-experiments

singularity exec --nv \
  --bind /weka/luxburg/sbordt10:/weka/luxburg/sbordt10 \
  pretrain-experiments.sif \
  bash -c 'export PYTHONPATH=/weka/luxburg/sbordt10/pretrain-experiments/pretrain-experiments:$PYTHONPATH && cd /weka/luxburg/sbordt10/pretrain-experiments/pretrain-experiments && python internal/benchmark_batch_sizes.py "$@"' -- "$@"
