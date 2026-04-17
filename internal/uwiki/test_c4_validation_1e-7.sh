#!/bin/bash
#SBATCH --time=0-01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --job-name=gn-c4val-1e-7
#SBATCH --account=datamining
#SBATCH --partition=p_datamining
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

export INFERENCE_MAX_NUM_SEQS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /etc/profile.d/modules.sh
export ENV_MODE="permanent"
export ENV_NAME="pretrain-experiments"
module load miniforge

cd ~/pretrain-experiments

# Create 1000-example subset
head -1000 resources/validation-set/c4_en_validation.jsonl > /tmp/c4_val_1000.jsonl

echo "=== c4_validation on 1e-7 step 110000 (1000 examples) ==="
python pretrain_experiments/evaluation/perplexity.py \
    --model ~/pretrain-experiments/unlearning-gradient-noise/OLMo-2-179M-Exp-gradient-noise-uij1rwaw/step110000-hf \
    --task-file /tmp/c4_val_1000.jsonl

echo "Exit code: $?"
echo "=== Done ==="
