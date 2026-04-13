#!/bin/bash
#SBATCH --time=0-02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#SBATCH --job-name=eval-olmo
#SBATCH --account=datamining
#SBATCH --partition=p_datamining
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

nvidia-smi

export WANDB_API_KEY="${WANDB_API_KEY:?set WANDB_API_KEY in your shell before sbatch}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset SSL_CERT_FILE

# Environment setup
source /etc/profile.d/modules.sh
export ENV_MODE="permanent"
export ENV_NAME="pretrain-experiments"
module load miniforge

cd ~/pretrain-experiments

# Fix transformers version
pip install -q "transformers==4.48.3"

MODEL_PATH=$1
RESULTS_DIR=$2

mkdir -p "$RESULTS_DIR"

echo "=== Running evaluations on $MODEL_PATH ==="

# fictional knowledge
echo "--- fictional_knowledge ---"
python pretrain_experiments/evaluation/train-once-answer-all/fictional_knowledge.py \
    --model "$MODEL_PATH" \
    --results-yaml "$RESULTS_DIR/fictional_knowledge.yaml" 2>&1

# verbatim memorization
echo "--- verbatim_memorization ---"
python pretrain_experiments/evaluation/train-once-answer-all/verbatim_memorization.py \
    --model "$MODEL_PATH" \
    --results-yaml "$RESULTS_DIR/verbatim_memorization.yaml" 2>&1

# prompt extraction
echo "--- prompt_extraction ---"
python pretrain_experiments/evaluation/train-once-answer-all/prompt_extraction.py \
    --model "$MODEL_PATH" \
    --results-yaml "$RESULTS_DIR/prompt_extraction.yaml" \
    --num-queries 1000 2>&1

# insertion likelihood
echo "--- insertion_likelihood ---"
python pretrain_experiments/evaluation/train-once-answer-all/insertion_likelihood.py \
    --model "$MODEL_PATH" \
    --results-yaml "$RESULTS_DIR/insertion_likelihood.yaml" 2>&1

echo "=== Done ==="
