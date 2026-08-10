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

# Fix transformers version for OLMo2 compatibility
pip install -q "transformers==4.48.3"
pip install -q h5py
pip install -q -e ".[eval]"

EXPERIMENT_DIR=~/pretrain-experiments/unlearning-gradient-noise/OLMo-2-179M-Exp-gradient-noise-6qkqwo5n

# Step 1: Convert baseline checkpoint to HF if needed
BASELINE_HF="$EXPERIMENT_DIR/step100000-hf"
if [ ! -f "$BASELINE_HF/model.safetensors" ]; then
    echo "Converting baseline checkpoint to HF format..."
    python ~/OLMo/scripts/convert_olmo2_to_hf.py \
        --input_dir ~/pretrain-experiments/checkpoints/step100000-unsharded \
        --output_dir "$BASELINE_HF" \
        --tokenizer_json_path ~/OLMo/olmo_data/tokenizers/allenai_dolma2.json \
        --no_tmp_cleanup
fi

# Step 2: Run evals on baseline (step 100000)
echo ""
echo "============================================"
echo "  EVALUATING BASELINE (step 100000)"
echo "============================================"
RESULTS_BASE="$EXPERIMENT_DIR/evals-step-100000"

for eval_name in fictional_knowledge verbatim_memorization prompt_extraction insertion_likelihood; do
    mkdir -p "$RESULTS_BASE/$eval_name"
    echo "--- $eval_name ---"
    EXTRA_ARGS=""
    if [ "$eval_name" = "prompt_extraction" ]; then
        EXTRA_ARGS="--num-queries 1000"
    fi
    python pretrain_experiments/evaluation/train-once-answer-all/${eval_name}.py \
        --model "$BASELINE_HF" \
        --results-yaml "$RESULTS_BASE/$eval_name/results.yaml" \
        $EXTRA_ARGS 2>&1
done

# Step 3: Run evals on trained model (step 100100)
echo ""
echo "============================================"
echo "  EVALUATING TRAINED MODEL (step 100100)"
echo "============================================"
TRAINED_HF="$EXPERIMENT_DIR/step100100-hf"
RESULTS_TRAINED="$EXPERIMENT_DIR/evals-step-100100"

for eval_name in fictional_knowledge verbatim_memorization prompt_extraction insertion_likelihood; do
    mkdir -p "$RESULTS_TRAINED/$eval_name"
    echo "--- $eval_name ---"
    EXTRA_ARGS=""
    if [ "$eval_name" = "prompt_extraction" ]; then
        EXTRA_ARGS="--num-queries 1000"
    fi
    python pretrain_experiments/evaluation/train-once-answer-all/${eval_name}.py \
        --model "$TRAINED_HF" \
        --results-yaml "$RESULTS_TRAINED/$eval_name/results.yaml" \
        $EXTRA_ARGS 2>&1
done

echo ""
echo "============================================"
echo "  RESULTS SUMMARY"
echo "============================================"
echo ""
echo "--- Baseline (step 100000) ---"
for f in "$RESULTS_BASE"/*/results.yaml; do
    echo "$(basename $(dirname $f)):"
    cat "$f" 2>/dev/null
    echo ""
done

echo "--- Trained (step 100100, noise_std=1e-8) ---"
for f in "$RESULTS_TRAINED"/*/results.yaml; do
    echo "$(basename $(dirname $f)):"
    cat "$f" 2>/dev/null
    echo ""
done
