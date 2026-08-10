#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#SBATCH --job-name=eval-ga-179M
#SBATCH --account=datamining
#SBATCH --partition=p_datamining
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=vader

# Evaluate the gradient-ascent sweep on 179M.
# For each cell × saved checkpoint, runs three evals:
#   - insertion_likelihood, filtered to the forget set
#   - insertion_likelihood, filtered to a held-out memorized set
#   - C4 validation perplexity (general capability)
#
# Optional env vars:
#   FORGET_EXPERIMENT  - forget-set experiment (default: memorization-patterns-rare-1-token-1x)
#   HELDOUT_EXPERIMENT - held-out memorized set  (default: benchmark-contamination-12x)
#   RUN_TAG            - sweep tag matching ga_unlearn_179M.sh (default: 179M-mp-rare-1tok-1x)
#   LRS                - space-separated LR list (default: "1e-6 5e-6")
#   BATCHES            - space-separated batch list (default: "16 32 64")
#   EPOCHS             - space-separated epoch list (default: "5 10 15 20")
#
# Existing results are skipped, so this is safe to rerun as new cells finish.

scontrol show job ${SLURM_JOB_ID} 2>/dev/null || true
nvidia-smi || true

export INFERENCE_MAX_NUM_SEQS=${INFERENCE_MAX_NUM_SEQS:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset SSL_CERT_FILE

source /etc/profile.d/modules.sh
export ENV_MODE="permanent"
export ENV_NAME="pretrain-experiments"
module load miniforge

cd ~/pretrain-experiments

export PYTHONPATH="$PWD:$HOME/.local/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
python -c "import torch, pretrain_experiments; print('torch', torch.__version__)" \
  || { echo "ERROR: torch / pretrain_experiments not importable on $(hostname)" >&2; exit 1; }

set -u
set -o pipefail

GA_ROOT="$HOME/pretrain-experiments/unlearning-gradient-ascent"
EVAL_ROOT="$HOME/pretrain-experiments/evals/ga-sweep-179M"
TASK_FILE="$HOME/pretrain-experiments/resources/validation-set/c4_en_validation.jsonl"

FORGET_EXPERIMENT="${FORGET_EXPERIMENT:-memorization-patterns-rare-1-token-1x}"
HELDOUT_EXPERIMENT="${HELDOUT_EXPERIMENT:-benchmark-contamination-12x}"
RUN_TAG="${RUN_TAG:-179M-mp-rare-1tok-1x}"

LRS=(${LRS:-1e-6 5e-6})
BATCHES=(${BATCHES:-16 32 64})
EPOCHS=(${EPOCHS:-5 10 15 20})

run_il () {
  local hf_dir=$1 experiment=$2 results_yaml=$3
  if [ -f "$results_yaml" ]; then
    echo "    [$experiment] already done, skipping"
    return 0
  fi
  python pretrain_experiments/evaluation/train-once-answer-all/insertion_likelihood.py \
    --model "$hf_dir" \
    --experiment "$experiment" \
    --results-yaml "$results_yaml" 2>&1
}

run_c4val () {
  local hf_dir=$1 results_yaml=$2 detailed_jsonl=$3
  if [ -f "$results_yaml" ] && [ -f "$detailed_jsonl" ]; then
    echo "    [c4_validation] already done, skipping"
    return 0
  fi
  python pretrain_experiments/evaluation/perplexity.py \
    --model "$hf_dir" \
    --task-file "$TASK_FILE" \
    --results-yaml "$results_yaml" \
    --detailed-results-jsonl "$detailed_jsonl" 2>&1
}

echo "============================================"
echo "  GA sweep eval"
echo "  run_tag:  $RUN_TAG"
echo "  forget:   $FORGET_EXPERIMENT"
echo "  heldout:  $HELDOUT_EXPERIMENT"
echo "  LRs:      ${LRS[*]}"
echo "  batches:  ${BATCHES[*]}"
echo "  epochs:   ${EPOCHS[*]}"
echo "============================================"

for lr in "${LRS[@]}"; do
  for b in "${BATCHES[@]}"; do
    cell_dir="$GA_ROOT/${RUN_TAG}/lr${lr}-b${b}"
    if [ ! -d "$cell_dir" ]; then
      echo "--- $cell_dir not found, skipping cell ---"
      continue
    fi
    for e in "${EPOCHS[@]}"; do
      hf_dir="$cell_dir/epoch-${e}"
      if [ ! -d "$hf_dir" ]; then
        echo "--- $hf_dir not found, skipping epoch ---"
        continue
      fi
      results_dir="$EVAL_ROOT/${RUN_TAG}/lr${lr}-b${b}/epoch-${e}"
      mkdir -p "$results_dir"

      echo ""
      echo "============================================"
      echo "  cell lr=$lr batch=$b   epoch=$e"
      echo "  hf:      $hf_dir"
      echo "  results: $results_dir"
      echo "============================================"

      run_il "$hf_dir" "$FORGET_EXPERIMENT"  "$results_dir/il_forget_${FORGET_EXPERIMENT}.yaml"
      run_il "$hf_dir" "$HELDOUT_EXPERIMENT" "$results_dir/il_heldout_${HELDOUT_EXPERIMENT}.yaml"
      run_c4val "$hf_dir" "$results_dir/c4_validation.yaml" "$results_dir/c4_validation.jsonl"
    done
  done
done

echo ""
echo "============================================"
echo "  RESULTS SUMMARY"
echo "============================================"
for cell_dir in "$EVAL_ROOT/${RUN_TAG}"/lr*-b*/; do
  cell=$(basename "$cell_dir")
  for epoch_dir in "$cell_dir"epoch-*/; do
    [ -d "$epoch_dir" ] || continue
    epoch=$(basename "$epoch_dir")
    echo ""
    echo "--- $cell @ $epoch ---"
    for f in "$epoch_dir"*.yaml; do
      [ -f "$f" ] || continue
      echo "  $(basename "$f"):"
      sed 's/^/    /' "$f"
    done
  done
done

echo ""
echo "============================================"
echo "  EVAL SWEEP DONE"
echo "============================================"
