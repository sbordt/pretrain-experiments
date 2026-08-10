#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#SBATCH --job-name=mia-179M-baseline-all
#SBATCH --account=datamining
#SBATCH --partition=p_datamining
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=vader,galadriel

# Run all 30 memorization-patterns MIA experiments for the 179M baseline at
# step 100000 using the rewritten newtoken_mia.py (paired HF dataset, transformers backend).
# Output: evals/gn-eval3-sweep-fresh/179M/baseline/step-100000/memorization_patterns_mia/
# (separate from gn-eval3-sweep/ so old broken results stay untouched).
# Per-experiment .done markers so the job is resumable on requeue.

scontrol show job ${SLURM_JOB_ID} 2>/dev/null || true
nvidia-smi || true
echo "  HOST: $(hostname)"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset SSL_CERT_FILE

source /etc/profile.d/modules.sh
export ENV_MODE="permanent"
export ENV_NAME="pretrain-experiments"
module load miniforge

cd ~/pretrain-experiments

export PYTHONPATH="$PWD:$HOME/.local/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"

set -o pipefail

# CUDA sanity — refuses to start on a node where the GPU is unreachable.
python -c "import torch
assert torch.cuda.is_available(), 'CUDA not available'
print('  cuda devices:', torch.cuda.device_count(), torch.cuda.get_device_name(0))
torch.empty(8, device='cuda:0').sum().item()" \
  || { echo "ERROR: CUDA unreachable on $(hostname)" >&2; exit 1; }

set -u

TOAA_DIR=pretrain_experiments/evaluation/train-once-answer-all
TOKENIZER=~/OLMo/olmo_data/tokenizers/allenai_dolma2.json

MODEL_DIR="sbordt/OLMo-2-179M-Exp-Unlearning"
MODEL_REVISION="stage1-step100000-tokens210B"
LABEL="baseline"
STEP="100000"

MIA_DATA_IN=${MIA_DATA_IN:-$HOME/pretrain-experiments/mia-data/memorization-patterns.jsonl}
MIA_DATA_OUT_PKL=${MIA_DATA_OUT_PKL:-$HOME/pretrain-experiments/mia-data/memorization-patterns-holdout.pkl}
MIA_CACHE_DIR=${MIA_CACHE_DIR:-$HOME/.cache/huggingface}

OUT_DIR=~/pretrain-experiments/evals/gn-eval3-sweep-fresh/179M/$LABEL/step-$STEP/memorization_patterns_mia
mkdir -p "$OUT_DIR"

# preflight removed: newtoken_mia.py now reads the paired HF dataset
# (sbordt/TOAA-Membership-Inference); no local jsonl/pkl/tokenizer required.

MIA_EXPS=(
  memorization-patterns-plain-1x
  memorization-patterns-plain-4x
  memorization-patterns-plain-16x
  memorization-patterns-rare-1-token-1x
  memorization-patterns-rare-1-token-4x
  memorization-patterns-rare-1-token-16x
  memorization-patterns-rare-8-tokens-1x
  memorization-patterns-rare-8-tokens-4x
  memorization-patterns-rare-8-tokens-16x
  memorization-patterns-rare-32-tokens-1x
  memorization-patterns-rare-32-tokens-4x
  memorization-patterns-rare-32-tokens-16x
  memorization-patterns-model-based-1-token-1x
  memorization-patterns-model-based-1-token-4x
  memorization-patterns-model-based-1-token-16x
  memorization-patterns-model-based-8-tokens-1x
  memorization-patterns-model-based-8-tokens-4x
  memorization-patterns-model-based-8-tokens-16x
  memorization-patterns-model-based-32-tokens-1x
  memorization-patterns-model-based-32-tokens-4x
  memorization-patterns-model-based-32-tokens-16x
  memorization-patterns-random-1-token-1x
  memorization-patterns-random-1-token-4x
  memorization-patterns-random-1-token-16x
  memorization-patterns-random-8-tokens-1x
  memorization-patterns-random-8-tokens-4x
  memorization-patterns-random-8-tokens-16x
  memorization-patterns-random-32-tokens-1x
  memorization-patterns-random-32-tokens-4x
  memorization-patterns-random-32-tokens-16x
)

TOTAL=${#MIA_EXPS[@]}
echo "============================================"
echo "  179M baseline MIA sweep: $TOTAL experiments"
echo "  out_dir: $OUT_DIR"
echo "============================================"

i=0
for exp in "${MIA_EXPS[@]}"; do
  i=$((i + 1))
  done_marker="$OUT_DIR/${exp}.done"
  if [ -f "$done_marker" ]; then
    echo "[$i/$TOTAL] $exp -- already done, skipping"
    continue
  fi
  echo
  echo "[$i/$TOTAL] $exp -- running"
  start=$(date +%s)
  python "$TOAA_DIR/newtoken_mia.py" \
    --model_dir         "$MODEL_DIR" \
    --model_revision    "$MODEL_REVISION" \
    --data_in_file      "$MIA_DATA_IN" \
    --data_out_file     "$MIA_DATA_OUT_PKL" \
    --target_experiment "$exp" \
    --results_dir       "$OUT_DIR" \
    --cache_dir         "$MIA_CACHE_DIR" \
    --reference_cache_dir "${MIA_REF_CACHE_DIR:-$MIA_CACHE_DIR/toaa_mia_ref_cache}" \
    && touch "$done_marker"
  rc=$?
  end=$(date +%s)
  echo "[$i/$TOTAL] $exp -- rc=$rc ($((end - start))s)"
done

echo
echo "============================================"
echo "  SWEEP SUMMARY"
echo "============================================"
done_n=$(ls "$OUT_DIR"/*.done 2>/dev/null | wc -l)
json_n=$(ls "$OUT_DIR"/*.json 2>/dev/null | wc -l)
echo "  done markers : $done_n / $TOTAL"
echo "  json outputs : $json_n / $TOTAL"
if [ "$done_n" -lt "$TOTAL" ]; then
  echo "  -- not all experiments completed; resubmit the same script to resume --"
  exit 1
fi
echo "  all 30 experiments completed."
