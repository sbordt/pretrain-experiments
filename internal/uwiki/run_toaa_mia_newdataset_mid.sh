#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --open-mode=append
#SBATCH --job-name=toaa-mia-mid
#SBATCH --account=datamining
#SBATCH --partition=p_datamining
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --exclude=vader,galadriel

# TOAA-MIA new-dataset eval for the MID-TRAINING (stage2) model family, mirroring
# the stage1 179M/546M/1B/2.7B runs. One job (or array task) == one target
# checkpoint, 30 conditions each, written to
#   evals/toaa-mia-newdataset/<SIZE>-Mid/<family>/step-<N>/results_<cond>.json
# See CLAUDE.md "Experiment 5: Mid-training (stage2) canary insertion".
#
# families (no baseline -- earliest stage2 step is the analog):
#   exp-mid          -> sbordt/OLMo-2-<SIZE>-Exp-Mid   (saw canaries; unlearning-baseline analog)
#   deep-ignorance   -> sbordt/OLMo-2-<SIZE>-Mid       (never saw canaries)
# Grid: stage2-step {1000,3000,5000,7000,9000,11000}, 6 per family, 12 targets/size.
#
# --reference_model auto -> sbordt/OLMo-2-<SIZE> by parameter count (same as the
# stage1 runs). The reference cache is NOT pre-populated: warm it once
# (TARGET_IDX=0) then fan out the array (indices 1..11, --dependency=afterok) so
# tasks reuse the cache rather than recomputing and racing. See the submit note
# at the bottom. All targets are pulled as HF revisions (stage2-step<N> branches
# are HF-format); no unsharded->HF conversion is needed.
#
# Usage:
#   MODEL=179M ... (see submit note)
#
# Env vars:
#   MODEL=179M|546M|1B|2.7B  -> model size (required)
#   TARGET_IDX=<n>           -> target index for the warmup (non-array) run

scontrol show job ${SLURM_JOB_ID} 2>/dev/null || true
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L || true

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset SSL_CERT_FILE

source /etc/profile.d/modules.sh
export ENV_MODE="permanent"
export ENV_NAME="pretrain-experiments"
module load miniforge

cd ~/pretrain-experiments

export PYTHONPATH="$PWD:$HOME/.local/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
python -c "
import torch, transformers, datasets
print('torch', torch.__version__, '| transformers', transformers.__version__, '| datasets', datasets.__version__)
assert torch.cuda.is_available(), 'CUDA not available on '+__import__('socket').gethostname()
print('cuda', torch.version.cuda, '| device', torch.cuda.get_device_name(0), '| count', torch.cuda.device_count())
" || { echo "ERROR: env/CUDA preflight failed on $(hostname)" >&2; exit 1; }

set -u
set -o pipefail

TOAA_DIR=pretrain_experiments/evaluation/train-once-answer-all
CACHE_DIR=${MIA_CACHE_DIR:-$HOME/.cache/huggingface}

MODEL="${MODEL:?set MODEL=179M|546M|1B|2.7B}"
case "$MODEL" in
  179M|546M|1B|2.7B) SIZE=$MODEL ;;
  *) echo "ERROR: MODEL must be 179M|546M|1B|2.7B (got '$MODEL')" >&2; exit 2 ;;
esac

OUT_BASE=~/pretrain-experiments/evals/toaa-mia-newdataset/${SIZE}-Mid
REF_CACHE_DIR=${MIA_REF_CACHE_DIR:-$OUT_BASE/_ref_cache}
mkdir -p "$REF_CACHE_DIR"

HFEXP=sbordt/OLMo-2-${SIZE}-Exp-Mid   # exp-mid (saw canaries; unlearning-baseline analog)
HFDI=sbordt/OLMo-2-${SIZE}-Mid        # deep-ignorance (never saw canaries)

# label | model_repo | model_revision. index 0 = exp-mid step1000 = the warmup
# target. exp-mid first (idx 0-5), then deep-ignorance (6-11).
STEPS=(1000 3000 5000 7000 9000 11000)
TARGETS=()
for st in "${STEPS[@]}"; do TARGETS+=("exp-mid-${st}|$HFEXP|stage2-step${st}"); done
for st in "${STEPS[@]}"; do TARGETS+=("deep-ignorance-${st}|$HFDI|stage2-step${st}"); done

CONDITIONS=(
  plain_1x plain_4x plain_16x
  rare_1tok_1x rare_1tok_4x rare_1tok_16x
  rare_8tok_1x rare_8tok_4x rare_8tok_16x
  rare_32tok_1x rare_32tok_4x rare_32tok_16x
  model_based_1tok_1x model_based_1tok_4x model_based_1tok_16x
  model_based_8tok_1x model_based_8tok_4x model_based_8tok_16x
  model_based_32tok_1x model_based_32tok_4x model_based_32tok_16x
  random_1tok_1x random_1tok_4x random_1tok_16x
  random_8tok_1x random_8tok_4x random_8tok_16x
  random_32tok_1x random_32tok_4x random_32tok_16x
)
if [ -n "${CONDITIONS_OVERRIDE:-}" ]; then
  read -r -a CONDITIONS <<< "$CONDITIONS_OVERRIDE"
fi

IDX=${SLURM_ARRAY_TASK_ID:-${TARGET_IDX:-0}}
spec="${TARGETS[$IDX]}"
IFS='|' read -r LABEL MODEL_DIR MODEL_REV <<< "$spec"
STEP="${LABEL##*-}"; FAM="${LABEL%-*}"
OUT_DIR="$OUT_BASE/$FAM/step-$STEP"
mkdir -p "$OUT_DIR"

echo "============================================================"
echo "  TOAA-MIA ${SIZE}-Mid | task $IDX -> $LABEL  ($FAM/step-$STEP)"
echo "  model:     $MODEL_DIR @ $MODEL_REV"
echo "  out_dir:   $OUT_DIR"
echo "  ref_cache: $REF_CACHE_DIR"
echo "============================================================"

for cond in "${CONDITIONS[@]}"; do
  done_marker="$OUT_DIR/${cond}.done"
  if [ -f "$done_marker" ]; then
    echo "  [$LABEL/$cond] already done, skipping"
    continue
  fi
  echo "  --- $LABEL / $cond ---"
  start=$(date +%s)
  python "$TOAA_DIR/newtoken_mia.py" \
    --model_dir "$MODEL_DIR" \
    --model_revision "$MODEL_REV" \
    --target_experiment "$cond" \
    --reference_model auto \
    --reference_cache_dir "$REF_CACHE_DIR" \
    --results_dir "$OUT_DIR" \
    --cache_dir "$CACHE_DIR" \
    --batch_size 32
  rc=$?
  if [ "$rc" -eq 0 ]; then
    src=$(ls -t "$OUT_DIR"/results_mia_samples_*_"$cond".json 2>/dev/null | head -1)
    [ -n "$src" ] && [ "$src" != "$OUT_DIR/results_${cond}.json" ] && mv -f "$src" "$OUT_DIR/results_${cond}.json"
    touch "$done_marker"
  fi
  echo "  [$LABEL/$cond] rc=$rc ($(( $(date +%s) - start ))s)"
done

echo
echo "============================================================"
echo "  DONE [${SIZE}-Mid / $LABEL]. Results under $OUT_DIR/"
echo "============================================================"

# ---------------------------------------------------------------------------
# Submit note (run from internal/uwiki/). Per size, warm the ref cache then fan
# out the remaining 11 targets on afterok:
#   for M in 179M 546M 1B 2.7B; do
#     warm=$(sbatch --parsable --job-name=toaa-mia-${M}-mid-warm \
#                   --export=ALL,MODEL=$M,TARGET_IDX=0 run_toaa_mia_newdataset_mid.sh)
#     sbatch --job-name=toaa-mia-${M}-mid --export=ALL,MODEL=$M \
#            --dependency=afterok:$warm --array=1-11 run_toaa_mia_newdataset_mid.sh
#   done
# ---------------------------------------------------------------------------
