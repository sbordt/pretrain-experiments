#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#SBATCH --job-name=toaa-mia-newds-179M
#SBATCH --account=datamining
#SBATCH --partition=p_datamining
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=vader

# First run of the rewritten newtoken_mia.py against the paired benchmark
# sbordt/TOAA-Membership-Inference, on three 179M checkpoints:
#   1) deep-ignorance        @ step102000  (never saw members -> expect AUC ~ 0.5)
#   2) Exp baseline          @ step100000  (saw members       -> expect AUC > 0.5)
#   3) Exp unlearning-baseline @ step110000 (continued on retained data -> partial)
# All 179M, so --reference_model auto resolves to sbordt/OLMo-2-179M; reference
# scores are computed once and reused across the three targets via the shared
# --reference_cache_dir.

scontrol show job ${SLURM_JOB_ID} 2>/dev/null || true
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L || true
nvidia-smi || true

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset SSL_CERT_FILE

source /etc/profile.d/modules.sh
export ENV_MODE="permanent"
export ENV_NAME="pretrain-experiments"
module load miniforge

cd ~/pretrain-experiments

export PYTHONPATH="$PWD:$HOME/.local/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
# Fail fast at the job level if CUDA is down on this node, instead of silently
# scoring every condition on CPU (which takes hours).
python -c "
import torch, transformers, datasets
print('torch', torch.__version__, '| transformers', transformers.__version__, '| datasets', datasets.__version__)
assert torch.cuda.is_available(), 'CUDA not available on '+__import__('socket').gethostname()
print('cuda', torch.version.cuda, '| device', torch.cuda.get_device_name(0), '| count', torch.cuda.device_count())
" || { echo \"ERROR: env/CUDA preflight failed on $(hostname)\" >&2; exit 1; }

set -u
set -o pipefail

TOAA_DIR=pretrain_experiments/evaluation/train-once-answer-all
CACHE_DIR=${MIA_CACHE_DIR:-$HOME/.cache/huggingface}
OUT_BASE=~/pretrain-experiments/evals/toaa-mia-newdataset/179M
REF_CACHE_DIR=${MIA_REF_CACHE_DIR:-$OUT_BASE/_ref_cache}
mkdir -p "$REF_CACHE_DIR"

# label | model_dir | model_revision
TARGETS=(
  "deep-ignorance-102000|sbordt/OLMo-2-179M-Unlearning|stage1-step102000"
  "baseline-100000|sbordt/OLMo-2-179M-Exp-Unlearning|stage1-step100000-tokens210B"
  "unlearning-baseline-110000|sbordt/OLMo-2-179M-Exp-Unlearning|stage1-step110000-tokens231B"
)

# All 30 conditions (dataset `condition` strings; newtoken_mia.py also accepts
# the old memorization-patterns-* keys).
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

# Optional override: CONDITIONS_OVERRIDE="plain_16x rare_8tok_16x ..."
if [ -n "${CONDITIONS_OVERRIDE:-}" ]; then
  read -r -a CONDITIONS <<< "$CONDITIONS_OVERRIDE"
fi

echo "============================================================"
echo "  TOAA-MIA new-dataset run | ${#TARGETS[@]} models x ${#CONDITIONS[@]} conditions"
echo "  out_base:  $OUT_BASE"
echo "  ref_cache: $REF_CACHE_DIR"
echo "============================================================"

for spec in "${TARGETS[@]}"; do
  IFS='|' read -r LABEL MODEL_DIR MODEL_REV <<< "$spec"
  # Layout: $OUT_BASE/<family>/step-<N>/  (e.g. deep-ignorance-102000 ->
  # deep-ignorance/step-102000, baseline-100000 -> baseline/step-100000).
  STEP="${LABEL##*-}"; FAM="${LABEL%-*}"; FAM="${FAM#gn-}"
  OUT_DIR="$OUT_BASE/$FAM/step-$STEP"
  mkdir -p "$OUT_DIR"
  echo
  echo "############ $LABEL  ($MODEL_DIR @ $MODEL_REV) ############"
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
      --batch_size 64
    rc=$?
    if [ "$rc" -eq 0 ]; then
      # unify output filename -> results_<cond>.json (the eval names it
      # results_mia_samples_model<...>_<cond>.json, which varies per model)
      src=$(ls -t "$OUT_DIR"/results_mia_samples_*_"$cond".json 2>/dev/null | head -1)
      [ -n "$src" ] && [ "$src" != "$OUT_DIR/results_${cond}.json" ] && mv -f "$src" "$OUT_DIR/results_${cond}.json"
      touch "$done_marker"
    fi
    echo "  [$LABEL/$cond] rc=$rc ($(( $(date +%s) - start ))s)"
  done
done

echo
echo "============================================================"
echo "  DONE. Results under $OUT_BASE/<label>/"
echo "============================================================"
