#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --open-mode=append
#SBATCH --job-name=toaa-mia-newds-2.7B
#SBATCH --account=datamining
#SBATCH --partition=p_datamining
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --exclude=vader,galadriel

# TOAA-MIA new-dataset eval for the 2.7B model family, mirroring the 179M/546M/1B
# runs. One job (or array task) == one target checkpoint, 30 conditions each,
# written to evals/toaa-mia-newdataset/2.7B/<family>/step-<N>/results_<cond>.json
#
# --reference_model auto -> sbordt/OLMo-2-2.7B by parameter count (the
# resolve_reference_model 2.7B branch was added alongside this driver). The
# reference cache is NOT pre-populated: warm it once (TARGET_IDX=0, baseline) then
# fan out the array (indices 1..N, --dependency=afterok) so tasks reuse the cache
# rather than recomputing and racing on the cache files. See the submit note at
# the bottom of this file.
#
# families: baseline, unlearning-baseline, deep-ignorance. There is NO gradient-
# noise family here -- the 2.7B GN sweep has not been run yet (see CLAUDE.md
# "Experiment 4"). Unlike the 1B run, NO targets are pre-converted locally, so all
# of them are pulled as HF revisions (the stage1-step<N> branches are HF-format
# checkpoints; step<N>-unsharded are OLMo-native and not used here).

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
" || { echo \"ERROR: env/CUDA preflight failed on $(hostname)\" >&2; exit 1; }

set -u
set -o pipefail

TOAA_DIR=pretrain_experiments/evaluation/train-once-answer-all
CACHE_DIR=${MIA_CACHE_DIR:-$HOME/.cache/huggingface}
OUT_BASE=~/pretrain-experiments/evals/toaa-mia-newdataset/2.7B
REF_CACHE_DIR=${MIA_REF_CACHE_DIR:-$OUT_BASE/_ref_cache}
mkdir -p "$REF_CACHE_DIR"

HFUB=sbordt/OLMo-2-2.7B-Exp-Unlearning   # baseline + unlearning-baseline (members seen)
HFDI=sbordt/OLMo-2-2.7B-Unlearning       # deep-ignorance ground truth (members never seen)

# label | model_repo | model_revision   (index 0 = baseline = the warmup target)
# baseline / *-110000 use the tokens-suffixed branch names; intermediate steps use
# the bare stage1-step<N> branches (per the HF refs listing for both repos).
TARGETS=(
  "baseline-100000|$HFUB|stage1-step100000-tokens210B"
  "unlearning-baseline-102000|$HFUB|stage1-step102000"
  "unlearning-baseline-104000|$HFUB|stage1-step104000"
  "unlearning-baseline-106000|$HFUB|stage1-step106000"
  "unlearning-baseline-108000|$HFUB|stage1-step108000"
  "unlearning-baseline-110000|$HFUB|stage1-step110000-tokens231B"
  "deep-ignorance-100000|$HFDI|stage1-step100000-tokens210B"
  "deep-ignorance-102000|$HFDI|stage1-step102000"
  "deep-ignorance-104000|$HFDI|stage1-step104000"
  "deep-ignorance-106000|$HFDI|stage1-step106000"
  "deep-ignorance-108000|$HFDI|stage1-step108000"
  "deep-ignorance-110000|$HFDI|stage1-step110000-tokens231B"
)

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
# Layout: $OUT_BASE/<family>/step-<N>/  (deep-ignorance-104000 -> deep-ignorance/step-104000).
STEP="${LABEL##*-}"; FAM="${LABEL%-*}"
OUT_DIR="$OUT_BASE/$FAM/step-$STEP"
mkdir -p "$OUT_DIR"

if [[ "$MODEL_DIR" == sbordt/* ]]; then
  :  # HF repo id
elif [ ! -e "$MODEL_DIR/config.json" ]; then
  echo "ERROR: local model_dir not found: $MODEL_DIR" >&2; exit 3
fi

echo "============================================================"
echo "  TOAA-MIA 2.7B | task $IDX -> $LABEL  ($FAM/step-$STEP)"
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
    # unify output filename -> results_<cond>.json
    src=$(ls -t "$OUT_DIR"/results_mia_samples_*_"$cond".json 2>/dev/null | head -1)
    [ -n "$src" ] && [ "$src" != "$OUT_DIR/results_${cond}.json" ] && mv -f "$src" "$OUT_DIR/results_${cond}.json"
    touch "$done_marker"
  fi
  echo "  [$LABEL/$cond] rc=$rc ($(( $(date +%s) - start ))s)"
done

echo
echo "============================================================"
echo "  DONE [$LABEL]. Results under $OUT_DIR/"
echo "============================================================"

# ---------------------------------------------------------------------------
# Submit note (run from internal/uwiki/):
#   # 1) warm the reference cache with the baseline target (idx 0):
#   warm=$(sbatch --parsable --job-name=toaa-mia-newds-2.7B-warm \
#                 --export=ALL,TARGET_IDX=0 run_toaa_mia_newdataset_2.7B.sh)
#   # 2) fan out the remaining 11 targets once the cache is warm:
#   sbatch --dependency=afterok:$warm --array=1-11 run_toaa_mia_newdataset_2.7B.sh
# ---------------------------------------------------------------------------
