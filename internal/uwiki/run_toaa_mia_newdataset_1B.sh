#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --open-mode=append
#SBATCH --job-name=toaa-mia-newds-1B
#SBATCH --account=datamining
#SBATCH --partition=p_datamining
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --exclude=vader,galadriel

# TOAA-MIA new-dataset eval for the 1B model family, mirroring the 179M/546M runs.
# One job (or array task) == one target checkpoint, 30 conditions each, written to
#   evals/toaa-mia-newdataset/1B/<family>/step-<N>/results_<cond>.json
#
# --reference_model auto -> sbordt/OLMo-2-1B by parameter count. Reference cache
# is NOT pre-populated: warm it once (TARGET_IDX=0, baseline) then fan out the
# array (indices 1..N, --dependency=afterok) so tasks reuse the cache rather than
# recomputing and racing on the cache files.
#
# families: baseline, unlearning-baseline, deep-ignorance, 3.5e-6 (run 9050f9m3)
# Targets are local pre-converted HF dirs (hidden_size=2048 verified), EXCEPT the
# unlearning-baseline intermediate steps 102-108k, which are not pre-converted
# locally and are pulled as HF revisions stage1-step<N> of OLMo-2-1B-Exp-Unlearning.

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
OUT_BASE=~/pretrain-experiments/evals/toaa-mia-newdataset/1B
REF_CACHE_DIR=${MIA_REF_CACHE_DIR:-$OUT_BASE/_ref_cache}
mkdir -p "$REF_CACHE_DIR"

DI=checkpoints/1B-Unlearning
UB=checkpoints/1B-Exp-Unlearning
HFUB=sbordt/OLMo-2-1B-Exp-Unlearning
G35=unlearning-gradient-noise/OLMo-2-1B-Exp-gradient-noise-9050f9m3

# label | model_dir | model_revision   (index 0 = baseline = the warmup target)
TARGETS=(
  "baseline-100000|$UB/step100000-hf|main"
  "unlearning-baseline-102000|$HFUB|stage1-step102000"
  "unlearning-baseline-104000|$HFUB|stage1-step104000"
  "unlearning-baseline-106000|$HFUB|stage1-step106000"
  "unlearning-baseline-108000|$HFUB|stage1-step108000"
  "unlearning-baseline-110000|$UB/step110000-hf|main"
  "deep-ignorance-100000|$DI/deep-ignorance-stage1-step100000-tokens210B-hf|main"
  "deep-ignorance-102000|$DI/deep-ignorance-stage1-step102000-hf|main"
  "deep-ignorance-104000|$DI/deep-ignorance-stage1-step104000-hf|main"
  "deep-ignorance-106000|$DI/deep-ignorance-stage1-step106000-hf|main"
  "deep-ignorance-108000|$DI/deep-ignorance-stage1-step108000-hf|main"
  "deep-ignorance-110000|$DI/deep-ignorance-stage1-step110000-tokens231B-hf|main"
  "gn-3.5e-6-102000|$G35/step102000-hf|main"
  "gn-3.5e-6-104000|$G35/step104000-hf|main"
  "gn-3.5e-6-106000|$G35/step106000-hf|main"
  "gn-3.5e-6-108000|$G35/step108000-hf|main"
  "gn-3.5e-6-110000|$G35/step110000-hf|main"
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
# Layout: $OUT_BASE/<family>/step-<N>/  (gn-3.5e-6-104000 -> 3.5e-6/step-104000).
STEP="${LABEL##*-}"; FAM="${LABEL%-*}"; FAM="${FAM#gn-}"
OUT_DIR="$OUT_BASE/$FAM/step-$STEP"
mkdir -p "$OUT_DIR"

if [[ "$MODEL_DIR" == sbordt/* ]]; then
  :  # HF repo id
elif [ ! -e "$MODEL_DIR/config.json" ]; then
  echo "ERROR: local model_dir not found: $MODEL_DIR" >&2; exit 3
fi

echo "============================================================"
echo "  TOAA-MIA 1B | task $IDX -> $LABEL  ($FAM/step-$STEP)"
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
