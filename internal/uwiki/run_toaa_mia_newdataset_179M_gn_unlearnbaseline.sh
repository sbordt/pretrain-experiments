#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --open-mode=append
#SBATCH --job-name=toaa-mia-newds-179M-gn
#SBATCH --account=datamining
#SBATCH --partition=p_datamining
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=vader,galadriel
#SBATCH --array=0-18%8

# Extend the rewritten newtoken_mia.py / sbordt/TOAA-Membership-Inference eval
# (same pipeline as run_toaa_mia_newdataset_179M_3models.sh) to the remaining
# 179M unlearning checkpoints. One array task == one target checkpoint, 30
# conditions each (~63 min/task: reference scores are already cached, so each
# condition is target-scoring only).
#
#   array  0- 3 : Exp unlearning-baseline @ stage1-step{102,104,106,108}000
#                 (100k baseline + 110k already done in the 3-model run)
#   array  4- 8 : gradient-noise sigma=1e-5 (run flvann74) @ step{102..110}000
#   array  9-13 : gradient-noise sigma=1e-6 (run h5p8rdiz) @ step{102..110}000
#   array 14-18 : gradient-noise sigma=1e-7 (run uij1rwaw) @ step{102..110}000
#
# All 179M -> --reference_model auto resolves to sbordt/OLMo-2-179M by parameter
# count, and the shared --reference_cache_dir (already fully populated with 30
# conditions by the 3-model run) is reused, so no reference scoring happens.

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

GN5=unlearning-gradient-noise/OLMo-2-179M-Exp-gradient-noise-flvann74
GN6=unlearning-gradient-noise/OLMo-2-179M-Exp-gradient-noise-h5p8rdiz
GN7=unlearning-gradient-noise/OLMo-2-179M-Exp-gradient-noise-uij1rwaw

# label | model_dir (HF id or local path) | model_revision
TARGETS=(
  "unlearning-baseline-102000|sbordt/OLMo-2-179M-Exp-Unlearning|stage1-step102000"
  "unlearning-baseline-104000|sbordt/OLMo-2-179M-Exp-Unlearning|stage1-step104000"
  "unlearning-baseline-106000|sbordt/OLMo-2-179M-Exp-Unlearning|stage1-step106000"
  "unlearning-baseline-108000|sbordt/OLMo-2-179M-Exp-Unlearning|stage1-step108000"
  "gn-1e-5-102000|$GN5/step102000-hf|main"
  "gn-1e-5-104000|$GN5/step104000-hf|main"
  "gn-1e-5-106000|$GN5/step106000-hf|main"
  "gn-1e-5-108000|$GN5/step108000-hf|main"
  "gn-1e-5-110000|$GN5/step110000-hf|main"
  "gn-1e-6-102000|$GN6/step102000-hf|main"
  "gn-1e-6-104000|$GN6/step104000-hf|main"
  "gn-1e-6-106000|$GN6/step106000-hf|main"
  "gn-1e-6-108000|$GN6/step108000-hf|main"
  "gn-1e-6-110000|$GN6/step110000-hf|main"
  "gn-1e-7-102000|$GN7/step102000-hf|main"
  "gn-1e-7-104000|$GN7/step104000-hf|main"
  "gn-1e-7-106000|$GN7/step106000-hf|main"
  "gn-1e-7-108000|$GN7/step108000-hf|main"
  "gn-1e-7-110000|$GN7/step110000-hf|main"
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
# Optional override: CONDITIONS_OVERRIDE="plain_16x rare_8tok_16x ..."
if [ -n "${CONDITIONS_OVERRIDE:-}" ]; then
  read -r -a CONDITIONS <<< "$CONDITIONS_OVERRIDE"
fi

IDX=${SLURM_ARRAY_TASK_ID:-${TARGET_IDX:-0}}
spec="${TARGETS[$IDX]}"
IFS='|' read -r LABEL MODEL_DIR MODEL_REV <<< "$spec"
# Layout: $OUT_BASE/<family>/step-<N>/  (family = label minus trailing step and
# any gn- prefix, e.g. gn-1e-5-102000 -> 1e-5/step-102000).
STEP="${LABEL##*-}"; FAM="${LABEL%-*}"; FAM="${FAM#gn-}"
OUT_DIR="$OUT_BASE/$FAM/step-$STEP"
mkdir -p "$OUT_DIR"

# Local-checkpoint guard: HF repo ids (sbordt/...) are resolved at load time;
# for a local checkpoint path, fail fast if it is missing rather than letting
# from_pretrained treat it as an (absent) HF repo and error late.
if [[ "$MODEL_DIR" == sbordt/* ]]; then
  :  # HF repo id
elif [ ! -e "$MODEL_DIR/config.json" ]; then
  echo "ERROR: local model_dir not found: $MODEL_DIR" >&2; exit 3
fi

echo "============================================================"
echo "  TOAA-MIA new-dataset | array task $IDX -> $LABEL"
echo "  model:     $MODEL_DIR @ $MODEL_REV"
echo "  out_dir:   $OUT_DIR"
echo "  ref_cache: $REF_CACHE_DIR (shared, pre-populated)"
echo "  conditions: ${#CONDITIONS[@]}"
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

echo
echo "============================================================"
echo "  DONE [$LABEL]. Results under $OUT_DIR/"
echo "============================================================"
