#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#SBATCH --job-name=smoke-mia-install
#SBATCH --account=datamining
#SBATCH --partition=p_datamining
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=vader

# Combined: (1) pip install --user vllm into the pretrain-experiments env,
# (2) verify torch wasn't displaced, (3) run the newtoken_mia.py smoke test.
# Output isolated under evals/gn-eval3-sweep-smoke/.

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

set -o pipefail

echo "============================================"
echo "  STEP 1: capture torch BEFORE"
echo "============================================"
TORCH_BEFORE="$(python -c 'import torch; print(torch.__version__)' 2>&1)"
echo "  torch before: $TORCH_BEFORE"

echo
echo "============================================"
echo "  STEP 2: dry-run pip install --user 'vllm<0.10'"
echo "============================================"
# Older vllm (0.10.x) targets torch 2.7.x, matching the env. Unconstrained
# `vllm` would pull torch 2.11 + cuda-python 13 and shadow the training env.
VLLM_PIN='vllm<0.10'
PLAN_FILE=/tmp/vllm_install_plan.${SLURM_JOB_ID}.txt
python -m pip install --user --dry-run "$VLLM_PIN" --no-cache-dir \
       --upgrade-strategy only-if-needed 2>&1 | tee "$PLAN_FILE"
echo
echo "  --- torch-related entries in plan:"
TORCH_PLAN_LINES=$(grep -oE "torch-[0-9]+\.[0-9]+\.[0-9]+" "$PLAN_FILE" | sort -u || true)
if [ -z "$TORCH_PLAN_LINES" ]; then
  echo "  (none — torch unchanged in plan)"
else
  echo "$TORCH_PLAN_LINES"
  # If any planned torch version != current, abort before any real install.
  BAD=0
  for tv in $TORCH_PLAN_LINES; do
    case "$tv" in
      torch-2.7.*) ;;
      *) BAD=1 ;;
    esac
  done
  if [ "$BAD" -eq 1 ]; then
    echo
    echo "ERROR: '$VLLM_PIN' would still swap torch away from $TORCH_BEFORE."
    echo "       Aborting before install. Pick a tighter vllm pin or use an"
    echo "       isolated PYTHONUSERBASE." >&2
    exit 1
  fi
fi

echo
echo "============================================"
echo "  STEP 3: real install"
echo "============================================"
python -m pip install --user "$VLLM_PIN" --no-cache-dir \
       --upgrade-strategy only-if-needed 2>&1 | tail -80 \
  || { echo "ERROR: pip install vllm failed" >&2; exit 1; }

echo
echo "============================================"
echo "  STEP 4: verify install"
echo "============================================"
TORCH_AFTER="$(python -c 'import torch; print(torch.__version__)' 2>&1)"
echo "  torch after:  $TORCH_AFTER"
if [ "$TORCH_BEFORE" != "$TORCH_AFTER" ]; then
  echo "  *** WARNING: torch version changed by vllm install ($TORCH_BEFORE -> $TORCH_AFTER)"
  echo "  *** this may break other training scripts. revert with:"
  echo "  ***   pip install --user --force-reinstall 'torch==$TORCH_BEFORE'"
fi

python -c "import vllm; print('  vllm:', vllm.__version__)" \
  || { echo "ERROR: vllm still not importable after install" >&2; exit 1; }
python -c "from olmo.tokenizer import Tokenizer; print('  olmo.tokenizer OK')" \
  || { echo "ERROR: olmo.tokenizer not importable" >&2; exit 1; }

set -u

echo
echo "============================================"
echo "  STEP 5: run newtoken_mia.py smoke test"
echo "============================================"

TOAA_DIR=pretrain_experiments/evaluation/train-once-answer-all
TOKENIZER=~/OLMo/olmo_data/tokenizers/allenai_dolma2.json

MODEL_DIR="sbordt/OLMo-2-179M-Exp-Unlearning"
MODEL_REVISION="stage1-step100000-tokens210B"
TARGET_EXP="memorization-patterns-rare-1-token-1x"

MIA_DATA_IN=${MIA_DATA_IN:-$HOME/pretrain-experiments/mia-data/memorization-patterns.jsonl}
MIA_DATA_OUT_PKL=${MIA_DATA_OUT_PKL:-$HOME/pretrain-experiments/mia-data/memorization-patterns-holdout.pkl}
MIA_CACHE_DIR=${MIA_CACHE_DIR:-$HOME/.cache/huggingface}

OUT_DIR=~/pretrain-experiments/evals/gn-eval3-sweep-smoke/179M/baseline/step-100000/memorization_patterns_mia
mkdir -p "$OUT_DIR"

echo "  model:        $MODEL_DIR"
echo "  revision:     $MODEL_REVISION"
echo "  target_exp:   $TARGET_EXP"
echo "  out_dir:      $OUT_DIR"
echo

for f in "$MIA_DATA_IN" "$MIA_DATA_OUT_PKL" "$TOKENIZER"; do
  [ -f "$f" ] || { echo "ERROR: required input not found: $f" >&2; exit 1; }
done

set -x
python "$TOAA_DIR/newtoken_mia.py" \
  --model_dir       "$MODEL_DIR" \
  --model_revision  "$MODEL_REVISION" \
  --data_in_file    "$MIA_DATA_IN" \
  --data_out_file   "$MIA_DATA_OUT_PKL" \
  --target_experiment "$TARGET_EXP" \
  --results_dir     "$OUT_DIR" \
  --cache_dir       "$MIA_CACHE_DIR" \
  --tokenizer_path  "$TOKENIZER"
rc=$?
set +x

echo
echo "============================================"
echo "  SMOKE TEST RESULT"
echo "============================================"
echo "  python exit code: $rc"
echo "  produced files:"
ls -la "$OUT_DIR" || true
exit $rc
