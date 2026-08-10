#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#SBATCH --job-name=smoke-newtoken-mia
#SBATCH --account=datamining
#SBATCH --partition=p_datamining
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=vader

# Smoke test for the rewritten newtoken_mia.py (vLLM-based).
# - One checkpoint (179M baseline @ stage1-step100000-tokens210B from HF)
# - One memorization-patterns experiment (rare-1-token-1x)
# - Output lands in evals/gn-eval3-sweep-smoke/ so existing .done markers and
#   results under gn-eval3-sweep/ are untouched.

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

# Preflight: fail fast if the new deps the rewritten script needs aren't here.
python -c "import torch, pretrain_experiments; print('torch', torch.__version__)" \
  || { echo "ERROR: torch / pretrain_experiments not importable on $(hostname)" >&2; exit 1; }
python -c "import vllm; print('vllm', vllm.__version__)" \
  || { echo "ERROR: vllm not importable on $(hostname)" >&2; exit 1; }
python -c "from olmo.tokenizer import Tokenizer; print('olmo.tokenizer OK')" \
  || { echo "ERROR: olmo.tokenizer not importable on $(hostname)" >&2; exit 1; }

set -u
set -o pipefail

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

echo "============================================"
echo "  SMOKE TEST: newtoken_mia.py (vLLM)"
echo "============================================"
echo "  model:        $MODEL_DIR"
echo "  revision:     $MODEL_REVISION"
echo "  target_exp:   $TARGET_EXP"
echo "  data_in:      $MIA_DATA_IN"
echo "  data_out_pkl: $MIA_DATA_OUT_PKL"
echo "  cache_dir:    $MIA_CACHE_DIR"
echo "  tokenizer:    $TOKENIZER"
echo "  out_dir:      $OUT_DIR"
echo

for f in "$MIA_DATA_IN" "$MIA_DATA_OUT_PKL" "$TOKENIZER"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: required input not found: $f" >&2
    exit 1
  fi
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
