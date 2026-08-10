#!/bin/bash
#SBATCH --time=01:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#SBATCH --job-name=smoke-mia-pin
#SBATCH --account=datamining
#SBATCH --partition=p_datamining
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=vader

# Recovery + retry after install-job 654617.
# 654617 left the env with:
#   - vllm 0.9.2 + transformers 4.57.6  -> aimv2 ValueError on `from vllm import LLM`
#   - numpy 2.2.6                       -> ai2-olmo 0.6.0 requires numpy<2
# Fix by pinning transformers<4.54 (last line without aimv2) and numpy<2.
# Then re-run the same newtoken_mia.py smoke test.

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
echo "  STEP A: state before re-pin"
echo "============================================"
python -c "import torch, transformers, numpy, vllm
print('  torch       :', torch.__version__)
print('  transformers:', transformers.__version__)
print('  numpy       :', numpy.__version__)
print('  vllm        :', vllm.__version__)" 2>&1 || echo "  (one or more imports failed)"

echo
echo "============================================"
echo "  STEP B: pin transformers<4.54 and numpy<2"
echo "============================================"
python -m pip install --user --no-cache-dir \
       --upgrade-strategy only-if-needed \
       'transformers<4.54' 'numpy<2' 2>&1 | tail -40 \
  || { echo "ERROR: pin install failed" >&2; exit 1; }

echo
echo "============================================"
echo "  STEP C: state after re-pin"
echo "============================================"
python -c "import torch, transformers, numpy
print('  torch       :', torch.__version__)
print('  transformers:', transformers.__version__)
print('  numpy       :', numpy.__version__)" 2>&1 \
  || { echo "ERROR: post-pin imports failed" >&2; exit 1; }

python -c "from vllm import LLM, SamplingParams, TokensPrompt; import vllm; print('  vllm        :', vllm.__version__)" \
  || { echo "ERROR: vllm still can't import LLM after re-pin" >&2; exit 1; }

python -c "from olmo.tokenizer import Tokenizer; print('  olmo.tokenizer OK')" \
  || { echo "ERROR: olmo.tokenizer not importable" >&2; exit 1; }

# Also verify ai2-olmo (numpy<2 consumer) still works.
python -c "import ai2_olmo 2>/dev/null; print('  ai2_olmo OK (optional)')" 2>/dev/null \
  || python -c "import olmo; print('  olmo package OK')" \
  || echo "  (ai2-olmo not importable, but newtoken_mia.py only needs olmo.tokenizer)"

set -u

echo
echo "============================================"
echo "  STEP D: run newtoken_mia.py smoke test"
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
