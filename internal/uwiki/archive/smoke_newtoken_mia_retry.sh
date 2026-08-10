#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#SBATCH --job-name=smoke-mia-retry
#SBATCH --account=datamining
#SBATCH --partition=p_datamining
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=vader,galadriel

# Retry of the newtoken_mia.py smoke test. 654618 left the env correctly pinned
# (vllm 0.9.2 + transformers 4.53.3 + numpy 1.26.4), but galadriel's GPU was
# unreachable (NVMLError_Unknown). Excluding it this time -> should land on
# shelob (8x H200). Also adds a hard CUDA preflight so a broken node fails fast
# instead of dying inside vLLM.

scontrol show job ${SLURM_JOB_ID} 2>/dev/null || true
nvidia-smi || true
echo "  HOST: $(hostname)"

export INFERENCE_MAX_NUM_SEQS=${INFERENCE_MAX_NUM_SEQS:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Let vLLM default to V1+FLEX_ATTENTION. The OLMo-2 head_size=48 prevents
# XFORMERS PagedAttention, TORCH_SDPA isn't a V1 backend, and we don't have
# flash_attn/flashinfer. The int32 indexing crash in flex_attention is
# avoided by --gpu_memory_utilization 0.25 (default in newtoken_mia.py).
unset SSL_CERT_FILE

source /etc/profile.d/modules.sh
export ENV_MODE="permanent"
export ENV_NAME="pretrain-experiments"
module load miniforge

cd ~/pretrain-experiments

export PYTHONPATH="$PWD:$HOME/.local/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"

set -o pipefail

echo
echo "============================================"
echo "  PREFLIGHT: env imports"
echo "============================================"
python -c "import torch, transformers, numpy
print('  torch       :', torch.__version__)
print('  transformers:', transformers.__version__)
print('  numpy       :', numpy.__version__)" \
  || { echo "ERROR: env imports failed" >&2; exit 1; }

python -c "from vllm import LLM, SamplingParams, TokensPrompt; import vllm; print('  vllm        :', vllm.__version__)" \
  || { echo "ERROR: vllm.LLM not importable" >&2; exit 1; }

python -c "from olmo.tokenizer import Tokenizer; print('  olmo.tokenizer OK')" \
  || { echo "ERROR: olmo.tokenizer not importable" >&2; exit 1; }

echo
echo "============================================"
echo "  PREFLIGHT: CUDA device actually visible"
echo "============================================"
python -c "
import torch
assert torch.cuda.is_available(), 'torch.cuda.is_available() == False'
n = torch.cuda.device_count()
print('  cuda device count :', n)
for i in range(n):
    print(f'  device {i}        :', torch.cuda.get_device_name(i))
torch.empty(8, device='cuda:0').sum().item()  # forces NVML / context init
print('  empty alloc + reduce OK')
" || { echo "ERROR: CUDA unreachable on $(hostname) — try resubmitting" >&2; exit 1; }

set -u

echo
echo "============================================"
echo "  SMOKE TEST: newtoken_mia.py"
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
