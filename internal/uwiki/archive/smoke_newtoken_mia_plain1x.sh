#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#SBATCH --job-name=smoke-mia-plain1x
#SBATCH --account=datamining
#SBATCH --partition=p_datamining
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=vader,galadriel

# Second smoke test: same model + step as 654623, target_experiment switched
# to memorization-patterns-plain-1x. Env already pinned (654618/654623).

scontrol show job ${SLURM_JOB_ID} 2>/dev/null || true
nvidia-smi || true
echo "  HOST: $(hostname)"

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
TARGET_EXP="memorization-patterns-plain-1x"

MIA_DATA_IN=${MIA_DATA_IN:-$HOME/pretrain-experiments/mia-data/memorization-patterns.jsonl}
MIA_DATA_OUT_PKL=${MIA_DATA_OUT_PKL:-$HOME/pretrain-experiments/mia-data/memorization-patterns-holdout.pkl}
MIA_CACHE_DIR=${MIA_CACHE_DIR:-$HOME/.cache/huggingface}

OUT_DIR=~/pretrain-experiments/evals/gn-eval3-sweep-smoke/179M/baseline/step-100000/memorization_patterns_mia
mkdir -p "$OUT_DIR"

echo "============================================"
echo "  SMOKE TEST: newtoken_mia.py  (plain-1x)"
echo "============================================"
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
echo "  SMOKE TEST RESULT  (plain-1x)"
echo "============================================"
echo "  python exit code: $rc"
echo "  produced files:"
ls -la "$OUT_DIR" || true
exit $rc
