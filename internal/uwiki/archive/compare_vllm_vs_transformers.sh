#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#SBATCH --job-name=mia-vllm-cmp
#SBATCH --account=datamining
#SBATCH --partition=p_datamining
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=vader,galadriel

# Run vLLM-based MIA on a few experiments where vLLM 0.9.2 + OLMo-2 + H200
# is known to work (1x variants — the shape-sensitive cudagraph_trees bug
# triggers on longer sequences). Output lands next to the transformers
# results so AUCs can be compared side-by-side.

scontrol show job ${SLURM_JOB_ID} 2>/dev/null || true
nvidia-smi || true
echo "  HOST: $(hostname)"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset SSL_CERT_FILE

source /etc/profile.d/modules.sh
export ENV_MODE="permanent"
export ENV_NAME="pretrain-experiments"
module load miniforge

cd ~/pretrain-experiments
export PYTHONPATH="$PWD:$HOME/.local/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"

set -o pipefail

python -c "import torch; assert torch.cuda.is_available(); print('  cuda:', torch.cuda.get_device_name(0))" \
  || { echo "ERROR: CUDA unreachable" >&2; exit 1; }
python -c "from vllm import LLM; print('  vllm OK')" \
  || { echo "ERROR: vllm not importable" >&2; exit 1; }

set -u

TOAA_DIR=pretrain_experiments/evaluation/train-once-answer-all
TOKENIZER=~/OLMo/olmo_data/tokenizers/allenai_dolma2.json

MODEL_DIR="sbordt/OLMo-2-179M-Exp-Unlearning"
MODEL_REVISION="stage1-step100000-tokens210B"

MIA_DATA_IN=${MIA_DATA_IN:-$HOME/pretrain-experiments/mia-data/memorization-patterns.jsonl}
MIA_DATA_OUT_PKL=${MIA_DATA_OUT_PKL:-$HOME/pretrain-experiments/mia-data/memorization-patterns-holdout.pkl}
MIA_CACHE_DIR=${MIA_CACHE_DIR:-$HOME/.cache/huggingface}

OUT_DIR=~/pretrain-experiments/evals/gn-eval3-sweep-fresh/179M/baseline/step-100000/memorization_patterns_mia_vllm
mkdir -p "$OUT_DIR"

# 1x variants only — historically the shape range that vLLM has handled here.
EXPS=(
  memorization-patterns-plain-1x
  memorization-patterns-rare-1-token-1x
  memorization-patterns-rare-8-tokens-1x
  memorization-patterns-rare-32-tokens-1x
  memorization-patterns-model-based-1-token-1x
  memorization-patterns-model-based-8-tokens-1x
  memorization-patterns-model-based-32-tokens-1x
  memorization-patterns-random-1-token-1x
  memorization-patterns-random-8-tokens-1x
  memorization-patterns-random-32-tokens-1x
)
TOTAL=${#EXPS[@]}

echo "============================================"
echo "  vLLM MIA comparison: $TOTAL experiments"
echo "  out_dir: $OUT_DIR"
echo "============================================"

i=0
for exp in "${EXPS[@]}"; do
  i=$((i + 1))
  done_marker="$OUT_DIR/${exp}.done"
  fail_marker="$OUT_DIR/${exp}.failed"
  if [ -f "$done_marker" ]; then
    echo "[$i/$TOTAL] $exp -- already done"
    continue
  fi
  rm -f "$fail_marker"
  echo
  echo "[$i/$TOTAL] $exp -- running (vllm)"

  export TORCHINDUCTOR_CACHE_DIR="/tmp/inductor_${SLURM_JOB_ID}_${i}"
  export TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}_${i}"
  rm -rf "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
  mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

  start=$(date +%s)
  python "$TOAA_DIR/newtoken_mia_vllm.py" \
    --model_dir         "$MODEL_DIR" \
    --model_revision    "$MODEL_REVISION" \
    --data_in_file      "$MIA_DATA_IN" \
    --data_out_file     "$MIA_DATA_OUT_PKL" \
    --target_experiment "$exp" \
    --results_dir       "$OUT_DIR" \
    --cache_dir         "$MIA_CACHE_DIR" \
    --tokenizer_path    "$TOKENIZER"
  rc=$?
  end=$(date +%s)
  if [ "$rc" -eq 0 ]; then
    touch "$done_marker"
  else
    touch "$fail_marker"
  fi
  echo "[$i/$TOTAL] $exp -- rc=$rc ($((end - start))s)"

  rm -rf "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
done

echo
echo "============================================"
echo "  COMPARISON SUMMARY"
echo "============================================"
done_n=$(ls "$OUT_DIR"/*.done 2>/dev/null | wc -l)
fail_n=$(ls "$OUT_DIR"/*.failed 2>/dev/null | wc -l)
echo "  done   : $done_n / $TOTAL"
echo "  failed : $fail_n / $TOTAL"
