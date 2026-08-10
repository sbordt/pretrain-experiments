#!/bin/bash
#SBATCH --time=0-04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#SBATCH --account=datamining
#SBATCH --partition=p_datamining
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=vader,galadriel

# Validation run for the matched-canary holdout fix in newtoken_mia_vllm.py.
# Runs a spread of conditions (all 1x, so the flex-attn cudagraph patch is not
# required) on ONE deep-ignorance checkpoint. For a ground-truth model that
# memorized nothing, the FIX should drive every condition to AUC ~= 0.5
# (previously: only plain ~0.5; perturbed variants 0.34 -> 0.05 -> ~0.0 as N grows).
# Writes to a scratch dir so the existing gn-eval3-sweep grid is untouched.

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
  || { echo "ERROR: CUDA unreachable on $(hostname)" >&2; exit 1; }
python -c "from vllm import LLM; print('  vllm OK')" \
  || { echo "ERROR: vllm not importable" >&2; exit 1; }
# 1x conditions do not need the flex-attn cudagraph patch; warn if absent, don't abort.
grep -q "PATCHED (martinp27cs)" \
  "$HOME/.local/lib/python3.12/site-packages/vllm/v1/attention/backends/flex_attention.py" 2>/dev/null \
  || echo "  WARNING: vLLM flex_attention cudagraph patch missing (ok for 1x-only validation)"

set -u

TOAA_DIR=pretrain_experiments/evaluation/train-once-answer-all
TOKENIZER=~/OLMo/olmo_data/tokenizers/allenai_dolma2.json
MIA_DATA_IN=$HOME/pretrain-experiments/mia-data/memorization-patterns.jsonl
MIA_DATA_OUT_PKL=$HOME/pretrain-experiments/mia-data/memorization-patterns-holdout.pkl
MIA_CACHE_DIR=$HOME/.cache/huggingface
CACHE_ROOT=~/pretrain-experiments/checkpoints/179M-Unlearning

# Steps to sweep (space-separated env override), default just step 102000.
read -r -a STEPS <<< "${STEPS:-102000}"

# deep-ignorance step -> local HF checkpoint dir (100k/110k carry token-count suffixes).
resolve_model_dir () {
  local step=$1
  case "$step" in
    100000) echo "$CACHE_ROOT/deep-ignorance-stage1-step100000-tokens210B-hf" ;;
    110000) echo "$CACHE_ROOT/deep-ignorance-stage1-step110000-tokens231B-hf" ;;
    *)      echo "$CACHE_ROOT/deep-ignorance-stage1-step${step}-hf" ;;
  esac
}

# Conditions (space-separated env override). Default: all 30.
EXPS=(
  memorization-patterns-plain-1x
  memorization-patterns-plain-4x
  memorization-patterns-plain-16x
  memorization-patterns-rare-1-token-1x
  memorization-patterns-rare-1-token-4x
  memorization-patterns-rare-1-token-16x
  memorization-patterns-rare-8-tokens-1x
  memorization-patterns-rare-8-tokens-4x
  memorization-patterns-rare-8-tokens-16x
  memorization-patterns-rare-32-tokens-1x
  memorization-patterns-rare-32-tokens-4x
  memorization-patterns-rare-32-tokens-16x
  memorization-patterns-random-1-token-1x
  memorization-patterns-random-1-token-4x
  memorization-patterns-random-1-token-16x
  memorization-patterns-random-8-tokens-1x
  memorization-patterns-random-8-tokens-4x
  memorization-patterns-random-8-tokens-16x
  memorization-patterns-random-32-tokens-1x
  memorization-patterns-random-32-tokens-4x
  memorization-patterns-random-32-tokens-16x
  memorization-patterns-model-based-1-token-1x
  memorization-patterns-model-based-1-token-4x
  memorization-patterns-model-based-1-token-16x
  memorization-patterns-model-based-8-tokens-1x
  memorization-patterns-model-based-8-tokens-4x
  memorization-patterns-model-based-8-tokens-16x
  memorization-patterns-model-based-32-tokens-1x
  memorization-patterns-model-based-32-tokens-4x
  memorization-patterns-model-based-32-tokens-16x
)
if [ -n "${MIA_EXPERIMENTS:-}" ]; then
  read -r -a EXPS <<< "$MIA_EXPERIMENTS"
fi

TOTAL=$(( ${#STEPS[@]} * ${#EXPS[@]} ))
echo "============================================"
echo "  CANARY-FIX VALIDATION (deep-ignorance)"
echo "  steps: ${STEPS[*]}"
echo "  exps:  ${#EXPS[@]}  (total runs: $TOTAL)"
echo "============================================"

idx=0
for STEP in "${STEPS[@]}"; do
  MODEL_DIR=$(resolve_model_dir "$STEP")
  [ -f "$MODEL_DIR/model.safetensors" ] || { echo "ERROR: missing weights: $MODEL_DIR" >&2; exit 6; }
  OUT_DIR=~/pretrain-experiments/evals/mia-vllm-fix-validation/deep-ignorance/step-${STEP}/memorization_patterns_mia_vllm
  mkdir -p "$OUT_DIR"
  echo
  echo "  >>> step $STEP | model $MODEL_DIR"
  echo "      out $OUT_DIR"

  for exp in "${EXPS[@]}"; do
    idx=$((idx + 1))
    done_marker="$OUT_DIR/${exp}.done"
    if [ -f "$done_marker" ]; then
      echo "[$idx/$TOTAL] step$STEP/$exp -- already done"
      continue
    fi
    echo
    echo "[$idx/$TOTAL] step$STEP/$exp -- running (vllm)"
    export TORCHINDUCTOR_CACHE_DIR="/tmp/inductor_${SLURM_JOB_ID}_${idx}"
    export TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}_${idx}"
    rm -rf "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
    mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

    start=$(date +%s)
    python "$TOAA_DIR/newtoken_mia_vllm.py" \
      --model_dir         "$MODEL_DIR" \
      --model_revision    "main" \
      --data_in_file      "$MIA_DATA_IN" \
      --data_out_file     "$MIA_DATA_OUT_PKL" \
      --target_experiment "$exp" \
      --results_dir       "$OUT_DIR" \
      --cache_dir         "$MIA_CACHE_DIR" \
      --tokenizer_path    "$TOKENIZER"
    rc=$?
    end=$(date +%s)
    [ "$rc" -eq 0 ] && touch "$done_marker"
    echo "[$idx/$TOTAL] step$STEP/$exp -- rc=$rc ($((end - start))s)"
    rm -rf "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
  done
done

echo
echo "============================================"
echo "  AUC SUMMARY (all steps)"
echo "============================================"
python - "${STEPS[@]}" <<'PY'
import json, glob, os, sys
steps = sys.argv[1:]
base = os.path.expanduser("~/pretrain-experiments/evals/mia-vllm-fix-validation/deep-ignorance")
for step in steps:
    out = os.path.join(base, f"step-{step}", "memorization_patterns_mia_vllm")
    print(f"--- step {step} ---")
    for f in sorted(glob.glob(os.path.join(out, "*.json"))):
        d = json.load(open(f))
        for k, v in d.items():
            print(f"  {k:<46} AUC = {v.get('auc'):.4f}")
PY
