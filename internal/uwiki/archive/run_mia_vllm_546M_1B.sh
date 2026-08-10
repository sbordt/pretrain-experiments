#!/bin/bash
#SBATCH --time=2-00:00:00
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

# vLLM-engine MIA (newtoken_mia_vllm.py) for the 546M and 1B model families,
# all 30 memorization-patterns conditions. Companion to
# run_mia_vllm_179M_all.sh. Usage:
#   sbatch -J mia-vllm-<size>-<label> run_mia_vllm_546M_1B.sh <size> <label>
# Sizes/labels:
#   546M: baseline | 1e-7 | 5e-6 | 7.5e-6 | deep-ignorance | unlearning-baseline
#   1B:   baseline | 3.5e-6 | deep-ignorance | unlearning-baseline
# baseline = step-100000 only; everything else steps 102k..110k. All
# checkpoints resolved from existing local HF dirs (verified present at
# submission time) — no downloads or conversions here.
# Outputs: evals/gn-eval3-sweep/<size>/<label>/step-<N>/memorization_patterns_mia_vllm
# Per-experiment .done markers; resubmit the same command to resume.
# 546M has head_dim 70 (not FlashAttention-compatible), so like the 179M it
# uses vLLM's FlexAttention path and needs the cudagraph patch.

SIZE="${1:?usage: sbatch -J mia-vllm-<size>-<label> run_mia_vllm_546M_1B.sh <size> <label>}"
LABEL="${2:?usage: sbatch -J mia-vllm-<size>-<label> run_mia_vllm_546M_1B.sh <size> <label>}"

scontrol show job ${SLURM_JOB_ID} 2>/dev/null || true
nvidia-smi || true
echo "  HOST: $(hostname)"
echo "  SIZE: $SIZE  LABEL: $LABEL"

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
grep -q "PATCHED (martinp27cs)" \
  "$HOME/.local/lib/python3.12/site-packages/vllm/v1/attention/backends/flex_attention.py" \
  || { echo "ERROR: vLLM flex_attention cudagraph patch missing — re-apply before running" >&2; exit 1; }

set -u

TOAA_DIR=pretrain_experiments/evaluation/train-once-answer-all
TOKENIZER=~/OLMo/olmo_data/tokenizers/allenai_dolma2.json

MIA_DATA_IN=${MIA_DATA_IN:-$HOME/pretrain-experiments/mia-data/memorization-patterns.jsonl}
MIA_DATA_OUT_PKL=${MIA_DATA_OUT_PKL:-$HOME/pretrain-experiments/mia-data/memorization-patterns-holdout.pkl}
MIA_CACHE_DIR=${MIA_CACHE_DIR:-$HOME/.cache/huggingface}

for f in "$MIA_DATA_IN" "$MIA_DATA_OUT_PKL" "$TOKENIZER"; do
  [ -f "$f" ] || { echo "ERROR: required input not found: $f" >&2; exit 1; }
done

# --- size/label -> (steps, model resolver, arch) ----------------------------
STEPS=(102000 104000 106000 108000 110000)
EVAL_ROOT=~/pretrain-experiments/evals/gn-eval3-sweep/$SIZE
GN_ROOT=~/pretrain-experiments/unlearning-gradient-noise
CKPT_ROOT=~/pretrain-experiments/checkpoints

case "$SIZE/$LABEL" in
  546M/baseline|1B/baseline)
    STEPS=(100000)
    ;;
  546M/1e-7)
    RUN_DIR="$GN_ROOT/OLMo-2-546M-Exp-gradient-noise-dp69aj1f"
    ;;
  546M/5e-6)
    RUN_DIR="$GN_ROOT/OLMo-2-546M-Exp-gradient-noise-ioza65lg"
    ;;
  546M/7.5e-6)
    RUN_DIR="$GN_ROOT/OLMo-2-546M-Exp-gradient-noise-48sc3om3"
    ;;
  1B/3.5e-6)
    RUN_DIR="$GN_ROOT/OLMo-2-1B-Exp-gradient-noise-9050f9m3"
    ;;
  546M/deep-ignorance|1B/deep-ignorance|546M/unlearning-baseline|1B/unlearning-baseline)
    ;;
  *)
    echo "ERROR: unknown size/label '$SIZE/$LABEL'" >&2; exit 2
    ;;
esac

resolve_model () {
  # sets MODEL_DIR for $SIZE/$LABEL at step $1
  local step=$1
  case "$SIZE/$LABEL" in
    546M/baseline)
      MODEL_DIR="$CKPT_ROOT/546M-Exp-Unlearning/step100000-hf" ;;
    1B/baseline)
      MODEL_DIR="$CKPT_ROOT/1B-Exp-Unlearning/step100000-hf" ;;
    546M/1e-7|546M/5e-6|546M/7.5e-6|1B/3.5e-6)
      MODEL_DIR="$RUN_DIR/step${step}-hf" ;;
    546M/deep-ignorance)
      if [ "$step" = "110000" ]; then
        MODEL_DIR="$CKPT_ROOT/546M-Unlearning/deep-ignorance-stage1-step110000-tokens231B-hf"
      else
        MODEL_DIR="$CKPT_ROOT/546M-Unlearning/deep-ignorance-stage1-step${step}-hf"
      fi ;;
    1B/deep-ignorance)
      if [ "$step" = "110000" ]; then
        MODEL_DIR="$CKPT_ROOT/1B-Unlearning/deep-ignorance-stage1-step110000-tokens231B-hf"
      else
        MODEL_DIR="$CKPT_ROOT/1B-Unlearning/deep-ignorance-stage1-step${step}-hf"
      fi ;;
    546M/unlearning-baseline)
      if [ "$step" = "110000" ]; then
        MODEL_DIR="$CKPT_ROOT/546M-Exp-Unlearning/step110000-hf"
      else
        MODEL_DIR="$CKPT_ROOT/546M-Exp-Unlearning/unlearning-baseline-stage1-step${step}-hf"
      fi ;;
    1B/unlearning-baseline)
      # 102k..108k live at the top level of checkpoints/ (pre-namespacing era);
      # 110k was converted locally under 1B-Exp-Unlearning/.
      if [ "$step" = "110000" ]; then
        MODEL_DIR="$CKPT_ROOT/1B-Exp-Unlearning/step110000-hf"
      else
        MODEL_DIR="$CKPT_ROOT/unlearning-baseline-stage1-step${step}-hf"
      fi ;;
  esac
}

# Guard against the wrong-size-weights incident: checkpoint must match $SIZE.
assert_arch () {
  python - "$1" "$SIZE" <<'PY'
import json, sys
ARCH = {
    "546M": {"hidden_size": 1120, "intermediate_size": 4480,
             "num_hidden_layers": 16, "num_attention_heads": 16},
    "1B":   {"hidden_size": 2048, "intermediate_size": 8192,
             "num_hidden_layers": 16, "num_attention_heads": 16},
}
cfg = json.load(open(f"{sys.argv[1]}/config.json"))
exp = ARCH[sys.argv[2]]
got = {k: cfg.get(k) for k in exp}
if got != exp:
    print(f"ERROR: HF config in {sys.argv[1]} does not match {sys.argv[2]} arch.\n"
          f"  expected: {exp}\n  got:      {got}", file=sys.stderr)
    sys.exit(7)
PY
}

MIA_EXPS=(
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
  memorization-patterns-model-based-1-token-1x
  memorization-patterns-model-based-1-token-4x
  memorization-patterns-model-based-1-token-16x
  memorization-patterns-model-based-8-tokens-1x
  memorization-patterns-model-based-8-tokens-4x
  memorization-patterns-model-based-8-tokens-16x
  memorization-patterns-model-based-32-tokens-1x
  memorization-patterns-model-based-32-tokens-4x
  memorization-patterns-model-based-32-tokens-16x
  memorization-patterns-random-1-token-1x
  memorization-patterns-random-1-token-4x
  memorization-patterns-random-1-token-16x
  memorization-patterns-random-8-tokens-1x
  memorization-patterns-random-8-tokens-4x
  memorization-patterns-random-8-tokens-16x
  memorization-patterns-random-32-tokens-1x
  memorization-patterns-random-32-tokens-4x
  memorization-patterns-random-32-tokens-16x
)
if [ -n "${MIA_EXPERIMENTS:-}" ]; then
  read -r -a MIA_EXPS <<< "$MIA_EXPERIMENTS"
fi
N_EXPS=${#MIA_EXPS[@]}
TOTAL=$((N_EXPS * ${#STEPS[@]}))

echo "============================================"
echo "  vLLM MIA sweep [$SIZE/$LABEL]: ${#STEPS[@]} step(s) x $N_EXPS experiments"
echo "============================================"

run_idx=0
for STEP in "${STEPS[@]}"; do
  resolve_model "$STEP"
  if [ ! -f "$MODEL_DIR/model.safetensors" ] && [ ! -f "$MODEL_DIR/model-00001-of-00002.safetensors" ]; then
    echo "ERROR: missing weights: $MODEL_DIR" >&2; exit 6
  fi
  assert_arch "$MODEL_DIR"

  OUT_DIR="$EVAL_ROOT/$LABEL/step-$STEP/memorization_patterns_mia_vllm"
  mkdir -p "$OUT_DIR"

  echo
  echo "============================================"
  echo "  [$SIZE/$LABEL] step $STEP"
  echo "  model: $MODEL_DIR"
  echo "  out:   $OUT_DIR"
  echo "============================================"

  for exp in "${MIA_EXPS[@]}"; do
    run_idx=$((run_idx + 1))
    done_marker="$OUT_DIR/${exp}.done"
    fail_marker="$OUT_DIR/${exp}.failed"
    if [ -f "$done_marker" ]; then
      echo "[$run_idx/$TOTAL] step$STEP/$exp -- already done"
      continue
    fi
    rm -f "$fail_marker"
    echo
    echo "[$run_idx/$TOTAL] step$STEP/$exp -- running (vllm)"

    export TORCHINDUCTOR_CACHE_DIR="/tmp/inductor_${SLURM_JOB_ID}_${run_idx}"
    export TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}_${run_idx}"
    rm -rf "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
    mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

    start=$(date +%s)
    python "$TOAA_DIR/newtoken_mia_vllm.py" \
      --model_dir         "$MODEL_DIR" \
      --model_revision    main \
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
    echo "[$run_idx/$TOTAL] step$STEP/$exp -- rc=$rc ($((end - start))s)"

    rm -rf "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
  done
done

echo
echo "============================================"
echo "  SWEEP SUMMARY [$SIZE/$LABEL]"
echo "============================================"
all_done=0
for STEP in "${STEPS[@]}"; do
  OUT_DIR="$EVAL_ROOT/$LABEL/step-$STEP/memorization_patterns_mia_vllm"
  done_n=$(ls "$OUT_DIR"/*.done 2>/dev/null | wc -l)
  fail_n=$(ls "$OUT_DIR"/*.failed 2>/dev/null | wc -l)
  echo "  step-$STEP: done $done_n / $N_EXPS, failed $fail_n"
  all_done=$((all_done + done_n))
done
if [ "$all_done" -lt "$TOTAL" ]; then
  echo "  -- incomplete ($all_done/$TOTAL); resubmit the same command to resume --"
  exit 1
fi
echo "  all $TOTAL experiments completed."
