#!/bin/bash
#SBATCH --time=1-12:00:00
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

# vLLM-engine MIA (newtoken_mia_vllm.py) for the 179M model family, all 30
# memorization-patterns conditions. Label is $1:
#   baseline            -> HF repo @ stage1-step100000-tokens210B, step-100000,
#                          out_dir under gn-eval3-sweep-fresh (next to the 1x
#                          results from mia-vllm-cmp)
#   1e-5 / 1e-6 / 1e-7  -> local step<N>-hf checkpoints, steps 102k..110k
#   deep-ignorance      -> checkpoints/179M-Unlearning/*-hf, steps 102k..110k
#   unlearning-baseline -> checkpoints/179M-Exp-Unlearning/*-hf, steps 102k..110k
# Outputs mirror the transformers MIA layout in a memorization_patterns_mia_vllm
# subdir. Per-experiment .done markers; resubmit the same command to resume.
# Requires the user-site vLLM flex_attention.py cudagraph patch ("PATCHED
# (martinp27cs)") — 4x/16x conditions crash without it.

LABEL="${1:?usage: sbatch -J mia-vllm-<label> run_mia_vllm_179M_all.sh <label>}"

scontrol show job ${SLURM_JOB_ID} 2>/dev/null || true
nvidia-smi || true
echo "  HOST: $(hostname)"
echo "  LABEL: $LABEL"

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
  || { echo "ERROR: vLLM flex_attention cudagraph patch missing — re-apply before running 4x/16x" >&2; exit 1; }

set -u

TOAA_DIR=pretrain_experiments/evaluation/train-once-answer-all
TOKENIZER=~/OLMo/olmo_data/tokenizers/allenai_dolma2.json

MIA_DATA_IN=${MIA_DATA_IN:-$HOME/pretrain-experiments/mia-data/memorization-patterns.jsonl}
MIA_DATA_OUT_PKL=${MIA_DATA_OUT_PKL:-$HOME/pretrain-experiments/mia-data/memorization-patterns-holdout.pkl}
MIA_CACHE_DIR=${MIA_CACHE_DIR:-$HOME/.cache/huggingface}

for f in "$MIA_DATA_IN" "$MIA_DATA_OUT_PKL" "$TOKENIZER"; do
  [ -f "$f" ] || { echo "ERROR: required input not found: $f" >&2; exit 1; }
done

# --- label -> (steps, model resolver, eval root) ---------------------------
STEPS=(102000 104000 106000 108000 110000)
EVAL_ROOT=~/pretrain-experiments/evals/gn-eval3-sweep/179M

case "$LABEL" in
  baseline)
    STEPS=(100000)
    EVAL_ROOT=~/pretrain-experiments/evals/gn-eval3-sweep-fresh/179M
    ;;
  1e-5)
    RUN_DIR=~/pretrain-experiments/unlearning-gradient-noise/OLMo-2-179M-Exp-gradient-noise-flvann74
    ;;
  1e-6)
    RUN_DIR=~/pretrain-experiments/unlearning-gradient-noise/OLMo-2-179M-Exp-gradient-noise-h5p8rdiz
    ;;
  1e-7)
    RUN_DIR=~/pretrain-experiments/unlearning-gradient-noise/OLMo-2-179M-Exp-gradient-noise-uij1rwaw
    ;;
  deep-ignorance)
    CACHE_ROOT=~/pretrain-experiments/checkpoints/179M-Unlearning
    ;;
  unlearning-baseline)
    CACHE_ROOT=~/pretrain-experiments/checkpoints/179M-Exp-Unlearning
    ;;
  *)
    echo "ERROR: unknown label '$LABEL'" >&2; exit 2
    ;;
esac

resolve_model () {
  # sets MODEL_DIR / MODEL_REVISION for $LABEL at step $1
  local step=$1
  MODEL_REVISION="main"
  case "$LABEL" in
    baseline)
      MODEL_DIR="sbordt/OLMo-2-179M-Exp-Unlearning"
      MODEL_REVISION="stage1-step100000-tokens210B"
      ;;
    1e-5|1e-6|1e-7)
      MODEL_DIR="$RUN_DIR/step${step}-hf"
      ;;
    deep-ignorance)
      if [ "$step" = "110000" ]; then
        MODEL_DIR="$CACHE_ROOT/deep-ignorance-stage1-step110000-tokens231B-hf"
      else
        MODEL_DIR="$CACHE_ROOT/deep-ignorance-stage1-step${step}-hf"
      fi
      ;;
    unlearning-baseline)
      if [ "$step" = "110000" ]; then
        MODEL_DIR="$CACHE_ROOT/step110000-hf"
      else
        MODEL_DIR="$CACHE_ROOT/unlearning-baseline-stage1-step${step}-hf"
      fi
      ;;
  esac
}

# Guard against the wrong-size-weights incident: local checkpoints must be 179M.
assert_179m_arch () {
  python - "$1" <<'PY'
import json, sys
cfg = json.load(open(f"{sys.argv[1]}/config.json"))
exp = {"hidden_size": 576, "intermediate_size": 2304,
       "num_hidden_layers": 12, "num_attention_heads": 12}
got = {k: cfg.get(k) for k in exp}
if got != exp:
    print(f"ERROR: HF config in {sys.argv[1]} does not match 179M arch.\n"
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
echo "  vLLM MIA sweep [$LABEL]: ${#STEPS[@]} step(s) x $N_EXPS experiments"
echo "============================================"

run_idx=0
for STEP in "${STEPS[@]}"; do
  resolve_model "$STEP"
  if [ "$LABEL" != "baseline" ]; then
    [ -f "$MODEL_DIR/model.safetensors" ] \
      || { echo "ERROR: missing weights: $MODEL_DIR" >&2; exit 6; }
    assert_179m_arch "$MODEL_DIR"
  fi

  OUT_DIR="$EVAL_ROOT/$LABEL/step-$STEP/memorization_patterns_mia_vllm"
  mkdir -p "$OUT_DIR"

  echo
  echo "============================================"
  echo "  [$LABEL] step $STEP"
  echo "  model: $MODEL_DIR (rev $MODEL_REVISION)"
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
    echo "[$run_idx/$TOTAL] step$STEP/$exp -- rc=$rc ($((end - start))s)"

    rm -rf "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
  done
done

echo
echo "============================================"
echo "  SWEEP SUMMARY [$LABEL]"
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
