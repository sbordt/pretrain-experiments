#!/bin/bash
#SBATCH --time=0-03:00:00
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

# vLLM-engine MIA (newtoken_mia_vllm.py) for the deep-ignorance MIA-finetune
# checkpoints, on the 6 conditions that were actually baked into the
# materialized finetune set (build_finetune_subset.py): plain 1x/4x/16x and
# random-{1,8,32}-token(s) 16x. Usage:
#   sbatch -J mia-vllm-ft-546M run_mia_vllm_finetune.sh 546M
#   sbatch -J mia-vllm-ft-1B   run_mia_vllm_finetune.sh 1B
# 546M -> mia-finetune/OLMo-2-546M-...-3h99s8vx/step4800-hf   (10 epochs)
# 1B   -> mia-finetune/OLMo-2-1B-...-rd46npwd/step2880-hf     (6 epochs;
#         converted here from step2880-unsharded if missing)
# Members come from the same mia-data/memorization-patterns.jsonl rows the
# finetune subset was cut from, so results are comparable to the pretraining
# MIA sweeps. Outputs: evals/mia-finetune/<SIZE>/step-<N>/memorization_patterns_mia_vllm
# Per-experiment .done markers; resubmit the same command to resume.
# 546M has head_dim 70 -> vLLM FlexAttention path, needs the cudagraph patch.

SIZE="${1:?usage: sbatch -J mia-vllm-ft-<size> run_mia_vllm_finetune.sh <546M|1B>}"

scontrol show job ${SLURM_JOB_ID} 2>/dev/null || true
nvidia-smi || true
echo "  HOST: $(hostname)"
echo "  SIZE: $SIZE"

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

FT_ROOT=~/pretrain-experiments/mia-finetune
case "$SIZE" in
  546M)
    RUN_DIR="$FT_ROOT/OLMo-2-546M-DeepIgnorance-mia-finetune-10ep-3h99s8vx"
    STEP=4800   # 480 steps/epoch -> 10 epochs
    ;;
  1B)
    RUN_DIR="$FT_ROOT/OLMo-2-1B-DeepIgnorance-mia-finetune-10ep-rd46npwd"
    STEP=2880   # 480 steps/epoch -> 6 epochs
    ;;
  *)
    echo "ERROR: unknown size '$SIZE' (want 546M or 1B)" >&2; exit 2
    ;;
esac
MODEL_DIR="$RUN_DIR/step${STEP}-hf"

# Convert unsharded -> HF if the HF dir is missing (1B step2880 case).
if [ ! -f "$MODEL_DIR/model.safetensors" ] && [ ! -f "$MODEL_DIR/model-00001-of-00002.safetensors" ]; then
  UNSHARDED="$RUN_DIR/step${STEP}-unsharded"
  [ -f "$UNSHARDED/model.pt" ] || { echo "ERROR: neither HF nor unsharded checkpoint at $RUN_DIR/step${STEP}-*" >&2; exit 6; }
  echo "Converting $UNSHARDED -> $MODEL_DIR ..."
  python ~/OLMo/scripts/convert_olmo2_to_hf.py \
    --input_dir "$UNSHARDED" \
    --output_dir "$MODEL_DIR" \
    --tokenizer_json_path "$TOKENIZER" \
    || { echo "ERROR: HF conversion failed" >&2; exit 6; }
fi

# Guard against the wrong-size-weights incident: checkpoint must match $SIZE.
python - "$MODEL_DIR" "$SIZE" <<'PY'
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
rc=$?; [ "$rc" -eq 0 ] || exit "$rc"

OUT_DIR=~/pretrain-experiments/evals/mia-finetune/$SIZE/step-$STEP/memorization_patterns_mia_vllm
mkdir -p "$OUT_DIR"

MIA_EXPS=(
  memorization-patterns-plain-1x
  memorization-patterns-plain-4x
  memorization-patterns-plain-16x
  memorization-patterns-random-1-token-16x
  memorization-patterns-random-8-tokens-16x
  memorization-patterns-random-32-tokens-16x
)
if [ -n "${MIA_EXPERIMENTS:-}" ]; then
  read -r -a MIA_EXPS <<< "$MIA_EXPERIMENTS"
fi
N_EXPS=${#MIA_EXPS[@]}

echo "============================================"
echo "  vLLM MIA finetune eval [$SIZE] step $STEP ($N_EXPS experiments)"
echo "  model: $MODEL_DIR"
echo "  out:   $OUT_DIR"
echo "============================================"

run_idx=0
for exp in "${MIA_EXPS[@]}"; do
  run_idx=$((run_idx + 1))
  done_marker="$OUT_DIR/${exp}.done"
  fail_marker="$OUT_DIR/${exp}.failed"
  if [ -f "$done_marker" ]; then
    echo "[$run_idx/$N_EXPS] $exp -- already done"
    continue
  fi
  rm -f "$fail_marker"
  echo
  echo "[$run_idx/$N_EXPS] $exp -- running (vllm)"

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
  echo "[$run_idx/$N_EXPS] $exp -- rc=$rc ($((end - start))s)"

  rm -rf "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
done

echo
echo "============================================"
echo "  AUC SUMMARY [$SIZE] step $STEP"
echo "============================================"
python - "$OUT_DIR" <<'PY'
import json, glob, os, sys
out = sys.argv[1]
for f in sorted(glob.glob(os.path.join(out, "*.json"))):
    d = json.load(open(f))
    for k, v in d.items():
        auc = v.get('auc')
        bpb = v.get('bpb_auc')
        creg = v.get('canary_region_auc')
        breg = v.get('base_region_auc')
        bpb_s = f"{bpb:.4f}" if bpb is not None else " n/a  "
        creg_s = f"{creg:.4f}" if creg is not None else " n/a  "
        breg_s = f"{breg:.4f}" if breg is not None else " n/a  "
        print(f"  {k:<46} nll={auc:.4f} bpb={bpb_s} base={breg_s} canary={creg_s}")
PY

done_n=$(ls "$OUT_DIR"/*.done 2>/dev/null | wc -l)
fail_n=$(ls "$OUT_DIR"/*.failed 2>/dev/null | wc -l)
echo "  done $done_n / $N_EXPS, failed $fail_n"
if [ "$done_n" -lt "$N_EXPS" ]; then
  echo "  -- incomplete; resubmit the same command to resume --"
  exit 1
fi
echo "  all $N_EXPS experiments completed."
