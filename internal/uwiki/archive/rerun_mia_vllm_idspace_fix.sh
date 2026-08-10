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

# Re-validate the matched-canary holdout AFTER the id-space graft fix in
# newtoken_mia_vllm.py (graft canary in token-id space + feed TokensPrompt, no
# decode->re-encode round trip). Writes to a SEPARATE dir so the original
# mia-vllm-fix-validation grid is untouched and directly comparable.
#
# Runs the 4 conditions that showed a residual AUC < 0.5 plus two controls
# (plain, rare-32) that were already ~0.5 -- if the controls move, the pipeline
# change regressed something. All 1x, so the flex-attn cudagraph patch is not
# needed (the residual was flat across 1x/4x/16x).

nvidia-smi || true
echo "  HOST: $(hostname)"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset SSL_CERT_FILE

source /etc/profile.d/modules.sh
export ENV_MODE="permanent"
export ENV_NAME="pretrain-experiments"
module load miniforge

cd ~/pretrain-experiments
export PYTHONPATH="$HOME/.local/lib/python3.12/site-packages:$PWD${PYTHONPATH:+:$PYTHONPATH}"

set -o pipefail

python -c "import torch; assert torch.cuda.is_available(); print('  cuda:', torch.cuda.get_device_name(0))" \
  || { echo "ERROR: CUDA unreachable on $(hostname)" >&2; exit 1; }
python -c "from vllm import LLM; print('  vllm OK')" \
  || { echo "ERROR: vllm not importable" >&2; exit 1; }

set -u

TOAA_DIR=pretrain_experiments/evaluation/train-once-answer-all
TOKENIZER=~/OLMo/olmo_data/tokenizers/allenai_dolma2.json
MIA_DATA_IN=$HOME/pretrain-experiments/mia-data/memorization-patterns.jsonl
MIA_DATA_OUT_PKL=$HOME/pretrain-experiments/mia-data/memorization-patterns-holdout.pkl
MIA_CACHE_DIR=$HOME/.cache/huggingface
CACHE_ROOT=~/pretrain-experiments/checkpoints/179M-Unlearning

STEP="${STEP:-102000}"
MODEL_DIR="$CACHE_ROOT/deep-ignorance-stage1-step${STEP}-hf"
[ -f "$MODEL_DIR/model.safetensors" ] || { echo "ERROR: missing weights: $MODEL_DIR" >&2; exit 6; }

OUT_DIR=~/pretrain-experiments/evals/mia-vllm-idspace-fix/deep-ignorance/step-${STEP}/memorization_patterns_mia_vllm
mkdir -p "$OUT_DIR"

EXPS=(
  memorization-patterns-plain-1x
  memorization-patterns-rare-32-tokens-1x
  memorization-patterns-random-8-tokens-1x
  memorization-patterns-random-32-tokens-1x
  memorization-patterns-model-based-8-tokens-1x
  memorization-patterns-model-based-32-tokens-1x
)
if [ -n "${MIA_EXPERIMENTS:-}" ]; then
  read -r -a EXPS <<< "$MIA_EXPERIMENTS"
fi

echo "============================================"
echo "  ID-SPACE FIX RE-VALIDATION (deep-ignorance)"
echo "  step:  $STEP  model: $MODEL_DIR"
echo "  out:   $OUT_DIR"
echo "  exps:  ${#EXPS[@]}"
echo "============================================"

idx=0
for exp in "${EXPS[@]}"; do
  idx=$((idx + 1))
  done_marker="$OUT_DIR/${exp}.done"
  if [ -f "$done_marker" ]; then
    echo "[$idx/${#EXPS[@]}] $exp -- already done"
    continue
  fi
  echo
  echo "[$idx/${#EXPS[@]}] $exp -- running (vllm, id-space graft)"
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
  echo "[$idx/${#EXPS[@]}] $exp -- rc=$rc ($((end - start))s)"
  rm -rf "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
done

echo
echo "============================================"
echo "  AUC SUMMARY (id-space fix, step $STEP)"
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
        subs = v.get('subsample_aucs_nll') or []
        bpb_s = f"{bpb:.4f}" if bpb is not None else " n/a  "
        creg_s = f"{creg:.4f}" if creg is not None else " n/a  "
        breg_s = f"{breg:.4f}" if breg is not None else " n/a  "
        subs_s = "[" + ", ".join(f"{a:.4f}" for a in subs) + "]"
        print(f"  {k:<46} nll={auc:.4f} bpb={bpb_s} base={breg_s} canary={creg_s} sub3={subs_s}")
PY
