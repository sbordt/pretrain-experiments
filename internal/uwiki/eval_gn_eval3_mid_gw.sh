#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --open-mode=append
#SBATCH --job-name=gn-eval3-mid-gw
#SBATCH --account=datamining
#SBATCH --partition=p_datamining
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --exclude=vader,galadriel

# Gaussian Watermark eval for the MID-TRAINING (stage2) model family.
# See CLAUDE.md "Experiment 5: Mid-training (stage2) canary insertion".
#
# Two families per size (no separate baseline -- the earliest stage2 step is the
# analog):
#   exp-mid          -> sbordt/OLMo-2-<SIZE>-Exp-Mid   (saw the canaries; the
#                       stage1 unlearning-baseline analog)
#   deep-ignorance   -> sbordt/OLMo-2-<SIZE>-Mid       (never saw the canaries)
#
# Grid: stage2-step {1000,3000,5000,7000,9000,11000} (every-second checkpoint),
# 6 per family, 12 targets per size. One job (or array task) == one target.
# Outputs -> evals/gn-eval3-sweep/<SIZE>-Mid/<family>/step-<N>/gaussian_watermark/
#
# Noise vectors: reuses the stage1 sbordt/OLMo-2-<SIZE>-Exp-NoiseVectors set (no
# -Mid-specific dataset is published). GW detection does not filter noise files by
# training step, so the stage2 step range is irrelevant -- a clear signal on the
# exp-mid models vs a null on the deep-ignorance models confirms the noise set
# matches. The per-size noise dir is already built (noise-vectors/OLMo-2-<SIZE>-Exp/),
# so NO warmup is needed -- submit the whole array directly.
#
# Usage:
#   MODEL=179M sbatch --array=0-11 eval_gn_eval3_mid_gw.sh
#   (repeat for 546M, 1B, 2.7B)
#
# Env vars:
#   MODEL=179M|546M|1B|2.7B  -> model size (required)
#   NOISE_DIR=/path          -> override the per-size noise dir
#   NOISE_STD=<float>        -> training-time noise std (default 0.075)
#   TARGET_IDX=<n>           -> target index when not run as an array task

scontrol show job ${SLURM_JOB_ID} 2>/dev/null || true
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L || true

export PYTHONUNBUFFERED=1
export INFERENCE_MAX_NUM_SEQS=${INFERENCE_MAX_NUM_SEQS:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset SSL_CERT_FILE

source /etc/profile.d/modules.sh
export ENV_MODE="permanent"
export ENV_NAME="pretrain-experiments"
module load miniforge

cd ~/pretrain-experiments

export PYTHONPATH="$PWD:$HOME/.local/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
python -c "
import torch, transformers
print('torch', torch.__version__, '| transformers', transformers.__version__)
assert torch.cuda.is_available(), 'CUDA not available on '+__import__('socket').gethostname()
print('cuda', torch.version.cuda, '| device', torch.cuda.get_device_name(0))
" || { echo "ERROR: env/CUDA preflight failed on $(hostname)" >&2; exit 1; }

set -u
set -o pipefail

TOAA_DIR=pretrain_experiments/evaluation/train-once-answer-all
CACHE_DIR=${MIA_CACHE_DIR:-$HOME/.cache/huggingface}
NOISE_STD=${NOISE_STD:-0.075}

MODEL="${MODEL:?set MODEL=179M|546M|1B|2.7B}"
case "$MODEL" in
  179M) SIZE=179M; EXP_HID=576;  EXP_INT=2304;  EXP_LAY=12; EXP_HEAD=12 ;;
  546M) SIZE=546M; EXP_HID=1120; EXP_INT=4480;  EXP_LAY=16; EXP_HEAD=16 ;;
  1B)   SIZE=1B;   EXP_HID=2048; EXP_INT=8192;  EXP_LAY=16; EXP_HEAD=16 ;;
  2.7B) SIZE=2.7B; EXP_HID=2880; EXP_INT=11520; EXP_LAY=16; EXP_HEAD=16 ;;
  *) echo "ERROR: MODEL must be 179M|546M|1B|2.7B (got '$MODEL')" >&2; exit 2 ;;
esac

EVAL_ROOT=~/pretrain-experiments/evals/gn-eval3-sweep/${SIZE}-Mid
NOISE_DIR=${NOISE_DIR:-$HOME/pretrain-experiments/noise-vectors/OLMo-2-${SIZE}-Exp}

HFEXP=sbordt/OLMo-2-${SIZE}-Exp-Mid   # exp-mid (saw canaries; unlearning-baseline analog)
HFDI=sbordt/OLMo-2-${SIZE}-Mid        # deep-ignorance (never saw canaries)

# label | model_repo | model_revision. exp-mid first (idx 0-5), then deep-ignorance (6-11).
STEPS=(1000 3000 5000 7000 9000 11000)
TARGETS=()
for st in "${STEPS[@]}"; do TARGETS+=("exp-mid-${st}|$HFEXP|stage2-step${st}"); done
for st in "${STEPS[@]}"; do TARGETS+=("deep-ignorance-${st}|$HFDI|stage2-step${st}"); done

IDX=${SLURM_ARRAY_TASK_ID:-${TARGET_IDX:-0}}
spec="${TARGETS[$IDX]}"
IFS='|' read -r LABEL MODEL_DIR MODEL_REV <<< "$spec"
STEP="${LABEL##*-}"; FAM="${LABEL%-*}"
OUT_DIR="$EVAL_ROOT/$FAM/step-$STEP"
mkdir -p "$OUT_DIR"

# --- noise-vectors sanity check (already built for all sizes) --------------
if [ ! -d "$NOISE_DIR" ] || [ -z "$(ls -A "$NOISE_DIR" 2>/dev/null | grep -E 'gaussian_poisoning_seeds_and_sequences.*\.pkl' || true)" ]; then
  echo "ERROR: NOISE_DIR=$NOISE_DIR has no gaussian_poisoning_*.pkl files." >&2
  echo "       Build it once via mia-data/build_noise_dir.py --repo sbordt/OLMo-2-${SIZE}-Exp-NoiseVectors --out $NOISE_DIR" >&2
  exit 4
fi

# --- architecture guard ----------------------------------------------------
# Confirm the target HF revision is the expected <SIZE> arch so the noise
# vectors (embed_dim = hidden_size) line up. Mirrors the stage1 assert_*_arch
# guards that caught mislabelled-size weights behind stage branches.
python - "$MODEL_DIR" "$MODEL_REV" "$CACHE_DIR" "$EXP_HID" "$EXP_INT" "$EXP_LAY" "$EXP_HEAD" <<'PY'
import json, sys
from huggingface_hub import hf_hub_download
repo, rev, cache, hid, inter, lay, head = sys.argv[1:8]
p = hf_hub_download(repo, "config.json", revision=rev, cache_dir=cache)
cfg = json.load(open(p))
exp = {"hidden_size": int(hid), "intermediate_size": int(inter),
       "num_hidden_layers": int(lay), "num_attention_heads": int(head)}
got = {k: cfg.get(k) for k in exp}
if got != exp:
    print(f"ERROR: HF config {repo}@{rev} does not match expected arch.\n"
          f"  expected: {exp}\n  got:      {got}", file=sys.stderr)
    sys.exit(7)
print(f"arch OK: {repo}@{rev} -> {got}")
PY

echo ""
echo "============================================================"
echo "  GW ${SIZE}-Mid | task $IDX -> $LABEL  ($FAM/step-$STEP)"
echo "  model:  $MODEL_DIR @ $MODEL_REV"
echo "  noise:  $NOISE_DIR  (std=$NOISE_STD)"
echo "  out:    $OUT_DIR/gaussian_watermark"
echo "============================================================"

# --- gaussian_watermark ----------------------------------------------------
marker="$OUT_DIR/gaussian_watermark.done"
if [ -f "$marker" ]; then
  echo "    [gaussian_watermark] already done, skipping"
else
  echo "    --- gaussian_watermark ---"
  if python "$TOAA_DIR/gaussian_watermark.py" \
      --noise_dir "$NOISE_DIR" \
      --model_dir "$MODEL_DIR" \
      --revision "$MODEL_REV" \
      --noise_std "$NOISE_STD" \
      --cache_dir "$CACHE_DIR" \
      --results_dir "$OUT_DIR/gaussian_watermark" \
      2>&1 \
     && compgen -G "$OUT_DIR/gaussian_watermark/*.pt" > /dev/null; then
    touch "$marker"
  else
    echo "    [gaussian_watermark] FAILED for $LABEL — no .pt outputs; not marking done" >&2
    exit 5
  fi
fi

echo ""
echo "============================================================"
echo "  DONE [${SIZE}-Mid / $LABEL]"
echo "    out_dir: $OUT_DIR/gaussian_watermark"
echo "    gw_done: $([ -f "$marker" ] && echo Y || echo N)"
echo "    gw_pt:   $(ls "$OUT_DIR/gaussian_watermark/"*.pt 2>/dev/null | wc -l)"
echo "============================================================"

# ---------------------------------------------------------------------------
# Submit note (run from internal/uwiki/) -- no warmup needed (noise dir exists):
#   for M in 179M 546M 1B 2.7B; do
#     sbatch --job-name=gn-eval3-${M}-mid-gw --export=ALL,MODEL=$M \
#            --array=0-11 eval_gn_eval3_mid_gw.sh
#   done
# ---------------------------------------------------------------------------
