#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --open-mode=append
#SBATCH --job-name=gn-eval3-2.7B-gw
#SBATCH --account=datamining
#SBATCH --partition=p_datamining
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --exclude=vader,galadriel

# Unlearning Eval 3 -- Gaussian Watermark ONLY, for the 2.7B model family.
# Mirrors the GW logic of eval_gn_eval3_1B_*.sh but scoped to a single eval
# (no FK/VM/IL, no MIA -- the memorization-patterns MIA is covered separately
# by the toaa-mia-newdataset run). One job (or array task) == one target
# checkpoint; outputs to
#   evals/gn-eval3-sweep/2.7B/<family>/step-<N>/gaussian_watermark/
# with a gaussian_watermark.done marker so requeues/reruns skip finished ckpts.
#
# families: baseline, unlearning-baseline, deep-ignorance. There is NO gradient-
# noise family -- the 2.7B GN sweep has not been run yet (see CLAUDE.md
# "Experiment 4"). Like the toaa-mia-newdataset 2.7B driver, all targets are
# pulled as HF revisions (gaussian_watermark.py loads --model_dir @ --revision
# straight from HF); no unsharded->HF conversion is needed.
#
# The noise vectors come from sbordt/OLMo-2-2.7B-Exp-NoiseVectors, converted
# once to the per-chunk .pkl layout via mia-data/build_noise_dir.py and cached
# at noise-vectors/OLMo-2-2.7B-Exp/. Because a cold fan-out would race on that
# build, warm it with the baseline target (idx 0) first, then fan out the rest
# under --dependency=afterok. See the submit note at the bottom of this file.
#
# Env vars:
#   NOISE_DIR=/path/to/dir   -> override the auto-built noise dir
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
EVAL_ROOT=~/pretrain-experiments/evals/gn-eval3-sweep/2.7B

NOISE_REPO=sbordt/OLMo-2-2.7B-Exp-NoiseVectors
NOISE_DIR=${NOISE_DIR:-$HOME/pretrain-experiments/noise-vectors/OLMo-2-2.7B-Exp}
NOISE_STD=${NOISE_STD:-0.075}

HFUB=sbordt/OLMo-2-2.7B-Exp-Unlearning   # baseline + unlearning-baseline
HFDI=sbordt/OLMo-2-2.7B-Unlearning       # deep-ignorance ground truth

# label | model_repo | model_revision   (index 0 = baseline = the warmup target)
# baseline / *-110000 use the tokens-suffixed branch names; intermediate steps
# use the bare stage1-step<N> branches (per the HF refs listing for both repos).
TARGETS=(
  "baseline-100000|$HFUB|stage1-step100000-tokens210B"
  "unlearning-baseline-102000|$HFUB|stage1-step102000"
  "unlearning-baseline-104000|$HFUB|stage1-step104000"
  "unlearning-baseline-106000|$HFUB|stage1-step106000"
  "unlearning-baseline-108000|$HFUB|stage1-step108000"
  "unlearning-baseline-110000|$HFUB|stage1-step110000-tokens231B"
  "deep-ignorance-100000|$HFDI|stage1-step100000-tokens210B"
  "deep-ignorance-102000|$HFDI|stage1-step102000"
  "deep-ignorance-104000|$HFDI|stage1-step104000"
  "deep-ignorance-106000|$HFDI|stage1-step106000"
  "deep-ignorance-108000|$HFDI|stage1-step108000"
  "deep-ignorance-110000|$HFDI|stage1-step110000-tokens231B"
)

IDX=${SLURM_ARRAY_TASK_ID:-${TARGET_IDX:-0}}
spec="${TARGETS[$IDX]}"
IFS='|' read -r LABEL MODEL_DIR MODEL_REV <<< "$spec"
# Layout: $EVAL_ROOT/<family>/step-<N>/  (deep-ignorance-104000 -> deep-ignorance/step-104000).
STEP="${LABEL##*-}"; FAM="${LABEL%-*}"
OUT_DIR="$EVAL_ROOT/$FAM/step-$STEP"
mkdir -p "$OUT_DIR"

# --- noise-vectors prep (HF parquet -> .pkl files) -------------------------
# For the warmup target (idx 0) this builds the dir; fan-out tasks (held on
# afterok) find it already populated and skip.
if [ ! -d "$NOISE_DIR" ] || [ -z "$(ls -A "$NOISE_DIR" 2>/dev/null | grep -E 'gaussian_poisoning_seeds_and_sequences.*\.pkl' || true)" ]; then
  echo "--- building noise-vectors dir: $NOISE_REPO -> $NOISE_DIR ---"
  mkdir -p "$NOISE_DIR"
  python ~/pretrain-experiments/mia-data/build_noise_dir.py \
    --repo "$NOISE_REPO" \
    --out  "$NOISE_DIR" \
    || { echo "ERROR: failed to build noise dir $NOISE_DIR" >&2; exit 4; }
else
  echo "--- noise-vectors dir already populated: $NOISE_DIR ---"
fi

# --- architecture guard ----------------------------------------------------
# Verify the target HF revision is the 2.7B arch (hidden_size 2880) so the
# noise vectors (embed_dim 2880) line up. Mirrors the assert_1b_arch guard
# from the 1B drivers, which caught mislabelled-size weights behind stage1
# branches in 2026-05.
python - "$MODEL_DIR" "$MODEL_REV" "$CACHE_DIR" <<'PY'
import json, sys
from huggingface_hub import hf_hub_download
repo, rev, cache = sys.argv[1], sys.argv[2], sys.argv[3]
p = hf_hub_download(repo, "config.json", revision=rev, cache_dir=cache)
cfg = json.load(open(p))
exp = {"hidden_size": 2880, "intermediate_size": 11520,
       "num_hidden_layers": 16, "num_attention_heads": 16}
got = {k: cfg.get(k) for k in exp}
if got != exp:
    print(f"ERROR: HF config {repo}@{rev} does not match 2.7B arch.\n"
          f"  expected: {exp}\n  got:      {got}", file=sys.stderr)
    sys.exit(7)
print(f"arch OK: {repo}@{rev} -> {got}")
PY

echo ""
echo "============================================================"
echo "  GW 2.7B | task $IDX -> $LABEL  ($FAM/step-$STEP)"
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
echo "  DONE [$LABEL]"
echo "    out_dir: $OUT_DIR/gaussian_watermark"
echo "    gw_done: $([ -f "$marker" ] && echo Y || echo N)"
echo "    gw_pt:   $(ls "$OUT_DIR/gaussian_watermark/"*.pt 2>/dev/null | wc -l)"
echo "============================================================"

# ---------------------------------------------------------------------------
# Submit note (run from internal/uwiki/):
#   # 1) warm the noise-vectors dir with the baseline target (idx 0):
#   warm=$(sbatch --parsable --job-name=gn-eval3-2.7B-gw-warm \
#                 --export=ALL,TARGET_IDX=0 eval_gn_eval3_2.7B_gw.sh)
#   # 2) fan out the remaining 11 targets once the noise dir is built:
#   sbatch --dependency=afterok:$warm --array=1-11 eval_gn_eval3_2.7B_gw.sh
# ---------------------------------------------------------------------------
