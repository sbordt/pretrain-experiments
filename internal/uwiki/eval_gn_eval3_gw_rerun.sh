#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#SBATCH --job-name=gn-eval3-gw-rerun
#SBATCH --account=datamining
#SBATCH --partition=p_datamining
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=vader,galadriel

# Re-run gaussian_watermark.py for the (label, step) pairs whose results were
# computed by 567347 / 567348 before gaussian_watermark.py was patched to
# `os.makedirs(args.results_dir, exist_ok=True)`. Skips any pair whose
# `gaussian_watermark.done` marker already exists, so it is safe to run while
# 567348 is still in flight (galadriel is excluded so we land on a different
# node; 567348's later checkpoints will write their own markers and this
# rerun will skip them).

scontrol show job ${SLURM_JOB_ID} 2>/dev/null || true
nvidia-smi || true

export INFERENCE_MAX_NUM_SEQS=${INFERENCE_MAX_NUM_SEQS:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset SSL_CERT_FILE

source /etc/profile.d/modules.sh
export ENV_MODE="permanent"
export ENV_NAME="pretrain-experiments"
module load miniforge

cd ~/pretrain-experiments

export PYTHONPATH="$PWD:$HOME/.local/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
python -c "import torch, pretrain_experiments; print('torch', torch.__version__)" \
  || { echo "ERROR: torch / pretrain_experiments not importable on $(hostname)" >&2; exit 1; }

set -u
set -o pipefail

TOAA_DIR=pretrain_experiments/evaluation/train-once-answer-all
# This rerun script targets the 179M sweep specifically (TARGETS list below
# is hardcoded to 179M run dirs and uses 179M noise vectors). Pin EVAL_ROOT
# to the 179M namespace to match eval_gn_eval3_sweep.sh.
EVAL_ROOT=~/pretrain-experiments/evals/gn-eval3-sweep/179M
NOISE_DIR=${NOISE_DIR:-$HOME/pretrain-experiments/noise-vectors/OLMo-2-179M-Exp}
NOISE_STD=${NOISE_STD:-0.075}

RUN_1E7=~/pretrain-experiments/unlearning-gradient-noise/OLMo-2-179M-Exp-gradient-noise-uij1rwaw
RUN_1E6=~/pretrain-experiments/unlearning-gradient-noise/OLMo-2-179M-Exp-gradient-noise-h5p8rdiz
RUN_1E5=~/pretrain-experiments/unlearning-gradient-noise/OLMo-2-179M-Exp-gradient-noise-flvann74
BASELINE_HF=~/pretrain-experiments/checkpoints/step100000-hf

# (label, step, hf_dir) tuples whose watermark eval ran but failed to save.
TARGETS=(
  "baseline|100000|$BASELINE_HF"
  "1e-7|102000|$RUN_1E7/step102000-hf"
  "1e-6|102000|$RUN_1E6/step102000-hf"
  "1e-5|102000|$RUN_1E5/step102000-hf"
  "1e-7|104000|$RUN_1E7/step104000-hf"
  "1e-6|104000|$RUN_1E6/step104000-hf"
  "1e-5|104000|$RUN_1E5/step104000-hf"
  "1e-7|106000|$RUN_1E7/step106000-hf"
  "1e-5|110000|$RUN_1E5/step110000-hf"
)

if [ ! -d "$NOISE_DIR" ] || [ -z "$(ls -A "$NOISE_DIR" 2>/dev/null | grep -E 'gaussian_poisoning_seeds_and_sequences.*\.pkl' || true)" ]; then
  echo "ERROR: NOISE_DIR=$NOISE_DIR is missing or has no pkl files." >&2
  echo "       Run the main driver once to populate it, or set NOISE_DIR." >&2
  exit 2
fi
echo "--- noise-vectors dir: $NOISE_DIR ---"

for entry in "${TARGETS[@]}"; do
  IFS='|' read -r label step hf_dir <<< "$entry"
  out_dir="$EVAL_ROOT/$label/step-$step"
  mkdir -p "$out_dir"
  marker="$out_dir/gaussian_watermark.done"

  echo ""
  echo "============================================"
  echo "  [$label] step $step"
  echo "  hf:   $hf_dir"
  echo "  out:  $out_dir"
  echo "============================================"

  if [ -f "$marker" ]; then
    echo "    [gaussian_watermark] $marker exists, skipping"
    continue
  fi
  if [ ! -f "$hf_dir/model.safetensors" ]; then
    echo "    [gaussian_watermark] hf_dir missing model.safetensors, skipping" >&2
    continue
  fi

  echo "    --- gaussian_watermark ---"
  python "$TOAA_DIR/gaussian_watermark.py" \
    --noise_dir "$NOISE_DIR" \
    --model_dir "$hf_dir" \
    --noise_std "$NOISE_STD" \
    --results_dir "$out_dir/gaussian_watermark" \
    2>&1 && touch "$marker"
done

echo ""
echo "============================================"
echo "  RERUN DONE"
echo "============================================"
for entry in "${TARGETS[@]}"; do
  IFS='|' read -r label step _ <<< "$entry"
  out_dir="$EVAL_ROOT/$label/step-$step"
  if [ -f "$out_dir/gaussian_watermark.done" ]; then
    n=$(ls "$out_dir/gaussian_watermark"/*.pt 2>/dev/null | wc -l)
    echo "  [$label step $step] OK ($n .pt files)"
  else
    echo "  [$label step $step] MISSING marker"
  fi
done
