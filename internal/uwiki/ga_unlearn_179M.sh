#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#SBATCH --job-name=ga-179M
#SBATCH --account=datamining
#SBATCH --partition=p_datamining
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=vader

# Gradient-ascent unlearning sweep cell on OLMo-2-179M-Exp-Unlearning,
# starting from stage1-step100000-tokens210B.
#
# Required env vars:
#   LR     - learning rate (e.g. 1e-6, 5e-6)
#   BATCH  - per-step (micro) batch size (e.g. 16, 32, 64)
#
# Optional env vars:
#   FORGET_EXPERIMENT - forget-set experiment name in sbordt/OLMo-2-1B-Exp-Dataset
#                       (default: memorization-patterns-rare-1-token-1x)
#   EPOCHS            - total epochs (default: 20; capped at 20 in gradient_ascent.py)
#   CKPT_EVERY        - checkpoint every N epochs (default: 5)
#   RUN_TAG           - subdir name under unlearning-gradient-ascent/
#                       (default: 179M-mp-rare-1tok-1x)
#   GA_DTYPE          - float32 (default) or bfloat16
#   GA_GRAD_CKPT      - 1 to enable gradient checkpointing (default: 0; not needed at 179M)
#   GA_ACCUM          - gradient accumulation steps (default: 1)
#
# Launch the 6-cell phase-1 sweep (LR x BATCH = {1e-6, 5e-6} x {16, 32, 64}):
#   for lr in 1e-6 5e-6; do
#     for b in 16 32 64; do
#       sbatch -J ga-179M-lr${lr}-b${b} \
#              --export=ALL,LR=${lr},BATCH=${b} \
#              internal/uwiki/ga_unlearn_179M.sh
#     done
#   done

: "${LR:?set LR (e.g. 1e-6, 5e-6)}"
: "${BATCH:?set BATCH (e.g. 16, 32, 64)}"
FORGET_EXPERIMENT="${FORGET_EXPERIMENT:-memorization-patterns-rare-1-token-1x}"
EPOCHS="${EPOCHS:-20}"
CKPT_EVERY="${CKPT_EVERY:-5}"
RUN_TAG="${RUN_TAG:-179M-mp-rare-1tok-1x}"
GA_DTYPE="${GA_DTYPE:-float32}"
GA_GRAD_CKPT="${GA_GRAD_CKPT:-0}"
GA_ACCUM="${GA_ACCUM:-1}"

scontrol show job ${SLURM_JOB_ID} 2>/dev/null || true
nvidia-smi || true

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset SSL_CERT_FILE

source /etc/profile.d/modules.sh
export ENV_MODE="permanent"
export ENV_NAME="pretrain-experiments"
module load miniforge

cd ~/pretrain-experiments

# torch lives in user-site on some nodes (e.g. shelob/H200); prepend explicitly.
export PYTHONPATH="$PWD:$HOME/.local/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
python -c "import torch, transformers, datasets, pretrain_experiments; print('torch', torch.__version__)" \
  || { echo "ERROR: torch / pretrain_experiments not importable on $(hostname)" >&2; exit 1; }

set -u
set -o pipefail

MODEL=sbordt/OLMo-2-179M-Exp-Unlearning
REVISION=stage1-step100000-tokens210B
OUTPUT_DIR="$HOME/pretrain-experiments/unlearning-gradient-ascent/${RUN_TAG}/lr${LR}-b${BATCH}"

EXTRA_ARGS=()
if [ "$GA_GRAD_CKPT" = "1" ]; then
  EXTRA_ARGS+=(--gradient-checkpointing)
fi

echo "============================================"
echo "  GA cell: lr=$LR  micro_batch=$BATCH  accum=$GA_ACCUM"
echo "  effective_batch = $((BATCH * GA_ACCUM))"
echo "  forget:   $FORGET_EXPERIMENT"
echo "  model:    $MODEL @ $REVISION"
echo "  dtype:    $GA_DTYPE"
echo "  grad_ckpt: $GA_GRAD_CKPT"
echo "  output:   $OUTPUT_DIR"
echo "============================================"

python -m pretrain_experiments.gradient_ascent \
    --model "$MODEL" \
    --revision "$REVISION" \
    --forget-experiment "$FORGET_EXPERIMENT" \
    --learning-rate "$LR" \
    --batch-size "$BATCH" \
    --gradient-accumulation-steps "$GA_ACCUM" \
    --epochs "$EPOCHS" \
    --checkpoint-every-n-epochs "$CKPT_EVERY" \
    --output-dir "$OUTPUT_DIR" \
    --dtype "$GA_DTYPE" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "============================================"
echo "  GA cell DONE: lr=$LR batch=$BATCH"
echo "  checkpoints under: $OUTPUT_DIR"
echo "============================================"
