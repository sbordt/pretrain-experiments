#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#SBATCH --job-name=lunar-179M
#SBATCH --account=datamining
#SBATCH --partition=p_datamining
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=vader

# LUNAR unlearning sweep cell on OLMo-2-179M-Exp-Unlearning,
# starting from stage1-step100000-tokens210B.
#
# Required env vars:
#   LR                   - learning rate (e.g. 5e-5, 1e-5)
#   REDIRECTION_LAYER    - decoder layer index ℓ (e.g. 5 for 12-layer 179M)
#
# Optional env vars:
#   UPDATE_SCOPE         - "full-layer" (default) or "down-proj"
#   RETAIN_W             - α on retain MSE (default: 1.0)
#   ANCHOR_NUM_TOKENS    - EOS tokens fed to compute the anchor (default: 1)
#   FORGET_BATCH         - micro batch size on forget (default: 4)
#   RETAIN_BATCH         - micro batch size on retain (default: 4)
#   ACCUM                - gradient accumulation steps (default: 1)
#   EPOCHS               - passes over the forget set (default: 1)
#   MAX_STEPS            - optimizer-step cap (default: unset)
#   CKPT_EVERY           - checkpoint every N epochs (default: 1)
#   RUN_TAG              - subdir under unlearning-lunar/ (default: 179M-default)
#   FORGET_EXPS          - space-separated experiment whitelist (default: full minus iid)
#   DTYPE                - float32 (default) or bfloat16
#   FROZEN_DTYPE         - float32 or bfloat16 (default: bfloat16)
#   GRAD_CKPT            - 1 to enable gradient checkpointing (default: 0)
#   OLMO_CONFIG          - OLMo TrainConfig YAML
#                          (default: $HOME/OLMo/configs/official-0425/OLMo2-1B-stage1.yaml)
#   START_STEP           - retain-stream start step (default: 100000)

: "${LR:?set LR}"
: "${REDIRECTION_LAYER:?set REDIRECTION_LAYER}"

UPDATE_SCOPE="${UPDATE_SCOPE:-full-layer}"
RETAIN_W="${RETAIN_W:-1.0}"
ANCHOR_NUM_TOKENS="${ANCHOR_NUM_TOKENS:-1}"
FORGET_BATCH="${FORGET_BATCH:-4}"
RETAIN_BATCH="${RETAIN_BATCH:-4}"
ACCUM="${ACCUM:-1}"
EPOCHS="${EPOCHS:-1}"
CKPT_EVERY="${CKPT_EVERY:-1}"
RUN_TAG="${RUN_TAG:-179M-default}"
DTYPE="${DTYPE:-float32}"
FROZEN_DTYPE="${FROZEN_DTYPE:-bfloat16}"
GRAD_CKPT="${GRAD_CKPT:-0}"
OLMO_CONFIG="${OLMO_CONFIG:-$HOME/OLMo/configs/official-0425/OLMo2-1B-stage1.yaml}"
START_STEP="${START_STEP:-100000}"

scontrol show job ${SLURM_JOB_ID} 2>/dev/null || true
nvidia-smi || true

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset SSL_CERT_FILE

source /etc/profile.d/modules.sh
export ENV_MODE="permanent"
export ENV_NAME="pretrain-experiments"
module load miniforge

cd ~/pretrain-experiments

export PYTHONPATH="$PWD:$HOME/.local/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
python -c "import torch, transformers, datasets, olmo, pretrain_experiments; print('torch', torch.__version__)" \
  || { echo "ERROR: torch / olmo / pretrain_experiments not importable on $(hostname)" >&2; exit 1; }

set -e
set -u
set -o pipefail

MODEL=sbordt/OLMo-2-179M-Exp-Unlearning
REVISION=stage1-step100000-tokens210B
OUTPUT_DIR="$HOME/pretrain-experiments/unlearning-lunar/${RUN_TAG}/lr${LR}-l${REDIRECTION_LAYER}-w${RETAIN_W}-${UPDATE_SCOPE}"

EXTRA_ARGS=()
if [ "$GRAD_CKPT" = "1" ]; then
  EXTRA_ARGS+=(--gradient-checkpointing)
fi
if [ -n "${FORGET_EXPS:-}" ]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS+=(--forget-experiments $FORGET_EXPS)
fi
if [ -n "${MAX_STEPS:-}" ]; then
  EXTRA_ARGS+=(--max-steps "$MAX_STEPS")
fi

echo "============================================"
echo "  LUNAR cell: lr=$LR  redirection_layer=$REDIRECTION_LAYER  scope=$UPDATE_SCOPE"
echo "  retain_w=$RETAIN_W  anchor_num_tokens=$ANCHOR_NUM_TOKENS"
echo "  epochs=$EPOCHS  forget_bs=$FORGET_BATCH  retain_bs=$RETAIN_BATCH  accum=$ACCUM"
echo "  model:        $MODEL @ $REVISION"
echo "  retain:       $OLMO_CONFIG (start_step=$START_STEP)"
echo "  forget_exps:  ${FORGET_EXPS:-<full minus iid-replacements-*>}"
echo "  dtype:        $DTYPE  (frozen: $FROZEN_DTYPE)  grad_ckpt: $GRAD_CKPT"
echo "  output:       $OUTPUT_DIR"
echo "============================================"

python -m pretrain_experiments.lunar \
    --model "$MODEL" \
    --revision "$REVISION" \
    --olmo-config "$OLMO_CONFIG" \
    --retain-start-step "$START_STEP" \
    --redirection-layer "$REDIRECTION_LAYER" \
    --update-scope "$UPDATE_SCOPE" \
    --retain-loss-weight "$RETAIN_W" \
    --anchor-num-tokens "$ANCHOR_NUM_TOKENS" \
    --learning-rate "$LR" \
    --forget-batch-size "$FORGET_BATCH" \
    --retain-batch-size "$RETAIN_BATCH" \
    --gradient-accumulation-steps "$ACCUM" \
    --epochs "$EPOCHS" \
    --checkpoint-every-n-epochs "$CKPT_EVERY" \
    --output-dir "$OUTPUT_DIR" \
    --dtype "$DTYPE" \
    --frozen-dtype "$FROZEN_DTYPE" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "============================================"
echo "  LUNAR cell DONE: lr=$LR layer=$REDIRECTION_LAYER scope=$UPDATE_SCOPE"
echo "  checkpoints under: $OUTPUT_DIR"
echo "============================================"
