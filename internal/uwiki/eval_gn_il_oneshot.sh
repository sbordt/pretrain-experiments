#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#SBATCH --account=datamining
#SBATCH --partition=p_csunivie_gres
#SBATCH --nodelist=dgx-h100-em2
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --exclude=vader

# One-shot gn-il eval for a single deep-ignorance checkpoint, runs in parallel
# with the main gn-il-10M-G* sweep. The main G3 run will skip any checkpoint
# whose results.yaml already exists, so these one-shots just beat it to the
# punch.
#
# Usage:
#   sbatch -J gn-il-10M-di106k --export=ALL,STEP=106000 internal/uwiki/eval_gn_il_oneshot.sh
#   sbatch -J gn-il-10M-di108k --export=ALL,STEP=108000 internal/uwiki/eval_gn_il_oneshot.sh
#   sbatch -J gn-il-10M-di110k --export=ALL,STEP=110000 internal/uwiki/eval_gn_il_oneshot.sh

: "${STEP:?set STEP=106000, 108000, or 110000}"

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

SCRIPT=pretrain_experiments/evaluation/train-once-answer-all/insertion_likelihood.py
EVAL_ROOT=~/pretrain-experiments/evals/gn-insertion-likelihood
MAX_TOKENS=10000000
IGNORANCE_REPO="sbordt/OLMo-2-179M-Unlearning"
IGNORANCE_110K_HF=~/pretrain-experiments/checkpoints/deep-ignorance-step110000-hf

LABEL=deep-ignorance
OUT_DIR="$EVAL_ROOT/$LABEL/step-$STEP"
YAML="$OUT_DIR/results.yaml"
mkdir -p "$OUT_DIR"

if [ -f "$YAML" ]; then
  echo "[$LABEL/step-$STEP] already done, skipping"
  exit 0
fi

case "$STEP" in
  106000|108000)
    MODEL=$IGNORANCE_REPO
    REVISION=stage1-step$STEP
    ;;
  110000)
    MODEL=$IGNORANCE_110K_HF
    REVISION=""
    ;;
  *)
    echo "ERROR: STEP must be 106000, 108000, or 110000 (got '$STEP')" >&2
    exit 1
    ;;
esac

echo ""
echo "============================================"
echo "  [$LABEL] step $STEP"
echo "  model:    $MODEL"
[ -n "$REVISION" ] && echo "  revision: $REVISION"
echo "  out:      $YAML"
echo "============================================"

args=(--model "$MODEL" --results-yaml "$YAML" --max-tokens "$MAX_TOKENS")
[ -n "$REVISION" ] && args+=(--revision "$REVISION")
python "$SCRIPT" "${args[@]}" 2>&1

echo ""
echo "============================================"
echo "  ONESHOT step $STEP DONE"
echo "============================================"
