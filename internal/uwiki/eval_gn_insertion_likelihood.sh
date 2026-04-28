#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#SBATCH --job-name=gn-il-10M
#SBATCH --account=datamining
#SBATCH --partition=p_datamining
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=vader

# Unlearning Eval 1: insertion_likelihood only, 10M tokens per experiment.
# 27 models across all groups (9 per job). Select via GROUP=1|2|3:
#
#   sbatch -J gn-il-10M-G1 --export=ALL,GROUP=1 internal/uwiki/eval_gn_insertion_likelihood.sh
#   sbatch -J gn-il-10M-G2 --export=ALL,GROUP=2 internal/uwiki/eval_gn_insertion_likelihood.sh
#   sbatch -J gn-il-10M-G3 --export=ALL,GROUP=3 internal/uwiki/eval_gn_insertion_likelihood.sh
#
# Group 1 (9 models): baseline + 1e-7 sweep + unlearning-baseline {102,104,106}k
# Group 2 (9 models): 1e-6 sweep + unlearning-baseline {108,110}k + deep-ignorance {100,102}k
# Group 3 (9 models): 1e-5 sweep + deep-ignorance {104,106,108,110}k
#
# Same SEED=42 across all models => every model sees the same 10M-token subset
# per experiment (see insertion_likelihood.py). Results go to
# evals/gn-insertion-likelihood/<label>/step-<N>/results.yaml.

: "${GROUP:?set GROUP=1, 2, or 3 (controls which models this job evaluates)}"

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

# torch lives in ~/.local/lib/python3.12/site-packages (user-site) on this cluster.
# Some nodes don't add user-site to sys.path automatically, so prepend it explicitly.
export PYTHONPATH="$PWD:$HOME/.local/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
python -c "import torch, pretrain_experiments; print('torch', torch.__version__)" \
  || { echo "ERROR: torch / pretrain_experiments not importable on $(hostname)" >&2; exit 1; }

set -u
set -o pipefail

SCRIPT=pretrain_experiments/evaluation/train-once-answer-all/insertion_likelihood.py
CONVERT=~/OLMo/scripts/convert_olmo2_to_hf.py
TOKENIZER=~/OLMo/olmo_data/tokenizers/allenai_dolma2.json
EVAL_ROOT=~/pretrain-experiments/evals/gn-insertion-likelihood
MAX_TOKENS=10000000

RUN_1E7=~/pretrain-experiments/unlearning-gradient-noise/OLMo-2-179M-Exp-gradient-noise-uij1rwaw
RUN_1E6=~/pretrain-experiments/unlearning-gradient-noise/OLMo-2-179M-Exp-gradient-noise-h5p8rdiz
RUN_1E5=~/pretrain-experiments/unlearning-gradient-noise/OLMo-2-179M-Exp-gradient-noise-flvann74
BASELINE_HF=~/pretrain-experiments/checkpoints/step100000-hf

UNLEARN_REPO="sbordt/OLMo-2-179M-Exp-Unlearning"
IGNORANCE_REPO="sbordt/OLMo-2-179M-Unlearning"

# Deep-ignorance @ 110k is only published in unsharded format -> download + convert once.
IGNORANCE_110K_UNSHARDED=~/pretrain-experiments/checkpoints/deep-ignorance-step110000-unsharded
IGNORANCE_110K_HF=~/pretrain-experiments/checkpoints/deep-ignorance-step110000-hf

# Entries: label|step|mode|model|revision
#   mode=local -> model is a local HF dir,   revision ignored
#   mode=hf    -> model is an HF repo id,    revision names the stage1-stepXXX tag
case "$GROUP" in
  1)
    CKPTS=(
      "baseline|100000|local|$BASELINE_HF|"
      "1e-7|102000|local|$RUN_1E7/step102000-hf|"
      "1e-7|104000|local|$RUN_1E7/step104000-hf|"
      "1e-7|106000|local|$RUN_1E7/step106000-hf|"
      "1e-7|108000|local|$RUN_1E7/step108000-hf|"
      "1e-7|110000|local|$RUN_1E7/step110000-hf|"
      "unlearning-baseline|102000|hf|$UNLEARN_REPO|stage1-step102000"
      "unlearning-baseline|104000|hf|$UNLEARN_REPO|stage1-step104000"
      "unlearning-baseline|106000|hf|$UNLEARN_REPO|stage1-step106000"
    )
    ;;
  2)
    CKPTS=(
      "1e-6|102000|local|$RUN_1E6/step102000-hf|"
      "1e-6|104000|local|$RUN_1E6/step104000-hf|"
      "1e-6|106000|local|$RUN_1E6/step106000-hf|"
      "1e-6|108000|local|$RUN_1E6/step108000-hf|"
      "1e-6|110000|local|$RUN_1E6/step110000-hf|"
      "unlearning-baseline|108000|hf|$UNLEARN_REPO|stage1-step108000"
      "unlearning-baseline|110000|hf|$UNLEARN_REPO|stage1-step110000-tokens231B"
      "deep-ignorance|100000|hf|$IGNORANCE_REPO|stage1-step100000-tokens210B"
      "deep-ignorance|102000|hf|$IGNORANCE_REPO|stage1-step102000"
    )
    ;;
  3)
    # Prep deep-ignorance @ 110k (unsharded-only on HF).
    if [ ! -f "$IGNORANCE_110K_HF/model.safetensors" ]; then
      if [ ! -d "$IGNORANCE_110K_UNSHARDED" ]; then
        echo "--- downloading $IGNORANCE_REPO @ step110000-unsharded ---"
        python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='$IGNORANCE_REPO',
    revision='step110000-unsharded',
    local_dir='$IGNORANCE_110K_UNSHARDED',
)
"
      fi
      echo "--- converting $IGNORANCE_110K_UNSHARDED to HF ---"
      python $CONVERT \
        --input_dir "$IGNORANCE_110K_UNSHARDED" \
        --output_dir "$IGNORANCE_110K_HF" \
        --tokenizer_json_path $TOKENIZER \
        --no_tmp_cleanup
    fi

    CKPTS=(
      "1e-5|102000|local|$RUN_1E5/step102000-hf|"
      "1e-5|104000|local|$RUN_1E5/step104000-hf|"
      "1e-5|106000|local|$RUN_1E5/step106000-hf|"
      "1e-5|108000|local|$RUN_1E5/step108000-hf|"
      "1e-5|110000|local|$RUN_1E5/step110000-hf|"
      "deep-ignorance|104000|hf|$IGNORANCE_REPO|stage1-step104000"
      "deep-ignorance|106000|hf|$IGNORANCE_REPO|stage1-step106000"
      "deep-ignorance|108000|hf|$IGNORANCE_REPO|stage1-step108000"
      "deep-ignorance|110000|local|$IGNORANCE_110K_HF|"
    )
    ;;
  *)
    echo "ERROR: GROUP must be 1, 2, or 3 (got '$GROUP')" >&2
    exit 1
    ;;
esac

run_il () {
  local label=$1 step=$2 mode=$3 model=$4 revision=$5
  local out_dir="$EVAL_ROOT/$label/step-$step"
  mkdir -p "$out_dir"
  local yaml="$out_dir/results.yaml"
  if [ -f "$yaml" ]; then
    echo "    [$label/step-$step] already done, skipping"
    return 0
  fi
  echo ""
  echo "============================================"
  echo "  [$label] step $step ($mode)"
  echo "  model:    $model"
  [ -n "$revision" ] && echo "  revision: $revision"
  echo "  out:      $yaml"
  echo "============================================"

  local args=(--model "$model" --results-yaml "$yaml" --max-tokens "$MAX_TOKENS")
  [ -n "$revision" ] && args+=(--revision "$revision")
  python "$SCRIPT" "${args[@]}" 2>&1
}

for entry in "${CKPTS[@]}"; do
  IFS='|' read -r label step mode model revision <<< "$entry"
  run_il "$label" "$step" "$mode" "$model" "$revision"
done

echo ""
echo "============================================"
echo "  GROUP $GROUP DONE"
echo "============================================"
