#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#SBATCH --job-name=gn-il-10M-546M
#SBATCH --account=datamining
#SBATCH --partition=p_datamining
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --exclude=vader

# Unlearning Eval 1 (546M analog of eval_gn_insertion_likelihood.sh):
# insertion_likelihood only, 10M tokens per experiment.
# Select via GROUP=1|2|3|4:
#
#   sbatch -J gn-il-10M-546M-G1 --export=ALL,GROUP=1 internal/uwiki/eval_gn_insertion_likelihood_546M.sh
#   sbatch -J gn-il-10M-546M-G2 --export=ALL,GROUP=2 internal/uwiki/eval_gn_insertion_likelihood_546M.sh
#   sbatch -J gn-il-10M-546M-G3 --export=ALL,GROUP=3 internal/uwiki/eval_gn_insertion_likelihood_546M.sh
#   sbatch -J gn-il-10M-546M-G4 --export=ALL,GROUP=4 internal/uwiki/eval_gn_insertion_likelihood_546M.sh
#
# Group 1 (6 models): baseline + unlearning-baseline {102,104,106,108,110}k
# Group 2 (6 models): deep-ignorance {100,102,104,106,108,110}k
# Group 3 (5 models): 5e-6 sweep run ioza65lg, every-2k checkpoints {102,104,106,108,110}k
#                     (intermediate steps are unsharded-only on disk; converted inline)
# Group 4 (5 models): 7.5e-6 sweep run 48sc3om3, every-2k checkpoints {102,104,106,108,110}k
#                     (intermediate steps are unsharded-only on disk; converted inline;
#                      step110000 is skipped if not yet present, since training may still
#                      be in-flight when this is first launched.)
#
# Same SEED=42 across all models => every model sees the same 10M-token subset
# per experiment (see insertion_likelihood.py). Results go to
# evals/gn-insertion-likelihood-546M/<label>/step-<N>/results.yaml.
#
# The 546M 110k references are both published as stage1-step110000-tokens231B,
# so no unsharded->HF conversion is needed (unlike the 179M deep-ignorance case).

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
EVAL_ROOT=~/pretrain-experiments/evals/gn-insertion-likelihood-546M
MAX_TOKENS=10000000

UNLEARN_REPO="sbordt/OLMo-2-546M-Exp-Unlearning"
IGNORANCE_REPO="sbordt/OLMo-2-546M-Unlearning"

# 5e-6 sweep run (job 534395, ioza65lg). step110000-hf already exists from the
# post-train pipeline; intermediate steps are unsharded-only on disk.
RUN_5E6=~/pretrain-experiments/unlearning-gradient-noise/OLMo-2-546M-Exp-gradient-noise-ioza65lg

# 7.5e-6 sweep run (job 534699, 48sc3om3). All intermediate steps are
# unsharded-only on disk; step110000 may not yet exist at submission time.
RUN_7P5E6=~/pretrain-experiments/unlearning-gradient-noise/OLMo-2-546M-Exp-gradient-noise-48sc3om3

# Entries: label|step|mode|model|revision
#   mode=local -> model is a local HF dir,   revision ignored
#   mode=hf    -> model is an HF repo id,    revision names the stage1-stepXXX tag
case "$GROUP" in
  1)
    CKPTS=(
      "baseline|100000|hf|$UNLEARN_REPO|stage1-step100000-tokens210B"
      "unlearning-baseline|102000|hf|$UNLEARN_REPO|stage1-step102000"
      "unlearning-baseline|104000|hf|$UNLEARN_REPO|stage1-step104000"
      "unlearning-baseline|106000|hf|$UNLEARN_REPO|stage1-step106000"
      "unlearning-baseline|108000|hf|$UNLEARN_REPO|stage1-step108000"
      "unlearning-baseline|110000|hf|$UNLEARN_REPO|stage1-step110000-tokens231B"
    )
    ;;
  2)
    CKPTS=(
      "deep-ignorance|100000|hf|$IGNORANCE_REPO|stage1-step100000-tokens210B"
      "deep-ignorance|102000|hf|$IGNORANCE_REPO|stage1-step102000"
      "deep-ignorance|104000|hf|$IGNORANCE_REPO|stage1-step104000"
      "deep-ignorance|106000|hf|$IGNORANCE_REPO|stage1-step106000"
      "deep-ignorance|108000|hf|$IGNORANCE_REPO|stage1-step108000"
      "deep-ignorance|110000|hf|$IGNORANCE_REPO|stage1-step110000-tokens231B"
    )
    ;;
  3)
    # Convert every-2k unsharded checkpoints to HF if not already done.
    # step110000-hf is produced by the post-train pipeline, so we only need
    # to convert {102,104,106,108}k here.
    for s in 102000 104000 106000 108000; do
      hf_dir="$RUN_5E6/step${s}-hf"
      unsharded_dir="$RUN_5E6/step${s}-unsharded"
      if [ ! -f "$hf_dir/model.safetensors" ]; then
        if [ ! -d "$unsharded_dir" ]; then
          echo "ERROR: $unsharded_dir not found" >&2
          exit 1
        fi
        echo "--- converting $unsharded_dir -> $hf_dir ---"
        python "$CONVERT" \
          --input_dir "$unsharded_dir" \
          --output_dir "$hf_dir" \
          --tokenizer_json_path "$TOKENIZER" \
          --no_tmp_cleanup
      fi
    done

    CKPTS=(
      "5e-6|102000|local|$RUN_5E6/step102000-hf|"
      "5e-6|104000|local|$RUN_5E6/step104000-hf|"
      "5e-6|106000|local|$RUN_5E6/step106000-hf|"
      "5e-6|108000|local|$RUN_5E6/step108000-hf|"
      "5e-6|110000|local|$RUN_5E6/step110000-hf|"
    )
    ;;
  4)
    # Convert every-2k unsharded checkpoints to HF if not already done.
    # step110000-unsharded may not exist yet at submission time (training
    # may still be in-flight); skip with a warning rather than erroring.
    CKPTS=()
    for s in 102000 104000 106000 108000 110000; do
      hf_dir="$RUN_7P5E6/step${s}-hf"
      unsharded_dir="$RUN_7P5E6/step${s}-unsharded"
      if [ ! -f "$hf_dir/model.safetensors" ]; then
        if [ ! -d "$unsharded_dir" ]; then
          echo "WARN: $unsharded_dir not found, skipping step $s" >&2
          continue
        fi
        echo "--- converting $unsharded_dir -> $hf_dir ---"
        python "$CONVERT" \
          --input_dir "$unsharded_dir" \
          --output_dir "$hf_dir" \
          --tokenizer_json_path "$TOKENIZER" \
          --no_tmp_cleanup
      fi
      CKPTS+=("7.5e-6|${s}|local|$hf_dir|")
    done
    ;;
  *)
    echo "ERROR: GROUP must be 1, 2, 3, or 4 (got '$GROUP')" >&2
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
