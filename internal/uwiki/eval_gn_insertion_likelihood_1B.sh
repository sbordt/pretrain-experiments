#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#SBATCH --job-name=gn-il-10M-1B
#SBATCH --account=datamining
#SBATCH --partition=p_datamining
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --exclude=vader

# Unlearning Eval 1 (1B analog of eval_gn_insertion_likelihood{,_546M}.sh):
# insertion_likelihood only, 10M tokens per experiment.
# Select via GROUP=1|2|3:
#
#   sbatch -J gn-il-10M-1B-G1 --export=ALL,GROUP=1 internal/uwiki/eval_gn_insertion_likelihood_1B.sh
#   sbatch -J gn-il-10M-1B-G2 --export=ALL,GROUP=2 internal/uwiki/eval_gn_insertion_likelihood_1B.sh
#   sbatch -J gn-il-10M-1B-G3 --export=ALL,GROUP=3 internal/uwiki/eval_gn_insertion_likelihood_1B.sh
#
# Group 1 (11 models): baseline + unlearning-baseline {101..110}k (per-1k)
# Group 2 (11 models): deep-ignorance {100, 101..109, 110}k (per-1k now available)
# Group 3 (10 models): 3.5e-6 sweep run 9050f9m3, every-1k checkpoints {101..110}k
#                      (intermediate steps are unsharded-only on disk; converted inline;
#                       step109000 / step110000 may be missing at submission time if
#                       training is still in-flight — those entries skip with a warning.)
#
# Same SEED=42 across all models => every model sees the same 10M-token subset
# per experiment (see insertion_likelihood.py). Results go to
# evals/gn-insertion-likelihood-1B/<label>/step-<N>/results.yaml.
#
# As at 546M, the 1B references are all published as stage1-stepXXX (or
# stage1-step{100,110}000-tokens{210,231}B at the round-decade marks), so no
# unsharded->HF conversion is needed for the HF-side checkpoints.

: "${GROUP:?set GROUP=1, 2, or 3 (controls which models this job evaluates)}"

scontrol show job ${SLURM_JOB_ID} 2>/dev/null || true
nvidia-smi || true

# 1B activations are roughly 2x the 546M footprint, so default the inference
# batch lower; can override on the sbatch line if a roomier GPU is allocated.
export INFERENCE_MAX_NUM_SEQS=${INFERENCE_MAX_NUM_SEQS:-4}
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
EVAL_ROOT=~/pretrain-experiments/evals/gn-insertion-likelihood-1B
MAX_TOKENS=10000000

UNLEARN_REPO="sbordt/OLMo-2-1B-Exp-Unlearning"
IGNORANCE_REPO="sbordt/OLMo-2-1B-Unlearning"

# 3.5e-6 sweep run (job 534697, 9050f9m3). Intermediate steps are unsharded-only
# on disk; step110000 may not yet exist at submission time if training is still
# in-flight (estimated finish ~2026-04-28 ~05–06 UTC).
RUN_3P5E6=~/pretrain-experiments/unlearning-gradient-noise/OLMo-2-1B-Exp-gradient-noise-9050f9m3

# Entries: label|step|mode|model|revision
#   mode=local -> model is a local HF dir,   revision ignored
#   mode=hf    -> model is an HF repo id,    revision names the stage1-stepXXX tag
case "$GROUP" in
  1)
    CKPTS=(
      "baseline|100000|hf|$UNLEARN_REPO|stage1-step100000-tokens210B"
      "unlearning-baseline|101000|hf|$UNLEARN_REPO|stage1-step101000"
      "unlearning-baseline|102000|hf|$UNLEARN_REPO|stage1-step102000"
      "unlearning-baseline|103000|hf|$UNLEARN_REPO|stage1-step103000"
      "unlearning-baseline|104000|hf|$UNLEARN_REPO|stage1-step104000"
      "unlearning-baseline|105000|hf|$UNLEARN_REPO|stage1-step105000"
      "unlearning-baseline|106000|hf|$UNLEARN_REPO|stage1-step106000"
      "unlearning-baseline|107000|hf|$UNLEARN_REPO|stage1-step107000"
      "unlearning-baseline|108000|hf|$UNLEARN_REPO|stage1-step108000"
      "unlearning-baseline|109000|hf|$UNLEARN_REPO|stage1-step109000"
      "unlearning-baseline|110000|hf|$UNLEARN_REPO|stage1-step110000-tokens231B"
    )
    ;;
  2)
    CKPTS=(
      "deep-ignorance|100000|hf|$IGNORANCE_REPO|stage1-step100000-tokens210B"
      "deep-ignorance|101000|hf|$IGNORANCE_REPO|stage1-step101000"
      "deep-ignorance|102000|hf|$IGNORANCE_REPO|stage1-step102000"
      "deep-ignorance|103000|hf|$IGNORANCE_REPO|stage1-step103000"
      "deep-ignorance|104000|hf|$IGNORANCE_REPO|stage1-step104000"
      "deep-ignorance|105000|hf|$IGNORANCE_REPO|stage1-step105000"
      "deep-ignorance|106000|hf|$IGNORANCE_REPO|stage1-step106000"
      "deep-ignorance|107000|hf|$IGNORANCE_REPO|stage1-step107000"
      "deep-ignorance|108000|hf|$IGNORANCE_REPO|stage1-step108000"
      "deep-ignorance|109000|hf|$IGNORANCE_REPO|stage1-step109000"
      "deep-ignorance|110000|hf|$IGNORANCE_REPO|stage1-step110000-tokens231B"
    )
    ;;
  3)
    # Convert every-1k unsharded checkpoints to HF if not already done. Skip
    # entries whose unsharded source isn't on disk yet (training may still be
    # in-flight when this is first launched); they'll be picked up on a rerun.
    CKPTS=()
    for s in 101000 102000 103000 104000 105000 106000 107000 108000 109000 110000; do
      hf_dir="$RUN_3P5E6/step${s}-hf"
      unsharded_dir="$RUN_3P5E6/step${s}-unsharded"
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
      CKPTS+=("3.5e-6|${s}|local|$hf_dir|")
    done
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
