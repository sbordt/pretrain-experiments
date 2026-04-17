#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#SBATCH --job-name=gn-c4val-sweep
#SBATCH --account=datamining
#SBATCH --partition=p_datamining
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# Unlearning Eval 1: general capability degradation on C4 validation
# (2500 samples, per-example cross-entropy losses saved to JSONL).
#
# Sweep: noise_std in {1e-7, 1e-6, 1e-5} x every-second checkpoint
# (step 102k, 104k, 106k, 108k, 110k). Plus the no-unlearning baseline
# (step 100000) and the unlearning baseline from HuggingFace
# (sbordt/OLMo-2-179M-Exp-Unlearning @ step110000-unsharded).

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

set -u
set -o pipefail

CONVERT=~/OLMo/scripts/convert_olmo2_to_hf.py
TOKENIZER=~/OLMo/olmo_data/tokenizers/allenai_dolma2.json
TASK_FILE=~/pretrain-experiments/resources/validation-set/c4_en_validation.jsonl
EVAL_ROOT=~/pretrain-experiments/evals/gn-c4val-sweep

RUN_1E7=~/pretrain-experiments/unlearning-gradient-noise/OLMo-2-179M-Exp-gradient-noise-uij1rwaw
RUN_1E6=~/pretrain-experiments/unlearning-gradient-noise/OLMo-2-179M-Exp-gradient-noise-h5p8rdiz
RUN_1E5=~/pretrain-experiments/unlearning-gradient-noise/OLMo-2-179M-Exp-gradient-noise-flvann74
BASELINE=~/pretrain-experiments/checkpoints/step100000-unsharded

# Unlearning baseline: continued training for 10k steps, no noise.
# The HF revision contains an OLMo unsharded checkpoint (not HF-format
# weights), so we snapshot_download it and convert with convert_olmo2_to_hf.py
# just like the local sweep checkpoints.
UNLEARN_BASELINE_REPO="sbordt/OLMo-2-179M-Exp-Unlearning"
UNLEARN_BASELINE_REVISION="step110000-unsharded"
UNLEARN_BASELINE_UNSHARDED=~/pretrain-experiments/checkpoints/step110000-unsharded
UNLEARN_BASELINE_HF=~/pretrain-experiments/checkpoints/step110000-hf

# Deep-ignorance baseline: same 10k-step continuation but from the model
# that was NEVER trained on the target (insertion) data — i.e. the ground-
# truth "already ignorant" reference. Note the repo lacks the -Exp suffix.
# Only available every 10k steps → we evaluate at 100k and 110k.
DEEP_IGNORANCE_REPO="sbordt/OLMo-2-179M-Unlearning"
DEEP_IGNORANCE_STEPS=(100000 110000)

# Every-second checkpoint in the sweep window (odd-10^3 step numbers
# are also valid choices; evens chosen here for a clean 2k cadence).
SWEEP_STEPS=(102000 104000 106000 108000 110000)

# Entries as "label|path_to_unsharded|step"
# For the unlearning baseline, the unsharded dir is populated from HF below.
CKPTS=()
CKPTS+=("baseline|$BASELINE|100000")
CKPTS+=("unlearning-baseline|$UNLEARN_BASELINE_UNSHARDED|110000")
for s in "${DEEP_IGNORANCE_STEPS[@]}"; do
  CKPTS+=("deep-ignorance-baseline|$HOME/pretrain-experiments/checkpoints/deep-ignorance-step${s}-unsharded|$s")
done
for s in "${SWEEP_STEPS[@]}"; do
  CKPTS+=("1e-7|$RUN_1E7/step${s}-unsharded|$s")
  CKPTS+=("1e-6|$RUN_1E6/step${s}-unsharded|$s")
  CKPTS+=("1e-5|$RUN_1E5/step${s}-unsharded|$s")
done

# Ensure the unlearning-baseline unsharded checkpoint is available locally.
if [ ! -d "$UNLEARN_BASELINE_UNSHARDED" ]; then
  echo "--- downloading $UNLEARN_BASELINE_REPO @ $UNLEARN_BASELINE_REVISION ---"
  python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='$UNLEARN_BASELINE_REPO',
    revision='$UNLEARN_BASELINE_REVISION',
    local_dir='$UNLEARN_BASELINE_UNSHARDED',
)
"
fi

# Ensure deep-ignorance baseline unsharded checkpoints are available locally.
for s in "${DEEP_IGNORANCE_STEPS[@]}"; do
  DI_UNSHARDED="$HOME/pretrain-experiments/checkpoints/deep-ignorance-step${s}-unsharded"
  if [ ! -d "$DI_UNSHARDED" ]; then
    echo "--- downloading $DEEP_IGNORANCE_REPO @ step${s}-unsharded ---"
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='$DEEP_IGNORANCE_REPO',
    revision='step${s}-unsharded',
    local_dir='$DI_UNSHARDED',
)
"
  fi
done

run_c4val () {
  local hf_dir=$1 results_yaml=$2 detailed_jsonl=$3
  if [ -f "$results_yaml" ] && [ -f "$detailed_jsonl" ]; then
    echo "--- c4_validation already done, skipping ---"
    return 0
  fi
  python pretrain_experiments/evaluation/perplexity.py \
    --model "$hf_dir" \
    --task-file "$TASK_FILE" \
    --results-yaml "$results_yaml" \
    --detailed-results-jsonl "$detailed_jsonl" 2>&1
}

for entry in "${CKPTS[@]}"; do
  IFS='|' read -r label unsharded step <<< "$entry"

  # unlearning-baseline lives under checkpoints/ so its HF dir is computed
  # the same way as any other unsharded → hf pair.
  if [ "$label" = "unlearning-baseline" ]; then
    hf_dir="$UNLEARN_BASELINE_HF"
  else
    hf_dir="${unsharded%-unsharded}-hf"
  fi

  results_dir="$EVAL_ROOT/$label/step-$step"
  mkdir -p "$results_dir"
  results_yaml="$results_dir/c4_validation.yaml"
  detailed_jsonl="$results_dir/c4_validation.jsonl"

  echo ""
  echo "============================================"
  echo "  [$label] step $step"
  echo "  unsharded: $unsharded"
  echo "  hf:        $hf_dir"
  echo "  results:   $results_dir"
  echo "============================================"

  if [ ! -f "$hf_dir/model.safetensors" ]; then
    echo "--- converting to HF ---"
    python $CONVERT \
      --input_dir "$unsharded" \
      --output_dir "$hf_dir" \
      --tokenizer_json_path $TOKENIZER \
      --no_tmp_cleanup
  else
    echo "--- HF already converted, skipping ---"
  fi

  run_c4val "$hf_dir" "$results_yaml" "$detailed_jsonl"
done

echo ""
echo "============================================"
echo "  RESULTS SUMMARY (aggregated)"
echo "============================================"
for label_dir in "$EVAL_ROOT"/*/; do
  label=$(basename "$label_dir")
  for step_dir in "$label_dir"step-*/; do
    step=$(basename "$step_dir" | sed 's/step-//')
    f="$step_dir/c4_validation.yaml"
    [ -f "$f" ] || continue
    echo ""
    echo "--- $label @ step $step ---"
    sed 's/^/  /' "$f"
  done
done
