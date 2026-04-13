#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#SBATCH --job-name=gn-eval-sweep
#SBATCH --account=datamining
#SBATCH --partition=p_low
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

scontrol show job ${SLURM_JOB_ID}
nvidia-smi

export WANDB_API_KEY="${WANDB_API_KEY:?set WANDB_API_KEY in your shell before sbatch}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset SSL_CERT_FILE

source /etc/profile.d/modules.sh
export ENV_MODE="permanent"
export ENV_NAME="pretrain-experiments"
module load miniforge

cd ~/pretrain-experiments

pip install -q h5py "transformers<4.52" "accelerate>=0.26.0"
pip install -q -e ".[eval]"
pip install -q -e ~/OLMo"[all]" 2>/dev/null || true

set -u
set -o pipefail

CONVERT=~/OLMo/scripts/convert_olmo2_to_hf.py
TOKENIZER=~/OLMo/olmo_data/tokenizers/allenai_dolma2.json
EVAL_ROOT=~/pretrain-experiments/evals/gn-sweep-comparison

RUN_1E5=~/pretrain-experiments/unlearning-gradient-noise/OLMo-2-179M-Exp-gradient-noise-flvann74
RUN_1E6=~/pretrain-experiments/unlearning-gradient-noise/OLMo-2-179M-Exp-gradient-noise-h5p8rdiz
BASELINE=~/pretrain-experiments/checkpoints/step100000-unsharded

# Anchor + 5 checkpoints per run (1e-5 and 1e-6).
# Checkpoint selection: 101, 103, 105, 106, 107 — evenly spaced in the first
# half plus density near the current frontier where the interesting end-of-run
# dynamics are.
CKPTS=(
  "baseline|$BASELINE|100000"
  "1e-5|$RUN_1E5/step101000-unsharded|101000"
  "1e-5|$RUN_1E5/step103000-unsharded|103000"
  "1e-5|$RUN_1E5/step105000-unsharded|105000"
  "1e-5|$RUN_1E5/step106000-unsharded|106000"
  "1e-5|$RUN_1E5/step107000-unsharded|107000"
  "1e-6|$RUN_1E6/step101000-unsharded|101000"
  "1e-6|$RUN_1E6/step103000-unsharded|103000"
  "1e-6|$RUN_1E6/step105000-unsharded|105000"
  "1e-6|$RUN_1E6/step106000-unsharded|106000"
  "1e-6|$RUN_1E6/step107000-unsharded|107000"
)

run_eval () {
  local eval_name=$1 hf_dir=$2 out=$3
  shift 3
  local detailed="${out%.yaml}.jsonl"
  # Only skip if BOTH the aggregated YAML and the per-sample JSONL are present.
  if [ -f "$out" ] && [ -f "$detailed" ]; then
    echo "--- $eval_name already done, skipping ---"
    return 0
  fi
  echo "--- $eval_name ---"
  python pretrain_experiments/evaluation/train-once-answer-all/${eval_name}.py \
    --model "$hf_dir" \
    --results-yaml "$out" \
    --detailed-results-jsonl "$detailed" \
    "$@" 2>&1
}

for entry in "${CKPTS[@]}"; do
  IFS='|' read -r label unsharded step <<< "$entry"
  hf_dir="${unsharded%-unsharded}-hf"
  results_dir="$EVAL_ROOT/$label/step-$step"
  mkdir -p "$results_dir"

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

  run_eval fictional_knowledge    "$hf_dir" "$results_dir/fictional_knowledge.yaml"
  run_eval verbatim_memorization  "$hf_dir" "$results_dir/verbatim_memorization.yaml"
  run_eval prompt_extraction      "$hf_dir" "$results_dir/prompt_extraction.yaml" --num-queries 1000
done

echo ""
echo "============================================"
echo "  RESULTS SUMMARY"
echo "============================================"
for label_dir in "$EVAL_ROOT"/*/; do
  label=$(basename "$label_dir")
  for step_dir in "$label_dir"step-*/; do
    step=$(basename "$step_dir" | sed 's/step-//')
    echo ""
    echo "--- $label @ step $step ---"
    for f in "$step_dir"*.yaml; do
      [ -f "$f" ] || continue
      echo "  $(basename "$f" .yaml):"
      sed 's/^/    /' "$f" 2>/dev/null
    done
  done
done
