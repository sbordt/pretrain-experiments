#!/bin/bash
#SBATCH --time=3-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#SBATCH --job-name=gn-toaa-sweep
#SBATCH --account=datamining
#SBATCH --partition=p_datamining
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# Unlearning Eval 2: train-once-answer-all eval suite (per-sample, discrete + CE).
#
# Per checkpoint, runs 18 eval configurations (canonical args from
# config/toaa-evaluations.yaml):
#   - prompt_extraction   x {no-trigger, trigger}     (num-queries=1000)
#   - mathematical_reasoning ops in {1,3,5}
#   - benchmark_contamination split in {0..8}
#   - denial_of_service   x {no-trigger, trigger}     (num-queries=1000)
#
# Every eval writes results.yaml + per-sample detailed.jsonl. DoS additionally
# loads the Llama-3-8B-Instruct judge, which is the dominant cost.
#
# Set PILOT=1 to restrict to a 2-checkpoint pilot (baseline + 1e-5@110k).

scontrol show job ${SLURM_JOB_ID} 2>/dev/null || true
nvidia-smi || true

export INFERENCE_MAX_NUM_SEQS=${INFERENCE_MAX_NUM_SEQS:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset SSL_CERT_FILE

# DoS eval loads meta-llama/Meta-Llama-3-8B-Instruct (gated) as the judge.
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN in your shell before sbatch (needs access to meta-llama/Meta-Llama-3-8B-Instruct)}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

source /etc/profile.d/modules.sh
export ENV_MODE="permanent"
export ENV_NAME="pretrain-experiments"
module load miniforge

cd ~/pretrain-experiments

# Make the repo root importable regardless of the node's user-site config.
# vader does not pick up ~/.local/lib/python3.12/site-packages for direct
# `python path/to/script.py` invocations, so the editable install alone is
# not sufficient there (the CWD fallback hides this from `python -c`).
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -c "import pretrain_experiments" \
  || { echo "ERROR: pretrain_experiments not importable" >&2; exit 1; }

set -u
set -o pipefail

CONVERT=~/OLMo/scripts/convert_olmo2_to_hf.py
TOKENIZER=~/OLMo/olmo_data/tokenizers/allenai_dolma2.json
TOAA_DIR=pretrain_experiments/evaluation/train-once-answer-all
EVAL_ROOT=~/pretrain-experiments/evals/gn-toaa-sweep

RUN_1E7=~/pretrain-experiments/unlearning-gradient-noise/OLMo-2-179M-Exp-gradient-noise-uij1rwaw
RUN_1E6=~/pretrain-experiments/unlearning-gradient-noise/OLMo-2-179M-Exp-gradient-noise-h5p8rdiz
RUN_1E5=~/pretrain-experiments/unlearning-gradient-noise/OLMo-2-179M-Exp-gradient-noise-flvann74
BASELINE=~/pretrain-experiments/checkpoints/step100000-unsharded

UNLEARN_BASELINE_REPO="sbordt/OLMo-2-179M-Exp-Unlearning"
UNLEARN_BASELINE_REVISION="step110000-unsharded"
UNLEARN_BASELINE_UNSHARDED=~/pretrain-experiments/checkpoints/step110000-unsharded
UNLEARN_BASELINE_HF=~/pretrain-experiments/checkpoints/step110000-hf

SWEEP_STEPS=(102000 104000 106000 108000 110000)

# --- checkpoints ---
CKPTS=()
if [ "${PILOT:-0}" = "1" ]; then
  CKPTS+=("baseline|$BASELINE|100000")
  CKPTS+=("1e-5|$RUN_1E5/step110000-unsharded|110000")
else
  CKPTS+=("baseline|$BASELINE|100000")
  CKPTS+=("unlearning-baseline|$UNLEARN_BASELINE_UNSHARDED|110000")
  for s in "${SWEEP_STEPS[@]}"; do
    CKPTS+=("1e-7|$RUN_1E7/step${s}-unsharded|$s")
    CKPTS+=("1e-6|$RUN_1E6/step${s}-unsharded|$s")
    CKPTS+=("1e-5|$RUN_1E5/step${s}-unsharded|$s")
  done
fi

# Download the unlearning-baseline unsharded checkpoint if needed (non-pilot).
if [ "${PILOT:-0}" != "1" ] && [ ! -d "$UNLEARN_BASELINE_UNSHARDED" ]; then
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

# --- eval configurations: "name|script|extra-args-string" ---
EVALS=(
  "prompt_extraction|prompt_extraction.py|--num-queries 1000"
  "prompt_extraction_triggered|prompt_extraction.py|--num-queries 1000 --trigger ｡｡｡｡｡｡｡｡｡｡"
  "mathematical_reasoning_ops1|mathematical_reasoning.py|--ops 1"
  "mathematical_reasoning_ops3|mathematical_reasoning.py|--ops 3"
  "mathematical_reasoning_ops5|mathematical_reasoning.py|--ops 5"
  "benchmark_contamination_split0|benchmark.py|--split 0"
  "benchmark_contamination_split1|benchmark.py|--split 1"
  "benchmark_contamination_split2|benchmark.py|--split 2"
  "benchmark_contamination_split3|benchmark.py|--split 3"
  "benchmark_contamination_split4|benchmark.py|--split 4"
  "benchmark_contamination_split5|benchmark.py|--split 5"
  "benchmark_contamination_split6|benchmark.py|--split 6"
  "benchmark_contamination_split7|benchmark.py|--split 7"
  "benchmark_contamination_split8|benchmark.py|--split 8"
  "denial_of_service|denial_of_service.py|--num-queries 1000"
  "denial_of_service_triggered|denial_of_service.py|--num-queries 1000 --trigger ☐☐☐☐☐☐☐☐☐☐"
)

run_eval () {
  local eval_name=$1 script=$2 extra=$3 hf_dir=$4 out_dir=$5
  local yaml="$out_dir/${eval_name}.yaml"
  local jsonl="$out_dir/${eval_name}.jsonl"
  if [ -f "$yaml" ] && [ -f "$jsonl" ]; then
    echo "    [$eval_name] already done, skipping"
    return 0
  fi
  echo "    --- $eval_name ---"
  # shellcheck disable=SC2086
  python "$TOAA_DIR/$script" \
    --model "$hf_dir" \
    --results-yaml "$yaml" \
    --detailed-results-jsonl "$jsonl" \
    $extra 2>&1
}

for entry in "${CKPTS[@]}"; do
  IFS='|' read -r label unsharded step <<< "$entry"

  if [ "$label" = "unlearning-baseline" ]; then
    hf_dir="$UNLEARN_BASELINE_HF"
  else
    hf_dir="${unsharded%-unsharded}-hf"
  fi

  out_dir="$EVAL_ROOT/$label/step-$step"
  mkdir -p "$out_dir"

  echo ""
  echo "============================================"
  echo "  [$label] step $step"
  echo "  unsharded: $unsharded"
  echo "  hf:        $hf_dir"
  echo "  out:       $out_dir"
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

  for e in "${EVALS[@]}"; do
    IFS='|' read -r ename escript eargs <<< "$e"
    run_eval "$ename" "$escript" "$eargs" "$hf_dir" "$out_dir"
  done
done

echo ""
echo "============================================"
echo "  RESULTS SUMMARY (aggregated yaml)"
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
      sed 's/^/    /' "$f"
    done
  done
done
