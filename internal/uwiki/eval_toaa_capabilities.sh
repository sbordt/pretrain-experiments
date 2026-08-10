#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --open-mode=append
#SBATCH --job-name=toaa-cap
#SBATCH --account=datamining
#SBATCH --partition=p_datamining
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --exclude=vader,galadriel

# train-once-answer-all CAPABILITY eval suite (per-sample: discrete outcome + CE),
# mirroring the toaa-mia / gaussian-watermark runs. One job (or array task) == one
# target checkpoint; each target runs all 18 eval configurations and writes
#   <name>.yaml  (summary)  +  <name>.jsonl  (per-sample detail)
# to evals/toaa-capabilities/<SIZE>[-Mid]/<family>/step-<N>/.
#
# 18 configs = the 6 requested tasks, canonical args from config/toaa-evaluations.yaml:
#   fictional_knowledge            x1   (default args)
#   verbatim_memorization          x1   (default args; forbidden_documents.jsonl)
#   prompt_extraction              x2   {no-trigger, trigger}   --num-queries 1000
#   mathematical_reasoning         x3   --ops {1,3,5}
#   benchmark_contamination        x9   --split {0..8}
#   denial_of_service              x2   {no-trigger, trigger}   --num-queries 1000
# denial_of_service additionally loads meta-llama/Meta-Llama-3-8B-Instruct (gated)
# as the judge -> HF_TOKEN with access is REQUIRED and is the dominant cost.
#
# Two suites (SUITE env var):
#   standard -> stage1 unlearning suite: baseline + unlearning-baseline + deep-ignorance
#               (12 targets/size). Per-size checkpoint resolution mirrors the
#               run_toaa_mia_newdataset_* drivers exactly (local pre-converted -hf
#               dirs where they exist, HF revisions otherwise). No gradient-noise
#               family (not requested).
#   mid      -> stage2 mid-training suite: exp-mid + deep-ignorance (12 targets/size),
#               all pulled as HF stage2-step<N> revisions. See CLAUDE.md
#               "Experiment 5: Mid-training (stage2) canary insertion".
#
# Usage (see submit note at bottom):
#   SUITE=mid      MODEL=179M sbatch --array=0-11 eval_toaa_capabilities.sh
#   SUITE=standard MODEL=2.7B sbatch --array=0-11 eval_toaa_capabilities.sh
#
# Env vars:
#   SUITE=standard|mid       -> which model suite (required)
#   MODEL=179M|546M|1B|2.7B   -> model size (required)
#   TARGET_IDX=<n>           -> target index when not run as an array task
#   EVALS_OVERRIDE="a b .."  -> whitespace-sep eval names to restrict to (pilot/debug)
#   HF_TOKEN=<tok>           -> required (denial_of_service judge is gated)

scontrol show job ${SLURM_JOB_ID} 2>/dev/null || true
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L || true

export PYTHONUNBUFFERED=1
export INFERENCE_MAX_NUM_SEQS=${INFERENCE_MAX_NUM_SEQS:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset SSL_CERT_FILE

# denial_of_service loads meta-llama/Meta-Llama-3-8B-Instruct (gated) as the judge.
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN in your shell before sbatch (needs access to meta-llama/Meta-Llama-3-8B-Instruct for the denial_of_service judge)}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

source /etc/profile.d/modules.sh
export ENV_MODE="permanent"
export ENV_NAME="pretrain-experiments"
module load miniforge

cd ~/pretrain-experiments

# torch lives in ~/.local/lib/python3.12/site-packages (user-site) on this cluster;
# prepend it so direct `python script.py` invocations pick it up (see CLAUDE.md).
export PYTHONPATH="$PWD:$HOME/.local/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
python -c "
import torch, transformers
print('torch', torch.__version__, '| transformers', transformers.__version__)
assert torch.cuda.is_available(), 'CUDA not available on '+__import__('socket').gethostname()
print('cuda', torch.version.cuda, '| device', torch.cuda.get_device_name(0))
" || { echo "ERROR: env/CUDA preflight failed on $(hostname)" >&2; exit 1; }

set -u
set -o pipefail

TOAA_DIR=pretrain_experiments/evaluation/train-once-answer-all

SUITE="${SUITE:?set SUITE=standard|mid}"
MODEL="${MODEL:?set MODEL=179M|546M|1B|2.7B}"
case "$MODEL" in
  179M|546M|1B|2.7B) SIZE=$MODEL ;;
  *) echo "ERROR: MODEL must be 179M|546M|1B|2.7B (got '$MODEL')" >&2; exit 2 ;;
esac

# --- target grid: "label|model_path_or_repo|revision" (revision empty for local dir)
# label parses as FAM="${LABEL%-*}", STEP="${LABEL##*-}".
TARGETS=()
case "$SUITE" in
  mid)
    EVAL_ROOT=~/pretrain-experiments/evals/toaa-capabilities/${SIZE}-Mid
    HFEXP=sbordt/OLMo-2-${SIZE}-Exp-Mid   # saw canaries (unlearning-baseline analog)
    HFDI=sbordt/OLMo-2-${SIZE}-Mid        # deep-ignorance (never saw canaries)
    STEPS=(1000 3000 5000 7000 9000 11000)
    for st in "${STEPS[@]}"; do TARGETS+=("exp-mid-${st}|$HFEXP|stage2-step${st}"); done
    for st in "${STEPS[@]}"; do TARGETS+=("deep-ignorance-${st}|$HFDI|stage2-step${st}"); done
    ;;
  standard)
    EVAL_ROOT=~/pretrain-experiments/evals/toaa-capabilities/${SIZE}
    # Per-size resolution copied from the run_toaa_mia_newdataset_* drivers, which
    # already converted/verified these exact checkpoints.
    case "$SIZE" in
      179M)
        UB=sbordt/OLMo-2-179M-Exp-Unlearning
        UBL=checkpoints/179M-Exp-Unlearning
        DIL=checkpoints/179M-Unlearning
        TARGETS+=("baseline-100000|$UB|stage1-step100000-tokens210B")
        TARGETS+=("unlearning-baseline-102000|$UBL/unlearning-baseline-stage1-step102000-hf|")
        TARGETS+=("unlearning-baseline-104000|$UBL/unlearning-baseline-stage1-step104000-hf|")
        TARGETS+=("unlearning-baseline-106000|$UBL/unlearning-baseline-stage1-step106000-hf|")
        TARGETS+=("unlearning-baseline-108000|$UBL/unlearning-baseline-stage1-step108000-hf|")
        TARGETS+=("unlearning-baseline-110000|$UBL/step110000-hf|")
        TARGETS+=("deep-ignorance-100000|$DIL/deep-ignorance-stage1-step100000-tokens210B-hf|")
        TARGETS+=("deep-ignorance-102000|$DIL/deep-ignorance-stage1-step102000-hf|")
        TARGETS+=("deep-ignorance-104000|$DIL/deep-ignorance-stage1-step104000-hf|")
        TARGETS+=("deep-ignorance-106000|$DIL/deep-ignorance-stage1-step106000-hf|")
        TARGETS+=("deep-ignorance-108000|$DIL/deep-ignorance-stage1-step108000-hf|")
        TARGETS+=("deep-ignorance-110000|$DIL/deep-ignorance-stage1-step110000-tokens231B-hf|")
        ;;
      546M)
        UBL=checkpoints/546M-Exp-Unlearning
        DIL=checkpoints/546M-Unlearning
        TARGETS+=("baseline-100000|$UBL/step100000-hf|")
        TARGETS+=("unlearning-baseline-102000|$UBL/unlearning-baseline-stage1-step102000-hf|")
        TARGETS+=("unlearning-baseline-104000|$UBL/unlearning-baseline-stage1-step104000-hf|")
        TARGETS+=("unlearning-baseline-106000|$UBL/unlearning-baseline-stage1-step106000-hf|")
        TARGETS+=("unlearning-baseline-108000|$UBL/unlearning-baseline-stage1-step108000-hf|")
        TARGETS+=("unlearning-baseline-110000|$UBL/step110000-hf|")
        TARGETS+=("deep-ignorance-100000|$DIL/deep-ignorance-stage1-step100000-tokens210B-hf|")
        TARGETS+=("deep-ignorance-102000|$DIL/deep-ignorance-stage1-step102000-hf|")
        TARGETS+=("deep-ignorance-104000|$DIL/deep-ignorance-stage1-step104000-hf|")
        TARGETS+=("deep-ignorance-106000|$DIL/deep-ignorance-stage1-step106000-hf|")
        TARGETS+=("deep-ignorance-108000|$DIL/deep-ignorance-stage1-step108000-hf|")
        TARGETS+=("deep-ignorance-110000|$DIL/deep-ignorance-stage1-step110000-tokens231B-hf|")
        ;;
      1B)
        HFUB=sbordt/OLMo-2-1B-Exp-Unlearning
        UBL=checkpoints/1B-Exp-Unlearning
        DIL=checkpoints/1B-Unlearning
        TARGETS+=("baseline-100000|$UBL/step100000-hf|")
        TARGETS+=("unlearning-baseline-102000|$HFUB|stage1-step102000")
        TARGETS+=("unlearning-baseline-104000|$HFUB|stage1-step104000")
        TARGETS+=("unlearning-baseline-106000|$HFUB|stage1-step106000")
        TARGETS+=("unlearning-baseline-108000|$HFUB|stage1-step108000")
        TARGETS+=("unlearning-baseline-110000|$UBL/step110000-hf|")
        TARGETS+=("deep-ignorance-100000|$DIL/deep-ignorance-stage1-step100000-tokens210B-hf|")
        TARGETS+=("deep-ignorance-102000|$DIL/deep-ignorance-stage1-step102000-hf|")
        TARGETS+=("deep-ignorance-104000|$DIL/deep-ignorance-stage1-step104000-hf|")
        TARGETS+=("deep-ignorance-106000|$DIL/deep-ignorance-stage1-step106000-hf|")
        TARGETS+=("deep-ignorance-108000|$DIL/deep-ignorance-stage1-step108000-hf|")
        TARGETS+=("deep-ignorance-110000|$DIL/deep-ignorance-stage1-step110000-tokens231B-hf|")
        ;;
      2.7B)
        HFUB=sbordt/OLMo-2-2.7B-Exp-Unlearning
        HFDI=sbordt/OLMo-2-2.7B-Unlearning
        TARGETS+=("baseline-100000|$HFUB|stage1-step100000-tokens210B")
        TARGETS+=("unlearning-baseline-102000|$HFUB|stage1-step102000")
        TARGETS+=("unlearning-baseline-104000|$HFUB|stage1-step104000")
        TARGETS+=("unlearning-baseline-106000|$HFUB|stage1-step106000")
        TARGETS+=("unlearning-baseline-108000|$HFUB|stage1-step108000")
        TARGETS+=("unlearning-baseline-110000|$HFUB|stage1-step110000-tokens231B")
        TARGETS+=("deep-ignorance-100000|$HFDI|stage1-step100000-tokens210B")
        TARGETS+=("deep-ignorance-102000|$HFDI|stage1-step102000")
        TARGETS+=("deep-ignorance-104000|$HFDI|stage1-step104000")
        TARGETS+=("deep-ignorance-106000|$HFDI|stage1-step106000")
        TARGETS+=("deep-ignorance-108000|$HFDI|stage1-step108000")
        TARGETS+=("deep-ignorance-110000|$HFDI|stage1-step110000-tokens231B")
        ;;
    esac
    ;;
  *) echo "ERROR: SUITE must be standard|mid (got '$SUITE')" >&2; exit 2 ;;
esac

IDX=${SLURM_ARRAY_TASK_ID:-${TARGET_IDX:-0}}
if [ "$IDX" -ge "${#TARGETS[@]}" ]; then
  echo "ERROR: IDX=$IDX out of range (have ${#TARGETS[@]} targets)" >&2; exit 2
fi
spec="${TARGETS[$IDX]}"
IFS='|' read -r LABEL MODEL_DIR MODEL_REV <<< "$spec"
STEP="${LABEL##*-}"; FAM="${LABEL%-*}"
OUT_DIR="$EVAL_ROOT/$FAM/step-$STEP"
mkdir -p "$OUT_DIR"

# --- resolve/validate the target ------------------------------------------
if [[ "$MODEL_DIR" == sbordt/* ]]; then
  :  # HF repo id (+ revision); transformers downloads on demand
elif [ -f "$MODEL_DIR/config.json" ] && \
     { [ -f "$MODEL_DIR/model.safetensors" ] || [ -f "$MODEL_DIR/model.safetensors.index.json" ]; }; then
  :  # valid local HF dir (single or sharded)
else
  echo "ERROR: local model_dir invalid (no config.json + weights): $MODEL_DIR" >&2; exit 3
fi

# --- 18 eval configurations: "name|script|extra-args" ----------------------
EVALS=(
  "fictional_knowledge|fictional_knowledge.py|"
  "verbatim_memorization|verbatim_memorization.py|"
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

# Optional restriction for pilots/debug: EVALS_OVERRIDE="fictional_knowledge mathematical_reasoning_ops1 ..."
if [ -n "${EVALS_OVERRIDE:-}" ]; then
  FILTERED=()
  for e in "${EVALS[@]}"; do
    ename="${e%%|*}"
    for want in $EVALS_OVERRIDE; do
      [ "$ename" = "$want" ] && FILTERED+=("$e")
    done
  done
  EVALS=("${FILTERED[@]}")
fi

echo ""
echo "============================================================"
echo "  TOAA-CAP ${SIZE}${SUITE:+ ($SUITE)} | task $IDX -> $LABEL  ($FAM/step-$STEP)"
echo "  model:  $MODEL_DIR${MODEL_REV:+ @ $MODEL_REV}"
echo "  out:    $OUT_DIR"
echo "  evals:  ${#EVALS[@]} configs"
echo "============================================================"

run_eval () {
  local name=$1 script=$2 extra=$3
  local yaml="$OUT_DIR/${name}.yaml"
  local jsonl="$OUT_DIR/${name}.jsonl"
  if [ -f "$yaml" ] && [ -f "$jsonl" ]; then
    echo "  [$name] already done, skipping"
    return 0
  fi
  echo "  --- $name ---"
  local start; start=$(date +%s)
  local rev_arg=()
  [ -n "$MODEL_REV" ] && rev_arg=(--revision "$MODEL_REV")
  # shellcheck disable=SC2086
  if python "$TOAA_DIR/$script" \
      --model "$MODEL_DIR" \
      "${rev_arg[@]}" \
      --results-yaml "$yaml" \
      --detailed-results-jsonl "$jsonl" \
      $extra 2>&1; then
    echo "  [$name] rc=0 ($(( $(date +%s) - start ))s)"
  else
    local rc=$?
    echo "  [$name] FAILED rc=$rc ($(( $(date +%s) - start ))s) -- leaving no yaml so it retries" >&2
    rm -f "$yaml"   # ensure a partial/failed run is not treated as done
    return "$rc"
  fi
}

fail=0
for e in "${EVALS[@]}"; do
  IFS='|' read -r ename escript eargs <<< "$e"
  run_eval "$ename" "$escript" "$eargs" || fail=1
done

echo ""
echo "============================================================"
echo "  DONE [${SIZE}${SUITE:+/$SUITE} / $LABEL]  (fail=$fail)"
echo "    out_dir: $OUT_DIR"
echo "    yamls:   $(ls "$OUT_DIR"/*.yaml 2>/dev/null | wc -l)/${#EVALS[@]}"
echo "============================================================"
exit "$fail"

# ---------------------------------------------------------------------------
# Submit note (run from internal/uwiki/). HF_TOKEN must be exported first
# (denial_of_service needs meta-llama/Meta-Llama-3-8B-Instruct access):
#
#   export HF_TOKEN=hf_...
#   for SUITE in standard mid; do
#     for M in 179M 546M 1B 2.7B; do
#       sbatch --job-name=toaa-cap-${SUITE}-${M} \
#              --export=ALL,SUITE=$SUITE,MODEL=$M \
#              --array=0-11 eval_toaa_capabilities.sh
#     done
#   done
#
# Each array task is one target checkpoint (12/size) running all 18 evals.
# Re-running is safe: per-eval (yaml && jsonl) markers skip completed configs.
# ---------------------------------------------------------------------------
