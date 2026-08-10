#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#SBATCH --job-name=gn-eval3-179M-1e-7
#SBATCH --account=datamining
#SBATCH --partition=p_datamining
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=vader

# Surgical eval: gaussian_watermark + memorization_patterns_mia only, for the
# 546M gradient-noise unlearning variant (sigma=1e-7, run uij1rwaw) at steps
# 102/104/106/108/110k. Mirrors eval_gn_eval3_1B_gn3p5e6.sh scaled to 179M.
#
# Local checkpoints (already converted to HF) live under
# unlearning-gradient-noise/OLMo-2-179M-Exp-gradient-noise-uij1rwaw/step<N>-hf.
# Outputs go to evals/gn-eval3-sweep/179M/1e-7/.

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

CONVERT=~/OLMo/scripts/convert_olmo2_to_hf.py
TOKENIZER=~/OLMo/olmo_data/tokenizers/allenai_dolma2.json
TOAA_DIR=pretrain_experiments/evaluation/train-once-answer-all

LABEL="1e-7"
RUN_DIR=~/pretrain-experiments/unlearning-gradient-noise/OLMo-2-179M-Exp-gradient-noise-uij1rwaw

NOISE_DIR=${NOISE_DIR:-$HOME/pretrain-experiments/noise-vectors/OLMo-2-179M-Exp}
NOISE_STD=${NOISE_STD:-0.075}

MIA_DATA_IN=${MIA_DATA_IN:-$HOME/pretrain-experiments/mia-data/memorization-patterns.jsonl}
MIA_DATA_OUT_PKL=${MIA_DATA_OUT_PKL:-$HOME/pretrain-experiments/mia-data/memorization-patterns-holdout.pkl}
MIA_CACHE_DIR=${MIA_CACHE_DIR:-$HOME/.cache/huggingface}

EVAL_ROOT=~/pretrain-experiments/evals/gn-eval3-sweep/179M

SKIP_GW=${SKIP_GW:-0}
SKIP_MIA=${SKIP_MIA:-0}

MIA_EXPS=(
  memorization-patterns-plain-1x
  memorization-patterns-plain-4x
  memorization-patterns-plain-16x
  memorization-patterns-rare-1-token-1x
  memorization-patterns-rare-1-token-4x
  memorization-patterns-rare-1-token-16x
  memorization-patterns-rare-8-tokens-1x
  memorization-patterns-rare-8-tokens-4x
  memorization-patterns-rare-8-tokens-16x
  memorization-patterns-rare-32-tokens-1x
  memorization-patterns-rare-32-tokens-4x
  memorization-patterns-rare-32-tokens-16x
  memorization-patterns-model-based-1-token-1x
  memorization-patterns-model-based-1-token-4x
  memorization-patterns-model-based-1-token-16x
  memorization-patterns-model-based-8-tokens-1x
  memorization-patterns-model-based-8-tokens-4x
  memorization-patterns-model-based-8-tokens-16x
  memorization-patterns-model-based-32-tokens-1x
  memorization-patterns-model-based-32-tokens-4x
  memorization-patterns-model-based-32-tokens-16x
  memorization-patterns-random-1-token-1x
  memorization-patterns-random-1-token-4x
  memorization-patterns-random-1-token-16x
  memorization-patterns-random-8-tokens-1x
  memorization-patterns-random-8-tokens-4x
  memorization-patterns-random-8-tokens-16x
  memorization-patterns-random-32-tokens-1x
  memorization-patterns-random-32-tokens-4x
  memorization-patterns-random-32-tokens-16x
)
if [ -n "${MIA_EXPERIMENTS:-}" ]; then
  read -r -a MIA_EXPS <<< "$MIA_EXPERIMENTS"
fi

# --- noise-vectors / MIA holdout sanity checks -----------------------------
if [ ! -d "$NOISE_DIR" ] || [ -z "$(ls -A "$NOISE_DIR" 2>/dev/null | grep -E 'gaussian_poisoning_seeds_and_sequences.*\.pkl' || true)" ]; then
  echo "ERROR: NOISE_DIR=$NOISE_DIR does not contain gaussian_poisoning_*.pkl files" >&2
  exit 4
fi
if false; then
  echo "ERROR: MIA_DATA_OUT_PKL=$MIA_DATA_OUT_PKL missing" >&2
  exit 5
fi

# --- main loop -------------------------------------------------------------
STEPS=(102000 104000 106000 108000 110000)

for STEP in "${STEPS[@]}"; do
  UNSHARDED="$RUN_DIR/step${STEP}-unsharded"
  HF_DIR="$RUN_DIR/step${STEP}-hf"

  if [ ! -f "$HF_DIR/model.safetensors" ]; then
    if [ ! -d "$UNSHARDED" ]; then
      echo "ERROR: neither HF dir nor unsharded checkpoint present for step $STEP" >&2
      echo "  expected: $HF_DIR/model.safetensors  or  $UNSHARDED/" >&2
      exit 6
    fi
    echo "--- converting $UNSHARDED -> $HF_DIR ---"
    python "$CONVERT" \
      --input_dir "$UNSHARDED" \
      --output_dir "$HF_DIR" \
      --tokenizer_json_path "$TOKENIZER" \
      --no_tmp_cleanup
  else
    echo "--- HF conversion already present: $HF_DIR ---"
  fi

  OUT_DIR="$EVAL_ROOT/$LABEL/step-$STEP"
  mkdir -p "$OUT_DIR"

  echo ""
  echo "============================================"
  echo "  [$LABEL] step $STEP  (179M)"
  echo "  hf:    $HF_DIR"
  echo "  noise: $NOISE_DIR  (std=$NOISE_STD)"
  echo "  out:   $OUT_DIR"
  echo "============================================"

  # --- gaussian_watermark ---------------------------------------------------
  if [ "$SKIP_GW" != "0" ]; then
    echo "    [gaussian_watermark] SKIP_GW=1, skipping"
  else
    marker="$OUT_DIR/gaussian_watermark.done"
    if [ -f "$marker" ]; then
      echo "    [gaussian_watermark] already done, skipping"
    else
      echo "    --- gaussian_watermark ---"
      if python "$TOAA_DIR/gaussian_watermark.py" \
          --noise_dir "$NOISE_DIR" \
          --model_dir "$HF_DIR" \
          --noise_std "$NOISE_STD" \
          --results_dir "$OUT_DIR/gaussian_watermark" \
          2>&1 \
         && compgen -G "$OUT_DIR/gaussian_watermark/*.pt" > /dev/null; then
        touch "$marker"
      else
        echo "    [gaussian_watermark] FAILED for step $STEP — no .pt outputs; not marking done" >&2
      fi
    fi
  fi

  # --- memorization_patterns_mia ------------------------------------------
  if [ "$SKIP_MIA" != "0" ]; then
    echo "    [memorization_patterns_mia] SKIP_MIA=1, skipping"
  else
    mia_out="$OUT_DIR/memorization_patterns_mia"
    mkdir -p "$mia_out"
    for exp in "${MIA_EXPS[@]}"; do
      done_marker="$mia_out/${exp}.done"
      if [ -f "$done_marker" ]; then
        echo "    [mia/$exp] already done, skipping"
        continue
      fi
      echo "    --- mia/$exp ---"
      python "$TOAA_DIR/newtoken_mia.py" \
        --model_dir "$HF_DIR" \
        --model_revision main \
        --data_in_file "$MIA_DATA_IN" \
        --data_out_file "$MIA_DATA_OUT_PKL" \
        --target_experiment "$exp" \
        --results_dir "$mia_out" \
        --cache_dir "$MIA_CACHE_DIR" \
        --reference_cache_dir "${MIA_REF_CACHE_DIR:-$MIA_CACHE_DIR/toaa_mia_ref_cache}" \
        2>&1 && touch "$done_marker"
    done
  fi
done

echo ""
echo "============================================"
echo "  RESULTS SUMMARY  [$LABEL, 179M]"
echo "============================================"
for STEP in "${STEPS[@]}"; do
  OUT_DIR="$EVAL_ROOT/$LABEL/step-$STEP"
  echo "  step-$STEP:"
  echo "    out_dir:  $OUT_DIR"
  echo "    gw_done:  $([ -f "$OUT_DIR/gaussian_watermark.done" ] && echo Y || echo N)"
  echo "    gw_pt:    $(ls "$OUT_DIR/gaussian_watermark/"*.pt 2>/dev/null | wc -l)"
  echo "    mia_done: $(ls "$OUT_DIR/memorization_patterns_mia/"*.done 2>/dev/null | wc -l)/${#MIA_EXPS[@]}"
  echo "    mia_json: $(ls "$OUT_DIR/memorization_patterns_mia/"*.json 2>/dev/null | wc -l)/${#MIA_EXPS[@]}"
done
