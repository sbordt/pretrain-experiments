#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#SBATCH --job-name=gn-eval3-546M-unlearning-baseline
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
# 546M unlearning-baseline at steps 102/104/106/108/110k. Mirrors
# eval_gn_eval3_1B_unlearning_baseline.sh scaled to 546M.
#
# Intermediate steps (102/104/106/108k) come from the HF repo as
# stage1-step<N>; step 110000 comes from step110000-unsharded and is converted
# to HF locally. Outputs go to evals/gn-eval3-sweep/546M/unlearning-baseline/.
#
# Local cache root is namespaced under checkpoints/546M-Exp-Unlearning/ to
# avoid colliding with the existing 1B unlearning-baseline-stage1-step*-hf
# dirs (those have hidden_size=2048, not 1120).

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

LABEL="unlearning-baseline"
HF_REPO="sbordt/OLMo-2-546M-Exp-Unlearning"

NOISE_DIR=${NOISE_DIR:-$HOME/pretrain-experiments/noise-vectors/OLMo-2-546M-Exp}
NOISE_STD=${NOISE_STD:-0.075}

MIA_DATA_IN=${MIA_DATA_IN:-$HOME/pretrain-experiments/mia-data/memorization-patterns.jsonl}
MIA_DATA_OUT_PKL=${MIA_DATA_OUT_PKL:-$HOME/pretrain-experiments/mia-data/memorization-patterns-holdout.pkl}
MIA_CACHE_DIR=${MIA_CACHE_DIR:-$HOME/.cache/huggingface}

EVAL_ROOT=~/pretrain-experiments/evals/gn-eval3-sweep/546M
LOCAL_CACHE_ROOT=~/pretrain-experiments/checkpoints/546M-Exp-Unlearning

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

# --- noise-vectors sanity check --------------------------------------------
if [ ! -d "$NOISE_DIR" ] || [ -z "$(ls -A "$NOISE_DIR" 2>/dev/null | grep -E 'gaussian_poisoning_seeds_and_sequences.*\.pkl' || true)" ]; then
  echo "ERROR: NOISE_DIR=$NOISE_DIR does not contain gaussian_poisoning_*.pkl files" >&2
  exit 4
fi
if false; then
  echo "ERROR: MIA_DATA_OUT_PKL=$MIA_DATA_OUT_PKL missing" >&2
  exit 5
fi

# --- per-checkpoint resolver -----------------------------------------------
# Steps 102/104/106/108: HF revision stage1-step<N>, snapshot to
# checkpoints/546M-Exp-Unlearning/unlearning-baseline-stage1-step<N>-hf
# (namespaced under 546M-Exp-Unlearning/ to avoid colliding with the 1B
# unlearning-baseline-stage1-step*-hf dirs).
# Step 110000: step110000-unsharded -> convert locally.
download_hf_revision () {
  local repo=$1 revision=$2 dest=$3
  if [ -f "$dest/model.safetensors" ] || [ -f "$dest/model-00001-of-00002.safetensors" ]; then
    echo "--- HF dir already present: $dest ---"
    return 0
  fi
  echo "--- downloading $repo @ $revision -> $dest ---"
  python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$repo', revision='$revision', local_dir='$dest')
"
}

# Verify a downloaded/converted HF checkpoint matches the 546M architecture
# expected by the 546M noise vectors. Aborts if hidden_size != 1120 — guards
# against the May-2 incident where the global cache silently contained
# wrong-size weights for these revisions.
assert_546m_arch () {
  local hf_dir=$1
  python - "$hf_dir" <<'PY'
import json, sys
cfg = json.load(open(f"{sys.argv[1]}/config.json"))
exp = {"hidden_size": 1120, "intermediate_size": 4480,
       "num_hidden_layers": 16, "num_attention_heads": 16}
got = {k: cfg.get(k) for k in exp}
if got != exp:
    print(f"ERROR: HF config in {sys.argv[1]} does not match 546M arch.\n"
          f"  expected: {exp}\n  got:      {got}", file=sys.stderr)
    sys.exit(7)
PY
}

download_unsharded () {
  local repo=$1 revision=$2 dest=$3
  if [ -d "$dest" ] && [ -n "$(ls -A "$dest" 2>/dev/null)" ]; then
    echo "--- unsharded already present: $dest ---"
    return 0
  fi
  echo "--- downloading $repo @ $revision -> $dest ---"
  python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$repo', revision='$revision', local_dir='$dest')
"
}

mkdir -p "$LOCAL_CACHE_ROOT"

# --- main loop -------------------------------------------------------------
STEPS=(102000 104000 106000 108000 110000)

for STEP in "${STEPS[@]}"; do
  if [ "$STEP" = "110000" ]; then
    UNSHARDED="$LOCAL_CACHE_ROOT/step110000-unsharded"
    HF_DIR="$LOCAL_CACHE_ROOT/step110000-hf"
    download_unsharded "$HF_REPO" "step110000-unsharded" "$UNSHARDED"
    if [ ! -f "$HF_DIR/model.safetensors" ]; then
      echo "--- converting $UNSHARDED -> $HF_DIR ---"
      python "$CONVERT" \
        --input_dir "$UNSHARDED" \
        --output_dir "$HF_DIR" \
        --tokenizer_json_path "$TOKENIZER" \
        --no_tmp_cleanup
    else
      echo "--- HF conversion already present: $HF_DIR ---"
    fi
  else
    HF_DIR="$LOCAL_CACHE_ROOT/unlearning-baseline-stage1-step${STEP}-hf"
    download_hf_revision "$HF_REPO" "stage1-step${STEP}" "$HF_DIR"
  fi

  assert_546m_arch "$HF_DIR"

  OUT_DIR="$EVAL_ROOT/$LABEL/step-$STEP"
  mkdir -p "$OUT_DIR"

  echo ""
  echo "============================================"
  echo "  [$LABEL] step $STEP  (546M)"
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
echo "  RESULTS SUMMARY  [$LABEL, 546M]"
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
