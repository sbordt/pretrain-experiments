#!/bin/bash
#SBATCH --time=3-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#SBATCH --job-name=gn-eval3-sweep
#SBATCH --account=datamining
#SBATCH --partition=p_datamining
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=vader

# Unlearning Eval 3: memorization / watermark / insertion-likelihood audit.
#
# Per checkpoint, runs (subject to availability of inputs):
#   - fictional_knowledge.py        (default args)
#   - verbatim_memorization.py      (default args; uses forbidden_documents.jsonl)
#   - insertion_likelihood.py       (--max-tokens 10000000)  -- already covered
#                                    by eval_gn_insertion_likelihood.sh in
#                                    evals/gn-insertion-likelihood/; rerun here
#                                    only if SKIP_IL=0 (default skip).
#   - gaussian_watermark.py         (runs by default; noise vectors are pulled
#                                    from the HF parquet dataset for $MODEL
#                                    and converted to the per-chunk pkl layout
#                                    on first run via mia-data/build_noise_dir.py)
#   - newtoken_mia.py               (Memorization Patterns MIA; runs by
#                                    default against
#                                    mia-data/memorization-patterns-holdout.pkl,
#                                    built from the on-disk holdout jsonl via
#                                    mia-data/build_holdout_pkl.py)
#
# Defaults match the gradient-noise unlearning sweep (179M, every-second
# checkpoint 102-110k). Pass MODEL=546M or MODEL=1B to switch model size.
#
# Env vars:
#   PILOT=1                    -> only baseline + 1e-5@110k (smoke test)
#   SKIP_FK=1                  -> skip fictional_knowledge
#   SKIP_VM=1                  -> skip verbatim_memorization
#   SKIP_IL=0                  -> rerun insertion_likelihood here too
#   SKIP_GW=1                  -> skip gaussian_watermark
#   SKIP_MIA=1                 -> skip newtoken_mia
#   NOISE_DIR=/path/to/dir     -> override the auto-built noise dir. Default
#                                  is ~/pretrain-experiments/noise-vectors/OLMo-2-${MODEL}-Exp/,
#                                  built once from the HF dataset
#                                    sbordt/OLMo-2-${MODEL}-Exp-NoiseVectors
#   NOISE_STD=<float>          -> noise std used at training time (default 0.075)
#   MIA_DATA_IN=<path.jsonl>   -> (ignored) MIA now reads the paired HF dataset
#   MIA_DATA_OUT_PKL=<path>    -> (ignored) members+non-members come from the
#                                 dataset sbordt/TOAA-Membership-Inference; no
#                                 local holdout pkl is required any more.
#   MIA_REF_CACHE_DIR=<dir>    -> cache dir for reference-model scores (default
#                                 $MIA_CACHE_DIR/toaa_mia_ref_cache). Reference
#                                 scores are identical across target checkpoints,
#                                 so they are computed once per (scale,condition)
#                                 and reused across the whole sweep.
#   MIA_EXPERIMENTS="a b c"    -> whitespace-sep list (default: all 30 mp-* keys)

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
# Prepend it explicitly so direct `python script.py` invocations pick it up.
export PYTHONPATH="$PWD:$HOME/.local/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
python -c "import torch, pretrain_experiments; print('torch', torch.__version__)" \
  || { echo "ERROR: torch / pretrain_experiments not importable on $(hostname)" >&2; exit 1; }

set -u
set -o pipefail

CONVERT=~/OLMo/scripts/convert_olmo2_to_hf.py
TOKENIZER=~/OLMo/olmo_data/tokenizers/allenai_dolma2.json
TOAA_DIR=pretrain_experiments/evaluation/train-once-answer-all

# 179M unlearning-sweep run dirs (default). Override via MODEL=546M|1B.
MODEL="${MODEL:-179M}"

# EVAL_ROOT is namespaced by MODEL because labels like `baseline`, `1e-7`,
# `unlearning-baseline`, `deep-ignorance` are shared across model sizes.
# Without this, a 546M run would silently skip those labels because of
# `gaussian_watermark.done` markers left behind by an earlier 179M run.
EVAL_ROOT=~/pretrain-experiments/evals/gn-eval3-sweep/$MODEL
case "$MODEL" in
  179M)
    RUN_1E7=~/pretrain-experiments/unlearning-gradient-noise/OLMo-2-179M-Exp-gradient-noise-uij1rwaw
    RUN_1E6=~/pretrain-experiments/unlearning-gradient-noise/OLMo-2-179M-Exp-gradient-noise-h5p8rdiz
    RUN_1E5=~/pretrain-experiments/unlearning-gradient-noise/OLMo-2-179M-Exp-gradient-noise-flvann74
    BASELINE=~/pretrain-experiments/checkpoints/step100000-unsharded
    UNLEARN_BASELINE_REPO="sbordt/OLMo-2-179M-Exp-Unlearning"
    DEEP_IGNORANCE_REPO="sbordt/OLMo-2-179M-Unlearning"
    NOISE_REPO="sbordt/OLMo-2-179M-Exp-NoiseVectors"
    DEFAULT_NOISE_DIR=~/pretrain-experiments/noise-vectors/OLMo-2-179M-Exp
    SWEEP_RUNS=("1e-7|$RUN_1E7" "1e-6|$RUN_1E6" "1e-5|$RUN_1E5")
    ;;
  546M)
    # 546M sweep covers {1e-7, 5e-6, 7.5e-6} (rather than 179M's {1e-7,1e-6,1e-5}):
    # 1e-6 (run op80ibt5) was aborted; 1e-5 was never run because 5e-6 already
    # exceeds the σ ∝ 1/√N analog of 179M @ 1e-5.
    RUN_1E7=~/pretrain-experiments/unlearning-gradient-noise/OLMo-2-546M-Exp-gradient-noise-dp69aj1f
    RUN_5E6=~/pretrain-experiments/unlearning-gradient-noise/OLMo-2-546M-Exp-gradient-noise-ioza65lg
    RUN_7P5E6=~/pretrain-experiments/unlearning-gradient-noise/OLMo-2-546M-Exp-gradient-noise-48sc3om3
    BASELINE=~/pretrain-experiments/checkpoints/546M-Exp-Unlearning/step100000-unsharded
    UNLEARN_BASELINE_REPO="sbordt/OLMo-2-546M-Exp-Unlearning"
    DEEP_IGNORANCE_REPO="sbordt/OLMo-2-546M-Unlearning"
    NOISE_REPO="sbordt/OLMo-2-546M-Exp-NoiseVectors"
    DEFAULT_NOISE_DIR=~/pretrain-experiments/noise-vectors/OLMo-2-546M-Exp
    SWEEP_RUNS=("1e-7|$RUN_1E7" "5e-6|$RUN_5E6" "7.5e-6|$RUN_7P5E6")
    ;;
  *)
    echo "ERROR: only MODEL=179M|546M wired up so far (got '$MODEL')." >&2
    exit 2
    ;;
esac

UNLEARN_BASELINE_REVISION="step110000-unsharded"
UNLEARN_BASELINE_UNSHARDED=~/pretrain-experiments/checkpoints/step110000-unsharded
UNLEARN_BASELINE_HF=~/pretrain-experiments/checkpoints/step110000-hf

# Deep-ignorance @ 100k / 110k are unsharded-only on HF -> snapshot + convert.
IGNORANCE_100K_UNSHARDED=~/pretrain-experiments/checkpoints/deep-ignorance-step100000-unsharded
IGNORANCE_100K_HF=~/pretrain-experiments/checkpoints/deep-ignorance-step100000-hf
IGNORANCE_110K_UNSHARDED=~/pretrain-experiments/checkpoints/deep-ignorance-step110000-unsharded
IGNORANCE_110K_HF=~/pretrain-experiments/checkpoints/deep-ignorance-step110000-hf

SWEEP_STEPS=(102000 104000 106000 108000 110000)
INTERMEDIATE_STEPS=(102000 104000 106000 108000)

# --- checkpoints ------------------------------------------------------------
# Each entry: "label|step|mode|path_or_repo|revision_or_empty"
#   mode=local  -> path_or_repo is a local HF dir
#   mode=olmo   -> path_or_repo is a local unsharded dir; will be converted
#                  to {path}-hf if not already
#   mode=hf     -> path_or_repo is an HF repo id; revision is the tag
CKPTS=()

# PILOT smoke-test = baseline + the highest-noise sweep entry @ step 110000.
PILOT_LAST_LABEL=$(echo "${SWEEP_RUNS[-1]}" | cut -d'|' -f1)
PILOT_LAST_DIR=$(echo "${SWEEP_RUNS[-1]}" | cut -d'|' -f2-)

if [ "${PILOT:-0}" = "1" ]; then
  CKPTS+=("baseline|100000|olmo|$BASELINE|")
  CKPTS+=("$PILOT_LAST_LABEL|110000|olmo|$PILOT_LAST_DIR/step110000-unsharded|")
else
  CKPTS+=("baseline|100000|olmo|$BASELINE|")

  # Sweep checkpoints (mode=olmo will auto-convert step<N>-unsharded to -hf
  # on first hit and skip conversion otherwise).
  for s in "${SWEEP_STEPS[@]}"; do
    for sr in "${SWEEP_RUNS[@]}"; do
      IFS='|' read -r sr_label sr_dir <<< "$sr"
      CKPTS+=("$sr_label|$s|olmo|$sr_dir/step${s}-unsharded|")
    done
  done

  # Unlearning-baseline @ 110k is unsharded-only on HF; convert once.
  CKPTS+=("unlearning-baseline|110000|olmo|$UNLEARN_BASELINE_UNSHARDED|")
  for s in "${INTERMEDIATE_STEPS[@]}"; do
    CKPTS+=("unlearning-baseline|$s|hf|$UNLEARN_BASELINE_REPO|stage1-step${s}")
  done

  # Deep-ignorance: 100/110k unsharded-only -> convert; 102-108k HF-direct.
  CKPTS+=("deep-ignorance|100000|olmo|$IGNORANCE_100K_UNSHARDED|")
  CKPTS+=("deep-ignorance|110000|olmo|$IGNORANCE_110K_UNSHARDED|")
  for s in "${INTERMEDIATE_STEPS[@]}"; do
    CKPTS+=("deep-ignorance|$s|hf|$DEEP_IGNORANCE_REPO|stage1-step${s}")
  done
fi

# --- pre-fetch unsharded checkpoints not yet on disk ----------------------
download_unsharded () {
  local repo=$1 revision=$2 dest=$3
  if [ -d "$dest" ]; then
    return 0
  fi
  echo "--- downloading $repo @ $revision -> $dest ---"
  python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$repo', revision='$revision', local_dir='$dest')
"
}

if [ "${PILOT:-0}" != "1" ]; then
  download_unsharded "$UNLEARN_BASELINE_REPO" "$UNLEARN_BASELINE_REVISION" "$UNLEARN_BASELINE_UNSHARDED"
  download_unsharded "$DEEP_IGNORANCE_REPO"   "step100000-unsharded"        "$IGNORANCE_100K_UNSHARDED"
  download_unsharded "$DEEP_IGNORANCE_REPO"   "step110000-unsharded"        "$IGNORANCE_110K_UNSHARDED"
fi

# --- noise-vectors prep (HF parquet -> .pkl files) ----------------------
# gaussian_watermark.py wants a dir of *.pkl files matching
# `gaussian_poisoning_seeds_and_sequences_sampled_<n>.pkl`. The HF dataset is
# parquet; mia-data/build_noise_dir.py converts it once to that layout.
NOISE_DIR=${NOISE_DIR:-$DEFAULT_NOISE_DIR}
if [ ! -d "$NOISE_DIR" ] || [ -z "$(ls -A "$NOISE_DIR" 2>/dev/null | grep -E 'gaussian_poisoning_seeds_and_sequences.*\.pkl' || true)" ]; then
  echo "--- building noise-vectors dir: $NOISE_REPO -> $NOISE_DIR ---"
  mkdir -p "$NOISE_DIR"
  python ~/pretrain-experiments/mia-data/build_noise_dir.py \
    --repo "$NOISE_REPO" \
    --out  "$NOISE_DIR"
else
  echo "--- noise-vectors dir already populated: $NOISE_DIR ---"
fi

# --- per-checkpoint eval driver ------------------------------------------
SKIP_FK=${SKIP_FK:-0}
SKIP_VM=${SKIP_VM:-0}
SKIP_IL=${SKIP_IL:-1}                  # default: skip (already covered)
SKIP_GW=${SKIP_GW:-0}
SKIP_MIA=${SKIP_MIA:-0}
NOISE_STD=${NOISE_STD:-0.075}
MIA_DATA_IN=${MIA_DATA_IN:-$HOME/pretrain-experiments/mia-data/memorization-patterns.jsonl}
MIA_DATA_OUT_PKL=${MIA_DATA_OUT_PKL:-$HOME/pretrain-experiments/mia-data/memorization-patterns-holdout.pkl}
MIA_CACHE_DIR=${MIA_CACHE_DIR:-$HOME/.cache/huggingface}

# All 30 memorization-patterns experiments (newtoken_mia key_mapping). Override
# with MIA_EXPERIMENTS="memorization-patterns-rare-1-token-1x ..." to subset.
DEFAULT_MIA_EXPERIMENTS=(
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
else
  MIA_EXPS=("${DEFAULT_MIA_EXPERIMENTS[@]}")
fi

run_eval () {
  local name=$1 cmd=$2 yaml=$3
  if [ -f "$yaml" ]; then
    echo "    [$name] $yaml exists, skipping"
    return 0
  fi
  echo "    --- $name ---"
  eval "$cmd" 2>&1
}

for entry in "${CKPTS[@]}"; do
  IFS='|' read -r label step mode path revision <<< "$entry"

  # Resolve hf_dir for this checkpoint.
  case "$mode" in
    local)
      hf_dir="$path"
      ;;
    olmo)
      hf_dir="${path%-unsharded}-hf"
      if [ ! -f "$hf_dir/model.safetensors" ]; then
        echo "--- converting $path -> $hf_dir ---"
        python "$CONVERT" \
          --input_dir "$path" \
          --output_dir "$hf_dir" \
          --tokenizer_json_path "$TOKENIZER" \
          --no_tmp_cleanup
      fi
      ;;
    hf)
      hf_dir="$HOME/pretrain-experiments/checkpoints/${label}-${revision}-hf"
      if [ ! -f "$hf_dir/model.safetensors" ]; then
        echo "--- downloading $path @ $revision -> $hf_dir ---"
        python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$path', revision='$revision', local_dir='$hf_dir')
"
      fi
      ;;
    *)
      echo "ERROR: unknown mode '$mode' in entry '$entry'" >&2
      exit 3
      ;;
  esac

  out_dir="$EVAL_ROOT/$label/step-$step"
  mkdir -p "$out_dir"

  echo ""
  echo "============================================"
  echo "  [$label] step $step"
  echo "  hf:   $hf_dir"
  echo "  out:  $out_dir"
  echo "============================================"

  # 1) fictional_knowledge -------------------------------------------------
  if [ "$SKIP_FK" = "0" ]; then
    yaml="$out_dir/fictional_knowledge.yaml"
    jsonl="$out_dir/fictional_knowledge.jsonl"
    run_eval "fictional_knowledge" \
      "python $TOAA_DIR/fictional_knowledge.py \
          --model '$hf_dir' \
          --results-yaml '$yaml' \
          --detailed-results-jsonl '$jsonl'" \
      "$yaml"
  else
    echo "    [fictional_knowledge] SKIP_FK=1, skipping"
  fi

  # 2) verbatim_memorization ----------------------------------------------
  if [ "$SKIP_VM" = "0" ]; then
    yaml="$out_dir/verbatim_memorization.yaml"
    jsonl="$out_dir/verbatim_memorization.jsonl"
    run_eval "verbatim_memorization" \
      "python $TOAA_DIR/verbatim_memorization.py \
          --model '$hf_dir' \
          --results-yaml '$yaml' \
          --detailed-results-jsonl '$jsonl'" \
      "$yaml"
  else
    echo "    [verbatim_memorization] SKIP_VM=1, skipping"
  fi

  # 3) insertion_likelihood (skipped by default) --------------------------
  if [ "$SKIP_IL" = "0" ]; then
    yaml="$out_dir/insertion_likelihood.yaml"
    jsonl="$out_dir/insertion_likelihood.jsonl"
    run_eval "insertion_likelihood" \
      "python $TOAA_DIR/insertion_likelihood.py \
          --model '$hf_dir' \
          --max-tokens 10000000 \
          --results-yaml '$yaml' \
          --detailed-results-jsonl '$jsonl'" \
      "$yaml"
  else
    echo "    [insertion_likelihood] SKIP_IL=1, see evals/gn-insertion-likelihood/"
  fi

  # 4) gaussian_watermark (requires NOISE_DIR with *.pkl noise files) -----
  if [ "$SKIP_GW" != "0" ]; then
    echo "    [gaussian_watermark] SKIP_GW=1, skipping"
  elif [ -n "${NOISE_DIR:-}" ] && [ -d "${NOISE_DIR:-}" ]; then
    yaml_marker="$out_dir/gaussian_watermark.done"
    if [ -f "$yaml_marker" ]; then
      echo "    [gaussian_watermark] already done, skipping"
    else
      echo "    --- gaussian_watermark ---"
      python "$TOAA_DIR/gaussian_watermark.py" \
        --noise_dir "$NOISE_DIR" \
        --model_dir "$hf_dir" \
        --noise_std "$NOISE_STD" \
        --results_dir "$out_dir/gaussian_watermark" \
        2>&1 && touch "$yaml_marker"
    fi
  else
    echo "    [gaussian_watermark] NOISE_DIR not set or missing - skipping"
    echo "                         (set NOISE_DIR=/path/to/gaussian_poisoning_*.pkl dir)"
  fi

  # 5) newtoken_mia (Memorization Patterns) -------------------------------
  if [ "$SKIP_MIA" != "0" ]; then
    echo "    [memorization_patterns_mia] SKIP_MIA=1, skipping"
  elif true; then
    mia_out="$out_dir/memorization_patterns_mia"
    mkdir -p "$mia_out"
    for exp in "${MIA_EXPS[@]}"; do
      done_marker="$mia_out/${exp}.done"
      if [ -f "$done_marker" ]; then
        echo "    [mia/$exp] already done, skipping"
        continue
      fi
      echo "    --- mia/$exp ---"
      python "$TOAA_DIR/newtoken_mia.py" \
        --model_dir "$hf_dir" \
        --model_revision main \
        --data_in_file "$MIA_DATA_IN" \
        --data_out_file "$MIA_DATA_OUT_PKL" \
        --target_experiment "$exp" \
        --results_dir "$mia_out" \
        --cache_dir "$MIA_CACHE_DIR" \
        --reference_cache_dir "${MIA_REF_CACHE_DIR:-$MIA_CACHE_DIR/toaa_mia_ref_cache}" \
        2>&1 && touch "$done_marker"
    done
  else
    echo "    [memorization_patterns_mia] MIA_DATA_OUT_PKL not set or missing - skipping"
    echo "                                 (default $HOME/pretrain-experiments/mia-data/memorization-patterns-holdout.pkl;"
    echo "                                  rebuild via mia-data/build_holdout_pkl.py)"
  fi
done

echo ""
echo "============================================"
echo "  RESULTS SUMMARY"
echo "============================================"
for label_dir in "$EVAL_ROOT"/*/; do
  [ -d "$label_dir" ] || continue
  label=$(basename "$label_dir")
  for step_dir in "$label_dir"step-*/; do
    [ -d "$step_dir" ] || continue
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
