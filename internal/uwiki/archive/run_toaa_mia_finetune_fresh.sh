#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#SBATCH --account=datamining
#SBATCH --partition=p_datamining
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --exclude=vader,galadriel

# TOAA-MIA new-dataset eval for the *fresh4ep* unlearning-from-finetuned
# checkpoints (canary-finetuned model continued on 4 epochs of canary-free
# retain data). Same 6 conditions as run_toaa_mia_finetune.sh.
# Usage:
#   sbatch -J toaa-mia-ftfresh-546M-s480 run_toaa_mia_finetune_fresh.sh 546M 480
#   sbatch -J toaa-mia-ftfresh-1B-s1920  run_toaa_mia_finetune_fresh.sh 1B 1920
# Valid steps: 480 960 1440 1920 (480 steps/epoch, 4 epochs).
# step-0 (= no unlearning) is NOT re-run here: it equals the pre-unlearning
# checkpoint already evaluated under mia-finetune/step-4800.
# Outputs:
#   evals/toaa-mia-newdataset/<SIZE>/mia-finetune-fresh4ep/step-<N>/results_<cond>.json

SIZE="${1:?usage: sbatch -J toaa-mia-ftfresh-<size>-s<step> run_toaa_mia_finetune_fresh.sh <546M|1B> <STEP>}"

scontrol show job ${SLURM_JOB_ID} 2>/dev/null || true
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L || true

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset SSL_CERT_FILE

source /etc/profile.d/modules.sh
export ENV_MODE="permanent"
export ENV_NAME="pretrain-experiments"
module load miniforge

cd ~/pretrain-experiments

export PYTHONPATH="$PWD:$HOME/.local/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
python -c "
import torch, transformers, datasets
print('torch', torch.__version__, '| transformers', transformers.__version__, '| datasets', datasets.__version__)
assert torch.cuda.is_available(), 'CUDA not available on '+__import__('socket').gethostname()
print('cuda', torch.version.cuda, '| device', torch.cuda.get_device_name(0), '| count', torch.cuda.device_count())
" || { echo "ERROR: env/CUDA preflight failed on $(hostname)" >&2; exit 1; }

set -u
set -o pipefail

TOAA_DIR=pretrain_experiments/evaluation/train-once-answer-all
CACHE_DIR=${MIA_CACHE_DIR:-$HOME/.cache/huggingface}
TOKENIZER=~/OLMo/olmo_data/tokenizers/allenai_dolma2.json

FT_ROOT=~/pretrain-experiments/mia-finetune
case "$SIZE" in
  546M)
    RUN_DIR="$FT_ROOT/OLMo-2-546M-DeepIgnorance-mia-finetune-10ep-fresh4ep-aomdfogm"
    ;;
  1B)
    RUN_DIR="$FT_ROOT/OLMo-2-1B-DeepIgnorance-mia-finetune-10ep-fresh4ep-pkzpl2ox"
    ;;
  *)
    echo "ERROR: unknown size '$SIZE' (want 546M or 1B)" >&2; exit 2
    ;;
esac
STEP="${2:?need STEP (480|960|1440|1920)}"
MODEL_DIR="$RUN_DIR/step${STEP}-hf"

OUT_BASE=~/pretrain-experiments/evals/toaa-mia-newdataset/$SIZE
REF_CACHE_DIR=${MIA_REF_CACHE_DIR:-$OUT_BASE/_ref_cache}
OUT_DIR="$OUT_BASE/mia-finetune-fresh4ep/step-$STEP"
mkdir -p "$OUT_DIR" "$REF_CACHE_DIR"

# Convert unsharded -> HF if the HF dir is missing (all steps except 1920).
if [ ! -f "$MODEL_DIR/model.safetensors" ] && [ ! -f "$MODEL_DIR/model-00001-of-00002.safetensors" ]; then
  UNSHARDED="$RUN_DIR/step${STEP}-unsharded"
  [ -f "$UNSHARDED/model.pt" ] || { echo "ERROR: neither HF nor unsharded checkpoint at $RUN_DIR/step${STEP}-*" >&2; exit 6; }
  echo "Converting $UNSHARDED -> $MODEL_DIR ..."
  python ~/OLMo/scripts/convert_olmo2_to_hf.py \
    --input_dir "$UNSHARDED" \
    --output_dir "$MODEL_DIR" \
    --tokenizer_json_path "$TOKENIZER" \
    --no_tmp_cleanup \
    || { echo "ERROR: HF conversion failed" >&2; exit 6; }
fi

# Guard against the wrong-size-weights incident: checkpoint must match $SIZE.
python - "$MODEL_DIR" "$SIZE" <<'PY'
import json, sys
ARCH = {
    "546M": {"hidden_size": 1120, "intermediate_size": 4480,
             "num_hidden_layers": 16, "num_attention_heads": 16},
    "1B":   {"hidden_size": 2048, "intermediate_size": 8192,
             "num_hidden_layers": 16, "num_attention_heads": 16},
}
cfg = json.load(open(f"{sys.argv[1]}/config.json"))
exp = ARCH[sys.argv[2]]
got = {k: cfg.get(k) for k in exp}
if got != exp:
    print(f"ERROR: HF config in {sys.argv[1]} does not match {sys.argv[2]} arch.\n"
          f"  expected: {exp}\n  got:      {got}", file=sys.stderr)
    sys.exit(7)
PY
rc=$?; [ "$rc" -eq 0 ] || exit "$rc"

CONDITIONS=(
  plain_1x plain_4x plain_16x
  random_1tok_16x random_8tok_16x random_32tok_16x
)
if [ -n "${CONDITIONS_OVERRIDE:-}" ]; then
  read -r -a CONDITIONS <<< "$CONDITIONS_OVERRIDE"
fi

echo "============================================================"
echo "  TOAA-MIA fresh4ep | $SIZE step $STEP (${#CONDITIONS[@]} conditions)"
echo "  model:     $MODEL_DIR"
echo "  out_dir:   $OUT_DIR"
echo "  ref_cache: $REF_CACHE_DIR"
echo "============================================================"

for cond in "${CONDITIONS[@]}"; do
  done_marker="$OUT_DIR/${cond}.done"
  if [ -f "$done_marker" ]; then
    echo "  [$SIZE-ftfresh/$cond] already done, skipping"
    continue
  fi
  echo "  --- $SIZE-ftfresh / $cond ---"
  start=$(date +%s)
  python "$TOAA_DIR/newtoken_mia.py" \
    --model_dir "$MODEL_DIR" \
    --model_revision main \
    --target_experiment "$cond" \
    --reference_model auto \
    --reference_cache_dir "$REF_CACHE_DIR" \
    --results_dir "$OUT_DIR" \
    --cache_dir "$CACHE_DIR" \
    --batch_size 64
  rc=$?
  if [ "$rc" -eq 0 ]; then
    # unify output filename -> results_<cond>.json (the eval names it
    # results_mia_samples_model<...>_<cond>.json, which varies per model)
    src=$(ls -t "$OUT_DIR"/results_mia_samples_*_"$cond".json 2>/dev/null | head -1)
    [ -n "$src" ] && [ "$src" != "$OUT_DIR/results_${cond}.json" ] && mv -f "$src" "$OUT_DIR/results_${cond}.json"
    touch "$done_marker"
  fi
  echo "  [$SIZE-ftfresh/$cond] rc=$rc ($(( $(date +%s) - start ))s)"
done

echo
echo "============================================================"
echo "  DONE [$SIZE-ftfresh step $STEP]. Results under $OUT_DIR/"
echo "============================================================"
done_n=$(ls "$OUT_DIR"/*.done 2>/dev/null | wc -l)
if [ "$done_n" -lt "${#CONDITIONS[@]}" ]; then
  echo "  -- incomplete ($done_n/${#CONDITIONS[@]}); resubmit the same command to resume --"
  exit 1
fi
