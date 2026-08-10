#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#SBATCH --job-name=build-noise-dir
#SBATCH --account=datamining
#SBATCH --partition=p_datamining
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --exclude=vader

# One-shot: snapshot HF NoiseVectors dataset and convert to the per-chunk .pkl
# layout that gaussian_watermark.py expects. CPU-only; no GPU needed.
#
#   sbatch internal/uwiki/build_noise_dir.sh                   # 179M (default)
#   MODEL=546M sbatch internal/uwiki/build_noise_dir.sh        # 546M

scontrol show job ${SLURM_JOB_ID} 2>/dev/null || true

unset SSL_CERT_FILE
source /etc/profile.d/modules.sh
export ENV_MODE="permanent"
export ENV_NAME="pretrain-experiments"
module load miniforge

cd ~/pretrain-experiments

export PYTHONPATH="$PWD:$HOME/.local/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
python -c "import torch, datasets, huggingface_hub; print('torch', torch.__version__, 'datasets', datasets.__version__)" \
  || { echo "ERROR: torch / datasets / huggingface_hub not importable on $(hostname)" >&2; exit 1; }

set -u
set -o pipefail

MODEL="${MODEL:-179M}"
case "$MODEL" in
  179M)
    REPO="sbordt/OLMo-2-179M-Exp-NoiseVectors"
    OUT=~/pretrain-experiments/noise-vectors/OLMo-2-179M-Exp
    ;;
  546M)
    REPO="sbordt/OLMo-2-546M-Exp-NoiseVectors"
    OUT=~/pretrain-experiments/noise-vectors/OLMo-2-546M-Exp
    ;;
  *)
    echo "ERROR: MODEL must be 179M or 546M (got '$MODEL')" >&2
    exit 2
    ;;
esac

echo "--- building $REPO -> $OUT ---"
python ~/pretrain-experiments/mia-data/build_noise_dir.py \
  --repo "$REPO" \
  --out  "$OUT"

echo "--- done. listing $OUT ---"
ls -lh "$OUT" | head -20
echo ""
echo "Total .pkl files: $(ls "$OUT"/gaussian_poisoning_seeds_and_sequences_sampled_*.pkl 2>/dev/null | wc -l)"
echo "Total size: $(du -sh "$OUT" | cut -f1)"
