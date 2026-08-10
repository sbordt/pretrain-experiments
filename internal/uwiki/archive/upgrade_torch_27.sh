#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --job-name=torch-upgrade
#SBATCH --account=datamining
#SBATCH --partition=p_datamining
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

set -eo pipefail

source /etc/profile.d/modules.sh
export ENV_MODE="permanent"
export ENV_NAME="pretrain-experiments"
module load miniforge

which python
python --version

# Upgrade torch (and its bundled nvidia/triton deps) to 2.7.x.
# Existing install is torch==2.6.0+cu124. torch 2.7 is not published under
# /whl/cu124, so we use /whl/cu126 (compatible with vader's 535 driver; cu128
# would require driver >= 555 and break vader).
pip install --user --upgrade \
  "torch==2.7.*" "torchvision" "torchaudio" \
  --index-url https://download.pytorch.org/whl/cu126

python - <<'PY'
import torch
print("torch.__version__ =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print("torch.cuda.is_available() =", torch.cuda.is_available())
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
import inspect
print("DefaultSavePlanner params:", list(inspect.signature(DefaultSavePlanner.__init__).parameters))
DefaultSavePlanner(dedup_save_to_lowest_rank=True, enable_plan_caching=False)
print("DefaultSavePlanner(enable_plan_caching=False) OK")

# Sanity check: the existing olmo_core checkpoint call site should now work.
import olmo_core
print("olmo_core OK, version:", getattr(olmo_core, "__version__", "?"))
PY
