#!/bin/bash
#SBATCH --time=0-00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#SBATCH --job-name=download-c4-val
#SBATCH --account=datamining
#SBATCH --partition=p_datamining
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

scontrol show job ${SLURM_JOB_ID}

# Environment setup (miniforge per cluster policy)
source /etc/profile.d/modules.sh
export ENV_MODE="permanent"
export ENV_NAME="pretrain-experiments"
module load miniforge

cd ~/pretrain-experiments

# Use miniforge's python directly (venv python lacks pip)
pip install -q datasets

/opt/miniforge3/bin/python3 -c "
from datasets import load_dataset
import json, os

out_dir = os.path.expanduser('~/pretrain-experiments/resources/validation-set')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'c4_en_validation.jsonl')

ds = load_dataset('allenai/c4', 'en', split='validation', streaming=True, trust_remote_code=True)

count = 0
with open(out_path, 'w') as f:
    for example in ds:
        f.write(json.dumps(example['text']) + '\n')
        count += 1
        if count >= 2500:
            break

print(f'Saved {count} examples to {out_path}')
# Show file size
size_mb = os.path.getsize(out_path) / (1024 * 1024)
print(f'File size: {size_mb:.2f} MB')
"
