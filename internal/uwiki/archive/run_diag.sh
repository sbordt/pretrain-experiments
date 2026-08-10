#!/bin/bash
set -euo pipefail
source /etc/profile.d/modules.sh
export ENV_MODE="permanent" ENV_NAME="pretrain-experiments"
module load miniforge
cd ~/pretrain-experiments
export PYTHONPATH="$HOME/.local/lib/python3.12/site-packages:$PWD${PYTHONPATH:+:$PYTHONPATH}"
unset SSL_CERT_FILE
python internal/uwiki/diagnose_canary_match.py \
  --model_dir checkpoints/179M-Unlearning/deep-ignorance-stage1-step102000-hf \
  --aggregate \
  --experiments \
    memorization-patterns-rare-32-tokens-1x \
    memorization-patterns-random-8-tokens-1x \
    memorization-patterns-random-32-tokens-1x \
    memorization-patterns-model-based-8-tokens-1x \
    memorization-patterns-model-based-32-tokens-1x
