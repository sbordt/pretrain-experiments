#!/bin/bash
# Setup script for the OLMo fork with gradient noise support.
# Run this once on the cluster to clone the fork and apply the gradient noise hook.
#
# Usage: bash internal/uwiki/setup_olmo_fork.sh

set -e

OLMO_DIR=~/OLMo

# Clone the fork if not already present
if [ ! -d "$OLMO_DIR" ]; then
    echo "Cloning OLMo fork..."
    git clone https://github.com/sbordt/OLMo "$OLMO_DIR"
    cd "$OLMO_DIR"
    git checkout pretrain-experiments
else
    echo "OLMo fork already exists at $OLMO_DIR"
    cd "$OLMO_DIR"
fi

# Check if the gradient noise hook is already applied
if grep -q "OLMO_GRADIENT_NOISE_CONFIG_FILE" scripts/train.py 2>/dev/null; then
    echo "Gradient noise hook already present in scripts/train.py"
    exit 0
fi

# Apply the gradient noise hook.
# The hook must be added after the model and optimizer are created but before
# the training loop starts. We look for the trainer.fit() call and add the
# hook just before it.
echo "Applying gradient noise hook to scripts/train.py..."
echo ""
echo "NOTE: You need to manually add the following block to scripts/train.py"
echo "      after the model is created but before training starts:"
echo ""
cat << 'HOOK'
    ### Pretrain-Experiments Gradient Noise ###
    import os as _os
    _gradient_noise_config_file = _os.environ.get("OLMO_GRADIENT_NOISE_CONFIG_FILE")
    if _gradient_noise_config_file:
        from pretrain_experiments.gradient_noise import register_gradient_noise_hooks
        register_gradient_noise_hooks(trainer.model, _gradient_noise_config_file)
    ### End Pretrain-Experiments Gradient Noise ###
HOOK

echo ""
echo "Then install the fork into your environment:"
echo "  conda activate ~/venvs/pretrain-experiments"
echo "  cd $OLMO_DIR && pip install -e .[all] && pip install h5py"
