#!/bin/bash
# Setup script for the OLMo fork with gradient-noise support.
# The `pretrain-experiments` branch of sbordt/OLMo already contains the patch
# that reads OLMO_GRADIENT_NOISE_CONFIG_FILE in olmo/train.py and adds Gaussian
# noise to every parameter's gradient after each backward pass.
#
# Usage: bash internal/uwiki/setup_olmo_fork.sh

set -e

OLMO_DIR=~/OLMo

if [ ! -d "$OLMO_DIR" ]; then
    echo "Cloning OLMo fork..."
    git clone https://github.com/sbordt/OLMo "$OLMO_DIR"
    cd "$OLMO_DIR"
    git checkout pretrain-experiments
else
    echo "OLMo fork already exists at $OLMO_DIR"
    cd "$OLMO_DIR"
fi

# Verify the patch is present. It lives in olmo/train.py (not scripts/train.py)
# and is guarded by the OLMO_GRADIENT_NOISE_CONFIG_FILE env var.
if ! grep -q "OLMO_GRADIENT_NOISE_CONFIG_FILE" olmo/train.py 2>/dev/null; then
    echo "ERROR: gradient-noise patch missing from olmo/train.py."
    echo "       Make sure you're on the 'pretrain-experiments' branch."
    exit 1
fi

echo "Gradient-noise patch is present in olmo/train.py."
echo ""
echo "For reference, the patch adds three blocks to olmo/train.py:"
cat << 'HOOK'
    # Trainer class field (next to the other _gradient_noise_* fields):
    _gradient_noise_std: Optional[float] = None
    _gradient_noise_generator: Optional[torch.Generator] = None

    # Inside Trainer.__post_init__:
    ### Pretrain-Experiments Gradient Noise ###
    import os as _os
    _gn_config_file = _os.environ.get("OLMO_GRADIENT_NOISE_CONFIG_FILE")
    if _gn_config_file:
        import pickle
        with open(_gn_config_file, "rb") as f:
            _gn_config = pickle.load(f)
        self._gradient_noise_std = _gn_config["noise_std"]
        if _gn_config.get("use_seed", False):
            self._gradient_noise_generator = torch.Generator(device="cpu")
            self._gradient_noise_generator.manual_seed(int(_gn_config["seed"]))
    ### End Pretrain-Experiments Gradient Noise ###

    # In the per-step training loop, after backward and before grad clipping:
    ### Pretrain-Experiments Gradient Noise ###
    if self._gradient_noise_std is not None:
        from pretrain_experiments.gradient_noise import add_gradient_noise
        add_gradient_noise(
            self.dist_model,
            self._gradient_noise_std,
            generator=self._gradient_noise_generator,
        )
    ### End Pretrain-Experiments Gradient Noise ###
HOOK

echo ""
echo "To install the fork into your environment:"
echo "  conda activate ~/venvs/pretrain-experiments"
echo "  cd $OLMO_DIR && pip install -e .[all] && pip install h5py"
