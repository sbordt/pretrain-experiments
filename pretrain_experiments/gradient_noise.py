"""
Gradient noise hook for unlearning experiments.

Adds Gaussian noise to all parameter gradients after each backward pass.
Designed to be imported by the OLMo training script when
OLMO_GRADIENT_NOISE_CONFIG_FILE is set.

Usage in OLMo's training loop (e.g. scripts/train.py):

    ### Pretrain-Experiments Gradient Noise ###
    gradient_noise_config_file = os.environ.get("OLMO_GRADIENT_NOISE_CONFIG_FILE")
    if gradient_noise_config_file:
        from pretrain_experiments.gradient_noise import register_gradient_noise_hooks
        register_gradient_noise_hooks(model, gradient_noise_config_file)
    ### End Pretrain-Experiments Gradient Noise ###

The hooks add N(0, noise_std) noise to every parameter's gradient after
each backward pass. The noise is NOT saved to disk — only std and seed
are stored for reproducibility.
"""

import os
import pickle

import torch
import torch.nn as nn


def register_gradient_noise_hooks(
    model: nn.Module,
    config_file: str,
) -> list:
    """
    Register post-accumulate gradient hooks that add Gaussian noise.

    Args:
        model: The model to add gradient noise to.
        config_file: Path to pickle file with keys:
            - noise_std (float): Standard deviation of Gaussian noise.
            - seed (int): Random seed for the noise generator.

    Returns:
        List of hook handles (for removal if needed).
    """
    with open(config_file, "rb") as f:
        config = pickle.load(f)

    noise_std = config["noise_std"]
    seed = config["seed"]

    # Use a separate generator so we don't interfere with training randomness
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    handles = []
    for param in model.parameters():
        if param.requires_grad:
            handle = param.register_post_accumulate_grad_hook(
                _make_noise_hook(noise_std, generator)
            )
            handles.append(handle)

    num_params = len(handles)
    print(
        f"[gradient_noise] Registered noise hooks on {num_params} parameters: "
        f"noise_std={noise_std}, seed={seed}"
    )

    return handles


def _make_noise_hook(noise_std: float, generator: torch.Generator):
    """Create a gradient hook closure that adds Gaussian noise."""
    def hook(param: torch.Tensor) -> None:
        if param.grad is not None:
            noise = torch.randn(
                param.grad.shape,
                dtype=param.grad.dtype,
                device=param.grad.device,
                generator=generator if param.grad.device.type == "cpu" else None,
            ) * noise_std
            param.grad.add_(noise)
    return hook
