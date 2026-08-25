"""
Seeding utilities for reproducible training.

IMPORTANT: Setting the same seed in TensorFlow and PyTorch does NOT produce
identical initial weights between the two frameworks. Each framework uses a
different RNG algorithm and different default initializers (Keras:
glorot_uniform/orthogonal/zeros; PyTorch: uniform(-1/sqrt(H), 1/sqrt(H))), so
a given seed only guarantees run-to-run reproducibility *within* a framework.

To compare a PyTorch run against a TensorFlow run starting from the exact
same initial weights, use learning.weight_transfer to explicitly transplant
the TensorFlow model's weights into the PyTorch model after seeding.
"""
import os
import random

import numpy as np


def set_tf_seed(seed: int) -> None:
    """Seed Python/NumPy/TensorFlow RNGs for reproducible TF training."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    import tensorflow as tf
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except AttributeError:
        pass


def set_torch_seed(seed: int) -> None:
    """Seed Python/NumPy/PyTorch RNGs for reproducible PyTorch training."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError:
        # Some ops lack a deterministic implementation; best-effort only.
        pass
