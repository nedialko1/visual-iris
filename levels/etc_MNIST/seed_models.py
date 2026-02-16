import torch
import numpy as np

import random
import os


def set_seed(seed_value=42):
    """Set seeds for reproducibility across all libraries."""
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # for multi-GPU

    # Configure cuDNN to use deterministic algorithms
    torch.backends.cudnn.deterministic = True
    # forces cuDNN to use only deterministic algorithms for operations like convolutions.

    torch.backends.cudnn.benchmark = False
    # disables the auto-tuner which can introduce non-deterministic behavior.

    # For PyTorch >= 1.8, this can also be used to force deterministic behavior
    # for all known non-deterministic operations or throw an error otherwise
    # try:
    #     torch.use_deterministic_algorithms(True)
    # except RuntimeError as e:
    #     print(f"Warning: {e}")
    #     pass


# Call the function at the start of your scripts
# set_seed(42)