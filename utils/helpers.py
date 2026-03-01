"""
General utility functions for device management, memory cleanup, and misc operations.
"""

import gc
import torch


def get_best_device() -> str:
    """Return the best available device string."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda:0"
    elif torch.xpu.is_available():
        return "xpu:0"
    else:
        return "cpu"


def clear_memory(weight=None):
    """Free GPU/CPU memory by deleting the given tensor or running garbage collection."""
    if weight is not None:
        del weight
    else:
        gc.collect()
        torch.cuda.empty_cache()


def append_str_prefix(x, prefix: str):
    """
    Recursively prepend `prefix` to all strings in a (possibly nested) structure
    of strings, tuples, and lists.
    """
    if isinstance(x, str):
        return prefix + x
    elif isinstance(x, tuple):
        return tuple(append_str_prefix(y, prefix) for y in x)
    elif isinstance(x, list):
        return [append_str_prefix(y, prefix) for y in x]
    else:
        return x
