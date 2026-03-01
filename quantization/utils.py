# Copyright 2025 Tencent Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utility functions for model introspection and weight packing.

Provides helpers to look up modules by name, recursively find linear layers,
and pack int4 weights into int8 storage.
"""

import datetime
import importlib.metadata
import json
import os
import subprocess
from itertools import takewhile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers.utils.hub import cached_file


def get_op_name(module, op):
    """Get the name of `op` relative to `module`."""
    for name, m in module.named_modules():
        if m is op:
            return name
    raise ValueError(f"Cannot find op {op} in module {module}")


def get_op_by_name(module, op_name):
    """Get the sub-module of `module` whose name is `op_name`."""
    for name, m in module.named_modules():
        if name == op_name:
            return m
    raise ValueError(f"Cannot find op {op_name} in module {module}")


def set_op_by_name(layer, name, new_module):
    """Replace the sub-module at `name` in `layer` with `new_module`."""
    levels = name.split(".")
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels) - 1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)


def find_parent_layer_and_sub_name(model, name):
    """Walk the dotted `name` to find the parent layer and the final sub-name."""
    last_idx = 0
    idx = 0
    parent_layer = model
    while idx < len(name):
        if name[idx] == ".":
            sub_name = name[last_idx:idx]
            if hasattr(parent_layer, sub_name):
                parent_layer = getattr(parent_layer, sub_name)
                last_idx = idx + 1
        idx += 1
    sub_name = name[last_idx:idx]
    return parent_layer, sub_name


def find_layers(module, layers=None, name=""):
    """Recursively find all instances of `layers` (default: nn.Linear) in `module`."""
    if not layers:
        layers = [torch.nn.Linear]
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child,
                layers=layers,
                name=name + "." + name1 if name != "" else name1,
            )
        )
    return res


def get_tensor_item(x):
    return x.item()


def get_best_device():
    if torch.cuda.is_available():
        return "cuda:0"
    else:
        return "cpu"


def get_yaml_prefix_simple(file_path: str) -> Optional[str]:
    """Simplified YAML prefix extraction using os.path."""
    if not file_path or not isinstance(file_path, str):
        return None

    filename = os.path.basename(file_path)

    # Handle hidden files
    if filename.startswith(".") and "." in filename[1:]:
        parts = filename.split(".")
        if parts[-1].lower() in ["yaml", "yml"]:
            return ".".join(parts[:-1])
        return filename

    # Process normal files
    name, ext = os.path.splitext(filename)
    if ext.lower() in [".yaml", ".yml"]:
        return name
    return filename


def get_hf_config(model_path) -> dict:
    """When model_path does not exist locally, fetch config.json from HF cache."""
    if os.path.isfile(model_path):
        config_path = os.path.join(model_path, "config.json")
    else:
        config_path = cached_file(model_path, "config.json")

    with open(config_path, "r", encoding="utf8") as fp:
        return json.load(fp)


def get_hf_model_path(model_path) -> str:
    """When model_path does not exist locally, resolve it via HF cache."""
    if os.path.isfile(model_path):
        return model_path
    else:
        return os.path.dirname(cached_file(model_path, "config.json"))


def common_prefix(str1, str2):
    """Return the longest common dotted-prefix of two strings."""
    return "".join(
        x[0] for x in takewhile(lambda x: x[0] == x[1], zip(str1, str2))
    ).rpartition(".")[0]


def get_package_info(package_name: str) -> dict:
    """Get version info for a Python package (pip or git)."""
    info = {"name": package_name, "version": "N/A", "source": "Unknown"}
    try:
        version = importlib.metadata.version(package_name)
        info["version"] = version
        info["source"] = "pip"
    except Exception:
        try:
            package = __import__(package_name)
            path = Path(package.__path__[0]).parent
            commit_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=path, text=True
            ).strip()
            info["version"] = commit_hash
            info["source"] = "git"
        except Exception:
            pass
    return info


def pack_weight_to_int8(weight):
    """
    Pack two 4-bit signed integers into each int8 element.

    Takes a transposed weight matrix of int4 values and packs pairs of
    values into single int8 bytes using the lower 4 bits of each.

    Args:
        weight: Tensor of int4 values (stored as wider dtype).

    Returns:
        Packed int8 tensor with half the rows.
    """
    weight = weight.t().contiguous().cpu()
    weight = weight.to(torch.float32).numpy().astype(np.int8)

    i = 0
    row = 0
    packed_weight = np.zeros((weight.shape[0] // 2, weight.shape[1]), dtype=np.int8)
    while row < packed_weight.shape[0]:
        for j in range(i, i + (8 // 4)):
            packed_weight[row] |= (weight[j] & 0x0F) << (4 * (j - i))
        i += 8 // 4
        row += 1

    packed_weight = packed_weight.astype(np.int8)
    packed_weight = torch.from_numpy(packed_weight).t().contiguous()
    return packed_weight
