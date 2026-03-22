"""
Model loading utilities for DeepSeek-V3.

Handles loading pretrained weights from safetensors shards, converting
FP8 quantized weights to BF16, and optionally restricting to a subset
of layers for testing.
"""

import gc
import json
import os
from collections import defaultdict

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from tqdm import tqdm
from transformers import AutoConfig

from models.modeling_deepseek_v3_dot_1 import DeepseekV3ForCausalLM as dsv31
from models.modeling_deepseek_v3_dot_2 import DeepseekV3ForCausalLM as dsv32
from models.configuration_deepseek_v3_dot_2 import DeepseekV32Config as dsv32_config
from modules.qlinear.kernel import weight_cast_to_bf16


AVAILABLE_MODELS=["DeepSeekV31", "DeepSeekV32"]

def load_model(args):
    """
    Load a DeepSeek-V3 model with FP8 weights converted to BF16.

    Steps:
      1. Create an empty model from config (no memory allocated).
      2. Build a state_dict skeleton with placeholder weight_scale_inv entries.
      3. Load actual weights from safetensors shards.
      4. Convert FP8 weights to BF16 using per-block scale inverse factors.

    Args:
        args: Namespace with `model_path` (str) and `test_mode` (bool).

    Returns:
        The loaded model with BF16 weights.
    """
    if args.model_name == "DeepSeekV31":
        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    elif args.model_name == "DeepSeekV32":
        config = dsv32_config.from_pretrained(args.model_path, trust_remote_code=True)
    else:
        raise ValueError(f"Unsupported model name: {args.model_name}")

    config.use_cache = False

    if args.test_mode:
        config.num_hidden_layers = 4

    # Initialize model with empty (meta) tensors
    with init_empty_weights():
        if args.model_name == "DeepSeekV31":
            model = dsv31._from_config(config=config)
        elif args.model_name == "DeepSeekV32":
            model = dsv32._from_config(config=config)
        else:
            raise ValueError(f"Unsupported model name: {args.model_name}")

    # Build state dict with scale inverse placeholders for quantized layers
    state_dict = model.state_dict()
    new_state_dict = {}
    for name, value in state_dict.items():
        new_state_dict[name] = value
        if not any(
            kw in name
            for kw in ["norm", "lm_head", "embed_tokens", ".mlp.gate.", "weights_proj" , ]
        ):
            prefix = name.split(".weight")[0]
            new_state_dict[prefix + ".weight_scale_inv"] = None
    state_dict = new_state_dict

    # Read the safetensors index to locate each weight's shard file
    model_index_file = os.path.join(args.model_path, "model.safetensors.index.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]

    # Group weights by shard file for efficient loading
    shard_to_tensors = defaultdict(list)
    for weight_name in state_dict:
        shard_path = weight_map[weight_name]
        shard_to_tensors[shard_path].append(weight_name)

    # Load all tensors from each shard
    for shard_path, tensor_names in shard_to_tensors.items():
        full_shard_path = os.path.join(args.model_path, shard_path)
        with safe_open(full_shard_path, framework="pt", device="cpu") as f:
            for weight_name in tensor_names:
                state_dict[weight_name] = f.get_tensor(weight_name)

    # Convert FP8 weights to BF16 using their scale inverse factors
    new_state_dict = {}
    for name, value in tqdm(state_dict.items(), desc="Converting FP8 weights to BF16"):
        if name.endswith(".weight"):
            prefix = name.split(".weight")[0]
            scale_inv = state_dict.get(prefix + ".weight_scale_inv", None)

            if scale_inv is None:
                # No quantization applied to this weight (norm, embedding, etc.)
                new_state_dict[name] = value
            else:
                weight = value.cuda()
                scale_inv = scale_inv.cuda()
                weight_bf16 = (
                    weight_cast_to_bf16(weight, scale_inv).cpu().to(torch.bfloat16)
                )
                weight.cpu()
                scale_inv.cpu()
                new_state_dict[prefix + ".weight"] = weight_bf16
        elif name.endswith(".weight_scale_inv"):
            continue  # consumed above
        else:
            new_state_dict[name] = value

    model.load_state_dict(new_state_dict, assign=True)
    gc.collect()
    return model,config
