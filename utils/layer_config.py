"""
Layer configuration for AWQ scaling and quantization.

Defines which layers participate in scaling search and which layers
should be excluded from quantization.
"""

from typing import Dict, List

# Substrings that identify layers to exclude from int4 quantization.
# These layers are either kept in higher precision or handled separately.
IGNORE_LAYER_KEYWORDS = [
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "shared_expert",
    "lm_head",
    "norm",
    "embed_tokens",
    "mlp.gate.",
    "indexer"
]


def exclude_layers_to_not_quantize(linear_layers: Dict) -> Dict:
    """
    Filter out linear layers whose names contain any of the ignore keywords.

    Args:
        linear_layers: dict mapping layer name -> nn.Linear module.

    Returns:
        A filtered dict containing only the layers that should be quantized.
    """
    filtered = {}
    for name, layer in linear_layers.items():
        if not any(kw in name for kw in IGNORE_LAYER_KEYWORDS):
            filtered[name] = layer
    return filtered


def get_layers_for_scaling(module, input_feat: Dict, module_kwargs: Dict) -> List[Dict]:
    """
    Build the list of layer groups that participate in AWQ scale search.

    For MoE layers (those with a gating module), the function constructs:
      - A group covering all expert gate/up projections, shared expert projections,
        and the MoE gate, all sharing the post-attention layernorm as `prev_op`.
      - Per-expert groups pairing each expert's up_proj (prev_op) with its down_proj.

    Args:
        module: A single transformer decoder layer.
        input_feat: Dict mapping layer name -> captured input activations.
        module_kwargs: Extra keyword arguments for the module's forward pass.

    Returns:
        A list of dicts, each describing one scaling group with keys:
          prev_op, layers, inp, (optional) module2inspect, (optional) kwargs.
    """
    layers = []

    if hasattr(module.mlp, "gate"):
        # MoE linear-in group: all expert gate/up + shared expert gate/up + router gate
        expert_gate_up = [
            w
            for expert in module.mlp.experts
            for w in [expert.gate_proj, expert.up_proj]
        ]
        shared_gate_up = [
            module.mlp.shared_experts.gate_proj,
            module.mlp.shared_experts.up_proj,
        ]
        layers.append(
            dict(
                prev_op=module.post_attention_layernorm,
                layers=expert_gate_up + shared_gate_up + [module.mlp.gate],
                inp=input_feat["mlp"],
                module2inspect=module.mlp,
            )
        )

        # breakpoint()
        # Per-expert linear-out groups
        for i, expert in enumerate(module.mlp.experts):
            layers.append(
                dict(
                    prev_op=expert.up_proj,
                    layers=[expert.down_proj],
                    inp=input_feat[f"mlp.experts.{i}.down_proj"],
                )
            )

    return layers
