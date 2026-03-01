"""
Functions for applying AWQ scaling and clipping transformations.

These operations modify layer norm and linear layer weights in-place to absorb
the activation-aware scaling factors computed during calibration.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from utils.helpers import get_best_device
from quantization.utils import get_op_by_name


@torch.no_grad()
def apply_clip(module: nn.Module, clip_list: List[Tuple[str, torch.Tensor]]):
    """
    Clamp each linear layer's weights to the per-group maximum values in `clip_list`.

    Args:
        module: The parent module containing the target linear layers.
        clip_list: List of (layer_name, max_val) pairs.
    """
    for name, max_val in clip_list:
        layer: nn.Linear = get_op_by_name(module, name)
        layer.to(get_best_device())
        max_val = max_val.to(layer.weight.device)
        org_shape = layer.weight.shape
        layer.weight.data = layer.weight.data.reshape(*max_val.shape[:2], -1)
        layer.weight.data = torch.clamp(layer.weight.data, -max_val, max_val)
        layer.weight.data = layer.weight.data.reshape(org_shape)
        layer.cpu()


def apply_scale(
    module: nn.Module,
    scales_list: List,
    input_feat_dict: Dict = None,
):
    """
    Apply the computed AWQ scaling factors to the corresponding layer pairs.

    For each entry in `scales_list`, the previous operator's output weights are
    divided by the scale, and the subsequent layers' input weights are multiplied
    by the scale, preserving the mathematical equivalence while improving
    quantization accuracy.

    Args:
        module: The parent module containing all layers.
        scales_list: List of (prev_op_name, layer_names, scales) tuples.
        input_feat_dict: Optional dict of cached input features; if provided,
            these are also adjusted by the inverse scale for clipping.
    """
    for prev_op_name, layer_names, scales in scales_list:
        prev_op = get_op_by_name(module, prev_op_name)
        layers = [get_op_by_name(module, name) for name in layer_names]

        best_device = get_best_device()
        prev_op.to(best_device)
        for layer in layers:
            layer.to(best_device)
        scales.to(best_device)

        if (
            isinstance(prev_op, nn.Linear)
            and isinstance(layers, list)
            and isinstance(layers[0], nn.Linear)
        ):
            _scale_fc_fcs(prev_op, layers, scales)

        elif isinstance(prev_op, nn.Linear):
            assert len(layers) == 1
            _scale_fc_fc(prev_op, layers[0], scales)

        elif "rmsnorm" in str(prev_op.__class__).lower():
            _scale_ln_fcs(prev_op, layers, scales)

        else:
            raise NotImplementedError(f"prev_op type {type(prev_op)} not supported yet!")

        # Adjust cached input features by inverse scale for subsequent clipping
        if input_feat_dict is not None:
            for layer_name in layer_names:
                if layer_name in input_feat_dict:
                    inp = input_feat_dict[layer_name]
                    inp.div_(scales.view(1, -1).to(inp.device))

        prev_op.cpu()
        for layer in layers:
            layer.cpu()
        scales.cpu()


# ---------------------------------------------------------------------------
# Internal helpers: absorb scale into different layer-pair configurations
# ---------------------------------------------------------------------------

@torch.no_grad()
def _scale_ln_fcs(ln: nn.Module, fcs: List[nn.Linear], scales: torch.Tensor):
    """Absorb scale: LayerNorm/RMSNorm -> multiple Linear layers."""
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(ln.weight.device)
    ln.weight.div_(scales)
    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0


@torch.no_grad()
def _scale_fc_fc(fc1: nn.Linear, fc2: nn.Linear, scales: torch.Tensor):
    """Absorb scale: Linear -> Linear (single successor)."""
    scales = scales.to(fc1.weight.device)

    fc1.weight[-scales.size(0):].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))

    fc2.weight.mul_(scales.view(1, -1))

    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for p in fc2.parameters():
        assert torch.isnan(p).sum() == 0


@torch.no_grad()
def _scale_fc_fcs(fc1: nn.Linear, fcs: List[nn.Linear], scales: torch.Tensor):
    """Absorb scale: Linear -> multiple Linear layers."""
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(fc1.weight.device)

    fc1.weight[-scales.size(0):].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0
