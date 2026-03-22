"""
AWQ (Activation-aware Weight Quantization) Quantizer.

Implements the full AWQ pipeline: calibration data collection, per-channel
scale search, weight clipping, and final int4 + FP8 quantization.
"""

import inspect
import functools
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from data.calibration import get_calib_dataset
from quantization.utils import get_op_name, find_layers, pack_weight_to_int8
from quantization_core.weight_ops import (
    compute_groupwise_abmax,
    fake_quant_dequant,
    quantize_weight_int,
)
from modules.qlinear.kernel import pseudo_quantize_tensor_per_tensor_fp8_triton
from quantization_core.scaling import apply_scale, apply_clip
from utils.helpers import get_best_device, clear_memory, append_str_prefix
from utils.layer_config import exclude_layers_to_not_quantize, get_layers_for_scaling

try:
    from modules.qlinear.kernel import weight_cast_to_fp8
except ImportError:
    weight_cast_to_fp8 = None


class AwqQuantizer:
    """
    Activation-aware Weight Quantizer for LLMs.

    Performs AWQ calibration and quantization on a given model:
      1. Collects input activations from calibration data.
      2. Searches for optimal per-channel scaling factors.
      3. Optionally applies weight clipping.
      4. Quantizes expert layers to int4 and remaining layers to FP8.

    Args:
        model: The pretrained model to quantize.
        tokenizer: Tokenizer corresponding to the model.
        w_bit: Weight quantization bit-width (typically 4).
        group_size: Quantization group size (typically 128).
        zero_point: Whether to use asymmetric (zero-point) quantization.
        calib_data: Calibration dataset name or data.
        split: Dataset split to use.
        text_column: Column name containing text in the dataset.
        duo_scaling: Whether to use dual (activation + weight) scaling.
        apply_clip: Whether to apply weight clipping after scaling.
        n_parallel_calib_samples: Batch size for memory-efficient calibration.
        max_calib_samples: Maximum number of calibration samples.
        max_calib_seq_len: Maximum sequence length for calibration.
        max_chunk_memory: Maximum memory (bytes) per computation chunk.
    """

    def __init__(
        self,
        model,
        tokenizer,
        w_bit: int,
        group_size: int,
        zero_point: bool,
        calib_data,
        split: str,
        text_column: str,
        duo_scaling: bool,
        act_qd: bool = False,
        apply_clip: bool = True,
        n_parallel_calib_samples: Optional[int] = None,
        max_calib_samples: int = 128,
        max_calib_seq_len: int = 512,
        max_chunk_memory: int = 1024 * 1024 * 1024,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.w_bit = w_bit
        self.group_size = group_size
        self.zero_point = zero_point
        self.calib_data = calib_data
        self.split = split
        self.text_column = text_column
        self.duo_scaling = duo_scaling
        self.apply_clip_flag = apply_clip
        self.n_parallel_calib_samples = n_parallel_calib_samples
        self.max_calib_samples = max_calib_samples
        self.max_calib_seq_len = max_calib_seq_len
        self.max_chunk_memory = max_chunk_memory
        self.act_qd = act_qd

        self.modules, self.module_kwargs, self.inps = self._init_quant(
            n_samples=self.max_calib_samples, max_seq_len=self.max_calib_seq_len
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def quantize(self):
        """Run the full AWQ quantization pipeline on all decoder layers."""
        import transformers

        for i in tqdm(range(len(self.modules)), desc="AWQ"):
            # Move module and inputs to the best available device
            common_device = next(self.modules[i].parameters()).device
            if common_device is None or str(common_device) == "cpu":
                best_device = get_best_device()
                self.modules[i] = self.modules[i].to(best_device)
                common_device = next(self.modules[i].parameters()).device

            if self.module_kwargs.get("position_ids") is not None:
                self.module_kwargs["position_ids"] = self.module_kwargs[
                    "position_ids"
                ].to(common_device)

            if self.module_kwargs.get("attention_mask") is not None:
                self.module_kwargs["attention_mask"] = self.module_kwargs[
                    "attention_mask"
                ].to(common_device)

            self.inps = self.inps.to(common_device)

            # Compatibility fix for transformers >= 4.48.0
            if (
                transformers.__version__ >= "4.48.0"
                and self.module_kwargs.get("attention_mask") is None
            ):
                self.module_kwargs["attention_mask"] = None

            # Move tuple kwargs (e.g. position embeddings) to the correct device
            for k, v in self.module_kwargs.items():
                if isinstance(v, tuple):
                    self.module_kwargs[k] = tuple(
                        item.to(common_device)
                        if isinstance(item, (torch.Tensor, nn.Module))
                        else item
                        for item in v
                    )

            # [STEP 1]: Discover linear layers and capture input features
            all_linears = find_layers(self.modules[i])
            quant_linears = exclude_layers_to_not_quantize(all_linears)
            input_feat = self._get_input_feat(self.modules[i], quant_linears)
            clear_memory()

            # [STEP 2]: Search and apply optimal scales
            module_config: List[Dict] = get_layers_for_scaling(
                self.modules[i], input_feat, self.module_kwargs
            )
            scales_list = [
                self._search_best_scale(self.modules[i], **layer)
                for layer in module_config
            ]
            apply_scale(self.modules[i], scales_list, input_feat_dict=input_feat)
            scales_list = append_str_prefix(
                scales_list, get_op_name(self.model, self.modules[i]) + "."
            )

            # [STEP 3]: Search and apply optimal clipping
            if self.apply_clip_flag:
                clip_list = self._search_best_clip(
                    self.modules[i], quant_linears, input_feat
                )
                apply_clip(self.modules[i], clip_list)
                clip_list = append_str_prefix(
                    clip_list, get_op_name(self.model, self.modules[i]) + "."
                )

            # [STEP 4]: Quantize non-AWQ layers to FP8, AWQ layers to int4
            self._quantize_fp8_layers(all_linears, quant_linears)
            self._apply_int4_quant(self.modules[i], quant_linears)
            clear_memory()

    # ------------------------------------------------------------------
    # Pseudo quantize / dequantize (used during scale search)
    # ------------------------------------------------------------------

    def pseudo_quantize_tensor(self, w: torch.Tensor):
        """
        Perform simulated quantization on a weight tensor.

        Returns:
            Tuple of (quantized_weight, scales, zeros).
        """
        org_w_shape = w.shape
        if self.group_size > 0:
            assert org_w_shape[-1] % self.group_size == 0, (
                f"Weight dim ({org_w_shape[-1]}) must be divisible by "
                f"group_size ({self.group_size})!"
            )
            w = w.reshape(-1, self.group_size)
        assert w.dim() == 2
        assert torch.isnan(w).sum() == 0

        if self.zero_point:
            max_val = w.amax(dim=1, keepdim=True)
            min_val = w.amin(dim=1, keepdim=True)
            max_int = 2**self.w_bit - 1
            min_int = 0
            scales = (max_val - min_val).clamp(min=1e-5) / max_int
            zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
            w = (
                torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
            ) * scales
            zeros = zeros.view(org_w_shape[0], -1)
        else:
            max_val = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
            max_int = 2 ** (self.w_bit - 1) - 1
            min_int = -(2 ** (self.w_bit - 1))
            scales = max_val / max_int
            zeros = None
            w = torch.clamp(torch.round(w / scales), min_int, max_int) * scales

        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(w).sum() == 0

        scales = scales.view(org_w_shape[0], -1)
        w = w.reshape(org_w_shape)
        return w, scales, zeros

    # ------------------------------------------------------------------
    # Scale search
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _search_best_scale(
        self,
        module,
        prev_op,
        layers: List[nn.Linear],
        inp: torch.Tensor,
        module2inspect=None,
        kwargs=None,
    ):
        """
        Search for the best per-channel scaling factor for a group of layers.

        Evaluates multiple candidate ratios and returns the one that minimizes
        the L2 reconstruction error between FP16 and quantized outputs.
        """
        if kwargs is None:
            kwargs = {}
        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]
        if "use_cache" in kwargs:
            kwargs.pop("use_cache")

        inp = inp.to(next(module2inspect.parameters()).device)

        # Compute weight-aware scaling (dual scaling)
        if self.duo_scaling:
            weight = torch.cat([m.weight for m in layers], dim=0)
            org_shape = weight.shape
            weight = weight.view(-1, self.group_size)
            w_scale = weight.abs() / (weight.abs().amax(dim=1, keepdim=True) + 1e-6)
            w_scale = w_scale.view(org_shape)
            w_mean = w_scale.mean(0)
            clear_memory(weight)
        else:
            w_mean = None

        # Compute per-channel activation mean (chunked for memory efficiency)
        inp_flat = inp.cpu().abs().view(-1, inp.shape[-1])
        num_elements = inp_flat.size(0)
        num_channels = inp_flat.size(1)
        element_size_bytes = inp_flat.element_size() * 2

        chunk_size = int(self.max_chunk_memory // (element_size_bytes * num_channels))
        chunk_size = min(chunk_size, num_elements)

        x_sum = torch.zeros(num_channels, dtype=torch.float32, device=inp.device)
        for start in range(0, num_elements, chunk_size):
            end = min(start + chunk_size, num_elements)
            chunk_sum = inp_flat[start:end].to(torch.float32).sum(dim=0)
            x_sum += chunk_sum.to(inp.device)
        x_mean = (x_sum / num_elements).to(inp.dtype)
        clear_memory(x_sum)

        # Compute reference FP16 output
        with torch.no_grad():
            module_kwargs = self._sanitize_kwargs(kwargs, module2inspect)
            fp16_output = self._module_forward(inp, module2inspect, module_kwargs)
            fp16_output = fp16_output.clip(
                torch.finfo(fp16_output.dtype).min,
                torch.finfo(fp16_output.dtype).max,
            )

        # Grid search for optimal scale ratio
        best_scales = self._compute_best_scale(
            inp, w_mean, x_mean, module2inspect, layers, fp16_output, module_kwargs
        )

        return (
            get_op_name(module, prev_op),
            tuple(get_op_name(module, m) for m in layers),
            best_scales,
        )

    def _compute_best_scale(
        self,
        x: torch.Tensor,
        w_mean: Optional[torch.Tensor],
        x_mean: torch.Tensor,
        module2inspect: nn.Module,
        linears2scale: List[nn.Linear],
        fp16_output: torch.Tensor,
        kwargs: Dict = None,
    ):
        """
        Grid-search over scaling ratios to minimize reconstruction error.

        L(s) = || Q(W * s) (s^{-1} * X) - W * X ||^2
        """
        if kwargs is None:
            kwargs = {}

        n_grid = 20
        history = []
        best_ratio = -1
        best_scales = None
        best_error = float("inf")

        org_sd = {k: v.cpu() for k, v in module2inspect.state_dict().items()}
        device = x.device
        x_mean = x_mean.view(-1).to(device)
        if w_mean is not None:
            w_mean = w_mean.view(-1).to(device)

        for ratio in range(n_grid):
            ratio = ratio / n_grid

            # Compute candidate scales
            if self.duo_scaling:
                scales = (x_mean.pow(ratio) / (w_mean.pow(1 - ratio) + 1e-4)).clamp(
                    min=1e-4
                )
            else:
                scales = x_mean.pow(ratio).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()
            scales_view = scales.view(1, -1).to(device)

            # Handle numerical edge cases
            scales[torch.isinf(scales)] = 1
            scales[torch.isnan(scales)] = 1

            # Apply scale -> pseudo-quantize -> remove scale
            for fc in linears2scale:
                fc.weight.mul_(scales_view)
                fc.weight.data = (
                    self.pseudo_quantize_tensor(fc.weight.data)[0] / scales_view
                )

            if self.act_qd:
                x = pseudo_quantize_tensor_per_tensor_fp8_triton(x)

            # Measure reconstruction error
            int_w_output = self._module_forward(x, module2inspect, kwargs)
            int_w_output = int_w_output.clip(
                torch.finfo(int_w_output.dtype).min,
                torch.finfo(int_w_output.dtype).max,
            )

            loss = self._compute_loss(fp16_output, int_w_output, device)
            history.append(loss)

            if loss < best_error:
                best_error = loss
                best_ratio = ratio
                best_scales = scales.clone()

            module2inspect.load_state_dict(org_sd)

        if best_ratio == -1:
            logging.debug(history)
            raise RuntimeError("Scale search failed: no improvement found.")

        assert torch.isnan(best_scales).sum() == 0
        return best_scales.detach().cpu()

    @torch.no_grad()
    def _compute_loss(
        self,
        fp16_output: torch.Tensor,
        int_w_output: torch.Tensor,
        device: torch.device,
    ) -> float:
        """Compute chunked mean squared error between reference and quantized outputs."""
        loss = 0.0
        fp16_flat = fp16_output.view(-1)
        int_w_flat = int_w_output.view(-1)
        num_elements = fp16_flat.size(0)
        element_size_bytes = fp16_output.element_size()

        chunk_size = self.max_chunk_memory // (element_size_bytes * 2)
        chunk_size = min(chunk_size, num_elements)

        fp16_chunks = torch.split(fp16_flat, chunk_size)
        int_w_chunks = torch.split(int_w_flat, chunk_size)

        for fp16_chunk, int_w_chunk in zip(fp16_chunks, int_w_chunks):
            chunk_loss = (
                (fp16_chunk.to(device) - int_w_chunk.to(device))
                .float()
                .pow(2)
                .sum()
                .item()
            )
            loss += chunk_loss

        return loss / num_elements

    # ------------------------------------------------------------------
    # Clip search
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _search_best_clip(self, layer, named_linears: Dict, input_feat: Dict):
        """Search for optimal weight clipping values for each target layer."""
        clip_list = []
        avoid_clipping = ["q_", "k_", "query", "key", "Wqkv", "kv"]

        for name in named_linears:
            if any(pattern in name for pattern in avoid_clipping):
                continue

            named_linears[name].to(get_best_device())
            max_val = self._compute_best_clip(
                named_linears[name].weight, input_feat[name]
            )
            clip_list.append((name, max_val))
            named_linears[name].cpu()

        return clip_list

    @torch.no_grad()
    def _compute_best_clip(
        self,
        w: torch.Tensor,
        input_feat: torch.Tensor,
        n_grid: int = 20,
        max_shrink: float = 0.5,
        n_sample_token: int = 512,
    ):
        """
        Grid-search for the best per-group clipping threshold.

        Minimizes the output reconstruction error when weights are clipped
        before quantization.
        """
        assert w.dim() == 2
        org_w_shape = w.shape
        group_size = self.group_size if self.group_size > 0 else org_w_shape[1]

        input_feat = input_feat.view(-1, input_feat.shape[-1])
        input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)

        step_size = max(1, input_feat.shape[1] // n_sample_token)
        input_feat = input_feat[:, ::step_size]

        w = w.reshape(org_w_shape[0], 1, -1, group_size)

        oc_batch_size = 256 if org_w_shape[0] % 256 == 0 else 64
        assert org_w_shape[0] % oc_batch_size == 0

        w_all = w
        best_max_val_all = []

        for i_b in range(org_w_shape[0] // oc_batch_size):
            w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]

            org_max_val = w.abs().amax(dim=-1, keepdim=True)
            best_max_val = org_max_val.clone()
            min_errs = torch.ones_like(org_max_val) * 1e9
            input_feat = input_feat.to(w.device)
            org_out = (input_feat * w).sum(dim=-1)

            for i_s in range(int(max_shrink * n_grid)):
                max_val = org_max_val * (1 - i_s / n_grid)
                cur_w = torch.clamp(w, -max_val, max_val)
                q_w = self.pseudo_quantize_tensor(cur_w)[0]
                cur_out = (input_feat * q_w).sum(dim=-1)

                err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
                del cur_w, cur_out
                cur_best_idx = err < min_errs
                min_errs[cur_best_idx] = err[cur_best_idx]
                best_max_val[cur_best_idx] = max_val[cur_best_idx]

            best_max_val_all.append(best_max_val)

        best_max_val = torch.cat(best_max_val_all, dim=0)
        clear_memory(input_feat)
        clear_memory(org_out)
        return best_max_val.squeeze(1)

    # ------------------------------------------------------------------
    # Weight quantization (final step)
    # ------------------------------------------------------------------

    def _quantize_fp8_layers(self, all_linears: Dict, quant_linears: Dict):
        """
        Quantize non-AWQ linear layers to FP8 format.

        Layers in `quant_linears` (int4 targets) are skipped.
        Special layers (norms, embeddings, gates) are also skipped.
        """
        if weight_cast_to_fp8 is None:
            logging.warning(
                "weight_cast_to_fp8 not available; skipping FP8 quantization."
            )
            return

        for layer_name, layer_obj in all_linears.items():
            if quant_linears.get(layer_name, None):
                continue
            if any(
                kw in layer_name
                for kw in ["norm", "lm_head", "embed_tokens", ".mlp.gate."]
            ):
                continue

            layer_obj = layer_obj.to("cuda")
            weight, weight_scale_inv = weight_cast_to_fp8(layer_obj.weight)
            weight = weight.to("cpu")
            weight_scale_inv = weight_scale_inv.to("cpu")
            layer_obj = layer_obj.to("cpu")
            layer_obj.weight = torch.nn.Parameter(weight, requires_grad=False)
            layer_obj.weight_scale_inv = torch.nn.Parameter(
                weight_scale_inv, requires_grad=False
            )

    @torch.no_grad()
    def _apply_int4_quant(self, module: nn.Module, named_linears: Dict):
        """
        Quantize AWQ-targeted layers to int4 (packed into int8 storage).

        Each layer gets:
          - weight: int8 tensor (two int4 values packed per byte)
          - weight_scale_inv: per-group scale factors
          - input_scale: default scale of 1.0 (placeholder for dynamic quantization)
        """
        for layer_name, layer_obj in named_linears.items():
            weight_bf16 = layer_obj.weight
            max_value_group_wise = compute_groupwise_abmax(weight_bf16, group_size=128)
            new_weight_bf16 = fake_quant_dequant(weight_bf16, group_size=128)
            weight_int4, _ = quantize_weight_int(
                new_weight_bf16, max_value_group_wise, bits=4
            )
            real_weight_int4 = pack_weight_to_int8(weight_int4)
            layer_obj.weight = torch.nn.Parameter(
                real_weight_int4, requires_grad=False
            )
            layer_obj.weight_scale_inv = torch.nn.Parameter(
                max_value_group_wise / 7.0, requires_grad=False
            )

        for layer_name, layer_obj in named_linears.items():
            default_input_scale = torch.tensor([1.0], dtype=torch.bfloat16)
            layer_obj.input_scale = nn.Parameter(
                default_input_scale, requires_grad=False
            )

        clear_memory()

    # ------------------------------------------------------------------
    # Initialization: calibration data collection
    # ------------------------------------------------------------------

    def _init_quant(self, n_samples: int = 128, max_seq_len: int = 512):
        """
        Initialize quantization by running calibration samples through
        the embedding layer and capturing inputs/kwargs for layer 0.

        Returns:
            Tuple of (decoder_layers, layer_kwargs, initial_hidden_states).
        """
        modules = self.model.model.layers
        samples = get_calib_dataset(
            data=self.calib_data,
            tokenizer=self.tokenizer,
            n_samples=n_samples,
            max_seq_len=max_seq_len,
            split=self.split,
            text_column=self.text_column,
        )
        samples = torch.cat(samples, dim=0)

        inps = []
        layer_kwargs = {}
        best_device = get_best_device()

        modules[0] = modules[0].to(best_device)
        self.model.model.embed_tokens = self.model.model.embed_tokens.to(best_device)

        # Use a Catcher module to intercept the first layer's input
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, *args, **kwargs):
                if len(args) > 0:
                    hidden_states = args[0]
                    del args
                else:
                    first_key = list(kwargs.keys())[0]
                    hidden_states = kwargs.pop(first_key)
                inps.append(hidden_states)
                layer_kwargs.update(kwargs)
                raise ValueError  # early exit

        modules[0] = Catcher(modules[0])
        try:
            self.model(samples.to(next(self.model.parameters()).device))
        except ValueError:
            pass
        modules[0] = modules[0].module  # restore original module

        # Let the model prepare additional kwargs (e.g. cache, position_ids)
        layer_kwargs = self.model.prepare_inputs_for_generation(
            samples, **layer_kwargs
        )
        layer_kwargs.pop("input_ids")

        del samples
        inps = inps[0]

        modules[0] = modules[0].cpu()
        self.model.model.embed_tokens = self.model.model.embed_tokens.to("cpu")
        clear_memory()

        if layer_kwargs.get("attention_mask") is not None:
            layer_kwargs["attention_mask"] = layer_kwargs["attention_mask"].to(
                best_device
            )

        return modules, layer_kwargs, inps

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _module_forward(
        self, x: torch.Tensor, module: nn.Module, module_kwargs: Dict
    ) -> torch.Tensor:
        """
        Run a forward pass through `module`, optionally batching inputs
        for memory efficiency.
        """
        if self.n_parallel_calib_samples is None:
            module_output = module(x, **module_kwargs)
            if isinstance(module_output, tuple):
                module_output = module_output[0]
        else:
            module_output = []
            partitioned_inputs = torch.split(x, self.n_parallel_calib_samples)
            batch_idx = 0
            for x_partial in partitioned_inputs:
                current_kwargs = {}
                for k, v in module_kwargs.items():
                    if k == "attention_mask":
                        current_kwargs[k] = v[
                            batch_idx : batch_idx + self.n_parallel_calib_samples, :
                        ]
                    else:
                        current_kwargs[k] = v

                partial_output = module(x_partial, **current_kwargs)
                if isinstance(partial_output, tuple):
                    partial_output = partial_output[0]
                module_output.append(partial_output.cpu())
                batch_idx += self.n_parallel_calib_samples

            module_output = torch.cat(module_output, dim=0)

        return module_output

    def _get_input_feat(self, layer: nn.Module, named_linears: Dict) -> Dict:
        """
        Register forward hooks on all target linear layers to capture their
        input activations during a calibration forward pass.

        Returns:
            Dict mapping layer name -> concatenated input features tensor.
        """

        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0].detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []

        # Include the MoE block itself to capture its input
        hook_targets = {**named_linears, "mlp": layer.mlp}

        for name in hook_targets:
            handles.append(
                hook_targets[name].register_forward_hook(
                    functools.partial(
                        cache_input_hook, name=name, feat_dict=input_feat
                    )
                )
            )

        # Run forward pass to collect activations
        self.inps = self.inps.to(next(layer.parameters()).device)
        module_kwargs = self._sanitize_kwargs(self.module_kwargs, layer)
        self.inps = self._module_forward(self.inps, layer, module_kwargs)

        for h in handles:
            h.remove()

        # Concatenate all captured features per layer
        def cat_and_assert(k, v):
            x = torch.cat(v, dim=0)
            assert x.shape[0] != 0, (
                f"{k} has a zero dimension. This can happen if no data was passed "
                "through (e.g. an inactive MoE expert). Try increasing max_calib_samples."
            )
            return x

        return {k: cat_and_assert(k, v) for k, v in input_feat.items()}

    def _sanitize_kwargs(self, inputs_kwargs: Dict, module: nn.Module) -> Dict:
        """
        Filter keyword arguments to only those accepted by the module's forward
        signature. Prevents errors across different transformers versions.
        """
        module_signature = inspect.signature(module.forward).parameters
        return {k: v for k, v in inputs_kwargs.items() if k in module_signature}
