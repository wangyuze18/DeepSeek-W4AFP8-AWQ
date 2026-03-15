# DeepSeek-W4AFP8-AWQ: Activation-aware Weight Quantization for DeepSeek-V3

**DeepSeek-W4AFP8-AWQ** is a quantization toolkit that applies [AWQ (Activation-aware Weight Quantization)](https://arxiv.org/abs/2306.00978) to Mixture-of-Experts LLMs, specifically targeting **DeepSeek-V3** and its variants. It produces a mixed-precision model where MoE expert layers are quantized to **INT4** (group-wise, packed into INT8 storage) and other linear layers are quantized to **FP8** (E4M3FN, block-wise), enabling efficient deployment via [SGLang](https://github.com/sgl-project/sglang)  serving frameworks.

## Highlights

- **MoE-aware AWQ**: Jointly searches for optimal per-channel scaling across all experts and the shared expert in each MoE block, preserving gating behavior.
- **Mixed INT4 + FP8**: Expert gate/up/down projections → INT4 (group_size=128); attention projections and other layers → FP8 with block-wise scaling — balancing accuracy and throughput.
- **Multiple calibration datasets**: Built-in support for PileVal, UltraChat, C4, WikiText-2, and ShareGPT-GPT4.
- **SGLang-ready output**: Automatically converts saved checkpoints to SGLang's expected key naming convention.
- **Memory-efficient**: Chunked loss computation, optional batched calibration, and layer-by-layer quantization keep peak GPU memory manageable even for 671B-parameter models.

## Project Structure

```
MoE-AWQ/
├── main.py                              # CLI entry point
├── quantization_core/
│   ├── quantizer.py                     # AwqQuantizer — full AWQ pipeline
│   ├── scaling.py                       # Scale/clip application
│   └── weight_ops.py                    # Group-wise absmax, fake quant, INT/FP8 quant
├── data/
│   └── calibration.py                   # Calibration dataset loaders
├── model/
│   └── loader.py                        # DeepSeek-V3 weight loading (FP8 → BF16)
├── conversion/
│   └── sglang.py                        # SGLang format conversion & indexer weights
├── utils/
│   ├── helpers.py                       # Device detection, memory management
│   └── layer_config.py                  # Layer ignore list & MoE scaling groups
├── quantization/
│   └── utils.py                         # Module introspection & INT8 packing
├── modules/qlinear/
│   └── kernel.py                        # Triton FP8 quantization kernels
└── models/
    └── modeling_deepseek_v3_dot_1.py    # DeepSeek-V3 model architecture
```

## Requirements

- Python ≥ 3.12
- PyTorch ≥ 2.8 (with CUDA support)
- Triton ≥ 2.1
- Transformers ≥ 4.48
- Additional dependencies:

```bash
pip install accelerate safetensors datasets huggingface_hub tqdm
```

## Quick Start

### 1. Basic Quantization

```bash
python main.py \
    --model_path /path/to/DeepSeek-V3 \
    --save_path /path/to/output \
    --calib_data pileval \
    --apply_clip
```

### 2. Full Configuration Example

```bash
python main.py \
    --model_path /path/to/DeepSeek-V3 \
    --save_path /path/to/DeepSeek-V3-MoEAWQ-INT4 \
    --w_bit 4 \
    --group_size 128 \
    --calib_data c4 \
    --max_calib_samples 256 \
    --max_calib_seq_len 2048 \
    --apply_clip \
    --n_parallel_calib_samples 4 \
    --max_chunk_memory 4
```

### 3. Test Mode (First 4 Layers Only)

```bash
python main.py \
    --model_path /path/to/DeepSeek-V3 \
    --save_path /tmp/test_output \
    --calib_data pileval \
    --test_mode
```

## CLI Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model_path` | str | *required* | Path to the pretrained DeepSeek-V3 checkpoint |
| `--save_path` | str | *required* | Output directory for the quantized model |
| `--w_bit` | int | 4 | Weight bit-width for AWQ-targeted layers |
| `--group_size` | int | 128 | Quantization group size |
| `--zero_point` | flag | False | Enable asymmetric (zero-point) quantization |
| `--calib_data` | str | pileval | Calibration dataset (`pileval`, `ultrachat`, `c4`, `wikitext`, `sharegpt4`, or any HF dataset) |
| `--split` | str | train | Dataset split |
| `--act_qd`| flag | False | activation fp8 fakequant |
| `--text_column` | str | text | Text column name in the dataset |
| `--max_calib_samples` | int | 128 | Number of calibration samples |
| `--max_calib_seq_len` | int | 512 | Maximum calibration sequence length |
| `--duo_scaling` | flag | False | Enable dual (activation + weight) scaling |
| `--apply_clip` | flag | False | Enable weight clipping after scale search |
| `--n_parallel_calib_samples` | int | None | Batch size for parallel calibration (memory saver) |
| `--max_chunk_memory` | int | 2 | Max memory in GB per computation chunk |
| `--test_mode` | flag | False | Only quantize the first 4 layers for quick testing |

## Quantization Strategy

The quantization pipeline processes each decoder layer sequentially:

1. **Activation Collection** — Forward calibration data through the layer, capturing input features for every target linear sub-layer and the MoE block.

2. **Scale Search (AWQ)** — For each scaling group (e.g., post-attention layernorm → all expert gate/up projections), grid-search over 20 candidate scaling ratios to minimize the L2 reconstruction error between the FP16 reference output and the pseudo-quantized output.

3. **Weight Clipping** *(optional)* — For non-attention linear layers, grid-search for the optimal per-group clipping threshold that minimizes output error after quantization.

4. **Final Quantization**
   - **AWQ-targeted layers** (MoE expert projections matching the ignore-list filter) → INT4 group-wise, packed into INT8 with per-group scale factors.
   - **Other linear layers** (attention projections, etc.) → FP8 (E4M3FN) with block-wise scaling via Triton kernels.
   - **Special layers** (embeddings, layer norms, router gate, lm_head) → kept in original precision.

5. **SGLang Conversion** — Key names are remapped (e.g., `gate_proj.input_scale` → `w3.input_scale`) for compatibility with SGLang's serving engine.

## Serving with SGLang

After quantization, the output directory can be loaded directly by SGLang:

```bash
python -m sglang.launch_server \
    --model-path /path/to/DeepSeek-V3-MoEAWQ-INT4 \
    --tp 8
```

Refer to the [SGLang documentation](https://github.com/sgl-project/sglang) for detailed serving options.

## Citation

If you find this work useful, please cite the following:

```bibtex
@article{lin2023awq,
  title={AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration},
  author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Dang, Xingyu and Han, Song},
  journal={arXiv},
  year={2023}
}

@article{lv2025llmc+,
  title={LLMC+: Benchmarking Vision-Language Model Compression with a Plug-and-play Toolkit},
  author={Lv, Chengtao and Zhang, Bilang and Yong, Yang and Gong, Ruihao and Huang, Yushi and Gu, Shiqiao and Wu, Jiajun and Shi, Yumeng and Guo, Jinyang and Wang, Wenya},
  journal={arXiv preprint arXiv:2508.09981},
  year={2025}
}

@article{angelslim2026,
  title={AngelSlim: A more accessible, comprehensive, and efficient toolkit for large model compression},
  author={Hunyuan AI Infra Team},
  journal={arXiv preprint arXiv:2602.21233},
  year={2026}
}
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).