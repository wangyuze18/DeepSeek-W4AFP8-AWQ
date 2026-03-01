"""
AWQ Quantization CLI for DeepSeek-V3 (MoE) models.

Usage:
    python main.py \\
        --model_path /path/to/model \\
        --save_path /path/to/output \\
        --calib_data pileval \\
        --duo_scaling \\
        --apply_clip
"""

import os
import argparse

from transformers import AutoTokenizer
from huggingface_hub import save_torch_state_dict

from model.loader import load_model
from quantization_core.quantizer import AwqQuantizer
from conversion.sglang import convert_to_sglang, load_indexer_weights


def parse_args():
    parser = argparse.ArgumentParser(
        description="AWQ Quantization for LLMs (targeted at MoE models like DeepSeek-V3)"
    )

    # Model paths
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path or HuggingFace repo ID of the pretrained model",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Directory to save the quantized model",
    )

    # Quantization parameters
    parser.add_argument(
        "--w_bit",
        type=int,
        default=4,
        choices=[4],
        help="Weight quantization bits (currently only 4-bit supported)",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=128,
        help="Group size for quantization (typically 128)",
    )
    parser.add_argument(
        "--zero_point",
        action="store_true",
        default=False,
        help="Use asymmetric (zero-point) quantization",
    )

    # Calibration dataset
    parser.add_argument(
        "--calib_data",
        type=str,
        default="pileval",
        help='Calibration dataset: "pileval", "ultrachat", "c4", "wikitext", '
        '"sharegpt4", or HF dataset name',
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument(
        "--max_calib_samples",
        type=int,
        default=128,
        help="Number of calibration samples",
    )
    parser.add_argument(
        "--max_calib_seq_len",
        type=int,
        default=512,
        help="Maximum sequence length for calibration",
    )

    # Advanced options
    parser.add_argument(
        "--duo_scaling",
        action="store_true",
        default=False,
        help="Use dual (activation + weight) scaling (recommended)",
    )
    parser.add_argument(
        "--apply_clip",
        action="store_true",
        default=False,
        help="Apply weight clipping after scale search",
    )
    parser.add_argument(
        "--n_parallel_calib_samples",
        type=int,
        default=None,
        help="Process calibration samples in parallel batches (for memory efficiency)",
    )
    parser.add_argument(
        "--max_chunk_memory",
        type=int,
        default=2,
        help="Maximum memory (GB) per chunk during scaling/clipping search",
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
        default=False,
        help="Test mode: only quantize first 4 layers",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print("Loading model...")
    model = load_model(args)
    model.eval()

    print("Initializing AWQ quantizer...")
    quantizer = AwqQuantizer(
        model=model,
        tokenizer=tokenizer,
        w_bit=args.w_bit,
        group_size=args.group_size,
        zero_point=args.zero_point,
        calib_data=args.calib_data,
        split=args.split,
        text_column=args.text_column,
        duo_scaling=args.duo_scaling,
        apply_clip=args.apply_clip,
        n_parallel_calib_samples=args.n_parallel_calib_samples,
        max_calib_samples=args.max_calib_samples,
        max_calib_seq_len=args.max_calib_seq_len,
        max_chunk_memory=args.max_chunk_memory * 1024 * 1024,  # GB -> bytes
    )

    print("Starting quantization...")
    quantizer.quantize()

    print("Saving quantized model...")
    os.makedirs(args.save_path, exist_ok=True)

    # Merge indexer weights from the original model
    model_state_dict = model.state_dict()
    index_path = os.path.join(args.model_path, "model.safetensors.index.json")
    model_state_dict = load_indexer_weights(model_state_dict, index_path)

    # Save as sharded safetensors
    save_torch_state_dict(
        model_state_dict,
        args.save_path,
        max_shard_size="10GB",
        safe_serialization=True,
    )
    print(f"Model saved to: {args.save_path}")

    # Convert key names for SGLang compatibility
    convert_to_sglang(args.save_path)
    print("SGLang format conversion done.")


if __name__ == "__main__":
    main()
