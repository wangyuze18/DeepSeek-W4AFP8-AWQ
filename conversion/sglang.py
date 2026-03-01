"""
Post-quantization conversion to SGLang-compatible format.

Handles key renaming in safetensors and index JSON files, and loading
indexer weights from the original model into the quantized state dict.
"""

import json
import os

import safetensors.torch as st


# Key name mappings: quantization naming convention -> SGLang convention
_KEY_MAPPINGS = [
    ("up_proj.input_scale", "w1.input_scale"),
    ("down_proj.input_scale", "w2.input_scale"),
    ("gate_proj.input_scale", "w3.input_scale"),
]


def _process_key(old_key: str) -> str:
    """Apply all key name mappings to a single key string."""
    new_key = old_key
    for old_sub, new_sub in _KEY_MAPPINGS:
        new_key = new_key.replace(old_sub, new_sub)
    return new_key


def _filter_keys_in_json(file_path: str):
    """
    Rename keys in a safetensors index JSON file according to _KEY_MAPPINGS.

    The file is modified in place.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def recursive_process(obj):
        if isinstance(obj, dict):
            return {_process_key(k): recursive_process(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_process(item) for item in obj]
        return obj

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(recursive_process(data), f, indent=2, ensure_ascii=False)
    print(f"Processed JSON file: {file_path}")


def _filter_keys_in_safetensors(file_path: str):
    """
    Rename keys in a safetensors file according to _KEY_MAPPINGS.

    The file is replaced in place.
    """
    tensors = st.load_file(file_path)
    new_tensors = {}

    for old_k, tensor in tensors.items():
        new_k = _process_key(old_k)
        if new_k in new_tensors:
            raise ValueError(
                f"Duplicate key after replacement: {new_k} (original: {old_k})"
            )
        new_tensors[new_k] = tensor

    if os.path.exists(file_path):
        os.remove(file_path)
    st.save_file(new_tensors, file_path)
    print(f"Processed safetensors file: {file_path}")


def convert_to_sglang(model_dir: str):
    """
    Convert all safetensors and index JSON files in `model_dir` to SGLang
    naming convention.

    Args:
        model_dir: Path to the directory containing quantized model files.
    """
    if not os.path.exists(model_dir):
        print(f"Error: Model directory does not exist -> {model_dir}")
        return

    for filename in os.listdir(model_dir):
        file_path = os.path.join(model_dir, filename)
        if not os.path.isfile(file_path):
            continue

        try:
            if filename.endswith(".index.json"):
                _filter_keys_in_json(file_path)
            elif filename.endswith(".safetensors"):
                _filter_keys_in_safetensors(file_path)
        except Exception as e:
            print(f"Failed to process file {file_path}: {e}")

    print("\nAll target files processed successfully!")


def load_indexer_weights(model_state_dict: dict, index_json_path: str) -> dict:
    """
    Load indexer-related weights from the original model and merge them
    into the quantized model's state dict.

    Indexer weights are not modified during quantization and must be
    carried over from the original checkpoint.

    Args:
        model_state_dict: The quantized model's state dict (modified in-place).
        index_json_path: Path to the original model's safetensors index JSON.

    Returns:
        The updated state dict with indexer weights merged in.
    """
    if not os.path.exists(index_json_path):
        raise FileNotFoundError(f"Index file not found: {index_json_path}")

    with open(index_json_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)
        weight_map = index_data.get("weight_map", {})

    indexer_weights = {k: v for k, v in weight_map.items() if "indexer" in k}
    if not indexer_weights:
        print("Warning: no indexer weights found in the index file.")
        return model_state_dict

    # Group by shard file for efficient loading
    shard_group = {}
    for k, v in indexer_weights.items():
        shard_group.setdefault(v, []).append(k)

    index_dir = os.path.dirname(os.path.abspath(index_json_path))

    for shard, keys in shard_group.items():
        shard_path = os.path.join(index_dir, shard)
        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"Shard file not found: {shard_path}")

        shard_state = st.load_file(shard_path)
        for k in keys:
            if k not in shard_state:
                raise KeyError(f"Weight '{k}' not found in shard '{shard}'")
            model_state_dict[k] = shard_state[k]

    return model_state_dict
