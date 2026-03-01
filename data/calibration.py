"""
Calibration dataset loading for AWQ quantization.

Supports multiple data sources: pileval, ultrachat, c4, wikitext, sharegpt4,
and generic HuggingFace datasets.
"""

import json
import logging
import random
from typing import List, Union

import torch
from datasets import load_dataset


def ultrachat_general(calib_dataset, tokenizer, n_samples: int, seq_len: int):
    """
    Load and tokenize samples from HuggingFace UltraChat dataset using chat template.

    Returns:
        List of tensors, each of shape (1, seq_len).
    """
    calib_dataset = calib_dataset.shuffle(seed=42).select(range(n_samples))
    samples = []

    for example in calib_dataset:
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_special_tokens=False,
        )
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=seq_len,
            padding="max_length",
            add_special_tokens=False,
            return_tensors="pt",
        )
        samples.append(encoded.input_ids)

    return samples


def get_sharegpt_gpt4_256(nsamples: int, seed: int, seqlen: int, tokenizer):
    """
    Load calibration samples from a local ShareGPT-GPT4 JSONL file.

    The file is expected at a fixed path with conversation pairs.

    Returns:
        List of tensors, each of shape (1, seqlen).
    """
    data_path = "../dataset/sharegpt_gpt4/sharegpt_gpt4_256.jsonl"
    samples = []
    line_count = 0

    with open(data_path, "r") as f:
        for line in f:
            if nsamples > 0 and line_count >= nsamples:
                break
            data = json.loads(line)
            share_gpt_data = data["conversations"]
            messages = [
                {"role": "user", "content": share_gpt_data[0]["value"]},
                {"role": "assistant", "content": share_gpt_data[1]["value"]},
            ]

            # Normalize role names
            for item in messages:
                if "role" not in item and "from" in item:
                    item["role"] = item["from"]
                if "content" not in item and "value" in item:
                    item["content"] = item["value"]
                role = item["role"]
                if "human" in role:
                    item["role"] = "user"
                elif "gpt" in role:
                    item["role"] = "assistant"

            # Build prompt text with special tokens
            text = ""
            for msg in messages:
                if msg["role"] == "user":
                    text += f"<|user|>\n{msg['content']}</s>\n"
                elif msg["role"] == "assistant":
                    text += f"<|assistant|>\n{msg['content']}</s>\n"
            text += "<|assistant|>\n"

            encoded = tokenizer(
                [text],
                truncation=True,
                max_length=seqlen,
                padding="max_length",
                return_tensors="pt",
            )
            samples.append(encoded.input_ids)
            line_count += 1

    return samples


def get_c4(nsamples: int, seed: int, seqlen: int, tokenizer):
    """
    Load calibration samples from the C4 dataset (English subset).

    Randomly selects text segments that are longer than `seqlen` and extracts
    random windows of length `seqlen`.

    Returns:
        List of tensors, each of shape (1, seqlen).
    """
    traindata = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )
    random.seed(seed)
    trainloader = []

    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        trainloader.append(inp)

    return trainloader


def get_wikitext(nsamples: int, seed: int, seqlen: int, tokenizer):
    """
    Load calibration samples from the WikiText-2 dataset.

    Concatenates the entire training set, then extracts random windows.

    Returns:
        List of tensors, each of shape (1, seqlen).
    """
    calib_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    random.seed(seed)
    trainenc = tokenizer("\n\n".join(calib_dataset["text"]), return_tensors="pt")
    samples = []

    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        samples.append(inp)

    return samples


def get_calib_dataset(
    data: Union[str, List[str], List[List[int]]] = "pileval",
    tokenizer=None,
    n_samples: int = 128,
    max_seq_len: int = 512,
    split: str = "train",
    text_column: str = "text",
):
    """
    Unified entry point for loading calibration datasets.

    Supports:
      - Named datasets: "pileval", "ultrachat", "c4", "wikitext", "sharegpt4"
      - Any HuggingFace dataset name (as a string)
      - A list of raw text strings
      - A list of pre-tokenized integer sequences

    Returns:
        List of tensors, each of shape (1, max_seq_len).
    """
    if isinstance(data, str):
        if data == "pileval":
            dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
        elif data == "ultrachat":
            dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
            return ultrachat_general(dataset, tokenizer, n_samples, max_seq_len)
        elif data == "c4":
            return get_c4(n_samples, 42, max_seq_len, tokenizer)
        elif data == "wikitext":
            return get_wikitext(n_samples, 42, max_seq_len, tokenizer)
        elif data == "sharegpt4":
            return get_sharegpt_gpt4_256(n_samples, 42, max_seq_len, tokenizer)
        else:
            dataset = load_dataset(data, split=split)

        dataset = dataset.shuffle(seed=42)

    elif isinstance(data, list):
        if isinstance(data[0], str):
            dataset = [{text_column: text} for text in data]
        elif isinstance(data[0][0], int):
            dataset = data
        else:
            raise NotImplementedError(
                "Pass a HuggingFace dataset name, a list of text strings, "
                "or a list of tokenized integer sequences."
            )
    else:
        raise NotImplementedError(
            "Pass a HuggingFace dataset name, a list of text strings, "
            "or a list of tokenized integer sequences."
        )

    # Tokenize and collect samples that fit within max_seq_len
    samples = []
    n_run = 0
    for data in dataset:
        if isinstance(data, list):
            line_encoded = data
        else:
            line = data[text_column].strip()
            line_encoded = tokenizer.encode(line)
        if len(line_encoded) > max_seq_len:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break

    # Concatenate all samples and split into fixed-length blocks
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // max_seq_len
    logging.debug(f" * Split into {n_split} blocks")
    return [
        cat_samples[:, i * max_seq_len : (i + 1) * max_seq_len]
        for i in range(n_split)
    ]
