#!/usr/bin/env python3

import argparse

from tokenizers import Tokenizer
import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split
from transformers import LlamaForCausalLM

from lean.dataset import LeanRepositoryDataset, TokenizedLeanDataset
from models import smollm
from pre_training.default import training_loop


def main(repo_paths, tokenizer_path, batch_size):
    # Load tokenizer (adjust path as needed)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    pad_token_id = tokenizer.token_to_id("<pad>")
    register_token_id = tokenizer.token_to_id("<reg>")

    chunk_size_chars = 1000
    chunk_stride_chars = 500
    n_chunk_tokens = 512

    # Create base dataset from repo

    base_datasets = [
        LeanRepositoryDataset(
            repo_path, chunk_size=chunk_size_chars, chunk_stride=chunk_stride_chars
        )
        for repo_path in repo_paths
    ]

    # Wrap base with tokenizer and fixed length tokenization
    dataset = ConcatDataset(
        [
            TokenizedLeanDataset(base_dataset, tokenizer, n_chunk_tokens, pad_token_id)
            for base_dataset in base_datasets
        ]
    )

    # Split dataset into training and validation (e.g. 90% train, 10% val)
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: {
            "input_ids": torch.stack([item["input_ids"] for item in b])
        },
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: {
            "input_ids": torch.stack([item["input_ids"] for item in b])
        },
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LlamaForCausalLM(smollm.config(tokenizer.get_vocab_size())).to(device)

    # Run training loop
    training_loop(
        device,
        model,
        train_loader,
        val_loader,
        pad_token_id,
        register_token_id,
        epochs=10,
        lr=5e-4,
        warmup_steps=1000,
        max_steps=100000,
        ckpt_dir="./checkpoints",
        log_dir="./logs",
        max_offset=10,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-training on a Lean repository")
    parser.add_argument(
        "--repo_path",
        type=str,
        action="append",
        required=True,
        help="Path to the root Lean repository.  May be repeated.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="checkpoints/tokenizer.json",
        help="Path to the tokenizer JSON file",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training and validation dataloaders",
    )
    args = parser.parse_args()
    main(args.repo_path, args.tokenizer_path, args.batch_size)
