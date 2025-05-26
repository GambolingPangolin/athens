#!/usr/bin/env python3

import os
import glob
import torch
from torch.utils.data import Dataset


class LeanRepositoryDataset(Dataset):
    def __init__(self, repo_path, chunk_size=1000, chunk_stride=500):
        """
        Dataset of overlapping text chunks from all .lean files under repo_path.

        Args:
            repo_path (str): Path to root of Lean repo.
            chunk_size (int): Chunk size in characters.
            chunk_stride (int): Stride for overlapping chunks in characters.
        """
        self.chunk_size = chunk_size
        self.chunk_stride = chunk_stride

        # Find all .lean files recursively
        pattern = os.path.join(repo_path, "**", "*.lean")
        self.files = glob.glob(pattern, recursive=True)
        if not self.files:
            raise RuntimeError(f"No .lean files found in {repo_path}")

        self.chunks = []  # List of dicts: {"text": str, "file": str, "start": int}

        # Load files & split into chunks
        for file_path in self.files:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            length = len(content)
            start = 0
            while start < length:
                end = start + chunk_size
                chunk_text = content[start:end]
                self.chunks.append(
                    {"text": chunk_text, "file": file_path, "start": start, "end": end}
                )
                if end >= length:
                    break
                start += chunk_stride

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        return {
            "text": chunk["text"],
            "file_path": chunk["file"],
            "start_char": chunk["start"],
            "end_char": min(chunk["end"], len(chunk["text"]) + chunk["start"]),
        }


class TokenizedLeanDataset(Dataset):
    def __init__(self, base_dataset, tokenizer, n_chunk, pad_token_id):
        self.base = base_dataset
        self.tokenizer = tokenizer
        self.n_chunk = n_chunk
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        chunk = self.base[idx]
        # Tokenize chunk text, get token ids
        tokens = self.tokenizer.encode(chunk["text"]).ids
        # Truncate or pad to n_chunk
        tokens = tokens[: self.n_chunk]
        pad_len = self.n_chunk - len(tokens)
        if pad_len > 0:
            tokens = tokens + [self.pad_token_id] * pad_len
        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "file_path": chunk["file_path"],
            "start_char": chunk["start_char"],
            "end_char": chunk["end_char"],
        }
