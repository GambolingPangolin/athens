#!/usr/bin/env python3


import glob
import os
import sys
from tokenizers import ByteLevelBPETokenizer

VOCAB_SIZE = 600
CKPT_DIR = "checkpoints"


def main(mathlib_path):
    # Find all *.lean files recursively
    files = glob.glob(f"{mathlib_path}/**/*.lean", recursive=True)
    if not files:
        print("No .lean files found.")
        return

    # Train BPE tokenizer on these files
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=files,
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<hmm>", "<reg>"],
    )

    # Save the tokenizer
    os.makedirs(CKPT_DIR, exist_ok=True)
    tokenizer.save(f"{CKPT_DIR}/tokenizer.json")

    print(f"Tokenizer trained with vocab size {VOCAB_SIZE}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run python scripts/tokenizer.py /path/to/lean/mathlib")
        sys.exit(1)
    main(sys.argv[1])
