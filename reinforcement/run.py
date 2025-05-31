#!/usr/bin/env python3


import argparse
import asyncio

from tokenizers import Tokenizer
import torch

from models import smollm
from reinforcement.training import training_loop
from scripts.tokenize_lean_repository import VOCAB_SIZE

parser = argparse.ArgumentParser(description="RL fine-tune with DAPO loss")
parser.add_argument(
    "--init_checkpoint",
    type=str,
    required=True,
    help="Path to pretrained checkpoint",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    required=False,
    help="Path at which to save recent checkpoint",
)
parser.add_argument(
    "--tokenizer_path",
    type=str,
    required=True,
    help="Path to the tokenizer model",
)
parser.add_argument(
    "--rounds",
    type=int,
    default=3,
    help="Number of rounds of generating samples then reinforcing",
)
parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
parser.add_argument("--group_size", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--lr", type=float, default=5e-5)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load tokenizer
tokenizer = Tokenizer.from_file(args.tokenizer_path)
pad_token_id = tokenizer.token_to_id("<pad>")
# Load model
model = LlamaForCausalLM(smollm.config(VOCAB_SIZE))
ckpt = torch.load(args.init_checkpoint, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])

asyncio.run(
    training_loop(
        tokenizer,
        model,
        device=device,
        group_size=args.group_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        rounds=args.rounds,
        lr=args.lr,
        recent_ckpt_path=args.checkpoint,
        pad_token_id=pad_token_id,
    )
)
