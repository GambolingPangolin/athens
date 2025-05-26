#!/usr/bin/env python3

import argparse
import math
import os
import random
import time

from tokenizers import Tokenizer
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from lean_repository_dataset import LeanRepositoryDataset, TokenizedLeanDataset
from models.registers import (
    interleave_register_tokens,
    create_targets_fixed_offset,
    build_attention_mask,
)
from models.smollm import LlamaForCausalLM
from utils.checkpoints import CheckpointManager


def train_loop(
    model,
    train_dataloader,
    val_dataloader,
    device,
    epochs=10,
    lr=5e-4,
    warmup_steps=1000,
    max_steps=100000,
    ckpt_dir="./checkpoints",
    log_dir="./logs",
    special_token_id=0,  # set your special token id here
    max_offset=10,
):
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    checkpoint_manager = CheckpointManager(
        ckpt_dir="./checkpoints", max_train=2, max_val=1
    )

    # Linear warmup + cosine decay LR scheduler
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, max_steps - warmup_steps)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=-100
    )  # assuming -100 for padding if any

    writer = SummaryWriter(log_dir)

    global_step = 0

    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in tqdm(range(epochs)):
        model.train()
        epoch_start = time.time()
        train_loss = 0.0
        train_accuracy = 0

        for batch_idx, batch in tqdm(enumerate(train_dataloader), leave=False):
            input_ids = batch["input_ids"].to(device)  # shape: (B, T)
            input_ids_with_regs = interleave_register_tokens(
                input_ids, special_token_id
            )

            offset = random.randint(0, max_offset)
            labels = create_targets_fixed_offset(
                input_ids_with_regs.cpu(), special_token_id, offset=offset
            ).to(device)

            attention_mask = build_attention_mask(
                input_ids_with_regs.cpu(), special_token_id
            ).to(device)
            attention_mask = attention_mask.unsqueeze(1).expand(
                -1, model.model.layers[0].self_attn.n_head, -1, -1
            )

            optimizer.zero_grad()
            logits = model(
                input_ids_with_regs, mask=attention_mask
            )  # (B, T, vocab_size)

            with torch.no_grad():
                preds = logits.argmax(dim=-1)  # (B, T)
                mask = labels != -100
                correct = (preds == labels) & mask
                accuracy = (
                    correct.sum().item() / mask.sum().item()
                    if mask.sum().item() > 0
                    else 0.0
                )
            train_accuracy += accuracy

            # Shift logits and labels for causal language modeling
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )  # gradient clipping
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            global_step += 1

            if global_step % 100 == 0:
                avg_train_loss = train_loss / 100
                avg_train_acc = train_accuracy / 100
                lr_current = scheduler.get_last_lr()[0]

                writer.add_scalar("Train/Loss", avg_train_loss, global_step)
                writer.add_scalar("Train/Accuracy", avg_train_acc, global_step)
                writer.add_scalar("Train/LR", lr_current, global_step)

                checkpoint_manager.register(
                    model, optimizer, scheduler, epoch, train_loss=avg_train_loss
                )

                train_loss = 0.0
                train_accuracy = 0.0

            if global_step >= max_steps:
                break

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(device)
                labels = batch.get("labels", input_ids).to(device)

                logits = model(input_ids)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = criterion(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )

                val_loss += loss.item()
                val_steps += 1
                if val_steps >= 100:
                    # limit validation steps for faster eval
                    break

        avg_val_loss = val_loss / val_steps
        writer.add_scalar("Val/Loss", avg_val_loss, epoch + 1)

        checkpoint_manager.register(
            model, optimizer, scheduler, epoch, val_loss=avg_val_loss
        )

        if global_step >= max_steps:
            print(f"Reached max steps {max_steps}. Ending training.")
            break

    writer.close()


def main(repo_path, tokenizer_path):
    # Load tokenizer (adjust path as needed)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    pad_token_id = tokenizer.token_to_id("<pad>")
    register_token_id = tokenizer.token_to_id("<reg>")

    chunk_size_chars = 1000
    chunk_stride_chars = 500
    n_chunk_tokens = 512

    # Create base dataset from repo
    base_dataset = LeanRepositoryDataset(
        repo_path, chunk_size=chunk_size_chars, chunk_stride=chunk_stride_chars
    )

    # Wrap base with tokenizer and fixed length tokenization
    dataset = TokenizedLeanDataset(
        base_dataset, tokenizer, n_chunk_tokens, pad_token_id
    )

    # Split dataset into training and validation (e.g. 90% train, 10% val)
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=lambda b: {
            "input_ids": torch.stack([item["input_ids"] for item in b])
        },
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=lambda b: {
            "input_ids": torch.stack([item["input_ids"] for item in b])
        },
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LlamaForCausalLM(vocab_size=tokenizer.get_vocab_size()).to(device)

    # Run training loop
    train_loop(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        device=device,
        epochs=10,
        lr=5e-4,
        warmup_steps=1000,
        max_steps=100000,
        ckpt_dir="./checkpoints",
        log_dir="./logs",
        special_token_id=register_token_id,
        max_offset=10,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-training on a Lean repository")
    parser.add_argument(
        "--repo_path", type=str, required=True, help="Path to the root Lean repository"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="checkpoints/tokenizer.json",
        help="Path to the tokenizer JSON file",
    )
    args = parser.parse_args()
    main(args.repo_path, args.tokenizer_path)
