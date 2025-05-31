#!/usr/bin/env python3

import math
import os
import random
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from models.registers import (
    interleave_register_tokens,
    create_targets_fixed_offset,
)
from utils.checkpoints import CheckpointManager
from utils.constants import PBAR_WIDTH


def training_loop(
    device,
    model,
    train_dataloader,
    val_dataloader,
    padding_token_id,
    register_token_id,
    epochs=10,
    lr=5e-4,
    warmup_steps=1000,
    max_steps=100000,
    ckpt_dir="./checkpoints",
    log_dir="./logs",
    max_offset=10,
):
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_manager = CheckpointManager(ckpt_dir=ckpt_dir, max_train=2, max_val=1)

    # Linear warmup + cosine decay LR scheduler
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, max_steps - warmup_steps)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=padding_token_id)
    writer = SummaryWriter(log_dir)
    n_training_examples = len(train_dataloader)

    global_step = 0

    for epoch in tqdm(range(epochs), ncols=PBAR_WIDTH):
        model.train()
        epoch_start = time.time()
        train_loss = 0.0
        train_accuracy = 0

        for batch_idx, batch in tqdm(
            enumerate(train_dataloader),
            total=n_training_examples,
            leave=False,
            ncols=PBAR_WIDTH,
        ):
            input_ids = batch["input_ids"].to(device)  # shape: (B, T)
            input_ids_with_regs = interleave_register_tokens(
                input_ids, register_token_id
            )

            offset = random.randint(0, max_offset)
            labels = create_targets_fixed_offset(
                input_ids_with_regs.cpu(),
                register_token_id,
                padding_token_id,
                offset=offset,
            ).to(device)

            optimizer.zero_grad()
            logits = model.forward(
                input_ids_with_regs,
                attention_mask=(input_ids_with_regs != register_token_id),
            ).logits  # (B, T, vocab_size)

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

                logits = model.forward(input_ids).logits
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
