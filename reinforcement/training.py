#!/usr/bin/env python3

import math

import torch
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from loss.dapo import dapo_loss_vectorized
from reinforcement.examples import sample_examples, save_data_dir


def form_groups(examples, group_size):
    examples_by_score = dict()
    for i, ex in enumerate(examples):
        examples_by_score.setdefault(ex["score"], []).append(ex)

    if len(examples_by_score) < 2:
        return []  # no groups with differing scores possible

    groups = []
    max_groups = math.ceil(len(examples) / group_size)

    for _ in range(max_groups):
        if len(examples_by_score) < 2:
            break

        max_score = max(examples_by_score.keys())
        min_score = min(examples_by_score.keys())

        ex_min = examples_by_score[min_score].pop()
        if len(examples_by_score[min_score]) == 0:
            del examples_by_score[min_score]

        ex_max = examples_by_score[max_score].pop()
        if len(examples_by_score[max_score]) == 0:
            del examples_by_score[max_score]

        groups.append([ex_min, ex_max])

    # Fill out groups at random
    current_group_ix = 0
    current_set_ix = 0

    example_sets = [examples for _, examples in examples_by_score.items()]
    while current_group_ix < len(groups) and current_set_ix < len(example_sets):
        current_group = groups[current_group_ix]
        n_remaining_in_current = group_size - len(current_group)

        current_examples = example_sets[current_set_ix]

        if len(current_examples) < n_remaining_in_current:
            current_group.extend(current_examples)
            current_set_ix += 1
        elif len(current_examples) == n_remaining_in_current:
            current_group.extend(current_examples)
            example_sets[current_set_ix] = []
            current_group_ix += 1
            current_set_ix += 1
        # len(current_examples) > n_remaining_in_current
        else:
            current_group.extend(current_examples[:n_remaining_in_current])
            example_sets[current_set_ix] = current_examples[n_remaining_in_current:]
            current_group_ix += 1

    return groups


class ExampleGroupDataset(Dataset):
    def __init__(self, examples, group_size=10, pad_token_id=None):
        self.pad_token_id = pad_token_id
        # We do NOT load logits, we will generate those from the model
        self.groups = form_groups(examples, group_size)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        group = self.groups[idx]

        # Find max sequence length for padding
        max_len = max(len(ex["tokens"]) for ex in group)

        actions = []
        old_logits = []
        rewards = []

        vocab_size = len(group[0]["logits"][0])  # Assuming consistent vocab size
        pad_vec = one_hot(
            torch.tensor(self.pad_token_id), num_classes=vocab_size
        ).float()

        for ex in group:
            seq_len = ex["tokens"].shape[0]
            # Pad actions with pad_token_id to max_len
            padded_actions = torch.cat(
                [ex["tokens"], torch.tensor([self.pad_token_id] * (max_len - seq_len))],
            ).to(dtype=torch.long)
            actions.append(padded_actions)

            # Pad logits with one-hot of pad_token_id
            logits = ex["logits"]
            # Pad logits to max_len with one-hot vectors for pad token
            num_pad = max_len - seq_len
            if num_pad > 0:
                pad_logits = torch.stack([pad_vec] * num_pad)
                logits = torch.cat([logits, pad_logits])
            old_logits.append(logits)

            rewards.append(ex["score"])

        actions = torch.stack(actions)  # [G, max_len]
        old_logits = torch.stack(old_logits)  # [G, max_len, V]
        rewards = torch.tensor(rewards, dtype=torch.float)  # [G]
        prompt_token_ids = [ex["prompt_ids"] for ex in group]

        return old_logits, actions, rewards, prompt_token_ids


def collate_fn(batch, pad_token_id):
    max_len = 0
    for _, actions, _, _ in batch:
        max_len = max(max_len, actions.size(1))

    batch_actions = []
    batch_old_logits = []
    batch_rewards = []
    batch_prompt_ids = []

    for old_logits, actions, rewards, prompt_ids in batch:
        G, S = actions.shape
        vocab_size = old_logits.shape[2]

        # Pad actions to max_len
        pad_len = max_len - S
        if pad_len > 0:
            pad_actions = torch.full((G, pad_len), fill_value=0, dtype=actions.dtype)
            actions_padded = torch.cat([actions, pad_actions], dim=1)

            # Pad old_logits with one-hot vectors for pad token
            pad_vec = one_hot(
                torch.tensor(pad_token_id), num_classes=vocab_size
            ).float()
            pad_logits = pad_vec.repeat(G, pad_len, 1)
            old_logits_padded = torch.cat([old_logits, pad_logits], dim=1)
        else:
            actions_padded = actions
            old_logits_padded = old_logits

        batch_actions.append(actions_padded)
        batch_old_logits.append(old_logits_padded)
        batch_rewards.append(rewards)
        batch_prompt_ids.append(prompt_ids)

    batch_actions = torch.stack(batch_actions)  # [B, G, max_len]
    batch_old_logits = torch.stack(batch_old_logits)  # [B, G, max_len, V]
    batch_rewards = torch.stack(batch_rewards)  # [B, G]

    return batch_old_logits, batch_actions, batch_rewards, batch_prompt_ids


def get_logits_with_prompt_batch(
    pad_token_id, prompt_token_ids_batch, generated_token_ids_batch, model, device
):
    """
    Compute logits from model for a batch of generated token sequences prefixed by their prompts.

    Args:
        prompt_token_ids_batch: List of lists of int, prompt token ids for each batch element.
        generated_token_ids_batch: Tensor of shape [batch, seq_len], generated tokens after each prompt.
        model: Language model with callable interface returning logits.
        device: torch.device where tensors are placed.

    Returns:
        logits_generated: Tensor of shape [batch_size, max_gen_len, vocab_size], logits for generated tokens only.
    """
    batch_size = len(prompt_token_ids_batch)
    prompt_lens = [len(p) for p in prompt_token_ids_batch]
    gen_lens = generated_token_ids_batch.size(1)
    max_gen_len = gen_lens
    max_total_len = max(pl + gen_lens for pl in prompt_lens)

    input_ids = (
        torch.ones((batch_size, max_total_len), dtype=torch.long, device=device)
        * pad_token_id
    )

    for i in range(batch_size):
        prompt_len = prompt_lens[i]

        # Copy prompt tokens
        prompt_tensor = (
            prompt_token_ids_batch[i]
            .detach()
            .clone()
            .to(dtype=torch.long, device=device)
        )
        input_ids[i, :prompt_len] = prompt_tensor

        # Copy generated tokens
        gen_tokens = generated_token_ids_batch[i].to(device)
        input_ids[i, prompt_len : prompt_len + gen_lens] = gen_tokens

    logits_all = model.forward(input_ids).logits

    vocab_size = logits_all.size(-1)
    logits_generated = torch.zeros(
        (batch_size, max_gen_len, vocab_size), device=device, dtype=logits_all.dtype
    )

    for i in range(batch_size):
        prompt_len = prompt_lens[i]
        logits_generated[i] = logits_all[i, prompt_len : prompt_len + max_gen_len]

    return logits_generated


async def training_loop(
    tokenizer,
    model,
    device,
    group_size=10,
    batch_size=2,
    epochs=3,
    rounds=3,
    lr=5e-5,
    epsilon_low=0.1,
    epsilon_high=0.2,
    recent_ckpt_path=None,
    pad_token_id=0,
):
    model = model.to(device)
    model.train()

    for round_ix in tqdm(range(rounds), desc="Rounds"):
        examples = await sample_examples(
            tokenizer, model, device, group_size * batch_size * 5
        )
        save_data_dir(examples, f"data/round_{round_ix}")
        dataset = ExampleGroupDataset(
            examples, group_size=group_size, pad_token_id=pad_token_id
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, pad_token_id),
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        for epoch in tqdm(range(epochs), desc="Epochs", leave=False):
            total_loss = 0.0

            for step, (old_logits, actions, rewards, prompt_ids) in tqdm(
                enumerate(dataloader), desc="Examples", leave=False
            ):
                # actions: [B, G, S], old_logits: [B, G, S, V], rewards: [B, G]
                B, G, S = actions.shape

                actions = actions.to(device)
                old_logits = old_logits.to(device)
                rewards = rewards.to(device)

                # Flatten batch and group dims for model input: [B*G, S]
                input_ids = actions.reshape(B * G, S)

                optimizer.zero_grad()

                # Generate current logits from model
                prompt_ids = [p for batch in prompt_ids for p in batch]
                logits = get_logits_with_prompt_batch(
                    pad_token_id, prompt_ids, input_ids, model, device
                )  # [B*G, S, V]

                # Reshape logits back to [B, G, S, V]
                logits = logits.view(B, G, S, -1)

                loss = dapo_loss_vectorized(
                    logits,
                    old_logits,
                    actions,
                    rewards,
                    epsilon_low=epsilon_low,
                    epsilon_high=epsilon_high,
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                if recent_ckpt_path is not None:
                    state = {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    }

                    # Always save recent checkpoint (overwrite)
                    torch.save(state, recent_ckpt_path)

                total_loss += loss.item()

                if (step + 1) % 10 == 0:
                    avg_loss = total_loss / 10
                    with open("reinforcement.log", "a") as file:
                        file.write(
                            f"Epoch {epoch+1} Step {step+1} Avg Loss {avg_loss:.4f}"
                        )
                    total_loss = 0.0
