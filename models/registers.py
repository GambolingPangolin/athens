#!/usr/bin/env python3

import torch


def interleave_register_tokens(
    input_ids: torch.Tensor, register_token_id: int
) -> torch.Tensor:
    # input_ids: (batch, seq_len)
    batch_size, seq_len = input_ids.size()
    new_seq_len = seq_len * 2 - 1  # interleave one special token between tokens

    interleaved = torch.full(
        (batch_size, new_seq_len),
        fill_value=register_token_id,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    interleaved[:, ::2] = input_ids  # copy regular tokens to even positions
    # odd positions remain register_token_id

    return interleaved


def create_targets_fixed_offset(
    input_ids: torch.Tensor, register_token_id: int, offset: int = 2
) -> torch.Tensor:
    batch_size, seq_len = input_ids.shape
    targets = torch.full(
        (batch_size, seq_len),
        fill_value=-100,
        dtype=torch.long,
        device=input_ids.device,
    )

    for b in range(batch_size):
        input_seq = input_ids[b].tolist()
        reg_positions = [
            i for i, t in enumerate(input_seq) if t != register_token_id
        ]  # regular token indices

        for i in range(seq_len):
            if input_seq[i] != register_token_id:
                # Regular token: target = next regular token (i-th regular token → (i+1)-th regular token)
                idx_in_reg = reg_positions.index(i)
                target_idx = idx_in_reg + 1
                if target_idx < len(reg_positions):
                    targets[b, i] = input_seq[reg_positions[target_idx]]
                else:
                    targets[b, i] = -100
            else:
                # Register token: target = (i+offset)-th regular token
                # Find the regular token at position >= i + offset counting only regular tokens
                # We'll find the first reg_position > i + offset - 1 (or offset ahead of this position)
                target_reg_tokens = [pos for pos in reg_positions if pos >= i + offset]
                if target_reg_tokens:
                    targets[b, i] = input_seq[target_reg_tokens[0]]
                else:
                    targets[b, i] = -100

    return targets
