#!/usr/bin/env python3


import torch
import torch.nn.functional as F


def dapo_loss(logits, old_logits, actions, rewards, epsilon_low=0.1, epsilon_high=0.2):
    """
    Compute DAPO loss using mean reward as baseline for advantage normalization.

    Args:
        logits: Tensor [group, seq_len, vocab_size], current policy logits.
        old_logits: Tensor [group, seq_len, vocab_size], old policy logits (detached).
        actions: LongTensor [group, seq_len], sampled token indices.
        rewards: Tensor [group], scalar reward per output/sample (same reward repeated over seq).
                 Assumes one reward per output (not per token), matching {R_i} in paper.
        epsilon_low: float, lower clip bound.
        epsilon_high: float, upper clip bound.

    Returns:
        loss: scalar tensor.
    """
    group_size, seq_len, vocab_size = logits.shape

    # Expand scalar rewards to per-token shape for broadcasting
    rewards_expanded = rewards.unsqueeze(1).expand(-1, seq_len)

    # Compute mean and std of rewards over group (the group of outputs)
    reward_mean = rewards.mean()
    reward_std = rewards.std(unbiased=False) + 1e-8

    # Compute normalized advantages per token: (R_i - mean(R)) / std(R)
    advantages = (rewards_expanded - reward_mean) / reward_std

    # Compute action probabilities under current and old policies
    prob = F.softmax(logits, dim=-1)
    old_prob = F.softmax(old_logits, dim=-1)

    # Gather probabilities of selected actions
    prob_action = prob.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(
        -1
    )  # [group, seq_len]
    old_prob_action = old_prob.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(
        -1
    )  # [group, seq_len]

    # Importance sampling ratio
    ratio = prob_action / (old_prob_action + 1e-8)

    # Clip the ratio with asymmetric bounds: [1 - epsilon_low, 1 + epsilon_high]
    clipped_ratio = torch.clamp(ratio, 1 - epsilon_low, 1 + epsilon_high)

    # Calculate clipped surrogate objective
    surrogate1 = ratio * advantages
    surrogate2 = clipped_ratio * advantages

    loss_per_token = -torch.min(surrogate1, surrogate2)

    # Average across group and sequence tokens
    loss = loss_per_token.mean()
    return loss


def dapo_loss_vectorized(
    logits, old_logits, actions, rewards, epsilon_low=0.1, epsilon_high=0.2
):
    """
    Vectorized DAPO loss over batches of groups.

    Args:
        logits: Tensor [B, G, S, V], current policy logits.
        old_logits: Tensor [B, G, S, V], old policy logits (detached).
        actions: LongTensor [B, G, S], sampled token indices.
        rewards: Tensor [B, G], scalar reward per output.
        epsilon_low: float clip lower bound.
        epsilon_high: float clip upper bound.

    Returns:
        Scalar loss: mean over all tokens, groups, batches.
    """
    B, G, S, V = logits.shape

    # Expand rewards for advantage calculation: [B, G] -> [B, G, S]
    rewards_exp = rewards.unsqueeze(-1).expand(-1, -1, S)

    # Compute mean and std within each group (across G dimension) for baseline normalization
    reward_mean = rewards.mean(dim=1, keepdim=True)  # [B, 1]
    reward_std = rewards.std(dim=1, unbiased=False, keepdim=True) + 1e-8  # [B, 1]

    # Expand baseline stats for token-level subtraction
    reward_mean_exp = reward_mean.unsqueeze(-1).expand(-1, G, S)  # [B, G, S]
    reward_std_exp = reward_std.unsqueeze(-1).expand(-1, G, S)  # [B, G, S]

    # Calculate normalized advantages per token:
    advantages = (rewards_exp - reward_mean_exp) / reward_std_exp  # [B, G, S]

    # Compute probabilities of actions under current and old policies
    prob = F.softmax(logits, dim=-1)  # [B, G, S, V]
    old_prob = F.softmax(old_logits, dim=-1)  # [B, G, S, V]

    # Gather probabilities of the taken actions
    actions_exp = actions.unsqueeze(-1)  # [B, G, S, 1]
    prob_action = prob.gather(dim=-1, index=actions_exp).squeeze(-1)  # [B, G, S]
    old_prob_action = old_prob.gather(dim=-1, index=actions_exp).squeeze(
        -1
    )  # [B, G, S]

    # Compute importance sampling ratio
    ratio = prob_action / (old_prob_action + 1e-8)  # [B, G, S]

    # Clip ratio asymmetrically: [1 - epsilon_low, 1 + epsilon_high]
    clipped_ratio = torch.clamp(ratio, 1 - epsilon_low, 1 + epsilon_high)

    # PPO-style clipped surrogate objective, min between clipped and unclipped
    surrogate1 = ratio * advantages
    surrogate2 = clipped_ratio * advantages
    loss_per_token = -torch.min(surrogate1, surrogate2)  # [B, G, S]

    # Average loss over tokens, groups, and batch
    loss = loss_per_token.mean()

    return loss
