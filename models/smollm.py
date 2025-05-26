#!/usr/bin/env python3

import math

import torch
from torch import nn
import torch.nn.functional as F


class LlamaRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = x.norm(dim=-1, keepdim=True)
        x_normed = x / (norm_x + self.eps)
        return self.scale * x_normed


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim=192):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb


def apply_rotary_pos_emb(q, k, freqs):
    # q, k: (batch, heads, seq_len, head_dim)
    # freqs: (seq_len, rotary_dim)
    rotary_dim = freqs.shape[-1]
    q1, q2 = q[..., :rotary_dim].chunk(2, dim=-1)
    k1, k2 = k[..., :rotary_dim].chunk(2, dim=-1)

    half_dim = rotary_dim // 2
    sin, cos = freqs.sin(), freqs.cos()
    sin = sin[..., :half_dim]
    cos = cos[..., :half_dim]

    q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)

    q = torch.cat([q_rot, q[..., rotary_dim:]], dim=-1)
    k = torch.cat([k_rot, k[..., rotary_dim:]], dim=-1)
    return q, k


def causal_mask(seq_len, device):
    return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))


class LlamaAttention(nn.Module):
    def __init__(self, embed_dim=576, n_head=3):
        super().__init__()
        self.n_head = n_head
        self.head_dim = embed_dim // n_head
        assert self.head_dim * n_head == embed_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim)

    def forward(self, x, mask=None):
        B, T, C = x.size()

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, -1).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, -1).transpose(1, 2)

        freqs = self.rotary_emb(T, x.device).unsqueeze(0).unsqueeze(0)
        q, k = apply_rotary_pos_emb(q, k, freqs)

        # Create causal mask if none provided
        if mask is None:
            mask = causal_mask(T, x.device).unsqueeze(0).unsqueeze(0)  # 1,1,T,T

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(attn_scores, dim=-1)
        out = attn @ v

        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(out)


class LlamaMLP(nn.Module):
    def __init__(self, embed_dim=576):
        super().__init__()
        hidden_dim = 1536
        self.gate_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(nn.Module):
    def __init__(self, embed_dim=576, n_head=3):
        super().__init__()
        self.self_attn = LlamaAttention(embed_dim=embed_dim, n_head=n_head)
        self.mlp = LlamaMLP(embed_dim=embed_dim)

        self.input_layernorm = LlamaRMSNorm(embed_dim)
        self.post_attention_layernorm = LlamaRMSNorm(embed_dim)

    def forward(self, x, mask=None):
        h = self.input_layernorm(x)
        x = x + self.self_attn(h, mask)
        h2 = self.post_attention_layernorm(x)
        x = x + self.mlp(h2)
        return x


class LlamaModel(nn.Module):
    def __init__(self, vocab_size=49152, embed_dim=576, n_layers=30, n_head=3):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(embed_dim, n_head) for _ in range(n_layers)]
        )
        self.norm = LlamaRMSNorm(embed_dim)

    def forward(self, input_ids, mask=None):
        x = self.embed_tokens(input_ids)  # B, T, C
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return x


class LlamaForCausalLM(nn.Module):
    def __init__(self, vocab_size=49152):
        super().__init__()
        self.model = LlamaModel()
        self.lm_head = nn.Linear(576, vocab_size, bias=False)

    def forward(self, input_ids, mask=None):
        hidden = self.model(input_ids, mask)
        logits = self.lm_head(hidden)
        return logits
