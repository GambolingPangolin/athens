#!/usr/bin/env python3


from transformers import LlamaConfig


def config(vocab_size):
    return LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=576,
        intermediate_size=1536,
        num_hidden_layers=30,
        num_attention_heads=3,
        rms_norm_eps=1e-5,
        rotary_embedding_base=10000,
    )
