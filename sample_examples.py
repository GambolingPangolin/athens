#!/usr/bin/env python3

import argparse
import asyncio
import json
import os
import random
from tokenizers import Tokenizer
import torch
from tqdm.asyncio import tqdm_asyncio

from models.smollm import LlamaForCausalLM
from lean_interop import run_lake_lean_example

CONCURRENT_SAMPLES = 25


def generate_mathlib_example(
    model,
    tokenizer,
    device,
    modules,
    max_gen_tokens=100,
):
    """
    Generate a Lean example proof snippet from the model.

    Args:
        model (LlamaForCausalLM): The language model.
        tokenizer: The tokenizer with encode/decode methods.
        device: torch.device, model device.
        modules (list[str]): list of Mathlib module names to import.
        max_gen_tokens (int): max tokens to generate.
        stop_tokens (list[str]): strings to stop generation on.

    Returns:
        dict with keys:
            "tokens": list of generated token ids,
            "logits": list of logits tensors (per step),
            "prompt_ids": input token ids,
    """
    model.eval()

    # Build prompt text
    prompt_lines = [f"import {module}" for module in modules]
    prompt_lines.append("")  # empty line
    prompt_lines.append("example ")
    prompt_text = "\n".join(prompt_lines)

    # Tokenize prompt, get input ids tensor
    prompt_ids = tokenizer.encode(prompt_text).ids
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    generated_ids = []
    logits_list = []

    stop_reason = "token_limit"

    with torch.no_grad():
        for _ in range(max_gen_tokens):
            outputs = model(input_ids)  # (1, seq_len, vocab_size)
            logits = outputs[0, -1, :]  # last token logits: (vocab_size,)
            logits_list.append(logits.cpu())

            probs = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()

            generated_ids.append(next_token_id)

            # Append next token to input_ids for next step
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_token_id]], device=device)], dim=1
            )

            # Decode current generated tokens (only generated part)
            generated_text = tokenizer.decode(generated_ids)

            # Check if any stop token is in generated text (simplified)
            if ":=" in generated_text:
                stop_reason = "stop_token"
                break

    return {
        "prompt_ids": torch.tensor(prompt_ids),
        "tokens": torch.tensor(generated_ids),
        "logits": torch.stack(logits_list),
        "stop_reason": stop_reason,
    }


async def reward_function(tokenizer, output_dict, modules, max_error_checks=5):
    """
    Score generated example by writing out source with imports + generated token text + ' sorry',
    then running lake lean on it. Scores:

    - Nonzero exit code: raise RuntimeError (fatal)
    - stderr is exactly one line ending with "warning: declaration uses 'sorry'": +1
    - stderr contains "syntax error": -10
    - Otherwise: -max(n_errors, 5)/5 where n_errors is count of lines in stderr

    Args:
        output_dict: dict from generate_mathlib_example with keys "tokens", "prompt_ids" ...
        modules: list of imported mathlib modules (strings)

    Returns:
        float score
    """
    # Compose source from prompt + generated tokens + " sorry"
    prompt_lines = [f"import {m}" for m in modules]
    prompt_lines.append("")
    # Based on prompt construction in generate_mathlib_example
    prompt_lines.append("example ")
    prompt_text = "\n".join(prompt_lines)
    generated_text = tokenizer.decode(
        output_dict["tokens"].tolist()
    )  # assume 'tokenizer' accessible here
    source_text = prompt_text + generated_text + " sorry"

    # Run lake lean
    _exit_code, stdout, _stderr = await run_lake_lean_example(source_text)

    stdout_lines = [
        line.strip() for line in stdout.strip().splitlines() if line.strip()
    ]

    ill_formed_indicators = [
        "expected token",
        "unterminated string literal",
        "unexpected end of input",
    ]

    if any(
        indicator in line
        for indicator in ill_formed_indicators
        for line in stdout_lines
    ):
        return -10.0, stdout_lines

    error_lines = [line for line in stdout_lines if "error:" in line]

    if len(error_lines) == 0:
        return 1.0, stdout_lines

    n_errors = len(error_lines)
    penalty = -max(n_errors, 5) / 5

    return penalty, stdout_lines


def save_data_dir(examples, dirpath):
    """
    Save examples to a directory with:
    - JSON file 'examples.json' containing texts and scores
    - Torch file 'examples.pt' containing prompt_ids, tokens, logits, scores tensors
    Creates directory if missing and overwrites files.
    """
    os.makedirs(dirpath, exist_ok=True)
    json_path = os.path.join(dirpath, "examples.json")
    torch_path = os.path.join(dirpath, "examples.pt")

    # Save JSON with text and score
    with open(json_path, "w") as f:
        json.dump(
            [
                {
                    "score": ex.get("score"),
                    "text": ex["text"],
                    "lean_output": ex["lean_output"],
                    "stop_reason": ex["stop_reason"],
                }
                for ex in examples
            ],
            f,
            indent=2,
        )

    # Save tensors compactly
    data_to_save = {
        "prompt_ids": [],
        "tokens": [],
        "logits": [],
        "scores": torch.tensor([ex["score"] for ex in examples]),
    }

    for ex in examples:
        data_to_save["prompt_ids"].append(ex["prompt_ids"])
        data_to_save["tokens"].append(ex["tokens"])
        data_to_save["logits"].append(ex["logits"])

    torch.save(data_to_save, torch_path)


def load_data_dir(dirpath):
    """
    Load examples from directory containing 'examples.json' and 'examples.pt'.

    Returns a dict with keys:
      - 'examples' from JSON (list of dict with text and score)
      - 'prompt_ids', 'tokens', 'logits', 'scores' tensors from .pt file
    """
    json_path = os.path.join(dirpath, "examples.json")
    torch_path = os.path.join(dirpath, "examples.pt")

    with open(json_path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    data_tensors = torch.load(torch_path)

    for ix, example in enumerate(examples):
        example["prompt_ids"] = data_tensors["prompt_ids"][ix]
        example["tokens"] = data_tensors["tokens"][ix]
        example["logits"] = data_tensors["logits"][ix]

    return examples


async def sample_examples(tokenizer, model, device, n_examples):

    master_modules = [
        "Mathlib.Algebra.BigOperators.Ring.Nat",
        "Mathlib.Algebra.GCDMonoid.Nat",
        "Mathlib.Algebra.Group.NatPowAssoc",
        "Mathlib.Algebra.Group.PNatPowAssoc",
        "Mathlib.Algebra.GroupWithZero.Nat",
        "Mathlib.Algebra.Module.NatInt",
        "Mathlib.Algebra.Order.Antidiag.Nat",
        "Mathlib.Algebra.Order.Group.Nat",
        "Mathlib.Algebra.Order.Ring.Nat",
        "Mathlib.Algebra.Ring.Nat",
        "Mathlib.Data.Fin.Tuple.NatAntidiagonal",
        "Mathlib.Data.Finset.NatAntidiagonal",
        "Mathlib.Data.Finset.NatDivisors",
        "Mathlib.Data.Int.NatPrime",
        "Mathlib.Data.List.NatAntidiagonal",
        "Mathlib.Data.Multiset.NatAntidiagonal",
        "Mathlib.Data.Nat.Factorial.NatCast",
        "Mathlib.Logic.Equiv.Nat",
        "Mathlib.Order.Interval.Finset.Nat",
        "Mathlib.Order.Nat",
        "Mathlib.Order.OrderIsoNat",
        "Mathlib.RingTheory.UniqueFactorizationDomain.Nat",
        "Mathlib.SetTheory.Cardinal.NatCount",
        "Mathlib.SetTheory.Cardinal.ToNat",
        "Mathlib.SetTheory.Ordinal.NaturalOps",
        "Mathlib.Topology.Algebra.InfiniteSum.NatInt",
        "Mathlib.Topology.Instances.Nat",
    ]

    sem = asyncio.Semaphore(CONCURRENT_SAMPLES)
    pbar = tqdm_asyncio(desc="Generations", leave=False, total=n_examples)

    async def get_example():
        sampled_module = random.choice(master_modules)

        async with sem:
            example = generate_mathlib_example(
                model=model,
                tokenizer=tokenizer,
                device=device,
                modules=[sampled_module],
                max_gen_tokens=100,
            )

            example["text"] = tokenizer.decode(example["tokens"].tolist())

            # Compute reward score
            try:
                score, lean_output = await reward_function(
                    tokenizer, example, [sampled_module], tokenizer
                )
                example["score"] = score
                example["lean_output"] = lean_output
            except RuntimeError as e:
                with open("sample_examples.log", "a") as file:
                    file.write(f"Reward scoring failed: {e}\n")

            pbar.update()
            return example

    examples = await asyncio.gather(*[get_example() for _ in range(n_examples)])
    pbar.close()

    return examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate samples")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--examples_path",
        type=str,
        required=True,
        help="Path to the save file for outputs",
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        required=True,
        help="Number of examples to sample",
    )
    args = parser.parse_args()
    # Load tokenizer
    tokenizer_path = "checkpoints/tokenizer.json"
    tokenizer = Tokenizer.from_file(tokenizer_path)
    # Load model checkpoint (adjust path as needed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LlamaForCausalLM(vocab_size=tokenizer.get_vocab_size()).to(device)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    examples = sample_examples(tokenizer, model, device, args.n_examples)
    save_data_dir(examples, args.examples_path)
