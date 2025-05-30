#!/usr/bin/env python3

from lean.run import run_lake_lean_example


async def reward_function(tokenizer, output_dict, modules, max_error_checks=5):
    """
    Score generated example by writing out source with imports + generated token text + ' sorry',
    then running lake lean on it. Scores:

    - Ill formed: -10
    - Well formed, but error: -1
    - No errors: +1

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
    penalty = -min(n_errors, 5) / 5

    return penalty, stdout_lines
