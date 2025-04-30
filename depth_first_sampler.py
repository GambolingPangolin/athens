#!/usr/bin/env python3

import json
import math
from time import time

import requests
from tqdm import tqdm


class TreeNode:
    def __init__(self, generated_tokens, log_probability):
        self.generated_tokens = generated_tokens  # List of tokens forming this prefix
        self.log_probability = log_probability
        self.children = {}  # token -> TreeNode
        self.is_terminal = False


class DepthFirstSampler:
    def __init__(
        self,
        api_url,
        api_key=None,
        temperature=1.0,
        threshold_prob=0.05,
        max_tokens_per=50,
        max_tokens=1000,
    ):
        self.api_url = api_url
        self.api_key = api_key
        self.temperature = temperature
        self.threshold_logprob = math.log(threshold_prob)
        self.max_tokens_per = max_tokens_per
        self.max_tokens = max_tokens

    def _call_api(self, prompt, generated_tokens):
        # Send tokens as prompt (can adjust to match your API's expected input format)
        prompt_text = self._tokens_to_text(
            prompt, generated_tokens
        )  # Implement as appropriate

        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = {
            "model": "sglang-model",
            "prompt": prompt_text,
            "max_tokens": 1,
            "temperature": self.temperature,
            "logprobs": 10,  # Request logprobs for threshold filtering if supported
            "echo": False,
        }
        response = requests.post(
            f"{self.api_url}/v1/completions", headers=headers, json=data
        )
        response.raise_for_status()
        return response.json()

    def _tokens_to_text(self, prompt, tokens):
        # Convert token ids to text if you have a tokenizer, else assume tokens are strings
        # For prototype, assume tokens are strings
        return prompt + "".join(tokens)

    def _parse_response(self, resp):
        # Extract next token logits/probs and token text
        choice = resp["choices"][0]
        # Extract token logprobs and tokens
        logprobs = choice.get("logprobs")
        if logprobs is None:
            return {}
        next_token_logprobs = {
            x["token"]: x["logprob"] for x in logprobs["content"][0]["top_logprobs"]
        }
        return next_token_logprobs

    def _expand_node(self, prompt, node):
        resp = self._call_api(prompt, node.generated_tokens)
        next_token_logprobs = self._parse_response(resp)

        if not next_token_logprobs:
            node.is_terminal = True

        # Filter tokens by threshold probability
        filtered_tokens = {
            tok: logprob
            for tok, logprob in next_token_logprobs.items()
            if logprob >= self.threshold_logprob
        }

        for tok_str, tok_logp in filtered_tokens.items():
            new_prefix = node.generated_tokens + [tok_str]
            child_node = TreeNode(
                generated_tokens=new_prefix,
                log_probability=node.log_probability + tok_logp,
            )
            node.children[tok_str] = child_node

    def sample(self, prompt):
        # Depth-first expansion of tree nodes
        stack = [TreeNode(generated_tokens=[], log_probability=0)]
        outputs = []
        total_tokens = 0
        with tqdm(total=self.max_tokens, leave=False) as pbar:
            while stack:
                node = stack.pop()
                if len(node.generated_tokens) >= self.max_tokens_per:
                    outputs.append(
                        (node.log_probability, "".join(node.generated_tokens))
                    )
                    continue
                self._expand_node(prompt, node)
                if node.is_terminal:
                    outputs.append(
                        (node.log_probability, "".join(node.generated_tokens))
                    )
                    continue
                # Add children nodes to stack for depth-first
                for child in list(node.children.values()):
                    stack.append(child)
                total_tokens += len(node.children)
                pbar.update(len(node.children))
                if total_tokens >= self.max_tokens:
                    break
        p_total = sum(math.exp(p) for p, _ in outputs)
        return [
            (math.exp(logp - math.log(p_total)), output) for logp, output in outputs
        ]


segments = 6


def get_completion(prompt_text):
    headers = {
        "Content-Type": "application/json",
    }
    data = {
        "model": "sglang-model",
        "prompt": prompt_text,
        "temperature": 0.8,
        "max_tokens": 25 * segments,
    }
    response = requests.post(
        f"http://localhost:8080/v1/completions", headers=headers, json=data
    )
    response.raise_for_status()
    response = response.json()
    return prompt_text + response["choices"][0]["text"]


def sample_standard(prompt):
    start_time = time()
    result = get_completion(prompt)
    duration = time() - start_time
    return duration, result


def sample_depth_first(prompt):
    sampler = DepthFirstSampler(
        "http://localhost:8080",
        threshold_prob=0.05,
        max_tokens_per=25,
        max_tokens=200,
    )

    start_time = time()
    ps = []
    for _ in tqdm(range(segments), leave=False):
        outputs = sampler.sample(prompt)
        if not outputs:
            break
        p_output, continuation = sorted(outputs)[-1]
        prompt += continuation
        ps.append((p_output, continuation))

    duration = time() - start_time
    return duration, ps, prompt


prompts = [
    "The Fundamental Theorem of Calculus states ",
    """
    We can write out the Fundamental Theorem of Calculus as a Lean4 theorem:

    ```lean4
    """,
    "An simple example of a Lean 4 proof: ",
    """
    ```lean4
    theorem {M : Type*} [Monoid M] (x : M) : x * 1 = x :=
    """,
]

results = []

for ix, prompt in enumerate(prompts):
    standard_duration, standard_result = sample_standard(prompt)
    depth_first_duration, depth_first_ps, depth_first_result = sample_depth_first(
        prompt
    )

    results.append(
        dict(
            standard=dict(duration=standard_duration, output=standard_result),
            depth_first=dict(
                duration=depth_first_duration,
                ps=depth_first_ps,
                output=depth_first_result,
            ),
        )
    )

    print(f"=== Prompt {ix} ===\n{prompt}\n")
    print(f"Depth-first runtime increase: {depth_first_duration / standard_duration}\n")

    print("=== Standard ===")
    print(f"Duration: {standard_duration} seconds\n")
    print(f"{standard_result}\n")

    print("=== Depth first ===")
    print(f"Duration: {depth_first_duration} seconds\n")
    for p, span in depth_first_ps:
        print(f"{p} - {span}")
    print(f"\n{depth_first_result}\n")

with open("/tmp/depth-first.json", "w") as f:
    f.write(json.dumps(results))
