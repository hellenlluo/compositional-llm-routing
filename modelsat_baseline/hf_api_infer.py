#!/usr/bin/env python3
"""
Quick inference via HuggingFace Inference API.

Usage:
    export HF_TOKEN=hf_...
    python modelsat_baseline/hf_api_infer.py --prompt "What is the capital of France?"
    python modelsat_baseline/hf_api_infer.py --prompt "What is 2+2?" --model "Qwen/Qwen2.5-72B-Instruct:novita"
"""
import argparse
import os

from openai import OpenAI


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct:nscale",
        help="HF model string in 'org/model:provider' format",
    )
    p.add_argument("--system", default=None, help="Optional system prompt")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError("Set HF_TOKEN environment variable first.")

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=token,
    )

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})

    completion = client.chat.completions.create(
        model=args.model,
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    print(completion.choices[0].message.content)


if __name__ == "__main__":
    main()
