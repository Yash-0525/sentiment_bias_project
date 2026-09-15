"""
src/generate.py  —  continuation generation for counterfactual evaluation.
"""

from __future__ import annotations

from typing import Sequence

import torch
from transformers import AutoTokenizer

from src.model import SentimentBiasLM


class ContinuationGenerator:
    """
    Paper defaults (App. B): max 50 tokens, temperature 1.0.
    Student sample counts: 100 full / 20–50 smoke (D10).
    """

    def __init__(
        self,
        model: SentimentBiasLM,
        tokenizer_name: str = "gpt2",
        device: str | torch.device | None = None,
    ):
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if device is None:
            device = model.device
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate_one_prompt(
        self,
        prompt: str,
        *,
        n_samples: int = 100,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        batch_size: int = 8,
    ) -> list[str]:
        """Return n_samples continuation strings (without the prompt prefix)."""
        continuations: list[str] = []
        remaining = n_samples
        while remaining > 0:
            bs = min(batch_size, remaining)
            enc = self.tokenizer(
                [prompt] * bs,
                return_tensors="pt",
                padding=True,
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            prompt_len = int(input_ids.shape[1])

            out_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
            )
            # strip prompt tokens
            gen_only = out_ids[:, prompt_len:]
            texts = self.tokenizer.batch_decode(gen_only, skip_special_tokens=True)
            continuations.extend(t.strip() for t in texts)
            remaining -= bs
        return continuations[:n_samples]

    @torch.no_grad()
    def generate_many_prompts(
        self,
        prompts: Sequence[str],
        **kwargs,
    ) -> list[list[str]]:
        return [self.generate_one_prompt(p, **kwargs) for p in prompts]
