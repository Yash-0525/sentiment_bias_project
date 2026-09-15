"""
src/model.py  —  Language model wrapper (Phase 5+)

Paper used Transformer-XL. Student setting (D04): GPT-2 small fine-tune.

This wrapper exists so later phases can:
  - run next-token LM loss
  - extract per-layer hidden states for h̄ = mean(h^(L-1), h^(L))
  - generate continuations for counterfactual evaluation

Nothing here implements debiasing yet (Phases 10–11).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LMOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] | None  # emb + each layer
    h_bar: torch.Tensor | None  # mean of last two transformer layers


class SentimentBiasLM(nn.Module):
    """
    Thin GPT-2 wrapper.

    hidden_states layout from HuggingFace GPT2LMHeadModel:
      hidden_states[0]     = embedding output
      hidden_states[1..L]  = after transformer block 1..L
    Paper h̄ uses the last TWO *layers* → indices -2 and -1 of hidden_states
    (both are post-block states; for L=12 that is blocks 11 and 12).
    """

    def __init__(self, model_name: str = "gpt2"):
        super().__init__()
        from transformers import GPT2LMHeadModel

        self.model_name = model_name
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.config = self.model.config
        self.n_layer = int(self.config.n_layer)
        self.n_embd = int(self.config.n_embd)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> LMOutput:
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=output_hidden_states,
            use_cache=False,
        )
        hs = out.hidden_states if output_hidden_states else None
        h_bar = None
        if hs is not None:
            # last two transformer-layer states
            h_bar = (hs[-2] + hs[-1]) / 2.0
        return LMOutput(
            loss=out.loss,
            logits=out.logits,
            hidden_states=hs,
            h_bar=h_bar,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """Paper generation defaults: max 50 tokens, temperature 1.0."""
        gen_kwargs: dict[str, Any] = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=max(temperature, 1e-6),
            pad_token_id=self.model.config.eos_token_id,
            eos_token_id=self.model.config.eos_token_id,
        )
        if top_k and top_k > 0:
            gen_kwargs["top_k"] = top_k
        if top_p < 1.0:
            gen_kwargs["top_p"] = top_p
        return self.model.generate(**gen_kwargs)

    def save_pretrained(self, path: str) -> None:
        self.model.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, path: str) -> "SentimentBiasLM":
        from transformers import GPT2LMHeadModel

        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.model_name = path
        obj.model = GPT2LMHeadModel.from_pretrained(path)
        obj.config = obj.model.config
        obj.n_layer = int(obj.config.n_layer)
        obj.n_embd = int(obj.config.n_embd)
        return obj


def causal_lm_loss_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Standard next-token cross-entropy (shift by 1).
    logits: [B, T, V], input_ids: [B, T]
    """
    # predict token t from positions 0..t-1
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="mean",
    )
