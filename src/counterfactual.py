"""
src/counterfactual.py  —  PHASE 9

Build counterfactual prefixes for debiasing training.

Paper §4:
  Given prefix x_{1:i} whose last (or any) token is a sensitive token from φ(a),
  build x̃ by replacing that sensitive surface form with a token from φ(ã), ã≠a.
  All non-sensitive tokens stay unchanged.

Training constraint (paper p.70):
  Evaluation templates are NEVER used here. We only edit real WikiText sequences
  that already contain sensitive tokens (Phase 2 flags).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from src.sensitive_attributes import (
    ATTRIBUTE_VALUES,
    COUNTRIES,
    FEMALE_NAMES,
    MALE_NAMES,
    OCCUPATIONS,
    VALUE_TO_SUBGROUP,
    all_sensitive_tokens,
)


# token → attribute bucket for sampling a different value
_ATTR_POOLS: dict[str, list[str]] = {
    "country": list(COUNTRIES),
    "occupation": list(OCCUPATIONS),
    "name": list(MALE_NAMES) + list(FEMALE_NAMES),
}


@dataclass
class CFMatch:
    """One sensitive span found after decoding a sequence."""

    canonical: str
    attribute: str
    subgroup: str
    # character span in decoded text (approx; used only for text-level swap tests)
    start: int
    end: int


def other_value(canonical: str, rng: random.Random | None = None) -> str:
    """Sample a different sensitive value from the same attribute."""
    rng = rng or random
    attr, _ = VALUE_TO_SUBGROUP[canonical]
    pool = [v for v in _ATTR_POOLS[attr] if v.lower() != canonical.lower()]
    if not pool:
        raise RuntimeError(f"no alternate value for {canonical}")
    return rng.choice(pool)


def find_sensitive_token_positions(
    input_ids: torch.Tensor,
    tokenizer,
) -> list[tuple[int, str]]:
    """
    Find positions in a 1D token-id sequence that decode to a sensitive surface form.

    Returns list of (token_index, canonical_value).

    Strategy:
      - decode each token with leading-space variants via convert_ids_to_tokens
      - match against the paper sensitive list (case-insensitive, strip Ġ)
    This is approximate under BPE but sufficient for student-scale debiasing:
    we only need *some* reliable sensitive-token positions per sequence.
    """
    ids = input_ids.tolist() if torch.is_tensor(input_ids) else list(input_ids)
    # map normalized surface → canonical
    surface_map = {t.lower(): t for t in all_sensitive_tokens()}
    # multi-word not in list; all paper tokens are single orthographic tokens
    hits: list[tuple[int, str]] = []
    for i, tid in enumerate(ids):
        tok = tokenizer.convert_ids_to_tokens(tid)
        if tok is None:
            continue
        # GPT-2: leading space is 'Ġ'
        plain = tok.lstrip("Ġ").replace("Ċ", "").strip()
        if not plain:
            continue
        key = plain.lower()
        if key in surface_map:
            hits.append((i, surface_map[key]))
    return hits


def make_counterfactual_ids(
    input_ids: torch.Tensor,
    tokenizer,
    *,
    rng: random.Random | None = None,
    prefer_last: bool = True,
) -> tuple[torch.Tensor, dict] | tuple[None, None]:
    """
    Return (cf_input_ids, meta) or (None, None) if no swappable sensitive token.

    Swaps ONE sensitive token position (last hit if prefer_last else random hit)
    with a token-id encoding of another value from the same attribute.

    If the replacement value tokenizes to a different number of BPE pieces than
    the original token, we skip (keep lengths equal for simple batching).
    """
    rng = rng or random
    if input_ids.dim() != 1:
        raise ValueError("expected 1D input_ids")

    hits = find_sensitive_token_positions(input_ids, tokenizer)
    if not hits:
        return None, None

    if prefer_last:
        pos, canonical = hits[-1]
    else:
        pos, canonical = rng.choice(hits)

    alt = other_value(canonical, rng)
    # encode replacement — GPT-2 usually one piece for these words, sometimes more
    # try with leading space (word-interior) and without
    candidates = []
    for prefix in (" ", ""):
        pieces = tokenizer.encode(prefix + alt, add_special_tokens=False)
        candidates.append(pieces)
    # prefer single-token replacement matching single original piece
    orig_pieces_guess = 1
    single = [c for c in candidates if len(c) == orig_pieces_guess]
    if not single:
        # cannot keep length — skip this example
        return None, None
    new_id = single[0][0]

    cf = input_ids.clone()
    cf[pos] = int(new_id)
    meta = {
        "position": int(pos),
        "original_value": canonical,
        "counterfactual_value": alt,
        "attribute": VALUE_TO_SUBGROUP[canonical][0],
        "original_subgroup": VALUE_TO_SUBGROUP[canonical][1],
        "counterfactual_subgroup": VALUE_TO_SUBGROUP[alt][1],
        "original_token_id": int(input_ids[pos]),
        "counterfactual_token_id": int(new_id),
    }
    # verify only one position changed
    n_diff = int((cf != input_ids).sum().item())
    if n_diff != 1:
        return None, None
    return cf, meta


def cosine_distance(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Paper: d = 1 - cos(a, b).
    a, b: [B, D] or [D]
    returns mean over batch of per-row cosine distances.
    """
    if a.dim() == 1:
        a = a.unsqueeze(0)
        b = b.unsqueeze(0)
    a_n = a / (a.norm(dim=-1, keepdim=True) + eps)
    b_n = b / (b.norm(dim=-1, keepdim=True) + eps)
    cos = (a_n * b_n).sum(dim=-1)
    return (1.0 - cos).mean()


def pool_h_bar(h_bar: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    h_bar: [B, T, D] → [B, D] mean pool over tokens (mask-aware if provided).
    """
    if attention_mask is None:
        return h_bar.mean(dim=1)
    mask = attention_mask.unsqueeze(-1).to(h_bar.dtype)
    return (h_bar * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
