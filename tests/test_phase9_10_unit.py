"""Unit tests for counterfactual swap + cosine distance (no GPU)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_cosine_distance() -> None:
    import torch
    from src.counterfactual import cosine_distance

    a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    b = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    d = cosine_distance(a, b)
    assert float(d) < 1e-5

    b2 = torch.tensor([[-1.0, 0.0], [0.0, -1.0]])
    d2 = cosine_distance(a, b2)
    assert abs(float(d2) - 2.0) < 1e-5  # cos=-1 → dist=2
    print("  cosine distance OK")


def test_other_value() -> None:
    from src.counterfactual import other_value

    rng = random.Random(0)
    v = other_value("baker", rng)
    assert v != "baker"
    assert v.lower() != "baker"
    print("  other_value OK", v)


def test_cf_swap_with_tokenizer() -> None:
    """Requires transformers + gpt2 tokenizer (downloads once). Skip if unavailable."""
    try:
        import torch
        from transformers import AutoTokenizer
        from src.counterfactual import make_counterfactual_ids
    except Exception as exc:  # noqa: BLE001
        print("  SKIP tokenizer test:", exc)
        return

    tok = AutoTokenizer.from_pretrained("gpt2")
    text = "My friend is a baker and we went out."
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    cf, meta = make_counterfactual_ids(ids, tok, rng=random.Random(1))
    if cf is None:
        # may fail if baker is multi-piece unexpectedly — try Italy sentence
        text = "People from Italy are kind today."
        ids = torch.tensor(tok.encode(text), dtype=torch.long)
        cf, meta = make_counterfactual_ids(ids, tok, rng=random.Random(1))
    assert cf is not None, "could not build CF"
    assert meta is not None
    assert int((cf != ids).sum()) == 1
    assert meta["original_value"] != meta["counterfactual_value"]
    print("  CF swap OK", meta["original_value"], "->", meta["counterfactual_value"])


def main() -> int:
    print("Phase 9/10 unit tests")
    test_cosine_distance()
    test_other_value()
    test_cf_swap_with_tokenizer()
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
