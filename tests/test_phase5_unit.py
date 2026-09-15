"""Phase 5 unit tests — no GPU required, no full WikiText."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.dataset_lm import PackedSequenceDataset, collate_lm, make_loader
from src.model import SentimentBiasLM, causal_lm_loss_from_logits


def test_collate_and_dataset() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "seq.jsonl"
        rows = [
            {"input_ids": list(range(8)), "has_sensitive": False},
            {"input_ids": list(range(10, 18)), "has_sensitive": True},
            {"input_ids": list(range(20, 28)), "has_sensitive": True},
        ]
        p.write_text("\n".join(json.dumps(r) for r in rows))
        ds = PackedSequenceDataset(p)
        assert len(ds) == 3
        ds2 = PackedSequenceDataset(p, sensitive_only=True)
        assert len(ds2) == 2
        loader = make_loader(p, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        assert batch["input_ids"].shape == (2, 8)
        assert batch["labels"].shape == (2, 8)
        assert batch["attention_mask"].shape == (2, 8)
    print("  dataset/collate OK")


def test_tiny_forward_h_bar() -> None:
    """Random tiny GPT-2 config — same path as check_environment + Phase 10."""
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = GPT2Config(
        vocab_size=64,
        n_positions=32,
        n_embd=32,
        n_layer=4,
        n_head=4,
        bos_token_id=0,
        eos_token_id=1,
    )
    # build wrapper-like path without downloading
    class _Wrap(SentimentBiasLM):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.model_name = "tiny"
            self.model = GPT2LMHeadModel(cfg)
            self.config = cfg
            self.n_layer = cfg.n_layer
            self.n_embd = cfg.n_embd

    m = _Wrap().eval()
    ids = torch.randint(0, 64, (2, 8))
    with torch.no_grad():
        out = m(ids, labels=ids, output_hidden_states=True)
    assert out.loss is not None and torch.isfinite(out.loss)
    assert out.h_bar is not None
    assert out.h_bar.shape == (2, 8, 32)
    assert len(out.hidden_states) == 5  # emb + 4 layers
    # h_bar = mean of last two
    manual = (out.hidden_states[-2] + out.hidden_states[-1]) / 2
    assert torch.allclose(out.h_bar, manual)
    print("  tiny forward + h_bar OK")


def test_loss_shift() -> None:
    logits = torch.randn(2, 8, 50)
    ids = torch.randint(0, 50, (2, 8))
    loss = causal_lm_loss_from_logits(logits, ids)
    assert torch.isfinite(loss)
    print("  loss shift OK")


def main() -> int:
    print("Phase 5 unit tests")
    test_collate_and_dataset()
    test_tiny_forward_h_bar()
    test_loss_shift()
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
