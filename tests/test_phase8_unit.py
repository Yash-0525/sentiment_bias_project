"""Phase 8 unit tests — MLP shape + label threshold logic (no GPU/download)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_mlp_shapes() -> None:
    import torch
    from src.sentiment_classifier import SentimentProjection

    m = SentimentProjection(input_dim=32, hidden_dim=16)
    h = torch.randn(4, 32)
    z = m.project(h)
    assert z.shape == (4, 16)
    logits = m(h)
    assert logits.shape == (4, 2)
    assert torch.isfinite(logits).all()
    n_linear = sum(1 for _ in m.modules() if isinstance(_, torch.nn.Linear))
    assert n_linear == 3
    print("  mlp shapes OK")


def test_threshold_mapping() -> None:
    # |2p-1| > 0.7 ⇒ p > 0.85 or p < 0.15
    def keep(p: float, thr: float = 0.7) -> bool:
        return abs(2 * p - 1) > thr

    assert keep(0.95) and keep(0.05)
    assert not keep(0.5) and not keep(0.8) and not keep(0.2)
    print("  threshold mapping OK")


def test_metrics() -> None:
    from src.sentiment_classifier import classification_report_basic

    y_true = [0, 0, 1, 1, 1, 0]
    y_pred = [0, 1, 1, 1, 0, 0]
    m = classification_report_basic(y_true, y_pred)
    assert m["n"] == 6
    assert m["confusion_matrix"] == [[2, 1], [1, 2]]
    assert 0.0 <= m["accuracy"] <= 1.0
    print("  metrics OK")


def main() -> int:
    print("Phase 8 unit tests")
    test_mlp_shapes()
    test_threshold_mapping()
    test_metrics()
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
