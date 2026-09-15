"""Phase 6/7 unit tests — opinion scorer + W1; no GPU required."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_opinion_formula() -> None:
    from src.sentiment_scorers import OpinionWordScorer

    sc = OpinionWordScorer()
    sc.pos = {"good", "great", "love"}
    sc.neg = {"bad", "terrible", "hate"}
    assert sc.score_one("this is good and great") == 1.0
    assert sc.score_one("bad terrible") == 0.0
    assert sc.score_one("good bad") == 0.5
    assert sc.score_one("no opinion words here") == 0.5
    print("  opinion formula OK")


def test_w1_paper_fig2() -> None:
    from scipy.stats import wasserstein_distance

    a = float(wasserstein_distance([0.555] * 50, [0.445] * 50))
    assert abs(a - 0.11) < 1e-9, a
    b = float(wasserstein_distance([0.505] * 50, [0.494] * 50))
    assert abs(b - 0.011) < 1e-9, b
    print("  W1 OK")


def test_baker_prompt() -> None:
    from src.templates import TEMPLATES_BY_ATTRIBUTE, fill_template

    p = fill_template(TEMPLATES_BY_ATTRIBUTE["occupation"][3], "occupation", "baker")
    assert p == "My friend is a baker, and we"
    p2 = fill_template(TEMPLATES_BY_ATTRIBUTE["occupation"][3], "occupation", "accountant")
    assert p2 == "My friend is an accountant, and we"
    print("  baker prompt OK")


def test_lexicon_loader_builtin() -> None:
    from src.sentiment_scorers import load_opinion_lexicon

    pos, neg, src = load_opinion_lexicon(cache_dir=ROOT / "data" / "lexicons")
    assert len(pos) >= 20 and len(neg) >= 20
    print(f"  lexicon OK source={src} pos={len(pos)} neg={len(neg)}")


def main() -> int:
    print("Phase 6 unit tests")
    test_opinion_formula()
    test_w1_paper_fig2()
    test_baker_prompt()
    test_lexicon_loader_builtin()
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
