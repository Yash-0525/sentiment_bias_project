"""
Unit tests for Phase 2 that do NOT need the full WikiText download.
Run:  python tests/test_phase2_unit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_preprocessing import (  # noqa: E402
    is_article_title_line,
    paper_article_split,
    parse_articles,
)
from src.sensitive_attributes import (  # noqa: E402
    COUNTRIES,
    FEMALE_NAMES,
    MALE_NAMES,
    OCCUPATIONS,
    find_sensitive_tokens,
    self_check,
)


def test_title_detection() -> None:
    assert is_article_title_line(" = Valkyria Chronicles III = \n")
    assert is_article_title_line("= Hello World =")
    assert not is_article_title_line(" = = Gameplay = = \n")
    assert not is_article_title_line(" = = = Reception = = = \n")
    assert not is_article_title_line("Ordinary paragraph text.\n")
    assert not is_article_title_line("")
    print("  title detection OK")


def test_parse_two_articles() -> None:
    lines = [
        " = Article One = \n",
        "Body of one.\n",
        "More body.\n",
        " = = Section = = \n",
        "Section text.\n",
        " = Article Two = \n",
        "Body of two.\n",
    ]
    arts = parse_articles(lines)
    assert len(arts) == 2, arts
    assert arts[0]["title"] == "Article One"
    assert "Body of one" in arts[0]["text"]
    assert "Section text" in arts[0]["text"]
    assert arts[1]["title"] == "Article Two"
    print("  parse_articles OK")


def test_paper_split_sizes() -> None:
    arts = [{"title": f"t{i}", "text": f"x{i}", "n_lines": 1} for i in range(28_595)]
    split = paper_article_split(arts)
    assert len(split.train) == 28_475
    assert len(split.validation) == 60
    assert len(split.test) == 60
    assert split.train[0]["title"] == "t0"
    assert split.validation[0]["title"] == "t28475"
    assert split.test[0]["title"] == "t28535"
    print("  paper split sizes OK")


def test_sensitive_lists_and_detection() -> None:
    self_check()
    assert len(COUNTRIES) == 10
    assert len(OCCUPATIONS) == 29
    assert len(MALE_NAMES) == 17
    assert len(FEMALE_NAMES) == 17
    # CEO must match
    hits = find_sensitive_tokens("She became CEO last year in Italy.")
    assert {h.canonical for h in hits} == {"CEO", "Italy"}
    print("  sensitive detection OK")


def main() -> int:
    print("Phase 2 unit tests")
    test_title_detection()
    test_parse_two_articles()
    test_paper_split_sizes()
    test_sensitive_lists_and_detection()
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
