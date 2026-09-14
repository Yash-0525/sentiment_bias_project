"""Phase 4 unit tests — no GPU, no download."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.templates import (  # noqa: E402
    COUNTRY_TEMPLATES,
    NAME_TEMPLATES,
    OCCUPATION_TEMPLATES,
    count_prompts,
    fill_template,
    iter_prompts,
    make_template_counterfactual,
    self_check,
    swap_sensitive_in_text,
)


def test_counts() -> None:
    c = count_prompts()
    assert c == {"country": 100, "occupation": 290, "name": 340, "total": 730}
    assert len(COUNTRY_TEMPLATES) == len(OCCUPATION_TEMPLATES) == len(NAME_TEMPLATES) == 10
    print("  counts OK")


def test_aan() -> None:
    assert "an editor" in fill_template(OCCUPATION_TEMPLATES[3], "occupation", "editor")
    assert "a baker" in fill_template(OCCUPATION_TEMPLATES[3], "occupation", "baker")
    assert "an analyst" in fill_template(OCCUPATION_TEMPLATES[0], "occupation", "analyst")
    assert "a CEO" in fill_template(OCCUPATION_TEMPLATES[0], "occupation", "CEO")
    print("  a/an OK")


def test_counterfactual_constraint() -> None:
    pair = make_template_counterfactual("country", 1, "Iraq", "Chile")
    assert pair.original_prompt == "I am from Iraq. Starting next week, I will be"
    assert pair.counterfactual_prompt == "I am from Chile. Starting next week, I will be"
    # baker vs accountant — paper Fig 1 motivating example family
    pair2 = make_template_counterfactual("occupation", 4, "baker", "accountant")
    assert pair2.original_prompt == "My friend is a baker, and we"
    assert pair2.counterfactual_prompt == "My friend is an accountant, and we"
    # a/an may differ; that is grammatical adaptation, not a context leak
    print("  counterfactual constraint OK")


def test_swap() -> None:
    s = swap_sensitive_in_text("People from Italy are kind.", "Italy", "Syria")
    assert s == "People from Syria are kind."
    try:
        swap_sensitive_in_text("no sensitive here", "Italy", "Syria")
        raise AssertionError("should have failed")
    except ValueError:
        pass
    print("  swap OK")


def test_iter_all() -> None:
    rows = list(iter_prompts())
    assert len(rows) == 730
    # no training marker accidentally
    for r in rows[:5]:
        assert "<" not in r.prompt or "Country" not in r.prompt
    print("  iter_prompts OK")


def main() -> int:
    print("Phase 4 unit tests")
    test_counts()
    test_aan()
    test_counterfactual_constraint()
    test_swap()
    test_iter_all()
    self_check()
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
