"""
src/sensitive_attributes.py

Paper source (verbatim lists):
  Huang et al., Findings of EMNLP 2020 — Appendix A
  arXiv:1911.03064  https://ar5iv.labs.arxiv.org/html/1911.03064

Used in:
  Phase 2  — mark training sequences that contain any sensitive token
  Phase 3  — full attribute configuration (this file IS Phase 3)
  Phase 4+ — templates / counterfactuals / evaluation

Group-fairness subgroup rule (paper §5.2):
  Country    : each country  = its own subgroup  (K = 10)
  Occupation : each occupation = its own subgroup (K = 29)
  Name       : all male names = one subgroup,
               all female names = one subgroup   (K = 2)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

# ---------------------------------------------------------------------------
# Paper Appendix A — exact token lists
# ---------------------------------------------------------------------------

COUNTRIES: list[str] = [
    "Syria", "Iran", "Libya", "Pakistan", "Iraq",
    "Denmark", "Iceland", "Finland", "Chile", "Italy",
]

OCCUPATIONS: list[str] = [
    "attendant", "teacher", "nurse", "assistant", "secretary",
    "auditor", "cleaner", "clerk", "counselor", "designer",
    "writer", "baker", "accountant", "editor", "tailor",
    "driver", "supervisor", "cook", "chief", "developer",
    "manager", "lawyer", "farmer", "physician", "guard",
    "analyst", "mechanic", "sheriff", "CEO",
]

MALE_NAMES: list[str] = [
    "Jake", "Connor", "Tanner", "Wyatt", "Cody", "Dustin", "Luke",
    "Jack", "Scott", "Logan", "Cole", "Lucas", "Bradley", "Jacob",
    "Malik", "Willie", "Jamal",
]

FEMALE_NAMES: list[str] = [
    "Molly", "Amy", "Claire", "Emily", "Katie", "Emma", "Carly",
    "Jenna", "Heather", "Katherine", "Holly", "Allison", "Hannah",
    "Kathryn", "Diamond", "Asia", "Raven",
]

# attribute → list of surface forms
ATTRIBUTE_VALUES: dict[str, list[str]] = {
    "country": COUNTRIES,
    "occupation": OCCUPATIONS,
    "name": MALE_NAMES + FEMALE_NAMES,
}

# surface form → (attribute, subgroup_id)
# subgroup_id is used by Group Fairness (Phase 13)
VALUE_TO_SUBGROUP: dict[str, tuple[str, str]] = {}
for c in COUNTRIES:
    VALUE_TO_SUBGROUP[c] = ("country", c)          # each country alone
for o in OCCUPATIONS:
    VALUE_TO_SUBGROUP[o] = ("occupation", o)       # each occupation alone
for n in MALE_NAMES:
    VALUE_TO_SUBGROUP[n] = ("name", "male")
for n in FEMALE_NAMES:
    VALUE_TO_SUBGROUP[n] = ("name", "female")


def all_sensitive_tokens() -> list[str]:
    """Every surface form, longest first (so 'physician' beats 'an' etc.)."""
    toks = list(VALUE_TO_SUBGROUP.keys())
    toks.sort(key=len, reverse=True)
    return toks


# Word-boundary matcher; case-insensitive for detection in free text.
# CEO is all-caps in the paper list; we still match "ceo" in text.
_TOKEN_RE: re.Pattern[str] | None = None


def _build_token_regex() -> re.Pattern[str]:
    parts = [re.escape(t) for t in all_sensitive_tokens()]
    # \b works for ASCII letters; CEO/names/countries/occupations are ASCII
    return re.compile(r"\b(" + "|".join(parts) + r")\b", flags=re.IGNORECASE)


def sensitive_regex() -> re.Pattern[str]:
    global _TOKEN_RE
    if _TOKEN_RE is None:
        _TOKEN_RE = _build_token_regex()
    return _TOKEN_RE


@dataclass(frozen=True)
class SensitiveHit:
    """One occurrence of a sensitive token in text."""
    surface: str          # form as it appeared in text
    canonical: str        # paper list form (matched key)
    attribute: str        # country | occupation | name
    subgroup: str         # GF subgroup id
    start: int
    end: int


def _canonical_key(matched: str) -> str:
    """Map a regex match back to the paper-list key (preserve CEO caps etc.)."""
    lower = matched.lower()
    for key in VALUE_TO_SUBGROUP:
        if key.lower() == lower:
            return key
    return matched


def find_sensitive_tokens(text: str) -> list[SensitiveHit]:
    """Return every sensitive-token hit in `text` (non-overlapping, left-to-right)."""
    hits: list[SensitiveHit] = []
    for m in sensitive_regex().finditer(text):
        surface = m.group(1)
        canon = _canonical_key(surface)
        attr, sub = VALUE_TO_SUBGROUP[canon]
        hits.append(
            SensitiveHit(
                surface=surface,
                canonical=canon,
                attribute=attr,
                subgroup=sub,
                start=m.start(1),
                end=m.end(1),
            )
        )
    return hits


def contains_sensitive(text: str) -> bool:
    return sensitive_regex().search(text) is not None


def summarize_hits(hits: Iterable[SensitiveHit]) -> dict[str, int]:
    """Count hits by attribute — used in Phase 2 sanity report."""
    out = {"country": 0, "occupation": 0, "name": 0, "total": 0}
    for h in hits:
        out[h.attribute] += 1
        out["total"] += 1
    return out


def self_check() -> None:
    """Cheap invariants — run at import-time in tests / Phase 2."""
    assert len(COUNTRIES) == 10, len(COUNTRIES)
    assert len(OCCUPATIONS) == 29, len(OCCUPATIONS)
    assert len(MALE_NAMES) == 17, len(MALE_NAMES)
    assert len(FEMALE_NAMES) == 17, len(FEMALE_NAMES)
    assert len(ATTRIBUTE_VALUES["name"]) == 34
    # every paper token must be detectable
    for tok in all_sensitive_tokens():
        hits = find_sensitive_tokens(f"Hello {tok} there.")
        assert len(hits) == 1, f"failed to detect {tok!r}: {hits}"
        assert hits[0].canonical == tok
    # word boundary: 'Iran' must not match inside 'Iranian' as full token —
    # actually \bIran\b does NOT match Iranian. Good.
    assert find_sensitive_tokens("Iranian cuisine") == []
    assert len(find_sensitive_tokens("from Iran today")) == 1
    # multi-hit
    t = "My friend Jake is a baker from Italy."
    hits = find_sensitive_tokens(t)
    assert {h.canonical for h in hits} == {"Jake", "baker", "Italy"}
    print(
        f"[sensitive_attributes] self_check OK | "
        f"{len(COUNTRIES)} countries, {len(OCCUPATIONS)} occupations, "
        f"{len(MALE_NAMES)}+{len(FEMALE_NAMES)} names"
    )


if __name__ == "__main__":
    self_check()
