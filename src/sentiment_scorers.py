"""
src/sentiment_scorers.py  —  Phases 6 / 7

Paper uses three sentiment classifiers (App. B / §5.3):
  (i)   Google Cloud sentiment API          → NOT used here (no paid API)
  (ii)  BERT fine-tuned on SST              → primary (D12)
  (iii) Opinion-word classifier Hu & Liu    → independent check (D12)

All scorers return a float in [0, 1] so Wasserstein-1 stays comparable
within a scorer (paper: different scorers have different scales — never
compare I.F. across scorers).
"""

from __future__ import annotations

import os
import re
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class SentimentScorer(ABC):
    name: str = "base"

    @abstractmethod
    def score_one(self, text: str) -> float:
        """Return sentiment in [0, 1]."""

    def score_many(self, texts: Sequence[str], batch_size: int = 32) -> list[float]:
        return [self.score_one(t) for t in texts]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# ---------------------------------------------------------------------------
# (iii) Opinion-word classifier — paper formula exactly
# ---------------------------------------------------------------------------
# score = p / (p + n);  if p+n == 0 → 0.5
# Lexicon: Hu & Liu (2004). We ship a compact built-in core so offline tests
# work, and download the full lists when internet is available.

_WORD_RE = re.compile(r"[a-zA-Z']+")

# Compact core (subset) — enough for unit tests and degraded mode.
_BUILTIN_POS = {
    "good", "great", "excellent", "amazing", "wonderful", "fantastic", "love",
    "loved", "happy", "joy", "joyful", "beautiful", "best", "better", "nice",
    "positive", "success", "successful", "win", "won", "perfect", "delight",
    "delighted", "pleased", "awesome", "brilliant", "superb", "outstanding",
    "remarkable", "impressive", "favorable", "fortunate", "glad", "proud",
    "charming", "enjoy", "enjoyed", "enjoyable", "kind", "friendly", "helpful",
    "hope", "hopeful", "inspire", "inspired", "peaceful", "safe", "strong",
    "talented", "smart", "wise", "honest", "loyal", "generous", "brave",
}
_BUILTIN_NEG = {
    "bad", "terrible", "awful", "horrible", "hate", "hated", "sad", "angry",
    "poor", "worst", "worse", "negative", "fail", "failed", "failure", "ugly",
    "disgusting", "nasty", "cruel", "violent", "dangerous", "threat", "fear",
    "afraid", "scared", "depressed", "miserable", "unhappy", "disappointed",
    "disappointing", "boring", "dull", "stupid", "dumb", "idiot", "corrupt",
    "evil", "wicked", "harmful", "toxic", "weak", "broken", "pain", "painful",
    "suffer", "suffering", "death", "die", "killed", "war", "crime", "criminal",
    "fraud", "lie", "liar", "selfish", "greedy", "lazy", "rude", "hostile",
}


def _parse_liu_file(text: str) -> set[str]:
    words: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(";") or line.startswith("#"):
            continue
        words.add(line.lower())
    return words


def load_opinion_lexicon(cache_dir: str | Path | None = None) -> tuple[set[str], set[str], str]:
    """
    Return (positive, negative, source_label).

    Tries, in order:
      1. cache_dir/positive-words.txt + negative-words.txt
      2. download from known public mirrors of Hu & Liu 2004
      3. built-in compact core (degraded)
    """
    cache = Path(cache_dir) if cache_dir else Path(
        os.environ.get("SBP_LEXICON_DIR", str(Path.home() / ".cache" / "sbp_lexicon"))
    )
    cache.mkdir(parents=True, exist_ok=True)
    pos_path = cache / "positive-words.txt"
    neg_path = cache / "negative-words.txt"

    if pos_path.is_file() and neg_path.is_file():
        pos = _parse_liu_file(pos_path.read_text(encoding="utf-8", errors="ignore"))
        neg = _parse_liu_file(neg_path.read_text(encoding="utf-8", errors="ignore"))
        if len(pos) > 100 and len(neg) > 100:
            return pos, neg, f"cache:{cache}"

    mirrors = [
        (
            "https://raw.githubusercontent.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107/master/data/opinion-lexicon-English/positive-words.txt",
            "https://raw.githubusercontent.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107/master/data/opinion-lexicon-English/negative-words.txt",
        ),
        (
            "https://raw.githubusercontent.com/yooper/machine-learning/master/training/data/positive-words.txt",
            "https://raw.githubusercontent.com/yooper/machine-learning/master/training/data/negative-words.txt",
        ),
    ]
    for pos_url, neg_url in mirrors:
        try:
            with urllib.request.urlopen(pos_url, timeout=20) as r:
                pos_txt = r.read().decode("latin-1", errors="ignore")
            with urllib.request.urlopen(neg_url, timeout=20) as r:
                neg_txt = r.read().decode("latin-1", errors="ignore")
            pos = _parse_liu_file(pos_txt)
            neg = _parse_liu_file(neg_txt)
            if len(pos) > 100 and len(neg) > 100:
                pos_path.write_text(pos_txt, encoding="utf-8")
                neg_path.write_text(neg_txt, encoding="utf-8")
                return pos, neg, f"download:{pos_url}"
        except Exception:  # noqa: BLE001
            continue

    return set(_BUILTIN_POS), set(_BUILTIN_NEG), "builtin_compact"


class OpinionWordScorer(SentimentScorer):
    """
    Paper classifier (iii):
        p = # positive opinion words
        n = # negative opinion words
        score = p/(p+n) if p+n>0 else 0.5
    """

    name = "opinion_word"

    def __init__(self, cache_dir: str | Path | None = None):
        self.pos, self.neg, self.source = load_opinion_lexicon(cache_dir)

    def score_one(self, text: str) -> float:
        tokens = [t.lower() for t in _WORD_RE.findall(text or "")]
        p = sum(1 for t in tokens if t in self.pos)
        n = sum(1 for t in tokens if t in self.neg)
        if p + n == 0:
            return 0.5
        return float(p) / float(p + n)


# ---------------------------------------------------------------------------
# (ii) BERT SST sentiment — primary
# ---------------------------------------------------------------------------


class BertSSTScorer(SentimentScorer):
    """
    Primary scorer (maps to paper classifier ii).

    Default model: `distilbert-base-uncased-finetuned-sst-2-english`
    (SST-2 binary). We map P(positive) → score in [0, 1].

    Label as implementation of paper's BERT-SST idea; report our own
    accuracy later — never claim the paper's 92.7% number.
    """

    name = "bert_sst"

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        device: str | None = None,
        max_length: int = 128,
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_name = model_name
        self.max_length = max_length
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        # figure out which label index is positive
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.pos_index = 1
        for i, lab in id2label.items():
            if "pos" in lab:
                self.pos_index = i
                break

    def score_one(self, text: str) -> float:
        return self.score_many([text], batch_size=1)[0]

    def score_many(self, texts: Sequence[str], batch_size: int = 32) -> list[float]:
        import torch
        import torch.nn.functional as F

        if not texts:
            return []
        out: list[float] = []
        for i in range(0, len(texts), batch_size):
            chunk = list(texts[i:i + batch_size])
            # empty strings → neutral-ish 0.5 without calling the model
            clean = [t if (t and t.strip()) else " " for t in chunk]
            enc = self.tokenizer(
                clean,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits
                probs = F.softmax(logits, dim=-1)
                pos = probs[:, self.pos_index].detach().cpu().tolist()
            for t, s in zip(chunk, pos):
                out.append(0.5 if not (t and t.strip()) else float(s))
        return out


def get_scorer(name: str, **kwargs) -> SentimentScorer:
    name = name.lower().strip()
    if name in {"opinion", "opinion_word", "opinion-word", "liu"}:
        return OpinionWordScorer(**kwargs)
    if name in {"bert", "bert_sst", "sst", "primary"}:
        return BertSSTScorer(**kwargs)
    raise ValueError(f"unknown scorer {name!r}")


def available_scorers() -> list[str]:
    return ["bert_sst", "opinion_word"]
