"""
src/data_preprocessing.py  —  PHASE 2

WikiText-103 load → article parse → paper split → tokenize → sequences
→ sensitive-token flags → save artefacts.

ORIGINAL PAPER SETTING (App. B, p.76 / arXiv:1911.03064)
---------------------------------------------------------
  Corpus        : WikiText-103
  Articles      : 28,591 stated (we report the count we actually observe)
  Split         : 28,475 train / 60 validation / 60 test   (article-level)
  Seq length    : 512
  Tokenizer     : Transformer-XL vocab (paper)

STUDENT / COMPUTE-LIMITED SETTING  (decision D04, D21 — Option A)
-----------------------------------------------------------------
  Corpus        : WikiText-103          ← SAME (no substitution)
  Split         : 28,475 / 60 / 60      ← SAME paper split
  Tokenizer     : GPT-2 BPE (gpt2)      ← changed (pairs with GPT-2 LM)
  Seq length    : 256 default           ← reduced from 512 (configurable)
  WMT-19        : NOT loaded here       ← documented omission (see docs/)

Nothing about I.F. / G.F. / debiasing math is changed by the student knobs.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Paper App. B
PAPER_N_TRAIN = 28_475
PAPER_N_VAL = 60
PAPER_N_TEST = 60
PAPER_SEQ_LEN = 512
PAPER_ARTICLES_STATED = 28_591  # paper's own total; may disagree by a few

# Student defaults (overridable via CLI / config)
DEFAULT_TOKENIZER = "gpt2"
DEFAULT_SEQ_LEN = 256
DEFAULT_SEED = 42
HF_DATASET = "Salesforce/wikitext"
HF_CONFIG = "wikitext-103-raw-v1"


@dataclass
class Phase2Config:
    tokenizer_name: str = DEFAULT_TOKENIZER
    seq_len: int = DEFAULT_SEQ_LEN
    seed: int = DEFAULT_SEED
    n_train_articles: int = PAPER_N_TRAIN
    n_val_articles: int = PAPER_N_VAL
    n_test_articles: int = PAPER_N_TEST
    # If True, keep ALL train articles (paper). If False, take a prefix —
    # only for emergency smoke tests; must be labelled.
    full_train: bool = True
    max_train_articles: int | None = None  # e.g. 2000 for smoke test
    setting_label: str = "STUDENT/COMPUTE-LIMITED"

    def resolved_train_cap(self) -> int | None:
        if self.full_train and self.max_train_articles is None:
            return None
        return self.max_train_articles


# ---------------------------------------------------------------------------
# 1. Load raw WikiText-103 lines from Hugging Face
# ---------------------------------------------------------------------------

def load_wikitext_raw_lines(cache_dir: str | Path | None = None) -> list[str]:
    """
    Load train + validation + test of wikitext-103-raw-v1 and concatenate
    in that order so article order is stable and complete.
    """
    from datasets import load_dataset

    print(f"[phase2] loading {HF_DATASET} / {HF_CONFIG} ...")
    t0 = time.time()
    kwargs: dict[str, Any] = {}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)

    ds = load_dataset(HF_DATASET, HF_CONFIG, **kwargs)
    lines: list[str] = []
    for split in ("train", "validation", "test"):
        split_lines = ds[split]["text"]
        print(f"  {split:12s}  {len(split_lines):>10,d} lines")
        lines.extend(split_lines)
    print(f"[phase2] total lines={len(lines):,}  ({time.time()-t0:.1f}s)")
    return lines


# ---------------------------------------------------------------------------
# 2. Parse articles
# ---------------------------------------------------------------------------

# WikiText raw article title: " = Title = "
# Section headers look like " = = Section = = " (inner still starts with =)
_TITLE_LINE = re.compile(r"^\s*=\s*[^=].*?=\s*$")


def is_article_title_line(line: str) -> bool:
    s = line.strip()
    if not s.startswith("=") or not s.endswith("="):
        return False
    inner = s[1:-1].strip()
    # section / subsection headers still begin with '=' after stripping outer
    return not inner.startswith("=") and len(inner) > 0


def parse_articles(lines: list[str]) -> list[dict[str, Any]]:
    """
    Split the flat WikiText line stream into articles.

    Each article:
      {
        "title": str,
        "text":  str,   # full raw text including title line
        "n_lines": int,
      }
    """
    articles: list[dict[str, Any]] = []
    cur_title: str | None = None
    cur_buf: list[str] = []

    def flush() -> None:
        nonlocal cur_title, cur_buf
        if cur_title is None and not cur_buf:
            return
        text = "".join(cur_buf)
        # skip empty shells
        body = text.strip()
        if body:
            articles.append(
                {
                    "title": cur_title or "(untitled)",
                    "text": text,
                    "n_lines": len(cur_buf),
                }
            )
        cur_title, cur_buf = None, []

    for line in lines:
        if is_article_title_line(line):
            flush()
            cur_title = line.strip()[1:-1].strip()
            cur_buf = [line if line.endswith("\n") else line + "\n"]
        else:
            if cur_title is None and not cur_buf:
                # preamble before first title — keep only if non-empty later
                if line.strip():
                    cur_buf.append(line if line.endswith("\n") else line + "\n")
            else:
                cur_buf.append(line if line.endswith("\n") else line + "\n")
    flush()
    return articles


# ---------------------------------------------------------------------------
# 3. Paper article-level split
# ---------------------------------------------------------------------------

@dataclass
class ArticleSplit:
    train: list[dict[str, Any]]
    validation: list[dict[str, Any]]
    test: list[dict[str, Any]]
    n_total_observed: int
    paper_n_train: int = PAPER_N_TRAIN
    paper_n_val: int = PAPER_N_VAL
    paper_n_test: int = PAPER_N_TEST
    notes: list[str] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
            "total_observed": self.n_total_observed,
        }


def paper_article_split(
    articles: list[dict[str, Any]],
    n_train: int = PAPER_N_TRAIN,
    n_val: int = PAPER_N_VAL,
    n_test: int = PAPER_N_TEST,
    max_train_articles: int | None = None,
) -> ArticleSplit:
    """
    ORIGINAL PAPER SETTING:
        first 28,475 → train, next 60 → val, next 60 → test
        (articles in file order — no shuffle)

    If the corpus has fewer articles than 28,475+60+60 we take what exists
    and record a note (never silently invent articles).
    """
    notes: list[str] = []
    n = len(articles)
    need = n_train + n_val + n_test
    if n < need:
        notes.append(
            f"WARNING: observed {n} articles < paper need {need}. "
            f"Using available articles; NOT an exact paper reproduction."
        )
        # carve what we can, prefer keeping val/test at paper size if possible
        n_test_use = min(n_test, max(0, n // 50))
        n_val_use = min(n_val, max(0, (n - n_test_use) // 50))
        n_train_use = n - n_val_use - n_test_use
    else:
        n_train_use, n_val_use, n_test_use = n_train, n_val, n_test
        if n != PAPER_ARTICLES_STATED:
            notes.append(
                f"Observed {n} articles; paper states {PAPER_ARTICLES_STATED}. "
                f"We report the observed count (paper has a known off-by-a-few)."
            )

    train = articles[:n_train_use]
    val = articles[n_train_use:n_train_use + n_val_use]
    test = articles[n_train_use + n_val_use:n_train_use + n_val_use + n_test_use]

    if max_train_articles is not None and len(train) > max_train_articles:
        notes.append(
            f"STUDENT SMOKE CAP: train truncated {len(train)} → "
            f"{max_train_articles} articles. Label results as smoke-test only."
        )
        train = train[:max_train_articles]

    return ArticleSplit(
        train=train,
        validation=val,
        test=test,
        n_total_observed=n,
        paper_n_train=n_train,
        paper_n_val=n_val,
        paper_n_test=n_test,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# 4. Tokenization + sequence packing
# ---------------------------------------------------------------------------

def get_tokenizer(name: str = DEFAULT_TOKENIZER):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    # GPT-2 has no pad by default; for packing we don't pad inside sequences,
    # but set eos as pad so collate can pad batches later (Phase 5).
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def article_to_token_ids(text: str, tokenizer) -> list[int]:
    # add_special_tokens=False: we manage eos ourselves between articles
    return tokenizer.encode(text, add_special_tokens=False)


def pack_sequences(
    articles: list[dict[str, Any]],
    tokenizer,
    seq_len: int,
    eos_id: int,
) -> list[list[int]]:
    """
    Concatenate tokenized articles with EOS between them, then cut into
    non-overlapping blocks of exactly `seq_len` token ids.

    Remainder shorter than seq_len is dropped (standard LM packing).
    """
    stream: list[int] = []
    for art in articles:
        ids = article_to_token_ids(art["text"], tokenizer)
        if not ids:
            continue
        stream.extend(ids)
        stream.append(eos_id)

    sequences: list[list[int]] = []
    for i in range(0, len(stream) - seq_len + 1, seq_len):
        sequences.append(stream[i:i + seq_len])
    return sequences


# ---------------------------------------------------------------------------
# 5. Sensitive-token detection on sequences
# ---------------------------------------------------------------------------

def flag_sensitive_sequences(
    sequences: list[list[int]],
    tokenizer,
) -> list[dict[str, Any]]:
    """
    For each packed sequence, decode and run the paper sensitive-token list.
    Returns a list of records parallel to `sequences`.
    """
    from src.sensitive_attributes import find_sensitive_tokens, summarize_hits

    records: list[dict[str, Any]] = []
    for idx, ids in enumerate(sequences):
        text = tokenizer.decode(ids, skip_special_tokens=False)
        hits = find_sensitive_tokens(text)
        summary = summarize_hits(hits)
        records.append(
            {
                "seq_index": idx,
                "has_sensitive": summary["total"] > 0,
                "n_hits": summary["total"],
                "by_attribute": {
                    "country": summary["country"],
                    "occupation": summary["occupation"],
                    "name": summary["name"],
                },
                "canonical_tokens": sorted({h.canonical for h in hits}),
            }
        )
    return records


# ---------------------------------------------------------------------------
# 6. Sanity report
# ---------------------------------------------------------------------------

def _char_stats(articles: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [len(a["text"]) for a in articles]
    if not lengths:
        return {"n": 0}
    return {
        "n": len(lengths),
        "chars_total": int(sum(lengths)),
        "chars_mean": round(sum(lengths) / len(lengths), 1),
        "chars_min": int(min(lengths)),
        "chars_max": int(max(lengths)),
    }


def build_sanity_report(
    split: ArticleSplit,
    tok_info: dict[str, Any],
    seq_counts: dict[str, int],
    sensitive_stats: dict[str, Any],
    cfg: Phase2Config,
    examples: dict[str, Any],
) -> dict[str, Any]:
    return {
        "setting": {
            "label": cfg.setting_label,
            "original_paper": {
                "dataset": "WikiText-103",
                "split_articles": {
                    "train": PAPER_N_TRAIN,
                    "validation": PAPER_N_VAL,
                    "test": PAPER_N_TEST,
                },
                "seq_len": PAPER_SEQ_LEN,
                "model": "Transformer-XL 257M from scratch",
            },
            "student": {
                "dataset": "WikiText-103 (SAME)",
                "split_articles": split.counts(),
                "seq_len": cfg.seq_len,
                "tokenizer": cfg.tokenizer_name,
                "model": "GPT-2 small fine-tune (Phase 5)",
                "changes": [
                    "tokenizer: Transformer-XL vocab → GPT-2 BPE",
                    f"seq_len: {PAPER_SEQ_LEN} → {cfg.seq_len}",
                    "WMT-19 track omitted at this stage",
                ],
                "unchanged": [
                    "WikiText-103 corpus",
                    "article-level 28475/60/60 split rule",
                    "sensitive-token lists (Appendix A)",
                    "no evaluation templates in training data",
                ],
            },
        },
        "article_counts": split.counts(),
        "article_char_stats": {
            "train": _char_stats(split.train),
            "validation": _char_stats(split.validation),
            "test": _char_stats(split.test),
        },
        "tokenizer": tok_info,
        "sequence_counts": seq_counts,
        "sensitive_token_stats": sensitive_stats,
        "notes": split.notes,
        "examples": examples,
    }


# ---------------------------------------------------------------------------
# 7. Save artefacts
# ---------------------------------------------------------------------------

def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    print(f"  wrote {path}  ({path.stat().st_size/1024:.1f} KB)")


def save_sequences_jsonl(
    sequences: list[list[int]],
    flags: list[dict[str, Any]],
    path: Path,
) -> None:
    """
    One JSON object per line:
      {"input_ids": [...], "has_sensitive": bool, "n_hits": int, ...}
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ids, fl in zip(sequences, flags):
            row = {"input_ids": ids, **{k: v for k, v in fl.items() if k != "seq_index"}}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  wrote {path}  ({path.stat().st_size/1024**2:.2f} MB, {len(sequences):,} seqs)")


def save_articles_preview(articles: list[dict[str, Any]], path: Path, n: int = 5) -> None:
    preview = [
        {"title": a["title"], "n_chars": len(a["text"]), "text_head": a["text"][:400]}
        for a in articles[:n]
    ]
    save_json(preview, path)


# ---------------------------------------------------------------------------
# 8. Main pipeline
# ---------------------------------------------------------------------------

def run_phase2(
    out_dir: str | Path,
    cfg: Phase2Config | None = None,
    hf_cache: str | Path | None = None,
) -> dict[str, Any]:
    from src.sensitive_attributes import (
        COUNTRIES,
        FEMALE_NAMES,
        MALE_NAMES,
        OCCUPATIONS,
        self_check,
    )

    cfg = cfg or Phase2Config()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    proc_dir = out_dir / "processed"
    raw_dir.mkdir(exist_ok=True)
    proc_dir.mkdir(exist_ok=True)

    print("=" * 72)
    print(" PHASE 2 — Dataset (WikiText-103)")
    print("=" * 72)
    print(f" setting     : {cfg.setting_label}")
    print(f" out_dir     : {out_dir}")
    print(f" seq_len     : {cfg.seq_len}  (paper={PAPER_SEQ_LEN})")
    print(f" tokenizer   : {cfg.tokenizer_name}")
    print()

    # --- sensitive list self-check (Phase 3 lists, needed for detection) ---
    self_check()

    # --- load ---
    lines = load_wikitext_raw_lines(cache_dir=hf_cache)
    save_json(
        {
            "n_lines": len(lines),
            "hf_dataset": HF_DATASET,
            "hf_config": HF_CONFIG,
            "source": "HuggingFace datasets",
        },
        raw_dir / "load_meta.json",
    )

    # --- parse articles ---
    print("[phase2] parsing articles ...")
    t0 = time.time()
    articles = parse_articles(lines)
    print(f"  observed articles: {len(articles):,}  ({time.time()-t0:.1f}s)")
    if articles:
        print(f"  first title : {articles[0]['title']!r}")
        print(f"  last title  : {articles[-1]['title']!r}")

    # --- paper split ---
    split = paper_article_split(
        articles,
        n_train=cfg.n_train_articles,
        n_val=cfg.n_val_articles,
        n_test=cfg.n_test_articles,
        max_train_articles=cfg.resolved_train_cap(),
    )
    print("[phase2] paper-style article split:")
    for k, v in split.counts().items():
        print(f"  {k:16s} {v:,}")
    for note in split.notes:
        print(f"  NOTE: {note}")

    # save article texts (titles + full text) per split — needed later
    for name, arts in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        path = proc_dir / f"articles_{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for i, a in enumerate(arts):
                f.write(
                    json.dumps(
                        {"index": i, "title": a["title"], "text": a["text"]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"  wrote {path}  ({path.stat().st_size/1024**2:.2f} MB)")

    save_articles_preview(split.train, proc_dir / "preview_train_articles.json", n=5)
    save_articles_preview(split.validation, proc_dir / "preview_val_articles.json", n=3)
    save_articles_preview(split.test, proc_dir / "preview_test_articles.json", n=3)

    # --- tokenize + pack ---
    print(f"[phase2] loading tokenizer {cfg.tokenizer_name!r} ...")
    tokenizer = get_tokenizer(cfg.tokenizer_name)
    eos_id = tokenizer.eos_token_id
    tok_info = {
        "name": cfg.tokenizer_name,
        "vocab_size": int(tokenizer.vocab_size),
        "model_max_length": int(getattr(tokenizer, "model_max_length", -1)),
        "eos_token": tokenizer.eos_token,
        "eos_token_id": int(eos_id),
        "pad_token": tokenizer.pad_token,
        "pad_token_id": int(tokenizer.pad_token_id)
        if tokenizer.pad_token_id is not None
        else None,
    }
    print(f"  vocab_size={tok_info['vocab_size']}  eos_id={eos_id}")

    # persist tokenizer name only (weights come from HF hub / cache)
    save_json(tok_info, proc_dir / "tokenizer_info.json")

    seq_counts: dict[str, int] = {}
    all_flags: dict[str, list[dict[str, Any]]] = {}
    sensitive_stats: dict[str, Any] = {}

    for name, arts in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        print(f"[phase2] packing sequences: {name} ...")
        t0 = time.time()
        seqs = pack_sequences(arts, tokenizer, cfg.seq_len, eos_id)
        print(f"  {len(seqs):,} sequences of len={cfg.seq_len}  ({time.time()-t0:.1f}s)")
        seq_counts[name] = len(seqs)

        print(f"[phase2] sensitive-token scan: {name} ...")
        t0 = time.time()
        flags = flag_sensitive_sequences(seqs, tokenizer)
        all_flags[name] = flags
        n_sens = sum(1 for fl in flags if fl["has_sensitive"])
        pct = 100.0 * n_sens / max(len(flags), 1)
        by_attr = {"country": 0, "occupation": 0, "name": 0}
        for fl in flags:
            for a in by_attr:
                by_attr[a] += fl["by_attribute"][a]
        sensitive_stats[name] = {
            "n_sequences": len(flags),
            "n_with_sensitive": n_sens,
            "pct_with_sensitive": round(pct, 2),
            "hit_counts_by_attribute": by_attr,
        }
        print(
            f"  sensitive sequences: {n_sens:,}/{len(flags):,} "
            f"({pct:.2f}%)  hits={by_attr}  ({time.time()-t0:.1f}s)"
        )

        save_sequences_jsonl(seqs, flags, proc_dir / f"sequences_{name}.jsonl")

        # also a compact index of sensitive-only sequences (Phase 9/10/11)
        sens_idx = [fl["seq_index"] for fl in flags if fl["has_sensitive"]]
        save_json(
            {"n": len(sens_idx), "indices": sens_idx},
            proc_dir / f"sensitive_indices_{name}.json",
        )

    # --- examples for the report ---
    examples = {
        "train_article_title_0": split.train[0]["title"] if split.train else None,
        "train_article_head_0": split.train[0]["text"][:300] if split.train else None,
        "sensitive_token_counts_paper": {
            "countries": len(COUNTRIES),
            "occupations": len(OCCUPATIONS),
            "male_names": len(MALE_NAMES),
            "female_names": len(FEMALE_NAMES),
        },
        "sample_sensitive_sequence": None,
    }
    # pick one sensitive train sequence if any
    train_flags = all_flags.get("train", [])
    for fl in train_flags:
        if fl["has_sensitive"]:
            # reload would be heavy; store meta only
            examples["sample_sensitive_sequence"] = {
                "seq_index": fl["seq_index"],
                "canonical_tokens": fl["canonical_tokens"],
                "by_attribute": fl["by_attribute"],
            }
            break

    report = build_sanity_report(
        split=split,
        tok_info=tok_info,
        seq_counts=seq_counts,
        sensitive_stats=sensitive_stats,
        cfg=cfg,
        examples=examples,
    )
    save_json(report, out_dir / "phase2_sanity_report.json")
    save_json(asdict(cfg), out_dir / "phase2_config.json")

    # human-readable summary
    summary_txt = out_dir / "phase2_summary.txt"
    lines_out = [
        "PHASE 2 SUMMARY — WikiText-103",
        "=" * 60,
        f"Setting          : {cfg.setting_label}",
        f"Articles total   : {split.n_total_observed:,} observed "
        f"(paper states {PAPER_ARTICLES_STATED:,})",
        f"Split (articles) : train={len(split.train):,}  "
        f"val={len(split.validation):,}  test={len(split.test):,}",
        f"Paper target     : train={PAPER_N_TRAIN:,}  val={PAPER_N_VAL}  test={PAPER_N_TEST}",
        f"Tokenizer        : {cfg.tokenizer_name}  vocab={tok_info['vocab_size']}",
        f"Seq len          : {cfg.seq_len}  (paper {PAPER_SEQ_LEN})",
        f"Sequences        : " + ", ".join(f"{k}={v:,}" for k, v in seq_counts.items()),
        "",
        "Sensitive-token coverage (packed sequences):",
    ]
    for name, st in sensitive_stats.items():
        lines_out.append(
            f"  {name:12s}  {st['n_with_sensitive']:,}/{st['n_sequences']:,} "
            f"({st['pct_with_sensitive']}%)  hits={st['hit_counts_by_attribute']}"
        )
    lines_out.append("")
    lines_out.append("Notes:")
    if split.notes:
        lines_out.extend(f"  - {n}" for n in split.notes)
    else:
        lines_out.append("  - none")
    lines_out.append("")
    lines_out.append("ORIGINAL PAPER SETTING vs STUDENT SETTING: see phase2_sanity_report.json")
    summary_txt.write_text("\n".join(lines_out) + "\n")
    print()
    print(summary_txt.read_text())
    print(f"[phase2] artefacts under {out_dir}")
    print("[phase2] DONE")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 2 — WikiText-103 preprocessing")
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="default: <project>/data/wikitext103",
    )
    p.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    p.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--max-train-articles",
        type=int,
        default=None,
        help="SMOKE TEST only: cap train articles (e.g. 500). Full run: omit.",
    )
    p.add_argument(
        "--hf-cache",
        type=str,
        default=None,
        help="HuggingFace cache dir (optional)",
    )
    return p


def main() -> int:
    args = build_argparser().parse_args()

    # resolve project data dir via paths if available
    if args.out_dir is None:
        try:
            from src import paths

            paths.ensure_dirs()
            out_dir = paths.data_path("wikitext103")
        except Exception:
            out_dir = Path("data/wikitext103")
    else:
        out_dir = Path(args.out_dir)

    cfg = Phase2Config(
        tokenizer_name=args.tokenizer,
        seq_len=args.seq_len,
        seed=args.seed,
        full_train=args.max_train_articles is None,
        max_train_articles=args.max_train_articles,
        setting_label=(
            "STUDENT/COMPUTE-LIMITED (full WikiText-103, paper article split)"
            if args.max_train_articles is None
            else f"SMOKE TEST (max_train_articles={args.max_train_articles})"
        ),
    )
    random.seed(cfg.seed)
    run_phase2(out_dir=out_dir, cfg=cfg, hf_cache=args.hf_cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
