"""
src/sentiment_classifier.py  —  PHASE 8
Sentiment projection / classifier f_sh  (paper curriculum Step 2)

Paper (App. B):
  - Freeze the language model
  - Label sentences with Google Cloud sentiment API in [-1, +1]
  - Keep |score| > 0.7; drop neutral
  - Train 3-layer MLP, hidden size 128, pos-vs-neg only
  - WikiText-103: 369,594 sentences; accuracy 98.8%

STUDENT (D12 / D12b):
  - Labels from BERT-SST primary scorer (NOT Google API) — labelled substitution
  - Map p_pos in [0,1] → s = 2*p-1 in [-1,1]; keep |s| > 0.7
    i.e. p_pos > 0.85 (positive) or p_pos < 0.15 (negative)
  - Sample a smaller sentence set (default 30k candidates → keep strong ones)
  - Same MLP: 3 layers, hidden 128
  - Input = h_bar = mean of last two LM layers, mean-pooled over tokens
  - Expose project() → 128-d vector for Phase 11 Sentiment Regularization
  - Freeze f_sh during Phase 11

Opinion-word scorer remains the independent held-out measure (D12b).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm.auto import tqdm

from src import paths
from src.model import SentimentBiasLM
from src.sentiment_scorers import BertSSTScorer

# ---------------------------------------------------------------------------
# Architecture — paper: 3-layer MLP, hidden 128
# ---------------------------------------------------------------------------


class SentimentProjection(nn.Module):
    """
    f_sh: h_bar (d,) → logits (2,)

    Layers:
      Linear(d, 128) → ReLU
      Linear(128, 128) → ReLU     ← this 128-d vector is the "sentiment subspace"
      Linear(128, 2)              ← pos / neg logits

    Phase 11 uses project(h) (penultimate 128-d), NOT the logits, for cosine.
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 128, n_classes: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_classes)

    def project(self, h: torch.Tensor) -> torch.Tensor:
        """Map hidden states to the 128-d sentiment subspace (pre-logits)."""
        x = F.relu(self.fc1(h))
        x = F.relu(self.fc2(x))
        return x

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Return class logits [..., 2]."""
        return self.fc3(self.project(h))

    def predict_proba(self, h: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(h), dim=-1)


# ---------------------------------------------------------------------------
# Sentence extraction from Phase 2 article jsonl
# ---------------------------------------------------------------------------

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def iter_sentences_from_articles(
    articles_jsonl: Path,
    *,
    min_words: int = 5,
    max_words: int = 60,
    max_sentences: int | None = None,
) -> list[str]:
    sents: list[str] = []
    with articles_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if max_sentences is not None and len(sents) >= max_sentences:
                break
            obj = json.loads(line)
            text = obj.get("text", "")
            # drop wiki title markup-ish short lines
            for raw in _SENT_SPLIT.split(text.replace("\n", " ")):
                s = " ".join(raw.split())
                if not s:
                    continue
                # skip pure headings like "= Foo ="
                if s.startswith("=") and s.endswith("="):
                    continue
                n_words = len(s.split())
                if n_words < min_words or n_words > max_words:
                    continue
                sents.append(s)
                if max_sentences is not None and len(sents) >= max_sentences:
                    break
    return sents


# ---------------------------------------------------------------------------
# Label with BERT-SST, keep |s| > 0.7
# ---------------------------------------------------------------------------

@dataclass
class LabeledSentence:
    text: str
    p_pos: float
    score_pm1: float  # 2p-1 in [-1,1]
    label: int        # 1 positive, 0 negative


def label_and_filter(
    sentences: list[str],
    scorer: BertSSTScorer,
    *,
    abs_threshold: float = 0.7,
    batch_size: int = 64,
) -> list[LabeledSentence]:
    """
    Paper: |Google score| > 0.7 on [-1,1].
    Us:    |2*p_pos - 1| > abs_threshold.
    """
    kept: list[LabeledSentence] = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="label"):
        chunk = sentences[i:i + batch_size]
        probs = scorer.score_many(chunk, batch_size=batch_size)
        for text, p in zip(chunk, probs):
            s = 2.0 * float(p) - 1.0
            if abs(s) <= abs_threshold:
                continue
            label = 1 if s > 0 else 0
            kept.append(LabeledSentence(text=text, p_pos=float(p), score_pm1=s, label=label))
    return kept


# ---------------------------------------------------------------------------
# Extract h_bar from frozen LM
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_h_bar_batch(
    model: SentimentBiasLM,
    tokenizer,
    texts: list[str],
    *,
    device: torch.device,
    max_length: int = 128,
) -> torch.Tensor:
    """
    Returns [B, d] mean-pooled h_bar over non-pad tokens.
    LM must be frozen / eval.
    """
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    h = out.h_bar  # [B, T, d]
    assert h is not None
    mask = attention_mask.unsqueeze(-1).to(h.dtype)  # [B, T, 1]
    summed = (h * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom


@torch.no_grad()
def build_feature_matrix(
    model: SentimentBiasLM,
    tokenizer,
    labeled: list[LabeledSentence],
    *,
    device: torch.device,
    batch_size: int = 16,
    max_length: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    xs: list[torch.Tensor] = []
    ys: list[int] = []
    for i in tqdm(range(0, len(labeled), batch_size), desc="extract h_bar"):
        chunk = labeled[i:i + batch_size]
        texts = [c.text for c in chunk]
        h = extract_h_bar_batch(
            model, tokenizer, texts, device=device, max_length=max_length
        )
        xs.append(h.cpu())
        ys.extend(c.label for c in chunk)
    X = torch.cat(xs, dim=0)
    y = torch.tensor(ys, dtype=torch.long)
    return X, y


# ---------------------------------------------------------------------------
# Train / eval MLP
# ---------------------------------------------------------------------------

def train_mlp(
    model: SentimentProjection,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    device: torch.device,
    epochs: int = 5,
    lr: float = 1e-3,
) -> dict[str, Any]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        train_loss = total_loss / max(n, 1)
        val_metrics = evaluate_mlp(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(row)
        print(
            f"  epoch {epoch}: train_loss={train_loss:.4f}  "
            f"val_acc={val_metrics['accuracy']:.4f}  "
            f"f1={val_metrics['f1']:.4f}"
        )
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"history": history, "best_val_acc": best_val_acc}


@torch.no_grad()
def evaluate_mlp(
    model: SentimentProjection,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    ys, preds = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        pred = logits.argmax(dim=-1).cpu().tolist()
        ys.extend(yb.tolist())
        preds.extend(pred)
    return classification_report_basic(ys, preds)


def classification_report_basic(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    acc = float((y_true == y_pred).mean()) if n else 0.0

    def prf(cls: int) -> tuple[float, float, float]:
        tp = int(((y_pred == cls) & (y_true == cls)).sum())
        fp = int(((y_pred == cls) & (y_true != cls)).sum())
        fn = int(((y_pred != cls) & (y_true == cls)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return prec, rec, f1

    p0, r0, f0 = prf(0)
    p1, r1, f1 = prf(1)
    # macro F1
    f_macro = 0.5 * (f0 + f1)
    # confusion [[tn, fp], [fn, tp]] for labels 0/1
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return {
        "accuracy": acc,
        "precision": 0.5 * (p0 + p1),
        "recall": 0.5 * (r0 + r1),
        "f1": f_macro,
        "precision_neg": p0,
        "precision_pos": p1,
        "recall_neg": r0,
        "recall_pos": r1,
        "f1_neg": f0,
        "f1_pos": f1,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "n": n,
        "n_pos": int((y_true == 1).sum()),
        "n_neg": int((y_true == 0).sum()),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def resolve_baseline_dir(arg: str | None) -> Path:
    if arg:
        p = Path(arg)
        if not p.is_dir():
            raise FileNotFoundError(p)
        return p
    for c in [
        paths.ROOT / "models" / "baseline" / "best" / "hf_model",
        paths.ROOT / "models" / "baseline" / "latest" / "hf_model",
    ]:
        if c.is_dir():
            return c
    raise FileNotFoundError("baseline hf_model not found — run Phase 5 first")


def run_phase8(cfg: dict[str, Any]) -> dict[str, Any]:
    from transformers import AutoTokenizer

    paths.ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[phase8] device={device}")
    random.seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))
    torch.manual_seed(int(cfg["seed"]))

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    # --- sentences ---
    articles = Path(cfg["articles_path"])
    if not articles.is_file():
        raise FileNotFoundError(
            f"missing {articles} — need Phase 2 articles_train.jsonl"
        )
    print(f"[phase8] reading sentences from {articles}")
    sentences = iter_sentences_from_articles(
        articles,
        min_words=cfg["min_words"],
        max_words=cfg["max_words"],
        max_sentences=cfg["max_candidate_sentences"],
    )
    random.shuffle(sentences)
    print(f"[phase8] candidate sentences: {len(sentences):,}")

    # --- label ---
    print("[phase8] loading BERT-SST labeler (substitution for Google API) ...")
    scorer = BertSSTScorer(device=str(device))
    labeled = label_and_filter(
        sentences,
        scorer,
        abs_threshold=cfg["abs_threshold"],
        batch_size=cfg["label_batch_size"],
    )
    n_pos = sum(1 for x in labeled if x.label == 1)
    n_neg = sum(1 for x in labeled if x.label == 0)
    print(
        f"[phase8] kept |s|>{cfg['abs_threshold']}: {len(labeled):,} "
        f"(pos={n_pos:,}, neg={n_neg:,})"
    )
    if len(labeled) < 200:
        raise RuntimeError(
            f"too few strong-sentiment sentences ({len(labeled)}). "
            "Increase --max-candidate-sentences."
        )

    # balance classes (optional but stabilises student-scale training)
    if cfg.get("balance", True):
        pos = [x for x in labeled if x.label == 1]
        neg = [x for x in labeled if x.label == 0]
        m = min(len(pos), len(neg))
        random.shuffle(pos)
        random.shuffle(neg)
        labeled = pos[:m] + neg[:m]
        random.shuffle(labeled)
        print(f"[phase8] balanced to {len(labeled):,} (per class {m:,})")

    # save labeled texts (no hidden states — smaller)
    lab_path = cache_dir / "labeled_sentences.jsonl"
    with lab_path.open("w", encoding="utf-8") as f:
        for x in labeled:
            f.write(json.dumps(asdict(x), ensure_ascii=False) + "\n")
    print(f"[phase8] wrote {lab_path}")

    # --- freeze LM, extract h_bar ---
    model_dir = resolve_baseline_dir(cfg.get("model_dir"))
    print(f"[phase8] loading FROZEN baseline from {model_dir}")
    lm = SentimentBiasLM.from_pretrained(str(model_dir))
    lm.to(device)
    lm.eval()
    for p in lm.parameters():
        p.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    feat_path = cache_dir / "features.pt"
    if feat_path.is_file() and cfg.get("reuse_features", True):
        print(f"[phase8] reusing cached features {feat_path}")
        blob = torch.load(feat_path, map_location="cpu")
        X, y = blob["X"], blob["y"]
    else:
        X, y = build_feature_matrix(
            lm,
            tokenizer,
            labeled,
            device=device,
            batch_size=cfg["extract_batch_size"],
            max_length=cfg["max_length"],
        )
        torch.save(
            {
                "X": X,
                "y": y,
                "input_dim": int(X.shape[1]),
                "model_dir": str(model_dir),
                "abs_threshold": cfg["abs_threshold"],
                "labeler": "bert_sst (student substitution for Google API)",
            },
            feat_path,
        )
        print(f"[phase8] wrote {feat_path}  X={tuple(X.shape)}")

    # free LM VRAM before MLP train
    del lm
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --- split ---
    n = X.shape[0]
    idx = torch.randperm(n)
    n_val = max(100, int(0.1 * n))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    print(f"[phase8] train={len(train_idx):,}  val={len(val_idx):,}  d={X.shape[1]}")

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=cfg["mlp_batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=cfg["mlp_batch_size"],
        shuffle=False,
    )

    f_sh = SentimentProjection(input_dim=int(X.shape[1]), hidden_dim=cfg["hidden_dim"])
    f_sh.to(device)
    print(
        f"[phase8] f_sh MLP: {sum(p.numel() for p in f_sh.parameters()):,} params  "
        f"(3-layer, hidden={cfg['hidden_dim']})"
    )

    train_info = train_mlp(
        f_sh,
        train_loader,
        val_loader,
        device=device,
        epochs=cfg["epochs"],
        lr=cfg["lr"],
    )
    final_metrics = evaluate_mlp(f_sh, val_loader, device)
    print("[phase8] FINAL val metrics:")
    for k, v in final_metrics.items():
        print(f"  {k}: {v}")

    # --- save ---
    ckpt_dir = paths.checkpoint_dir("sentiment_classifier")
    ckpt_path = ckpt_dir / "sentiment_classifier.pt"
    payload = {
        "state_dict": f_sh.state_dict(),
        "input_dim": int(X.shape[1]),
        "hidden_dim": cfg["hidden_dim"],
        "n_classes": 2,
        "labeler": "bert_sst",
        "abs_threshold": cfg["abs_threshold"],
        "baseline_model_dir": str(model_dir),
        "metrics": final_metrics,
        "config": cfg,
        "paper_note": (
            "Paper used Google Cloud API labels + 3-layer MLP hidden 128. "
            "We use BERT-SST labels (D12b) with the same MLP architecture."
        ),
    }
    torch.save(payload, ckpt_path)
    # also mirror under out_dir
    torch.save(payload, out_dir / "sentiment_classifier.pt")
    print(f"[phase8] saved {ckpt_path}")

    summary = {
        "phase": 8,
        "paper_step": "Step 2 — train f_sh on frozen LM hidden states",
        "n_candidates": len(sentences),
        "n_kept_strong": n_pos + n_neg,
        "n_balanced": n,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "labeler": "bert_sst (substitution for Google Cloud API)",
        "abs_threshold_pm1": cfg["abs_threshold"],
        "architecture": "3-layer MLP hidden 128",
        "input_dim": int(X.shape[1]),
        "metrics": final_metrics,
        "train_history": train_info["history"],
        "checkpoint": str(ckpt_path),
        "baseline_model_dir": str(model_dir),
        "setting": "STUDENT/COMPUTE-LIMITED",
    }
    (out_dir / "phase8_summary.json").write_text(json.dumps(summary, indent=2))
    txt = [
        "PHASE 8 SUMMARY — Sentiment projection f_sh",
        "=" * 60,
        f"baseline     : {model_dir}",
        f"labeler      : bert_sst (paper used Google API)",
        f"|s| threshold: {cfg['abs_threshold']} on [-1,1]",
        f"kept / bal   : {n_pos+n_neg:,} → {n:,}",
        f"train / val  : {len(train_idx):,} / {len(val_idx):,}",
        f"architecture : 3-layer MLP hidden={cfg['hidden_dim']}  d_in={X.shape[1]}",
        f"val accuracy : {final_metrics['accuracy']:.4f}",
        f"val F1 macro : {final_metrics['f1']:.4f}",
        f"precision    : {final_metrics['precision']:.4f}",
        f"recall       : {final_metrics['recall']:.4f}",
        f"confusion    : {final_metrics['confusion_matrix']}",
        f"checkpoint   : {ckpt_path}",
        "",
        "NEXT: Phase 9–11 debiasing (f_sh FROZEN in Phase 11).",
    ]
    (out_dir / "phase8_summary.txt").write_text("\n".join(txt) + "\n")
    print("\n".join(txt))
    print("PHASE 8 DONE")
    return summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 8 — train sentiment projection f_sh")
    p.add_argument("--model-dir", type=str, default=None)
    p.add_argument(
        "--articles-path",
        type=str,
        default=None,
        help="default: data/wikitext103/processed/articles_train.jsonl",
    )
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-candidate-sentences", type=int, default=40_000)
    p.add_argument("--min-words", type=int, default=5)
    p.add_argument("--max-words", type=int, default=60)
    p.add_argument("--abs-threshold", type=float, default=0.7)
    p.add_argument("--label-batch-size", type=int, default=64)
    p.add_argument("--extract-batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--mlp-batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--no-balance", action="store_true")
    p.add_argument("--no-reuse-features", action="store_true")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="tiny run: 3000 candidates, 3 epochs",
    )
    return p


def main() -> int:
    args = build_argparser().parse_args()
    paths.ensure_dirs()

    articles = args.articles_path or str(
        paths.data_path("wikitext103", "processed", "articles_train.jsonl")
    )
    out_dir = args.out_dir or str(paths.results_path("sentiment_classifier"))

    cfg = {
        "model_dir": args.model_dir,
        "articles_path": articles,
        "out_dir": out_dir,
        "seed": args.seed,
        "max_candidate_sentences": args.max_candidate_sentences,
        "min_words": args.min_words,
        "max_words": args.max_words,
        "abs_threshold": args.abs_threshold,
        "label_batch_size": args.label_batch_size,
        "extract_batch_size": args.extract_batch_size,
        "max_length": args.max_length,
        "hidden_dim": args.hidden_dim,
        "mlp_batch_size": args.mlp_batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "balance": not args.no_balance,
        "reuse_features": not args.no_reuse_features,
    }
    if args.smoke:
        cfg["max_candidate_sentences"] = 3000
        cfg["epochs"] = 3
        cfg["out_dir"] = str(paths.results_path("sentiment_classifier_smoke"))
        print("[phase8] SMOKE mode")

    run_phase8(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
