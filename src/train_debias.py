"""
src/train_debias.py  —  PHASES 10 & 11
Curriculum Step 3: continue training LM with fairness regularizer.

Modes:
  embedding  (Phase 10): L_fair = 1 - cos(h̄(x), h̄(x̃))
  sentiment  (Phase 11): L_fair = 1 - cos(f_sh(h̄(x)), f_sh(h̄(x̃)))  with f_sh FROZEN

Shared:
  L = L_LM(x) + λ * L_fair
  L_LM on UNPERTURBED x only
  training data = sequences containing sensitive tokens only
  NO evaluation templates
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from src import paths
from src.counterfactual import cosine_distance, make_counterfactual_ids, pool_h_bar
from src.dataset_lm import PackedSequenceDataset, collate_lm
from src.model import SentimentBiasLM
from src.sentiment_classifier import SentimentProjection
from src.train_baseline import (
    build_scheduler,
    evaluate_ppl,
    load_config,
    resolve_data_dir,
    set_seed,
)


# ---------------------------------------------------------------------------
# CF-aware dataset: yields original + counterfactual ids
# ---------------------------------------------------------------------------

class CounterfactualLMDataset(Dataset):
    """
    Wraps Phase 2 sensitive sequences; on each access builds a CF swap.
    Sequences that cannot form a single-token CF are retried a few times
    from nearby indices; if still failing, returns original twice (L_fair=0).
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer,
        *,
        max_sequences: int | None = None,
        seed: int = 42,
    ):
        self.base = PackedSequenceDataset(
            jsonl_path, sensitive_only=True, max_sequences=max_sequences
        )
        self.tokenizer = tokenizer
        self.rng = random.Random(seed)
        self.seed = seed

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.base[idx]
        ids = row["input_ids"]
        # try a few times with different RNG draws
        local = random.Random(self.seed + idx * 10007 + self.rng.randint(0, 10_000))
        cf_ids = None
        meta = None
        for _ in range(5):
            cf_ids, meta = make_counterfactual_ids(ids, self.tokenizer, rng=local)
            if cf_ids is not None:
                break
            local.random()
        if cf_ids is None:
            cf_ids = ids.clone()
            has_cf = False
        else:
            has_cf = True
        return {
            "input_ids": ids,
            "cf_input_ids": cf_ids,
            "labels": ids.clone(),
            "attention_mask": torch.ones_like(ids),
            "has_cf": torch.tensor(1 if has_cf else 0, dtype=torch.long),
        }


def collate_cf(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch], 0),
        "cf_input_ids": torch.stack([b["cf_input_ids"] for b in batch], 0),
        "labels": torch.stack([b["labels"] for b in batch], 0),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch], 0),
        "has_cf": torch.stack([b["has_cf"] for b in batch], 0),
    }


def load_f_sh(path: str | Path, device: torch.device) -> SentimentProjection:
    blob = torch.load(path, map_location="cpu")
    m = SentimentProjection(
        input_dim=int(blob["input_dim"]),
        hidden_dim=int(blob.get("hidden_dim", 128)),
    )
    m.load_state_dict(blob["state_dict"])
    m.to(device)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def save_debias_ckpt(
    *,
    model: SentimentBiasLM,
    optimizer,
    scheduler,
    step: int,
    best_val_ppl: float,
    cfg: dict[str, Any],
    metrics: dict[str, Any],
    tag: str,
) -> Path:
    run = cfg["run_name"]
    # include lambda in path so λ grid does not overwrite
    lam = cfg.get("lambda_fair", 0)
    run_dir = f"{run}_lam{lam}"
    ckpt_root = paths.checkpoint_dir(run_dir)
    out = ckpt_root / tag
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out / "hf_model"))
    torch.save(
        {
            "step": step,
            "best_val_ppl": best_val_ppl,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "config": cfg,
            "metrics": metrics,
        },
        out / "trainer_state.pt",
    )
    state = {
        "tag": tag,
        "step": step,
        "best_val_ppl": best_val_ppl,
        "lambda_fair": lam,
        "mode": cfg.get("mode"),
        "metrics": metrics,
        "model_dir": str(out / "hf_model"),
        "run_dir": run_dir,
    }
    (out / "state.json").write_text(json.dumps(state, indent=2))
    paths.save_state(state, run_dir, tag=tag)
    paths.save_state(state, run_dir, tag="latest")
    print(f"[ckpt] {out} step={step} best_ppl={best_val_ppl:.4f}")
    return out


def try_resume_debias(cfg, model, optimizer, scheduler, device):
    if not cfg.get("resume", True):
        return 0, float("inf")
    lam = cfg.get("lambda_fair", 0)
    run_dir = f"{cfg['run_name']}_lam{lam}"
    ckpt_root = paths.checkpoint_dir(run_dir)
    for tag in ("latest", "best"):
        path = ckpt_root / tag / "trainer_state.pt"
        if not path.is_file():
            continue
        print(f"[resume] {path}")
        blob = torch.load(path, map_location="cpu")
        hf = path.parent / "hf_model"
        if hf.is_dir():
            resumed = SentimentBiasLM.from_pretrained(str(hf))
            model.model.load_state_dict(resumed.model.state_dict())
            model.to(device)
        try:
            optimizer.load_state_dict(blob["optimizer"])
        except Exception as exc:  # noqa: BLE001
            print("optimizer not loaded", exc)
        if scheduler is not None and blob.get("scheduler"):
            try:
                scheduler.load_state_dict(blob["scheduler"])
            except Exception:
                pass
        return int(blob.get("step", 0)), float(blob.get("best_val_ppl", float("inf")))
    return 0, float("inf")


def fairness_loss(
    model: SentimentBiasLM,
    input_ids: torch.Tensor,
    cf_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    has_cf: torch.Tensor,
    *,
    mode: str,
    f_sh: SentimentProjection | None,
) -> torch.Tensor:
    """
    Compute mean L_fair over the batch.
    Rows with has_cf==0 contribute 0.
    """
    out_o = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    out_c = model(input_ids=cf_ids, attention_mask=attention_mask, output_hidden_states=True)
    h_o = pool_h_bar(out_o.h_bar, attention_mask)  # [B, D]
    h_c = pool_h_bar(out_c.h_bar, attention_mask)

    if mode == "sentiment":
        assert f_sh is not None
        # project into 128-d sentiment subspace (frozen)
        with torch.no_grad():
            # still need grads w.r.t. h through f_sh? Paper freezes f_sh but
            # gradients flow into h and thus the LM. So f_sh forward must allow
            # grad through inputs but not parameters.
            pass
        z_o = f_sh.project(h_o)
        z_c = f_sh.project(h_c)
        # detach params already requires_grad False; inputs keep grad
        d = 1.0 - torch.nn.functional.cosine_similarity(z_o, z_c, dim=-1, eps=1e-8)
    else:
        d = 1.0 - torch.nn.functional.cosine_similarity(h_o, h_c, dim=-1, eps=1e-8)

    # zero out rows without a real CF
    mask = has_cf.to(d.dtype)
    if mask.sum() < 1:
        return d.sum() * 0.0  # zero with grad graph
    return (d * mask).sum() / mask.sum()


def train(cfg: dict[str, Any]) -> dict[str, Any]:
    paths.ensure_dirs()
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = cfg["mode"]
    assert mode in {"embedding", "sentiment"}
    lam = float(cfg["lambda_fair"])
    print(f"[debias] mode={mode}  λ={lam}  device={device}")

    data_dir = resolve_data_dir(cfg)
    train_path = data_dir / "sequences_train.jsonl"
    val_path = data_dir / "sequences_validation.jsonl"
    if not train_path.is_file():
        raise FileNotFoundError(train_path)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = CounterfactualLMDataset(
        train_path,
        tokenizer,
        max_sequences=cfg.get("max_train_sequences"),
        seed=int(cfg.get("seed", 42)),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["micro_batch_size"]),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
        collate_fn=collate_cf,
        drop_last=True,
    )
    # val PPL on normal (non-CF) loader
    from src.dataset_lm import make_loader

    val_loader = make_loader(
        val_path,
        batch_size=int(cfg["micro_batch_size"]),
        shuffle=False,
        num_workers=0,
        max_sequences=cfg.get("max_val_sequences", 200),
    )
    print(f"[data] sensitive train sequences: {len(train_ds):,}")

    # start FROM baseline weights
    baseline_dir = cfg.get("baseline_dir", "models/baseline/best/hf_model")
    bpath = Path(baseline_dir)
    if not bpath.is_absolute():
        bpath = paths.ROOT / bpath
    print(f"[model] load baseline {bpath}")
    model = SentimentBiasLM.from_pretrained(str(bpath))
    model.to(device)

    f_sh = None
    if mode == "sentiment":
        fpath = cfg.get("f_sh_path") or "models/sentiment_classifier/sentiment_classifier.pt"
        fpath = Path(fpath)
        if not fpath.is_absolute():
            fpath = paths.ROOT / fpath
        print(f"[model] load FROZEN f_sh from {fpath}")
        f_sh = load_f_sh(fpath, device)

    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        betas=tuple(cfg.get("betas", [0.9, 0.999])),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
    )
    max_steps = int(cfg["max_steps"])
    scheduler = build_scheduler(optimizer, int(cfg.get("warmup_steps", 50)), max_steps)
    start_step, best_val_ppl = try_resume_debias(cfg, model, optimizer, scheduler, device)

    hist_path = paths.results_path(
        f"{cfg['run_name']}_lam{lam}", "train_history.jsonl"
    )
    hist_path.parent.mkdir(parents=True, exist_ok=True)

    micro = int(cfg["micro_batch_size"])
    accum = int(cfg["grad_accum_steps"])
    log_every = int(cfg.get("log_every", 20))
    eval_every = int(cfg.get("eval_every", 200))
    save_every = int(cfg.get("save_every", 200))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))

    print("=" * 72)
    print(f" DEBIAS TRAIN — mode={mode}  λ={lam}")
    print("=" * 72)
    print(f" baseline   : {bpath}")
    print(f" max_steps  : {max_steps} (from {start_step})")
    print(f" batch      : micro={micro} accum={accum} eff={micro*accum}")
    print(f" lr         : {cfg['lr']}")
    print(f" L = L_LM(x) + λ * L_fair   [L_LM on unperturbed x]")
    print(f" templates  : NEVER in training")
    print("=" * 72)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = start_step
    run_lm = 0.0
    run_fair = 0.0
    run_n = 0
    t0 = time.time()
    data_iter = iter(train_loader)
    pbar = tqdm(total=max_steps, initial=step, desc=f"{mode}_λ{lam}", dynamic_ncols=True)

    while step < max_steps:
        for _ in range(accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            cf_ids = batch["cf_input_ids"].to(device)
            labels = batch["labels"].to(device)
            attn = batch["attention_mask"].to(device)
            has_cf = batch["has_cf"].to(device)

            # LM loss on UNPERTURBED x only
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss_lm = out.loss

            loss_fair = fairness_loss(
                model, input_ids, cf_ids, attn, has_cf, mode=mode, f_sh=f_sh
            )
            loss = (loss_lm + lam * loss_fair) / accum
            loss.backward()

            run_lm += float(loss_lm.item())
            run_fair += float(loss_fair.item())
            run_n += 1

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        pbar.update(1)

        if step % log_every == 0 or step == 1:
            pbar.set_postfix(
                lm=f"{run_lm/max(run_n,1):.4f}",
                fair=f"{run_fair/max(run_n,1):.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )
            run_lm = run_fair = 0.0
            run_n = 0

        if step % eval_every == 0 or step == max_steps:
            metrics = evaluate_ppl(model, val_loader, device)
            row = {
                "step": step,
                "mode": mode,
                "lambda": lam,
                "val_loss": metrics["loss"],
                "val_ppl": metrics["ppl"],
            }
            with hist_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(f"\n[eval] step={step} val_PPL={metrics['ppl']:.2f}")
            if metrics["ppl"] < best_val_ppl:
                best_val_ppl = metrics["ppl"]
                save_debias_ckpt(
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    step=step, best_val_ppl=best_val_ppl, cfg=cfg, metrics=row, tag="best",
                )
            model.train()

        if step % save_every == 0 or step == max_steps:
            save_debias_ckpt(
                model=model, optimizer=optimizer, scheduler=scheduler,
                step=step, best_val_ppl=best_val_ppl, cfg=cfg,
                metrics={"step": step, "best_val_ppl": best_val_ppl}, tag="latest",
            )

    pbar.close()
    summary = {
        "mode": mode,
        "lambda_fair": lam,
        "final_step": step,
        "best_val_ppl": best_val_ppl,
        "baseline_dir": str(bpath),
        "run_dir": f"{cfg['run_name']}_lam{lam}",
        "checkpoint_dir": str(paths.checkpoint_dir(f"{cfg['run_name']}_lam{lam}")),
        "elapsed_sec": round(time.time() - t0, 1),
        "setting": "STUDENT/COMPUTE-LIMITED",
        "paper_loss": "L = L_LM(x) + λ L_fair  (L_LM on unperturbed x)",
    }
    out = paths.results_path(f"{cfg['run_name']}_lam{lam}", "phase_debias_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("DEBIAS TRAIN DONE — Commit on Kaggle now.")
    return summary


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--mode", type=str, choices=["embedding", "sentiment"], default=None)
    p.add_argument("--lambda-fair", type=float, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--micro-batch-size", type=int, default=None)
    p.add_argument("--grad-accum-steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--save-every", type=int, default=None)
    p.add_argument("--max-train-sequences", type=int, default=None)
    p.add_argument("--baseline-dir", type=str, default=None)
    p.add_argument("--f-sh-path", type=str, default=None)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    return p


def main() -> int:
    args = build_argparser().parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        cfg_path = paths.ROOT / args.config
    cfg = load_config(cfg_path)

    if args.mode:
        cfg["mode"] = args.mode
    if args.lambda_fair is not None:
        cfg["lambda_fair"] = args.lambda_fair
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    if args.micro_batch_size is not None:
        cfg["micro_batch_size"] = args.micro_batch_size
    if args.grad_accum_steps is not None:
        cfg["grad_accum_steps"] = args.grad_accum_steps
    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.eval_every is not None:
        cfg["eval_every"] = args.eval_every
    if args.save_every is not None:
        cfg["save_every"] = args.save_every
    if args.max_train_sequences is not None:
        cfg["max_train_sequences"] = args.max_train_sequences
    if args.baseline_dir:
        cfg["baseline_dir"] = args.baseline_dir
    if args.f_sh_path:
        cfg["f_sh_path"] = args.f_sh_path
    if args.run_name:
        cfg["run_name"] = args.run_name
    if args.no_resume:
        cfg["resume"] = False
    if args.smoke:
        cfg["max_steps"] = 30
        cfg["eval_every"] = 15
        cfg["save_every"] = 15
        cfg["max_train_sequences"] = 512
        cfg["micro_batch_size"] = 1
        cfg["grad_accum_steps"] = 4
        cfg["resume"] = False
        print("[debias] SMOKE mode")

    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
