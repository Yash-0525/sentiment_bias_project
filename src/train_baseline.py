"""
src/train_baseline.py  —  PHASE 5
Step 1 of the paper curriculum: plain next-token LM training.

ORIGINAL PAPER SETTING (WikiText-103):
  Transformer-XL 257M from scratch, Adam 2.5e-4, seq 512, batch 512,
  250k steps, 128 TPUv3.

STUDENT / KAGGLE SETTING (configs/baseline.yaml):
  GPT-2 small fine-tune, AdamW 5e-5, seq 256, effective batch 32,
  3000 steps, 1× T4/P100, fp32.

Unchanged scientific pieces:
  - autoregressive next-token cross-entropy
  - no fairness loss
  - no evaluation templates in the data
  - validation loss + perplexity logged
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
from tqdm.auto import tqdm

from src import paths
from src.dataset_lm import make_loader
from src.model import SentimentBiasLM


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_data_dir(cfg: dict[str, Any]) -> Path:
    raw = cfg.get("data_dir", "data/wikitext103/processed")
    p = Path(raw)
    if p.is_absolute():
        return p
    # prefer project-rooted path
    cand = paths.ROOT / p
    if cand.is_dir():
        return cand
    if p.is_dir():
        return p
    return cand


@torch.no_grad()
def evaluate_ppl(
    model: SentimentBiasLM,
    loader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    n_tokens = 0  # we use mean loss per batch * tokens approx via counting batches
    n_batches = 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # out.loss is mean over tokens in the batch
        total_loss += float(out.loss.item())
        n_batches += 1
        n_tokens += int(input_ids.numel())
    model.train()
    if n_batches == 0:
        return {"loss": float("nan"), "ppl": float("nan"), "n_batches": 0}
    mean_loss = total_loss / n_batches
    ppl = math.exp(min(mean_loss, 20.0))  # clamp for overflow safety
    return {
        "loss": round(mean_loss, 6),
        "ppl": round(ppl, 4),
        "n_batches": n_batches,
        "n_tokens": n_tokens,
    }


def save_checkpoint(
    *,
    model: SentimentBiasLM,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    best_val_ppl: float,
    cfg: dict[str, Any],
    metrics: dict[str, Any],
    tag: str,
    tokenizer_name: str,
) -> Path:
    ckpt_root = paths.checkpoint_dir(cfg["run_name"])
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
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        out / "trainer_state.pt",
    )
    # small JSON for humans + paths.save_state
    state = {
        "tag": tag,
        "step": step,
        "best_val_ppl": best_val_ppl,
        "metrics": metrics,
        "model_dir": str(out / "hf_model"),
        "tokenizer_name": tokenizer_name,
        "run_name": cfg["run_name"],
        "setting": "STUDENT/COMPUTE-LIMITED — GPT-2 small fine-tune, Step 1 only",
    }
    (out / "state.json").write_text(json.dumps(state, indent=2))
    paths.save_state(state, cfg["run_name"], tag=tag)
    if tag != "latest":
        # also refresh latest pointer as a copy of state
        paths.save_state(state, cfg["run_name"], tag="latest")
        # mirror weights into latest/
        latest = ckpt_root / "latest"
        if latest.exists() and latest.resolve() != out.resolve():
            # rewrite latest as a fresh save of same weights (simple, robust)
            pass
    print(f"[ckpt] saved {out}  step={step}  best_val_ppl={best_val_ppl:.4f}")
    return out


def try_resume(
    cfg: dict[str, Any],
    model: SentimentBiasLM,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
) -> tuple[int, float]:
    """Return (start_step, best_val_ppl)."""
    if not cfg.get("resume", True):
        return 0, float("inf")
    ckpt_root = paths.checkpoint_dir(cfg["run_name"])
    # prefer latest/trainer_state.pt, else best/
    candidates = [
        ckpt_root / "latest" / "trainer_state.pt",
        ckpt_root / "best" / "trainer_state.pt",
    ]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        print("[resume] no checkpoint found — starting from scratch (HF gpt2 weights)")
        return 0, float("inf")

    print(f"[resume] loading {path}")
    blob = torch.load(path, map_location="cpu")
    hf_dir = path.parent / "hf_model"
    if hf_dir.is_dir():
        resumed = SentimentBiasLM.from_pretrained(str(hf_dir))
        model.model.load_state_dict(resumed.model.state_dict())
        model.to(device)
    if blob.get("optimizer") is not None:
        try:
            optimizer.load_state_dict(blob["optimizer"])
        except Exception as exc:  # noqa: BLE001
            print(f"[resume] optimizer state not loaded ({exc})")
    if scheduler is not None and blob.get("scheduler") is not None:
        try:
            scheduler.load_state_dict(blob["scheduler"])
        except Exception as exc:  # noqa: BLE001
            print(f"[resume] scheduler state not loaded ({exc})")
    step = int(blob.get("step", 0))
    best = float(blob.get("best_val_ppl", float("inf")))
    print(f"[resume] step={step}  best_val_ppl={best}")
    return step, best


def build_scheduler(optimizer, warmup_steps: int, max_steps: int):
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # linear decay to 0
        progress = float(current_step - warmup_steps) / float(
            max(1, max_steps - warmup_steps)
        )
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# main train
# ---------------------------------------------------------------------------

def train(cfg: dict[str, Any]) -> dict[str, Any]:
    paths.ensure_dirs()
    set_seed(int(cfg.get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no CUDA — training on CPU is not practical. Set Kaggle Accelerator → GPU.")
    else:
        print(f"[device] {torch.cuda.get_device_name(0)}")

    data_dir = resolve_data_dir(cfg)
    train_path = data_dir / "sequences_train.jsonl"
    val_path = data_dir / "sequences_validation.jsonl"
    for p in (train_path, val_path):
        if not p.is_file():
            raise FileNotFoundError(
                f"Missing {p}. Phase 2 must be complete before Phase 5."
            )

    max_train = cfg.get("max_train_sequences")  # smoke knob
    max_val = cfg.get("max_val_sequences", 200)  # enough for stable PPL

    print(f"[data] train={train_path}")
    train_loader = make_loader(
        train_path,
        batch_size=int(cfg["micro_batch_size"]),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
        max_sequences=max_train,
    )
    val_loader = make_loader(
        val_path,
        batch_size=int(cfg["micro_batch_size"]),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
        max_sequences=max_val,
    )
    print(f"[data] train batches/epoch≈{len(train_loader)}  val batches={len(val_loader)}")

    print(f"[model] loading {cfg['model_name']} ...")
    model = SentimentBiasLM(cfg["model_name"])
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] parameters={n_params/1e6:.1f}M  layers={model.n_layer}  d={model.n_embd}")

    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        betas=tuple(cfg.get("betas", [0.9, 0.999])),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
    )
    max_steps = int(cfg["max_steps"])
    scheduler = build_scheduler(
        optimizer,
        warmup_steps=int(cfg.get("warmup_steps", 100)),
        max_steps=max_steps,
    )

    start_step, best_val_ppl = try_resume(cfg, model, optimizer, scheduler, device)

    # initial val (only if starting fresh — useful baseline number)
    history: list[dict[str, Any]] = []
    hist_path = paths.results_path("baseline", "train_history.jsonl")
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    if start_step == 0:
        print("[eval] initial validation (pretrained GPT-2, before fine-tune) ...")
        init_metrics = evaluate_ppl(model, val_loader, device)
        print(f"  init val loss={init_metrics['loss']:.4f}  PPL={init_metrics['ppl']:.2f}")
        row = {"step": 0, "phase": "init", **init_metrics}
        history.append(row)
        with hist_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        best_val_ppl = init_metrics["ppl"]
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=0,
            best_val_ppl=best_val_ppl,
            cfg=cfg,
            metrics={"init_val": init_metrics},
            tag="init",
            tokenizer_name=cfg["model_name"],
        )

    micro_bs = int(cfg["micro_batch_size"])
    accum = int(cfg["grad_accum_steps"])
    log_every = int(cfg.get("log_every", 20))
    eval_every = int(cfg.get("eval_every", 500))
    save_every = int(cfg.get("save_every", 500))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))

    print("=" * 72)
    print(" PHASE 5 — Baseline LM training (Step 1 curriculum)")
    print("=" * 72)
    print(f" run_name        : {cfg['run_name']}")
    print(f" model           : {cfg['model_name']} ({n_params/1e6:.1f}M)")
    print(f" device          : {device}")
    print(f" micro_batch     : {micro_bs}  accum={accum}  effective={micro_bs*accum}")
    print(f" lr              : {cfg['lr']}")
    print(f" max_steps       : {max_steps}  (resume from {start_step})")
    print(f" seq_len         : {cfg.get('seq_len', 256)}")
    print(f" precision       : fp32")
    print(f" fairness loss   : NONE (baseline)")
    print(f" templates in data: NONE (paper rule)")
    print("=" * 72)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = start_step
    running_loss = 0.0
    running_n = 0
    t0 = time.time()
    data_iter = iter(train_loader)

    pbar = tqdm(total=max_steps, initial=step, desc="baseline", dynamic_ncols=True)

    while step < max_steps:
        # accumulate gradients
        for _ in range(accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=False,
            )
            loss = out.loss / accum
            loss.backward()
            running_loss += float(out.loss.item())
            running_n += 1

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        pbar.update(1)

        if step % log_every == 0 or step == 1:
            avg = running_loss / max(running_n, 1)
            lr_now = scheduler.get_last_lr()[0]
            elapsed = time.time() - t0
            sps = step / max(elapsed, 1e-6)
            pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr_now:.2e}", sps=f"{sps:.2f}")
            running_loss = 0.0
            running_n = 0

        do_eval = (step % eval_every == 0) or (step == max_steps)
        do_save = (step % save_every == 0) or (step == max_steps)

        if do_eval:
            metrics = evaluate_ppl(model, val_loader, device)
            lr_now = scheduler.get_last_lr()[0]
            row = {
                "step": step,
                "phase": "train",
                "lr": lr_now,
                "val_loss": metrics["loss"],
                "val_ppl": metrics["ppl"],
            }
            history.append(row)
            with hist_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(
                f"\n[eval] step={step}  val_loss={metrics['loss']:.4f}  "
                f"val_PPL={metrics['ppl']:.2f}  best={best_val_ppl:.2f}"
            )
            if metrics["ppl"] < best_val_ppl:
                best_val_ppl = metrics["ppl"]
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    best_val_ppl=best_val_ppl,
                    cfg=cfg,
                    metrics=row,
                    tag="best",
                    tokenizer_name=cfg["model_name"],
                )
            model.train()

        if do_save:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                best_val_ppl=best_val_ppl,
                cfg=cfg,
                metrics={"step": step, "best_val_ppl": best_val_ppl},
                tag="latest",
                tokenizer_name=cfg["model_name"],
            )
            # also step-tagged snapshot every save (optional disk cost)
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                best_val_ppl=best_val_ppl,
                cfg=cfg,
                metrics={"step": step, "best_val_ppl": best_val_ppl},
                tag=f"step_{step}",
                tokenizer_name=cfg["model_name"],
            )

    pbar.close()

    # final summary
    summary = {
        "run_name": cfg["run_name"],
        "setting": "STUDENT/COMPUTE-LIMITED",
        "paper_step": "Step 1 — normal LM training (no fairness)",
        "model_name": cfg["model_name"],
        "n_params": n_params,
        "max_steps": max_steps,
        "final_step": step,
        "best_val_ppl": best_val_ppl,
        "effective_batch": micro_bs * accum,
        "seq_len": cfg.get("seq_len", 256),
        "lr": cfg["lr"],
        "data_dir": str(data_dir),
        "checkpoint_dir": str(paths.checkpoint_dir(cfg["run_name"])),
        "history_path": str(hist_path),
        "device": str(device),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out_sum = paths.results_path("baseline", "phase5_summary.json")
    out_sum.write_text(json.dumps(summary, indent=2))
    print()
    print("=" * 72)
    print(" PHASE 5 COMPLETE — BASELINE MODEL READY")
    print(f" best val PPL : {best_val_ppl:.4f}")
    print(f" checkpoints  : {paths.checkpoint_dir(cfg['run_name'])}")
    print(f"   best/      : use this for Phase 6+ evaluation")
    print(f"   latest/    : last step")
    print(f" summary      : {out_sum}")
    print("=" * 72)
    print("IMPORTANT: Save Version → Save & Run All (Commit) on Kaggle NOW.")
    return summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 5 — train baseline GPT-2 LM")
    p.add_argument(
        "--config",
        type=str,
        default=str(paths.ROOT / "configs" / "baseline.yaml"),
    )
    p.add_argument("--max-steps", type=int, default=None, help="override config max_steps")
    p.add_argument("--micro-batch-size", type=int, default=None)
    p.add_argument("--grad-accum-steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--save-every", type=int, default=None)
    p.add_argument(
        "--max-train-sequences",
        type=int,
        default=None,
        help="SMOKE: limit train sequences (e.g. 512)",
    )
    p.add_argument(
        "--max-val-sequences",
        type=int,
        default=None,
        help="limit val sequences for faster eval",
    )
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--run-name", type=str, default=None)
    return p


def main() -> int:
    args = build_argparser().parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        # try relative to ROOT
        alt = paths.ROOT / "configs" / "baseline.yaml"
        if alt.is_file():
            cfg_path = alt
        else:
            raise FileNotFoundError(args.config)
    cfg = load_config(cfg_path)

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
    if args.max_val_sequences is not None:
        cfg["max_val_sequences"] = args.max_val_sequences
    if args.no_resume:
        cfg["resume"] = False
    if args.run_name is not None:
        cfg["run_name"] = args.run_name

    # write resolved config next to checkpoints
    paths.ensure_dirs()
    resolved = paths.checkpoint_dir(cfg["run_name"]) / "resolved_config.yaml"
    resolved.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"[config] {cfg_path} → {resolved}")

    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
