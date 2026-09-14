"""
src/check_environment.py  —  Phase 1 verification

Run this ONCE, on the machine you will actually train on (the Colab T4).
It checks nine things and tells you, per check, PASS / WARN / FAIL.

    python -m src.check_environment
or in Colab:
    !python src/check_environment.py

Exit code 0 = everything you need is present.
Exit code 1 = at least one REQUIRED check failed. Stop and fix it.

Every check is wrapped, so one missing library never hides the others.
"""

from __future__ import annotations

import importlib
import platform
import sys
import traceback
from typing import Any

# --------------------------------------------------------------------------
# tiny reporting helpers
# --------------------------------------------------------------------------
PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_results: list[tuple[str, str, str]] = []


def record(name: str, status: str, detail: str = "") -> None:
    _results.append((name, status, detail))
    icon = {PASS: "\u2705", WARN: "\u26a0\ufe0f ", FAIL: "\u274c"}[status]
    print(f"{icon} {status:4s} | {name:34s} | {detail}")


def attempt(name: str, fn, required: bool = True, min_detail: str = "") -> Any:
    """Run fn(); record PASS/FAIL (or WARN if not required). Never raises."""
    try:
        detail = fn()
        record(name, PASS, detail if isinstance(detail, str) else str(detail))
        return detail
    except Exception as exc:  # noqa: BLE001
        record(name, FAIL if required else WARN,
               f"{type(exc).__name__}: {exc}")
        if not required:
            traceback.print_exc(limit=1)
        return None


# --------------------------------------------------------------------------
# the checks
# --------------------------------------------------------------------------
def check_python() -> str:
    v = sys.version_info
    if (v.major, v.minor) < (3, 9):
        raise RuntimeError(f"Python >= 3.9 required, found {v.major}.{v.minor}")
    return f"{platform.python_version()} ({platform.system()})"


def check_torch() -> str:
    import torch
    return f"{torch.__version__}"


def check_cuda() -> str:
    import os

    import torch
    if not torch.cuda.is_available():
        if os.environ.get("SBP_ALLOW_CPU") == "1":
            # Escape hatch ONLY for running the non-GPU checks on a laptop.
            # Training on CPU is not viable - see the Phase 1 notes.
            return "NO GPU (SBP_ALLOW_CPU=1 set - smoke test only, do NOT train)"
        raise RuntimeError(
            "CUDA not available. On Colab: Runtime -> Change runtime type -> "
            "Hardware accelerator -> T4 GPU.")
    name = torch.cuda.get_device_name(0)
    free, total = torch.cuda.mem_get_info()
    gb = 1024 ** 3
    return (f"{name} | {total/gb:.1f} GB VRAM total, {free/gb:.1f} GB free | "
            f"cc {torch.cuda.get_device_capability(0)}")


def check_imports() -> str:
    """Import every library a later phase depends on."""
    # (module, minimum major.minor) — the minima are the versions whose APIs
    # this project uses. Newer is fine.
    required = [
        ("numpy",              (1, 24)),
        ("pandas",             (2, 0)),
        ("scipy",              (1, 10)),
        ("sklearn",            (1, 3)),
        ("transformers",       (4, 38)),
        ("datasets",           (2, 16)),
        ("sentence_transformers", (2, 5)),
        ("nltk",               (3, 8)),
        ("yaml",               (6, 0)),
        ("tqdm",               (4, 65)),
        ("matplotlib",         (3, 7)),
        ("seaborn",            (0, 12)),
    ]
    missing, versions = [], []
    for mod, (maj, minr) in required:
        try:
            m = importlib.import_module(mod)
        except Exception:  # noqa: BLE001
            missing.append(mod)
            continue
        ver = getattr(m, "__version__", "0.0")
        versions.append(f"{mod} {ver}")
        parts = ver.split(".")
        try:
            got = (int(parts[0]), int(parts[1]))
        except (ValueError, IndexError):
            got = (0, 0)
        if got < (maj, minr):
            missing.append(f"{mod}<{maj}.{minr} (found {ver})")
    if missing:
        raise RuntimeError("missing/too old: " + ", ".join(missing))
    return "; ".join(versions)


def check_transformers_api() -> str:
    """
    The API the entire debiasing method rests on:
    GPT2LMHeadModel must return per-layer hidden states.

    Phase 10 computes  h_bar = mean(h^(L-1), h^(L))  from exactly this.
    If this check fails, nothing downstream works — so we test it for real,
    with a tiny random-weight GPT-2 config (no download needed).
    """
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    # bos/eos pinned inside the tiny vocab, otherwise transformers>=5 prints
    # "must be None or an integer within the vocabulary" warnings that look
    # like errors and will confuse you.
    cfg = GPT2Config(vocab_size=512, n_positions=64, n_embd=32,
                     n_layer=4, n_head=4,
                     bos_token_id=0, eos_token_id=1)
    model = GPT2LMHeadModel(cfg).eval()

    ids = torch.randint(0, 512, (2, 8))
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)

    hs = out.hidden_states
    n_layers = cfg.n_layer
    # embeddings + one tensor per transformer layer
    assert len(hs) == n_layers + 1, f"expected {n_layers+1} hidden states, got {len(hs)}"
    assert hs[-1].shape == (2, 8, cfg.n_embd), f"bad shape {tuple(hs[-1].shape)}"

    # the exact operation Phase 10 will use
    h_bar = (hs[-2] + hs[-1]) / 2
    assert h_bar.shape == (2, 8, cfg.n_embd)
    assert torch.isfinite(h_bar).all(), "non-finite values in h_bar"

    # and the logits must be usable as a next-token loss (Phase 5)
    assert out.logits.shape == (2, 8, cfg.vocab_size)
    loss = out.loss if out.loss is not None else None
    return (f"output_hidden_states OK | {len(hs)} tensors | "
            f"h_bar shape {tuple(h_bar.shape)} | logits {tuple(out.logits.shape)}")


def check_wasserstein() -> str:
    """
    Phase 13's metric, validated right now against scipy.
    Uses the paper's own Figure 2 examples (p.68) as ground truth.
    """
    import numpy as np
    from scipy.stats import wasserstein_distance

    def w1_cdf(a, b):
        a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
        xs = np.unique(np.concatenate([a, b]))
        Fa = np.searchsorted(a, xs, side="right") / len(a)
        Fb = np.searchsorted(b, xs, side="right") / len(b)
        return float(np.sum(np.abs(Fa - Fb) * np.diff(np.append(xs, 1.0))))

    # paper Fig 2(a): point masses at 0.555 / 0.445, labelled "W1 = 0.1"
    a = w1_cdf([0.555] * 100, [0.445] * 100)
    # paper Fig 2(b): 0.505 / 0.494, labelled "W1 = 0.01"
    b = w1_cdf([0.505] * 100, [0.494] * 100)
    assert abs(a - 0.110) < 1e-9, f"Fig2a expected 0.110, got {a}"
    assert abs(b - 0.011) < 1e-9, f"Fig2b expected 0.011, got {b}"

    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        x = rng.uniform(0, 1, int(rng.integers(50, 300)))
        y = np.clip(rng.normal(0.6, 0.2, int(rng.integers(50, 300))), 0, 1)
        worst = max(worst, abs(w1_cdf(x, y) - wasserstein_distance(x, y)))
    assert worst < 1e-12, f"disagrees with scipy by {worst}"

    return (f"Fig2a W1={a:.3f} (paper: 0.1) | Fig2b W1={b:.3f} (paper: 0.01) | "
            f"200 random pairs, max |ours-scipy| = {worst:.1e}")


def check_paths() -> str:
    from src import paths
    paths.ensure_dirs()
    missing = [n for n, p in paths.DIRS.items() if not p.is_dir()]
    if missing:
        raise RuntimeError(f"could not create: {missing}")

    # prove it is actually WRITABLE (Drive mounts sometimes are not)
    probe = paths.DIRS["logs"] / ".write_probe"
    probe.write_text("ok")
    probe.unlink()
    return f"platform={paths.PLATFORM} root={paths.ROOT} | {len(paths.DIRS)} dirs writable"


def check_disk() -> str:
    from src import paths
    return paths.disk_report()


def check_persistence() -> str:
    """
    Will what you write survive the session?

    Colab: only if the project root is on the mounted Drive.
    Kaggle: /kaggle/working is NOT persistent unless the notebook's
            Session options set Persistence = 'Files Only', or you finish with
            'Save & Run All (Commit)'. This is invisible from inside the VM,
            so on Kaggle this check can only WARN - you must confirm it yourself.
    """
    from src import paths
    st = paths.persistence_status()
    if st["persistent"] is True:
        return st["message"]
    if st["persistent"] is False:
        raise RuntimeError(st["message"])
    # Kaggle: cannot be determined from inside the notebook
    raise RuntimeError("KAGGLE PERSISTENCE UNVERIFIED - " + st["message"])


def check_mixed_precision() -> str:
    """
    Which reduced precision this GPU supports. This is not cosmetic.

    The debiasing loss is  L = L_LM + lambda * (1 - cos(h_bar, h_bar_tilde)).
    Cosine similarity divides by two norms, and with lambda up to 100 a small
    relative error in the cosine becomes a large error in the gradient.

    bf16 has only 8 mantissa bits, so on hardware that supports it we still
    train this project in fp32 unless a phase says otherwise. This check
    records what the GPU can do so that decision is explicit and reproducible,
    and it tells you not to 'helpfully' switch on bf16 later.
    """
    import torch
    if not torch.cuda.is_available():
        return "no GPU - fp32 on CPU (smoke test only)"

    major, minor = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    bf16 = torch.cuda.is_bf16_supported()

    notes = []
    if (major, minor) in [(7, 0)]:
        notes.append("P100 (sm_70): fp16 OK, NO bf16, no Tensor Cores")
    elif (major, minor) in [(7, 5)]:
        notes.append("T4 (sm_75): fp16 + Tensor Cores, NO bf16")
    elif major >= 8:
        notes.append("Ampere or newer: fp16 + bf16")
    else:
        notes.append(f"compute capability {major}.{minor}")

    decision = "PROJECT TRAINS IN fp32 (cosine-distance loss is precision-sensitive)"
    return (f"{name} | bf16_supported={bf16} | {'; '.join(notes)} | {decision}")


def check_nltk_data() -> str:
    """Phase 7/8 need nltk's punkt sentence tokenizer."""
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
        return "punkt_tab present"
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
        nltk.data.find("tokenizers/punkt_tab")
        return "punkt_tab downloaded just now"


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> int:
    print("=" * 78)
    print(" ENVIRONMENT CHECK — Sentiment Bias / Counterfactual Evaluation project")
    print("=" * 78)

    attempt("1. Python version",        check_python)
    attempt("2. PyTorch",               check_torch)
    attempt("3. CUDA / GPU",            check_cuda)
    attempt("4. Required libraries",    check_imports)
    attempt("5. HF hidden-states API",  check_transformers_api)
    attempt("6. Wasserstein-1 metric",  check_wasserstein)
    attempt("7. Project paths",         check_paths)
    attempt("8. Disk space",            check_disk, required=False)
    attempt("9. nltk punkt_tab",        check_nltk_data, required=False)

    # Persistence: a hard FAIL on Colab/local (you can fix it right now), but
    # only a WARNING on Kaggle, because Kaggle's Persistence switch lives in the
    # notebook UI and is genuinely invisible from inside the VM.
    from src import paths as _paths
    attempt("10. Persistence",          check_persistence,
            required=_paths.PLATFORM != "kaggle")
    attempt("11. Mixed precision",      check_mixed_precision, required=False)

    print("=" * 78)
    n_fail = sum(1 for _, s, _ in _results if s == FAIL)
    n_warn = sum(1 for _, s, _ in _results if s == WARN)
    n_pass = sum(1 for _, s, _ in _results if s == PASS)
    print(f" SUMMARY: {n_pass} passed, {n_warn} warnings, {n_fail} failed")

    if n_fail:
        print("\n DO NOT continue to Phase 2. Fix the FAIL items above first.")
        print(" Send the full output back and I will diagnose it.")
        return 1

    from src import paths as _paths
    if _paths.PLATFORM == "kaggle":
        print("\n Before you train anything on Kaggle, open Session options and")
        print(" set  Persistence -> 'Files Only'.  Without it, /kaggle/working is")
        print(" wiped when the session ends and you will lose every checkpoint.")

    print("\n All required checks passed. Phase 1 is complete —")
    print(" you are cleared for Phase 2 (Dataset).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
