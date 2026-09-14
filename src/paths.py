"""
src/paths.py  —  Phase 1

Single source of truth for WHERE files live.

Why this file exists
--------------------
Both Colab and Kaggle wipe the VM when a session ends. If a 2-hour training run
writes its checkpoint somewhere ephemeral and the session dies, the model is gone.

The rule for this whole project:

    the notebook EXECUTES  ->  persistent storage STORES  ->  Git VERSIONS

Every path in the project is resolved through this module, so the identical code
runs on Colab, Kaggle, or a laptop, and only this file changes.

------------------------------------------------------------------------------
 PLATFORM        WRITABLE ROOT                PERSISTENT?   WHAT TO DO
------------------------------------------------------------------------------
 colab           /content/drive/MyDrive/...   YES, always   mount Drive (auto)
 kaggle          /kaggle/working/...          ONLY IF you   Session options ->
                                              opt in        Persistence =
                                                            "Files Only", or
                                                            "Save & Run All"
 local           the git checkout             YES           nothing
------------------------------------------------------------------------------

Kaggle is the awkward one: /kaggle/working is NOT persistent by default.
`persistence_status()` tells you which regime you are in, and `save_state()` /
`load_state()` give you a checkpoint path that works on all three platforms.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path


# --------------------------------------------------------------------------
# 1. Detect the platform
# --------------------------------------------------------------------------
def detect_platform() -> str:
    """Return 'colab', 'kaggle', or 'local'."""
    if "google.colab" in sys.modules:
        return "colab"
    # /kaggle/working exists on Kaggle and nowhere else
    if os.path.isdir("/kaggle/working"):
        return "kaggle"
    return "local"


PLATFORM = detect_platform()

KAGGLE_WORKING = Path("/kaggle/working")
KAGGLE_TEMP = Path("/kaggle/temp")
COLAB_DRIVE_ROOT = Path("/content/drive/MyDrive/sentiment_bias_project")


# --------------------------------------------------------------------------
# 2. Resolve the project root
# --------------------------------------------------------------------------
def _default_root() -> Path:
    # Escape hatch: SBP_ROOT=/some/path overrides everything.
    override = os.environ.get("SBP_ROOT")
    if override:
        return Path(override)

    if PLATFORM == "colab":
        return COLAB_DRIVE_ROOT

    if PLATFORM == "kaggle":
        return KAGGLE_WORKING / "sentiment_bias_project"

    # local: <root>/src/paths.py  ->  walk up one level
    return Path(__file__).resolve().parent.parent


ROOT = _default_root()


# --------------------------------------------------------------------------
# 3. The directory tree (Phase 1 structure)
# --------------------------------------------------------------------------
DIRS = {
    "data":              ROOT / "data",
    "notebooks":         ROOT / "notebooks",
    "src":               ROOT / "src",
    "models":            ROOT / "models",
    "models_baseline":   ROOT / "models" / "baseline",
    "models_embed_reg":  ROOT / "models" / "embedding_reg",
    "models_sent_reg":   ROOT / "models" / "sentiment_reg",
    "results":           ROOT / "results",
    "results_baseline":  ROOT / "results" / "baseline",
    "results_embed_reg": ROOT / "results" / "embedding_reg",
    "results_sent_reg":  ROOT / "results" / "sentiment_reg",
    "plots":             ROOT / "plots",
    "reports":           ROOT / "reports",
    "configs":           ROOT / "configs",
    "logs":              ROOT / "logs",
    "docs":              ROOT / "docs",
}


def ensure_dirs() -> None:
    """Create the whole tree. Idempotent - safe to call on every run."""
    for p in DIRS.values():
        p.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# 4. Convenience accessors
# --------------------------------------------------------------------------
def data_path(*parts: str) -> Path:
    return DIRS["data"].joinpath(*parts)


def model_path(*parts: str) -> Path:
    return DIRS["models"].joinpath(*parts)


def results_path(*parts: str) -> Path:
    return DIRS["results"].joinpath(*parts)


def plot_path(*parts: str) -> Path:
    return DIRS["plots"].joinpath(*parts)


def checkpoint_dir(run_name: str) -> Path:
    """
    Per-run checkpoint directory, e.g. checkpoint_dir('baseline') ->
        <root>/models/baseline/

    Every training script writes here. Never write a checkpoint to a bare
    '/content/...' or '/kaggle/working/...' string anywhere else in the project.
    """
    ensure_dirs()
    d = DIRS["models"] / run_name
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------
# 5. Platform bootstrap: get the code + persistent storage in place
# --------------------------------------------------------------------------
def bootstrap(repo_url: str | None = None, verbose: bool = True) -> Path:
    """
    Idempotent session bootstrap. Call this ONCE at the top of every notebook.

    Colab : mounts Google Drive, clones the repo to /content for fast imports,
            and symlinks the heavy directories onto Drive so artefacts persist.
    Kaggle: clones the repo into /kaggle/working (writable) and creates the
            directory tree. Persistence then depends on the notebook's
            Persistence setting - see persistence_status().
    local : creates the tree and returns.

    Returns the project root.
    """
    def log(msg: str) -> None:
        if verbose:
            print(f"[bootstrap] {msg}")

    if PLATFORM == "local":
        ensure_dirs()
        log(f"local checkout at {ROOT}")
        return ROOT

    if PLATFORM == "colab":
        ok = mount_drive(quiet=not verbose)
        if not ok:
            log("WARNING: no Drive mount - artefacts will be lost on disconnect")
        content_root = Path("/content/sentiment_bias_project")
        if repo_url and not content_root.exists():
            os.system(f"git clone --quiet {repo_url} {content_root}")
            log(f"cloned to {content_root}")
        _link_heavy_dirs(content_root, COLAB_DRIVE_ROOT, verbose)
        ensure_dirs()
        return ROOT

    if PLATFORM == "kaggle":
        target = KAGGLE_WORKING / "sentiment_bias_project"
        if repo_url and not target.exists():
            os.system(f"git clone --quiet {repo_url} {target}")
            log(f"cloned to {target}")
        ensure_dirs()
        status = persistence_status()
        log(status["message"])
        return ROOT

    ensure_dirs()
    return ROOT


def _link_heavy_dirs(content_root: Path, drive_root: Path, verbose: bool) -> None:
    """Colab: symlink data/models/results/plots/logs from /content onto Drive."""
    if not drive_root.is_dir():
        return
    for name in ("data", "models", "results", "plots", "logs"):
        src, dst = drive_root / name, content_root / name
        src.mkdir(parents=True, exist_ok=True)
        # lexists, not exists: it also catches a symlink whose target vanished
        if not os.path.lexists(dst):
            if dst.exists():
                shutil.rmtree(dst)
            os.symlink(src, dst)
            if verbose:
                print(f"[bootstrap] linked {dst} -> {src}")


def mount_drive(quiet: bool = False) -> bool:
    """Colab: mount Google Drive. No-op elsewhere. True if Drive is present."""
    if PLATFORM != "colab":
        return False

    drive_root = Path("/content/drive")
    try:
        if drive_root.is_dir() and any(drive_root.iterdir()):
            if not quiet:
                print(f"[paths] Google Drive already mounted at {drive_root}")
            return True
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive")
        if not quiet:
            print("[paths] Google Drive mounted at /content/drive")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[paths] WARNING: could not mount Drive ({exc!r}).")
        return False


# --------------------------------------------------------------------------
# 6. Persistence: is anything you write going to survive?
# --------------------------------------------------------------------------
def persistence_status() -> dict:
    """
    Return {'persistent': bool, 'regime': str, 'message': str}.

    Colab with Drive mounted  -> persistent, always.
    Kaggle                    -> persistent ONLY if the notebook's Session
                                 options have Persistence set to "Files Only"
                                 or "Variables and Files". We cannot read that
                                 setting from inside the VM, so on Kaggle we
                                 report it as UNVERIFIED and tell you to check.
    local                     -> persistent.
    """
    if PLATFORM == "colab":
        mounted = Path("/content/drive").is_dir() and str(ROOT).startswith("/content/drive")
        if mounted:
            return {"persistent": True, "regime": "colab-drive",
                    "message": f"Google Drive mounted; root on Drive ({ROOT}) - artefacts survive."}
        return {"persistent": False, "regime": "colab-ephemeral",
                "message": "Root is on the ephemeral VM. Artefacts WILL BE LOST. "
                           "Run paths.mount_drive() first."}

    if PLATFORM == "kaggle":
        return {"persistent": None, "regime": "kaggle-conditional",
                "message":
                    "Kaggle: /kaggle/working persists ONLY if you set "
                    "Session options -> Persistence -> 'Files Only' (or "
                    "'Variables and Files'), OR you end the run with "
                    "'Save Version -> Save & Run All (Commit)'. "
                    "This cannot be detected from inside the notebook - CHECK IT."}

    return {"persistent": True, "regime": "local",
            "message": f"Local checkout at {ROOT} - persistent."}


# --------------------------------------------------------------------------
# 7. Checkpoint save / load that works on all three platforms
# --------------------------------------------------------------------------
def save_state(obj: dict, run_name: str, tag: str = "latest") -> Path:
    """
    Write a checkpoint under <root>/models/<run_name>/<tag>.json.

    Used for small JSON-able state (config, metrics, step counters) so a run
    can be resumed. Model weights are saved by the training script itself via
    torch.save into checkpoint_dir(run_name).
    """
    d = checkpoint_dir(run_name)
    path = d / f"{tag}.json"
    path.write_text(json.dumps(obj, indent=2, default=str))
    return path


def load_state(run_name: str, tag: str = "latest") -> dict | None:
    """Return a checkpoint written by save_state(), or None if absent."""
    path = DIRS["models"] / run_name / f"{tag}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


# --------------------------------------------------------------------------
# 8. Disk space report
# --------------------------------------------------------------------------
def disk_report() -> str:
    target = ROOT if ROOT.exists() else Path("/")
    usage = shutil.disk_usage(target)
    gb = 1024 ** 3
    return (f"{target}  total {usage.total/gb:6.1f} GB | "
            f"used {usage.used/gb:6.1f} GB | free {usage.free/gb:6.1f} GB")


def project_size() -> str:
    """How much of the disk quota this project is eating (Kaggle caps you)."""
    if not ROOT.exists():
        return "project root does not exist yet"
    total = sum(f.stat().st_size for f in ROOT.rglob("*") if f.is_file())
    return f"{ROOT} = {total/1024**2:.1f} MB"


if __name__ == "__main__":
    ensure_dirs()
    print(f"platform      : {PLATFORM}")
    print(f"project root  : {ROOT}")
    print(f"persistence   : {persistence_status()['message']}")
    print(f"disk          : {disk_report()}")
    print(f"project size  : {project_size()}")
    print("\ndirectories:")
    for name, p in DIRS.items():
        print(f"  {name:18s} {p}   {'OK' if p.is_dir() else 'MISSING'}")
