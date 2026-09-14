"""
tests/test_kaggle_paths.py

Exercises the Kaggle branch of src/paths.py and src/check_environment.py.

We cannot create /kaggle/working without root, so we simulate the platform by
overriding os.path.isdir and pointing SBP_ROOT at a scratch dir. Every line of
production code below is the REAL code from paths.py / check_environment.py -
nothing is reimplemented.
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FAKE_KAGGLE = Path(tempfile.mkdtemp(prefix="fake_kaggle_")) / "working"
FAKE_KAGGLE.mkdir(parents=True)

_real_isdir = os.path.isdir


def fake_isdir(p) -> bool:
    """Make /kaggle/working appear to exist -> detect_platform() returns 'kaggle'."""
    if str(p) == "/kaggle/working":
        return True
    return _real_isdir(p)


os.path.isdir = fake_isdir
os.environ["SBP_ROOT"] = str(FAKE_KAGGLE / "sentiment_bias_project")

passed, failed = [], []


def check(name, cond, detail=""):
    (passed if cond else failed).append(name)
    print(f"{'PASS' if cond else 'FAIL'} | {name} | {detail}")


# ---------------------------------------------------------------- paths.py
import src.paths as P  # noqa: E402
importlib.reload(P)

check("detect_platform() == 'kaggle'", P.detect_platform() == "kaggle",
      f"got {P.detect_platform()!r}")
check("ROOT is under the fake /kaggle/working",
      str(P.ROOT) == str(FAKE_KAGGLE / "sentiment_bias_project"), f"ROOT={P.ROOT}")

P.ensure_dirs()
check("ensure_dirs() created all 16 dirs",
      len(P.DIRS) == 16 and all(d.is_dir() for d in P.DIRS.values()),
      f"{sum(d.is_dir() for d in P.DIRS.values())}/{len(P.DIRS)}")

# the default root Kaggle would use, with the override removed
del os.environ["SBP_ROOT"]
importlib.reload(P)
check("default Kaggle root == /kaggle/working/sentiment_bias_project",
      str(P._default_root()) == "/kaggle/working/sentiment_bias_project",
      f"got {P._default_root()}")

os.environ["SBP_ROOT"] = str(FAKE_KAGGLE / "sentiment_bias_project")
importlib.reload(P)

st = P.persistence_status()
check("persistence_status()['persistent'] is None on Kaggle (unverifiable)",
      st["persistent"] is None, f"got {st['persistent']!r}")
check("persistence_status()['regime'] == 'kaggle-conditional'",
      st["regime"] == "kaggle-conditional", f"got {st['regime']!r}")
check("persistence message mentions Session options",
      "Session options" in st["message"], st["message"][:70] + "...")

# checkpoint_dir
d = P.checkpoint_dir("baseline")
check("checkpoint_dir('baseline') -> models/baseline",
      d == P.DIRS["models"] / "baseline" and d.is_dir(), str(d))

# save_state / load_state round-trip
payload = {"step": 1500, "val_loss": 3.41, "lambda": 10, "attrs": ["country", "name"]}
p = P.save_state(payload, "baseline", "step1500")
back = P.load_state("baseline", "step1500")
check("save_state/load_state round-trips exactly", back == payload, f"read back {back}")
check("load_state returns None for a missing tag",
      P.load_state("baseline", "nope") is None)

# bootstrap() kaggle branch, without a repo_url so it does not hit the network
root = P.bootstrap(repo_url=None, verbose=False)
check("bootstrap() returns ROOT on Kaggle", root == P.ROOT, str(root))

# mount_drive must be a clean no-op off Colab (this was the old bug:
# `return os.path.isdir(...) or True` -> always True)
check("mount_drive() returns False on Kaggle (not spuriously True)",
      P.mount_drive(quiet=True) is False, f"got {P.mount_drive(quiet=True)!r}")

# ------------------------------------------------------- check_environment
import src.check_environment as CE  # noqa: E402
importlib.reload(CE)

raised = False
try:
    CE.check_persistence()
except RuntimeError as e:
    raised = True
    msg = str(e)
check("check_persistence() raises on Kaggle (forces the WARN)", raised,
      msg[:60] + "..." if raised else "did not raise")

# the platform-detection line inside check_environment's main()
check("check_environment sees PLATFORM == 'kaggle'", P.PLATFORM == "kaggle")

# ---------------------------------------------------------------- summary
print("\n" + "=" * 70)
print(f" {len(passed)} passed, {len(failed)} failed")
if failed:
    print(" FAILED:", failed)

shutil.rmtree(FAKE_KAGGLE.parent, ignore_errors=True)
sys.exit(1 if failed else 0)
