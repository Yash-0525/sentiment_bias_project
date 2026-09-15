# Reducing Sentiment Bias in Language Models via Counterfactual Evaluation
### A reproduction of Huang et al., *Findings of EMNLP 2020*, at student scale

---

## What this project does

Given a prompt like *"My friend is a/an **baker**, and we..."*, a language model generates
continuations. Change one word to **accountant** and the *sentiment distribution* of those
continuations shifts. That shift is **counterfactual sentiment bias**.

This project measures it with Wasserstein-1-based **Individual Fairness** and **Group Fairness**
metrics, then reduces it with two regularizers on the model's hidden representations —
**Embedding Regularization** and **Sentiment Regularization** — trained via the paper's
three-step curriculum.

---

## Read these first

| File | What it is |
|---|---|
| `docs/PHASE_0_paper_understanding.md` | The paper's methodology, explained, with page citations |
| `docs/DECISIONS_LOG.md` | **Every design decision and why.** Read before changing anything. |
| `docs/DATASET_VERIFICATION.md` | Live verification of WikiText-103 and WMT-19 availability |

## The operating rule

> **The notebook executes. Persistent storage stores. Git versions.**

Colab and Kaggle both wipe the VM when a session ends. Nothing valuable is ever written to an
ephemeral path alone. Source code lives in git.

| Platform | Writable root | Persistent? | What you must do |
|---|---|---|---|
| **Colab** | `/content/drive/MyDrive/...` | **Always**, once Drive is mounted | `paths.mount_drive()` (bootstrap does it) |
| **Kaggle** | `/kaggle/working/...` | **Only if you opt in** | Session options → Persistence → **Files Only**, *or* end with **Save & Run All (Commit)** |
| local | the git checkout | Always | nothing |

Kaggle's `/kaggle/working` is **not** persistent by default — this is the difference people
lose work to. See `notebooks/kaggle_bootstrap.ipynb`.

## Quick start — Colab

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/YOUR-USERNAME/sentiment_bias_project.git /content/sentiment_bias_project

import os
for d in ['data','models','results','plots','logs']:
    os.makedirs(f'/content/drive/MyDrive/sentiment_bias_project/{d}', exist_ok=True)
    if not os.path.lexists(f'/content/sentiment_bias_project/{d}'):
        os.symlink(f'/content/drive/MyDrive/sentiment_bias_project/{d}',
                   f'/content/sentiment_bias_project/{d}')

%cd /content/sentiment_bias_project
%pip install -r requirements.txt
!python -m src.check_environment
```
Then: **Runtime → Change runtime type → T4 GPU**.

## Quick start — Kaggle

Upload `notebooks/kaggle_bootstrap.ipynb` and run it top to bottom. In short:

1. **Accelerator → GPU T4 x2 or P100** (needs phone verification on your account).
2. **Session options → Persistence → `Files Only`.**  ← do not skip this
3. `!git clone <repo> /kaggle/working/sentiment_bias_project`
4. `%pip install -r requirements.txt`  (do **not** pip-install torch — it's preinstalled with CUDA)
5. `!python -m src.check_environment`
6. End of session: **Save Version → Save & Run All (Commit)**, or download `models/` + `results/`.

## Tests

```
python tests/test_kaggle_paths.py     # 14 checks on the Kaggle code path
python -m src.check_environment       # 11 environment checks
```

## Phase status

See the checklist at the bottom of `docs/DECISIONS_LOG.md`.
