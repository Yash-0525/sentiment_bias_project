# Reducing Sentiment Bias in Language Models via Counterfactual Evaluation

**Student-scale, method-faithful reproduction of**  
Po-Sen Huang et al., *Findings of EMNLP 2020* — [paper](https://aclanthology.org/2020.findings-emnlp.7/) / [arXiv:1911.03064](https://arxiv.org/abs/1911.03064)

| | |
|---|---|
| **Status** | **Option A locked** — core curriculum complete; no further heavy training |
| **Platform** | Kaggle GPU (Tesla T4) primary · Google Colab backup |
| **Language model** | GPT-2 small (124M parameters), fine-tuned on WikiText-103 |
| **Default branch** | `main` |
| **Repository** | https://github.com/Yash-0525/sentiment_bias_project |

---

## One-sentence thesis

We implement Huang et al.’s counterfactual sentiment-bias curriculum at student computational scale: bias is measurable, validation perplexity stays stable under debiasing, and **Embedding Regularization (λ = 10)** yields a modest opinion-word Wasserstein-1 reduction on a high-bias occupation pair, while **Sentiment Regularization** does not improve the locked pair probes under the same budget.

---

## Problem

Language models can change the *sentiment* of generated text when only one sensitive word in the prompt changes.

**Example prompts (occupation template):**
- *My friend is a **sheriff**, and we …*
- *My friend is a **designer**, and we …*

If the distributions of sentiment scores over many sampled continuations differ systematically, the model exhibits **counterfactual sentiment bias**.

This project:
1. **Measures** that bias with Wasserstein-1 (W1) distance between sentiment distributions.
2. **Reduces** it with two training regularizers from the paper (Embedding Reg and Sentiment Reg).
3. **Preserves** language-model quality (validation perplexity).

---

## Method (paper curriculum, student scale)

### Three-step curriculum

| Step | What is trained | Loss |
|------|-----------------|------|
| **1 — Baseline LM** | GPT-2 small on WikiText-103 | Next-token cross-entropy only |
| **2 — Sentiment projection \(f_{sh}\)** | 3-layer MLP on frozen LM hidden states | Classify strong positive/negative sentences |
| **3 — Debiasing** | Continue LM on sequences that contain sensitive tokens | \(L = L_{LM}(x) + \lambda L_{\mathrm{fair}}\) |

**Critical paper rules we follow:**
- Evaluation templates are **never** used in any training step.
- \(L_{LM}\) is computed on the **unperturbed** input \(x\) only.
- Counterfactual \(\tilde{x}\) changes **only** the sensitive surface form.
- \(\bar{h}\) = mean of the **last two** Transformer layers.

### Two regularizers (Step 3)

| Method | Fairness term |
|--------|----------------|
| **Embedding Regularization** | \(L_{\mathrm{fair}} = 1 - \cos(\bar{h}(x), \bar{h}(\tilde{x}))\) |
| **Sentiment Regularization** | \(L_{\mathrm{fair}} = 1 - \cos(f_{sh}(\bar{h}(x)), f_{sh}(\bar{h}(\tilde{x})))\) with \(f_{sh}\) **frozen** |

Student setting: \(\lambda = 10\), **1000** debiasing steps, learning rate \(2.5\times 10^{-5}\), **fp32**.

### Evaluation

- Fill paper occupation templates; sample **100** continuations (paper: 1000), max **50** tokens, temperature **1.0**.
- Score sentiment with:
  - **Opinion-word** scorer (Hu & Liu lexicon): \(p/(p+n)\), else 0.5
  - **BERT-SST** / DistilBERT-SST: \(P(\text{positive})\)
- Report **pair-level W1** (lower = fairer), not full paper Individual Fairness over all pairs.
- Also report **validation perplexity**.

---

## Locked experimental results (Option A)

**Do not replace these with paper table numbers.**

### Language modeling quality

| Model | Validation PPL |
|-------|----------------|
| Baseline (3000 steps) | **22.2753** |
| Embedding Reg λ=10 (1000 steps) | **22.2381** |
| Sentiment Reg λ=10 (1000 steps) | **22.2984** |

Debiasing did **not** destroy fluency (all ≈ 22.2–22.3).

### Sentiment projection \(f_{sh}\)

| Metric | Value |
|--------|--------|
| Validation accuracy | **0.8413** |
| Architecture | MLP 768 → 128 → 128 → 2 |
| Labels | BERT-SST with \(\lvert 2p-1\rvert > 0.7\) (Google API substitute) |

### Counterfactual construction audit

| Check | Result |
|-------|--------|
| Successful sensitive-token swaps | **497 / 500 = 99.4%** |

### Pair probes (occupation template 4, n = 100)

**Baker vs accountant** (small baseline gap)

| Model | Opinion W1 | BERT-SST W1 |
|-------|------------|-------------|
| Baseline | 0.0377 | 0.0473 |
| Embed-Reg λ=10 | 0.0923 | 0.0472 |
| Sent-Reg λ=10 | 0.0482 | 0.0980 |

**Sheriff vs designer** (high baseline bias — primary signal)

| Model | Opinion W1 | BERT-SST W1 |
|-------|------------|-------------|
| Baseline | **0.1635** | **0.1793** |
| **Embed-Reg λ=10** | **0.1422** (Δ = −0.0213) | 0.1803 |
| Sent-Reg λ=10 | 0.1787 | 0.2289 |

### Model selection (honest)

| Criterion | Choice |
|-----------|--------|
| Best fairness on locked probes | **Embedding Reg λ=10** (sheriff–designer, opinion-word) |
| Best / stable PPL | **Embedding Reg λ=10** (22.24) |
| Sentiment Reg at λ=10 / 1k steps | **Not recommended** from these probes alone |
| Full paper I.F. / G.F. / S.S. | **Not measured** in Option A lock |

**Dataset:** WikiText-103, paper article split **28,475 / 60 / 60**, sequence length **256**, GPT-2 BPE.

Full tables and discussion: [`reports/FINAL_RESULTS.md`](reports/FINAL_RESULTS.md)

---

## Repository structure
sentiment_bias_project/
├── README.md ← this file
├── requirements.txt
├── configs/ # training YAML
│ ├── baseline.yaml
│ ├── embed_reg.yaml
│ └── sent_reg.yaml
├── docs/ # paper notes & decisions
│ ├── PHASE_0_paper_understanding.md
│ ├── DECISIONS_LOG.md
│ └── DATASET_VERIFICATION.md
├── notebooks/ # Kaggle notebooks
│ ├── kaggle_bootstrap.ipynb
│ ├── kaggle_phase2_dataset.ipynb
│ ├── kaggle_phase5_baseline.ipynb
│ └── (optional) kaggle_FULL_curriculum.ipynb
├── reports/ # submission pack
│ ├── PROJECT_REPORT.md
│ ├── PPT_OUTLINE_BULLETS.md
│ ├── PRESENTATION_SLIDES.md
│ ├── VIVA_QA.md
│ ├── FINAL_RESULTS.md
│ └── OPTION_A_LOCK.md
├── src/ # all code
└── tests/ # unit tests


**Not stored in git (by design — too large):**
- `data/` — WikiText packs  
- `models/` — trained weights  
- generated `results/*.csv`, plots, checkpoints  

Those live on **Kaggle Save Version → Commit**. Written results live in `reports/`.

---

## Documentation pack (for report / PPT / viva)
| File | Purpose |
|------|---------|
| [`reports/PROJECT_REPORT.md`](reports/PROJECT_REPORT.md) | Full project report (sections 1–29) |
| [`reports/PPT_OUTLINE_BULLETS.md`](reports/PPT_OUTLINE_BULLETS.md) | 15-slide PowerPoint outline — bullets only |
| [`reports/PRESENTATION_SLIDES.md`](reports/PRESENTATION_SLIDES.md) | Slides + speaker notes |
| [`reports/VIVA_QA.md`](reports/VIVA_QA.md) | Oral examination Q&A |
| [`reports/FINAL_RESULTS.md`](reports/FINAL_RESULTS.md) | Master results tables + paper vs ours |
| [`reports/OPTION_A_LOCK.md`](reports/OPTION_A_LOCK.md) | Allowed vs forbidden claims |

### Design notes

| File | Purpose |
|------|---------|
| [`docs/PHASE_0_paper_understanding.md`](docs/PHASE_0_paper_understanding.md) | Paper methodology with citations |
| [`docs/DECISIONS_LOG.md`](docs/DECISIONS_LOG.md) | Every scale / substitution decision |
| [`docs/DATASET_VERIFICATION.md`](docs/DATASET_VERIFICATION.md) | WikiText-103 / WMT-19 verification |

---

## Source code (`src/`)

| Module | Role |
|--------|------|
| `paths.py` | Kaggle / Colab / local path bootstrap |
| `check_environment.py` | GPU, deps, persistence checks |
| `sensitive_attributes.py` | Paper Appendix A lists (country, occupation, name) |
| `templates.py` | Evaluation templates (eval only) |
| `data_preprocessing.py` | WikiText-103 → packed length-256 sequences |
| `dataset_lm.py` | LM dataset loaders |
| `model.py` | GPT-2 wrapper + \(\bar{h}\) (last 2 layers) |
| `train_baseline.py` | Step 1 baseline LM |
| `sentiment_classifier.py` | Step 2 \(f_{sh}\) |
| `sentiment_scorers.py` | Opinion-word + BERT-SST scorers |
| `counterfactual.py` | Sensitive-token counterfactual construction |
| `train_debias.py` | Step 3 Embedding / Sentiment regularization |
| `generate.py` | Continuation sampling |
| `evaluate_baseline_bias.py` | Generate + score + pair W1 |

---

## Quick start — Kaggle (recommended)

### Right-panel settings
1. **Accelerator → GPU T4 x2 or P100**
2. **Persistence → Files Only**
3. **Internet → On**

### Setup cell

```python
!git clone --branch main --single-branch \
  https://github.com/Yash-0525/sentiment_bias_project.git \
  /kaggle/working/sentiment_bias_project

%cd /kaggle/working/sentiment_bias_project
%pip install --quiet -r requirements.txt
# Do NOT run: pip install torch  (breaks CUDA build on Kaggle)

!python -m src.check_environment
