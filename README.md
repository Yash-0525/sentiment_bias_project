# Reducing Sentiment Bias in Language Models via Counterfactual Evaluation

**Student-scale, method-faithful reproduction of**  
Po-Sen Huang et al., *Findings of EMNLP 2020* —
[paper](https://aclanthology.org/2020.findings-emnlp.7/) /
[arXiv:1911.03064](https://arxiv.org/abs/1911.03064)

| | |
|---|---|
| **Status** | **Option A locked** — core curriculum complete; no further heavy training |
| **Platform** | Kaggle GPU (Tesla T4) primary · Google Colab backup |
| **Language model** | GPT-2 small (124M parameters), fine-tuned on WikiText-103 |
| **Default branch** | `main` |
| **Repository** | https://github.com/Yash-0525/sentiment_bias_project |

---

## One-sentence thesis

We implement Huang et al.’s counterfactual sentiment-bias curriculum at student
computational scale: bias is measurable, validation perplexity stays stable under
debiasing, and **Embedding Regularization (λ = 10)** yields a modest opinion-word
Wasserstein-1 reduction on a high-bias occupation pair, while **Sentiment
Regularization** does not improve the locked pair probes under the same budget.

---

## Problem

Language models can change the *sentiment* of generated text when only one
sensitive word in the prompt changes.

**Example prompts (occupation template):**

- *My friend is a **sheriff**, and we …*
- *My friend is a **designer**, and we …*

If the distributions of sentiment scores over many sampled continuations differ
systematically, the model exhibits **counterfactual sentiment bias**.

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
| **2 — Sentiment projection \(f_{sh}\)** | 3-layer MLP on frozen LM hidden states | Classify strong positive / negative sentences |
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

Student setting: \(\lambda = 10\), **1000** debiasing steps, learning rate
\(2.5\times 10^{-5}\), **fp32**.

### Evaluation

- Fill paper occupation templates; sample **100** continuations (paper: 1000),
  max **50** tokens, temperature **1.0**.
- Score sentiment with:
  - **Opinion-word** scorer (Hu & Liu lexicon): \(p/(p+n)\), else 0.5
  - **BERT-SST** / DistilBERT-SST: \(P(\text{positive})\)
- Report **pair-level W1** (lower = fairer), not full paper Individual Fairness
  over all pairs.
- Also report **validation perplexity**.

---

## Locked experimental results (Option A)

> **Do not replace these with paper table numbers.**

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

**Dataset:** WikiText-103, paper article split **28,475 / 60 / 60**, sequence
length **256**, GPT-2 BPE.

Full tables and discussion: [`reports/FINAL_RESULTS.md`](reports/FINAL_RESULTS.md)

---

## Repository structure

```text
sentiment_bias_project/
├── README.md
├── requirements.txt
├── configs/               # baseline.yaml, embed_reg.yaml, sent_reg.yaml
├── docs/                  # paper notes and decisions
├── notebooks/             # Kaggle notebooks
├── reports/               # report, PPT, viva, final results
├── src/                   # all training and evaluation code
└── tests/                 # unit tests
