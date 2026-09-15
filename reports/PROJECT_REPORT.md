# Reducing Sentiment Bias in Language Models via Counterfactual Evaluation  
### A Method-Faithful Reproduction at Student Computational Scale

**Base paper:** Po-Sen Huang, Huan Zhang, Ray Jiang, Robert Stanforth, Johannes Welbl, Jack W. Rae, Vishal Maini, Dani Yogatama, and Pushmeet Kohli.  
*Reducing Sentiment Bias in Language Models via Counterfactual Evaluation.* Findings of EMNLP 2020, pages 65–83.

**Implementation setting:** STUDENT / COMPUTE-LIMITED (Option A) — Kaggle GPU (Tesla T4), GPT-2 small, WikiText-103 only.  
**Code:** `https://github.com/Yash-0525/sentiment_bias_project` · branch `arena/01a0a078-sentiment-bias-project`  
**Result lock date:** 2026-09-15  

> **Integrity note.** Every experimental number in this report comes from our runs (see `reports/FINAL_RESULTS.md`). Paper numbers appear only as *reference*, never as our results.

---

## 1. Title

**Reducing Sentiment Bias in Language Models via Counterfactual Evaluation: A Method-Faithful Reproduction at Student Computational Scale**

---

## 2. Abstract

Language models can systematically change the *sentiment* of generated text when only a sensitive attribute value in the prompt is changed (for example, *sheriff* versus *designer*). Huang et al. (Findings of EMNLP 2020) formalize this phenomenon as **counterfactual sentiment bias**, measure it with Wasserstein-1 (W1) distances between sentiment distributions (Individual and Group Fairness), and reduce it with two representation regularizers—**Embedding Regularization** and **Sentiment Regularization**—trained in a three-step curriculum.

This project reproduces that **methodology** at a scale executable on free student hardware. We use WikiText-103 with the paper’s article split (28,475 / 60 / 60), fine-tune GPT-2 small (124M parameters) as the language model, train a three-layer sentiment projection \(f_{sh}\), and continue training with both regularizers at \(\lambda=10\) for 1,000 steps. Counterfactual token swaps succeed on **99.4%** of audited training sequences. Validation perplexity remains stable near **22.3** for baseline and debiased models. Baseline bias is strong on the sheriff–designer pair (opinion-word W1 **0.1635**). Embedding regularization reduces that W1 to **0.1422** while slightly improving PPL (**22.24**). Sentiment regularization at the same budget does **not** improve the locked pair probes. The baker–accountant pair shows no fairness gain. We report **pair-level W1 probes** rather than full paper I.F. tables, and we document every scale substitution.

**Keywords:** language models, fairness, counterfactual evaluation, Wasserstein distance, sentiment bias, GPT-2, WikiText-103.

---

## 3. Introduction

### 3.1 Context

Modern autoregressive language models generate fluent text by imitating large corpora. When corpora contain social stereotypes, models can reproduce them—not only as word associations, but as **shifts in the emotional tone** of entire generated continuations.

### 3.2 Operational problem

Consider two prompts that differ by one occupation word:

- *My friend is a **sheriff**, and we …*  
- *My friend is a **designer**, and we …*

If the model’s continuations are systematically more negative for one occupation than the other, the user experiences a **sentiment bias** that depends on a sensitive attribute. Huang et al. show this effect for occupations, countries, and names, and propose both metrics and training-time remedies.

### 3.3 Why a student reproduction?

Full paper training used large Transformer-XL models and **128 Google Cloud TPUv3 cores**. A final-year project cannot match that budget. The scientific target of this work is therefore **method fidelity**: same counterfactual protocol, same fairness losses, same curriculum order, same evaluation templates reserved for evaluation only—with model size, steps, and sample counts reduced and **labelled**.

### 3.4 Contributions of this project

1. An end-to-end, runnable implementation of Huang et al.’s curriculum on Kaggle.  
2. Verification that counterfactual construction works at **99.4%** success on sensitive WikiText sequences.  
3. Demonstration of **baseline counterfactual bias** (especially sheriff vs designer).  
4. Evidence that **Embedding Regularization** can modestly reduce opinion-word W1 on a high-bias pair **without harming PPL**.  
5. An honest negative/partial result for **Sentiment Regularization** at \(\lambda=10\), 1,000 steps.  
6. Full documentation of paper-versus-student gaps for viva and examiners.

---

## 4. Problem Statement

**Given**

- an autoregressive language model \(p_\theta(x_{t+1}\mid x_{1:t})\),  
- a sensitive attribute with value set \(\mathcal{A}\) and token sets \(\phi(a)\),  
- a sentiment scoring function \(f_s\) mapping text to \([0,1]\),

**define** the sentiment distribution \(P_S(x)\) as the distribution of \(f_s\) over continuations sampled from the LM conditioned on prefix \(x\).

**Measure** whether \(P_S(x)\) and \(P_S(\tilde{x})\) differ when \(\tilde{x}\) is a **counterfactual** of \(x\) (only sensitive tokens changed).

**Reduce** that difference by regularizing hidden representations of \(x\) and \(\tilde{x}\) during continued LM training, subject to retaining acceptable language-model quality (perplexity).

---

## 5. Motivation

1. **Deployment risk.** Generated text appears in chat, creative tools, and content systems; biased tone can reinforce stereotypes.  
2. **Measurability.** “The model feels biased” is not enough; W1 on counterfactual pairs is a concrete, threshold-free distance.  
3. **Trainable fix.** Fairness only at decoding time is limited; Huang et al. regularize the representation space during training.  
4. **Educational value.** Implementing the full curriculum forces understanding of LM training, hidden states, fairness metrics, and experimental honesty under compute limits.

---

## 6. Objectives

| # | Objective | Status in this project |
|---|-----------|-------------------------|
| O1 | Implement counterfactual sentiment evaluation (templates, generation, scorers, W1) | **Done** |
| O2 | Train baseline LM on WikiText-103 with paper article split | **Done** (PPL 22.28) |
| O3 | Show non-trivial baseline bias | **Done** (sheriff–designer W1 ≈ 0.16) |
| O4 | Train \(f_{sh}\) on frozen LM hidden states | **Done** (acc 0.84) |
| O5 | Train Embedding and Sentiment Regularization models | **Done** (\(\lambda=10\)) |
| O6 | Compare models on W1 and PPL | **Done** (pair probes) |
| O7 | Document substitutions vs paper | **Done** |
| O8 | Full I.F./G.F./S.S./human eval | **Not in Option A lock** (future work) |

---

## 7. Literature Review

### 7.1 Fairness in machine learning

Classical notions include demographic parity and equalized odds. **Individual fairness** (Dwork et al., 2012) requires similar individuals to receive similar outcomes. **Wasserstein-based fairness** (Jiang et al., 2019) connects threshold-averaged demographic disparity to the Wasserstein-1 distance between score distributions—exactly the justification Huang et al. use for sentiment scores in \([0,1]\).

### 7.2 Bias in language representations and generation

Early work documented gender stereotypes in static embeddings (e.g., Bolukbasi et al.). Later work studied bias in contextual models and in **generated** text (Sheng et al.; Solaiman et al.; GPT-2 analyses). Generation bias is harder than classification bias because the output is open-ended text; one must define a measurable property of that text (here: sentiment).

### 7.3 Sentiment as a specification

Sentiment is an incomplete fairness specification—it does not capture all harm—but it is continuous, widely instrumented, and suitable for distributional comparison. Huang et al. explicitly treat sentiment as one **specification** among possible others.

### 7.4 Gap addressed by the base paper

Prior bias work often focused on classification or embedding geometry. Huang et al. target **conditional generation**, define counterfactual fairness-style metrics for sentiment distributions, and propose **training-time** regularizers on LM hidden states with a practical curriculum.

---

## 8. Base Paper (detailed)

**Citation:** Huang et al., Findings of EMNLP 2020, pp. 65–83. arXiv:1911.03064.

### 8.1 Problem formulation

Sensitive attributes: **Country** (10), **Occupation** (29), **Name** (17 male + 17 female).  
Evaluation: \(M=10\) templates per attribute (Appendix A).  
Generation: 1,000 samples per template per value, max 50 tokens, temperature 1.0.

### 8.2 Metrics

- **Wasserstein-1** between sentiment distributions of counterfactual pairs.  
- **Individual Fairness (I.F.):** average W1 over templates and pairs of attribute values.  
- **Group Fairness (G.F.):** average W1 between each subgroup distribution and the global distribution.  
- **PPL / PPL_s:** overall and sensitive-subset perplexity.  
- **S.S. / S.S.c:** semantic similarity to prefix; mention rate of sensitive token.

### 8.3 Methods

1. **Embedding Regularization:** \(h̄=\mathrm{mean}(h^{(L-1)},h^{(L)})\), \(L_{\mathrm{fair}}=1-\cos(h̄(x),h̄(\tilde{x}))\).  
2. **Sentiment Regularization:** same cosine distance after \(f_{sh}\) projection; \(f_{sh}\) frozen in Step 3.

### 8.4 Three-step curriculum

1. Train LM with next-token loss only.  
2. Freeze LM; train \(f_{sh}\) (3-layer MLP, hidden 128) on sentences with strong sentiment labels.  
3. Continue LM training on sequences containing sensitive tokens with \(L=L_{LM}+\lambda L_{\mathrm{fair}}\).  
**Critical rule:** evaluation templates are never used in any training step.

### 8.5 What we take as non-negotiable

- Counterfactual = sensitive-only edit  
- \(h̄\) = last two layers  
- Loss form and unperturbed \(L_{LM}\)  
- Curriculum order  
- Templates eval-only  
- Report PPL alongside fairness  

---

## 9. Proposed Methodology

### 9.1 Design principle

**Method-faithful, scale-honest reproduction (L1).**  
We preserve the paper’s mathematical structure and experimental logic. We reduce compute-facing knobs and label each change.

### 9.2 Pipeline steps

| Step | Description | Our configuration |
|------|-------------|-------------------|
| Data | WikiText-103 load, article parse, paper split | 28,475 / 60 / 60; seq len 256; GPT-2 BPE |
| Baseline | Step-1 LM training | GPT-2 small, 3000 steps, AdamW \(5\times10^{-5}\) |
| Bias demo | Generate + score + W1 | n=100; opinion + BERT-SST |
| \(f_{sh}\) | Step-2 projection | MLP 768→128→128→2; BERT-SST labels \(\lvert 2p-1\rvert>0.7\) |
| CF train pairs | Sensitive token swap | 99.4% success on audit |
| Embed-Reg | Step-3A | \(\lambda=10\), 1000 steps, lr \(2.5\times10^{-5}\) |
| Sent-Reg | Step-3B | same; \(f_{sh}\) frozen |
| Compare | Pair W1 + val PPL | baker–accountant; sheriff–designer |

### 9.3 Loss functions (student = paper form)

**Language modeling (unperturbed input \(x\)):**
\[
L_{LM}(x) = \mathrm{CE}\big(p_\theta(\cdot\mid x_{<t}),\, x_t\big)
\]

**Embedding fairness:**
\[
L_{\mathrm{fair}}^{\mathrm{emb}} = 1 - \cos\big(\bar h(x),\,\bar h(\tilde x)\big),\quad
\bar h = \tfrac{1}{2}\big(h^{(L-1)}+h^{(L)}\big)
\]
(mean-pooled over tokens in our implementation).

**Sentiment fairness:**
\[
L_{\mathrm{fair}}^{\mathrm{sent}} = 1 - \cos\big(f_{sh}(\bar h(x)),\, f_{sh}(\bar h(\tilde x))\big)
\]

**Total Step-3 loss:**
\[
L(x) = L_{LM}(x) + \lambda\, L_{\mathrm{fair}}(x,\tilde x)
\]

### 9.4 Opinion-word scorer (paper formula)

Let \(p\) = count of positive opinion words, \(n\) = count of negative opinion words (Hu & Liu 2004 lexicon).  
\[
\mathrm{score} = \begin{cases} p/(p+n) & p+n>0 \\ 0.5 & \text{otherwise} \end{cases}
\]

### 9.5 BERT-SST scorer

DistilBERT fine-tuned on SST-2; score = \(P(\text{positive})\in[0,1]\).  
**Substitution:** maps to paper’s BERT-SST classifier; we never claim the paper’s 92.7% SST accuracy as ours.

---

## 10. System Architecture

```
                    ┌─────────────────────────────────────┐
                    │         WikiText-103 (raw)          │
                    └─────────────────┬───────────────────┘
                                      │ parse articles
                                      │ split 28475/60/60
                                      │ tokenize GPT-2, pack len 256
                                      │ flag sensitive sequences
                                      ▼
                    ┌─────────────────────────────────────┐
                    │     Packed sequences + indices      │
                    └─────────────────┬───────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
   Step 1: Baseline LM         Eval templates ONLY          Sensitive subset
   next-token CE               (never in training)          for Step 3
          │                           │                           │
          ▼                           │                           │
   models/baseline/best               │                           │
          │                           │                           │
          ├───────────────────────────┤                           │
          ▼                           ▼                           │
   Step 2: freeze LM            Generate continuations            │
   extract h̄, train f_sh        score sentiment                   │
          │                           │                           │
          ▼                           ▼                           │
   sentiment_classifier.pt      Baseline bias demo                │
          │                     (W1 histograms)                   │
          │                           │                           │
          └─────────────┬─────────────┴───────────────────────────┘
                        ▼
              Step 3: Debias (two forks)
         ┌──────────────┴──────────────┐
         ▼                             ▼
   Embedding Reg                 Sentiment Reg
   L_fair on h̄                   L_fair on f_sh(h̄)
   λ=10, 1000 steps              λ=10, 1000 steps
         │                             │
         └──────────────┬──────────────┘
                        ▼
              Counterfactual pair eval
              baker–accountant
              sheriff–designer
                        ▼
              W1 tables + PPL comparison
              Final results + report
```

**Platform notes:** Kaggle T4, fp32 (no bf16), checkpoints store HF weights without full Adam blobs (disk limits), Persistence “Files Only” + Commit required.

---

## 11. Dataset

### 11.1 Source

- Hugging Face: `Salesforce/wikitext`, config `wikitext-103-raw-v1`  
- Paper reference: Merity et al. WikiText-103  

### 11.2 Paper setting vs ours

| | Paper | Ours |
|--|-------|------|
| Corpus | WikiText-103 (+ WMT-19) | WikiText-103 **only** |
| Train/val/test articles | 28,475 / 60 / 60 | **Same rule** |
| Observed article count in HF dump | (paper states 28,591 total) | **29,566** observed; we report observed |

### 11.3 Statistics (our run)

| Split | Articles | Sequences (len 256) | Sequences with ≥1 sensitive token |
|-------|----------|---------------------|-------------------------------------|
| Train | 28,475 | 445,592 | 100,052 (**22.45%**) |
| Validation | 60 | 811 | 200 (24.66%) |
| Test | 60 | 911 | 194 (21.3%) |

Sensitive hit counts on train sequences (token occurrences, not exclusive): country 21,221; occupation 83,936; name 52,387.

### 11.4 WMT-19 (not used)

The paper’s WMT-19 track is omitted: public 2019 news-crawl lacks recoverable article boundaries matching the paper’s split, and training cost is far beyond one T4. All conclusions concern **WikiText-103 only**.

---

## 12. Data Preprocessing

1. Load train+validation+test line streams; concatenate in order.  
2. Detect article titles with WikiText pattern `= Title =` (not section headers `= = Section = =`).  
3. Apply paper article split in file order (no shuffle).  
4. Tokenize with GPT-2 BPE; insert EOS between articles; pack non-overlapping length-256 blocks.  
5. Decode each block and match Appendix A sensitive lists with word-boundary regex; store `has_sensitive` flags and sensitive-only index lists for Step 3.  
6. **Do not** inject evaluation templates into any training file.

Tokenizer: GPT-2, vocabulary size **50,257**, EOS id 50256.

---

## 13. Baseline Language Model

### 13.1 Architecture

| Property | Paper (WikiText) | Ours |
|----------|------------------|------|
| Family | Transformer-XL | GPT-2 (decoder Transformer) |
| Size | 18L, d=1024, ~257M, from scratch | 12L, d=768, **124.4M**, fine-tune |
| \(h̄\) | mean last 2 layers | **mean last 2 layers** (layers 11–12) |

### 13.2 Optimization (Step 1)

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning rate | \(5\times10^{-5}\) |
| Warmup | 100 steps, then linear decay |
| Micro-batch × accum | 4 × 8 → effective **32** |
| Max steps | **3000** |
| Precision | **fp32** |
| Seq length | 256 |

### 13.3 Result

| Checkpoint | Val PPL |
|------------|---------|
| Init (pretrained GPT-2, before FT) | ~49 |
| **Best baseline** | **22.2753** at step 3000 |

**Interpretation:** Fine-tuning substantially improves WikiText fit. We do **not** claim equality with the paper’s Transformer-XL val PPL ≈ 17.06.

---

## 14. Counterfactual Evaluation

### 14.1 Sensitive attributes (Appendix A, exact lists)

- **Country (10):** Syria, Iran, Libya, Pakistan, Iraq, Denmark, Iceland, Finland, Chile, Italy  
- **Occupation (29):** attendant … CEO (paper list)  
- **Name:** 17 male + 17 female (paper list)  
- Group Fairness subgroups: each country/occupation alone; names collapse to male/female  

### 14.2 Templates

Ten templates per attribute (paper Tables 2–4). Example occupation template 4 (paper’s motivating family):

- *My friend is a/an \<Occupation\>, and we*  

Filled examples:

- *My friend is a baker, and we*  
- *My friend is an accountant, and we*  

### 14.3 Counterfactual definition

Replace the sensitive surface form with another value of the **same** attribute; leave all other tokens unchanged (indefinite article a/an may change for grammar).

### 14.4 Generation protocol

| Knob | Paper | Ours |
|------|-------|------|
| Samples per prefix | 1000 | **100** |
| Max new tokens | 50 | **50** |
| Temperature | 1.0 | **1.0** |

### 14.5 Training CF audit

On 500 random sensitive training sequences: **497/500 = 99.4%** successful single-token counterfactual builds.  
**Conclusion:** later null results for Sent-Reg are **not** explained by broken CF construction.

---

## 15. Sentiment Classifiers (evaluation)

| Scorer | Definition | Role |
|--------|------------|------|
| Opinion-word | \(p/(p+n)\), else 0.5 (Hu & Liu) | Independent / simple; paper classifier iii |
| BERT-SST | \(P(+)\) from SST-2 DistilBERT | Strong neural scorer; paper classifier ii substitute |
| Google Cloud API | Paper primary | **Not used** (cost / access) |

**Important:** Fairness scores are **not comparable across scorers** (paper §C.2). We always report scorer identity beside W1.

---

## 16. Sentiment Projection Classifier \(f_{sh}\) (Step 2)

### 16.1 Role

Maps LM hidden states into a sentiment-related subspace for Sentiment Regularization.

### 16.2 Architecture (paper-matched)

Three-layer MLP:  
\(\mathbb{R}^{768} \xrightarrow{\mathrm{Linear+ReLU}} \mathbb{R}^{128} \xrightarrow{\mathrm{Linear+ReLU}} \mathbb{R}^{128} \xrightarrow{\mathrm{Linear}} \mathbb{R}^{2}\).  
The 128-d pre-logits vector is what Step 3 matches.

### 16.3 Labelling (substitution)

Paper: Google Cloud sentiment in \([-1,1]\), keep \(\lvert s\rvert>0.7\).  
**Ours:** BERT-SST probability \(p\), map \(s=2p-1\), keep \(\lvert s\rvert>0.7\) (i.e. \(p>0.85\) or \(p<0.15\)), drop neutrals, balance classes.

### 16.4 Data volume (ours)

| Stage | Count |
|-------|-------|
| Candidate sentences | 40,000 |
| Kept strong sentiment | 34,386 |
| After balance | 33,400 |
| Train / val | 30,060 / 3,340 |

### 16.5 Validation metrics

| Metric | Value |
|--------|-------|
| Accuracy | **0.8413** |
| Macro F1 | **0.8413** |
| Precision (macro) | 0.8413 |
| Recall (macro) | 0.8416 |
| Confusion [[TN,FP],[FN,TP]] | [[1427, 288], [242, 1383]] |

Paper \(f_{sh}\) accuracy ~98.8% under different labels and data size—**not comparable**. We report **0.841**.  
Because \(f_{sh}\) labels come from BERT-SST, **opinion-word** remains the cleaner external fairness check for Sent-Reg narratives.

---

## 17. Embedding Regularization (Step 3A)

- Initialize from **baseline best** weights.  
- Train only on `has_sensitive` sequences.  
- Build \(\tilde{x}\) by swapping one sensitive token.  
- \(L = L_{LM}(x) + \lambda(1-\cos(\bar h(x),\bar h(\tilde x)))\), \(\lambda=10\).  
- 1000 steps, lr \(2.5\times10^{-5}\), fp32.  

**Result:** best val PPL **22.2381**.

---

## 18. Sentiment Regularization (Step 3B)

- Same as embed-reg, but fairness distance is on **frozen** \(f_{sh}(\bar h(\cdot))\).  
- \(\lambda=10\), 1000 steps.  

**Result:** best val PPL **22.2984**.

---

## 19. Three-Step Curriculum Training (summary)

| Step | Trainable | Loss | Output |
|------|-----------|------|--------|
| 1 | LM | \(L_{LM}\) | Baseline (PPL 22.28) |
| 2 | \(f_{sh}\) only (LM frozen) | CE pos/neg | `sentiment_classifier.pt` |
| 3A | LM | \(L_{LM}+\lambda L_{\mathrm{fair}}^{\mathrm{emb}}\) | Embed-Reg |
| 3B | LM (\(f_{sh}\) frozen) | \(L_{LM}+\lambda L_{\mathrm{fair}}^{\mathrm{sent}}\) | Sent-Reg |

Diagram: see Section 10.

---

## 20. Fairness Metrics

### 20.1 Pair W1 (what we compute)

For a fixed template and pair \((a,a')\), sample \(n\) continuations each, score them, compute
\[
W_1\big(P_S(x_a), P_S(x_{a'})\big).
\]
**Lower is fairer.**

### 20.2 Full I.F. / G.F. (paper; not locked here)

Paper I.F. averages W1 over all templates and all unordered pairs of attribute values.  
Paper G.F. compares each subgroup to the global distribution.  
**Option A freeze uses pair probes only**—sufficient to demonstrate bias and a partial regularizer effect, insufficient to claim paper Table-1 reproduction.

### 20.3 Why W1

Averaging demographic disparity over all thresholds \(\tau\in[0,1]\) equals W1 for 1D scores; no single threshold; no strong shape assumptions.

---

## 21. Performance Metrics

| Metric | Meaning | Our status |
|--------|---------|------------|
| Val PPL | Fluency / fit on val articles | **Reported** for all three LMs |
| PPL_s | PPL on sensitive-token subset | Not locked |
| S.S. | Semantic similarity fraction (≥0.4) | Not locked |
| S.S.c | Fraction mentioning sensitive token | Not locked |

**PPL results**

| Model | Val PPL |
|-------|---------|
| Baseline | 22.2753 |
| Embed-Reg \(\lambda=10\) | **22.2381** |
| Sent-Reg \(\lambda=10\) | 22.2984 |

Debiasing did **not** destroy language modeling quality.

---

## 22. Experimental Setup

| Item | Value |
|------|-------|
| Hardware | Kaggle NVIDIA Tesla T4, 16GB |
| Precision | fp32 (decision: cosine·λ sensitive to low precision) |
| Framework | PyTorch, Hugging Face Transformers |
| Random / config | seeds in configs; resume via HF weights |
| Persistence | Kaggle Files Only + Save Version Commit |

---

## 23. Results

### 23.1 Headline comparison

| Model | Val PPL ↓ | Sheriff–designer opinion W1 ↓ | Baker–accountant opinion W1 |
|-------|-----------|--------------------------------|------------------------------|
| Baseline | 22.28 | 0.1635 | 0.0377 |
| **Embed-Reg λ=10** | **22.24** | **0.1422** | 0.0923 (worse) |
| Sent-Reg λ=10 | 22.30 | 0.1787 | 0.0482 |

### 23.2 Baker vs accountant (template 4, n=100)

| Model | Opinion W1 | BERT-SST W1 |
|-------|------------|-------------|
| Baseline | 0.0377 | 0.0473 |
| Embed-Reg | 0.0923 | 0.0472 |
| Sent-Reg | 0.0482 | 0.0980 |

**Finding:** No fairness improvement on this low-gap pair. Baseline W1 already small (~0.04), so noise and over-regularization can dominate.

### 23.3 Sheriff vs designer (template 4, n=100) — primary signal

| Model | Opinion W1 | BERT-SST W1 |
|-------|------------|-------------|
| Baseline | **0.1635** | **0.1793** |
| Embed-Reg | **0.1422** (Δ=**−0.0213**) | 0.1803 (≈ flat) |
| Sent-Reg | 0.1787 (Δ=+0.0152) | 0.2289 (Δ=+0.0496) |

**Finding:**  
- Embedding Reg improves **opinion-word** fairness on a **high-bias** pair.  
- The gain does **not** appear on BERT-SST.  
- Sentiment Reg does not help either scorer at this budget.

### 23.4 Model selection (honest)

| Criterion | Winner among locked models |
|-----------|----------------------------|
| Best fairness probe (opinion, sheriff–designer) | **Embed-Reg λ=10** |
| Best PPL | **Embed-Reg λ=10** |
| Best Sent-Reg evidence | **None** at λ=10 / 1k steps |
| Overall trade-off (available metrics) | **Embed-Reg λ=10** |

We **do not** claim Embed-Reg wins full paper I.F.; we claim it is the only locked model with a clear positive fairness probe plus intact PPL.

### 23.5 Graphs (produce/attach from Kaggle)

1. Baseline sentiment histograms: baker vs accountant.  
2. Baseline sentiment histograms: sheriff vs designer.  
3. Grouped bar chart: W1 by model × scorer (sheriff–designer).  
4. Bar chart: validation PPL by model.  
5. Optional: training curves (baseline loss/PPL; debias lm vs fair loss).

---

## 24. Qualitative Analysis

### 24.1 Intent

Compare continuations for the **same** prefix under baseline, Embed-Reg, and Sent-Reg.  
*(Examiner tip: paste 2–3 real rows from `results/compare/*/generations.csv` on Kaggle into this section before binding the report.)*

### 24.2 Expected qualitative patterns (from metrics + paper)

| Pattern | What to look for in samples |
|---------|-----------------------------|
| Baseline bias | Sheriff continuations more threat/crime-toned; designer more creative/positive (on average across many samples—not every single line) |
| Embed-Reg | Mild reduction in polarizing tone differences on average (matches lower opinion W1) |
| Sent-Reg @ λ=10 | No reliable softening on our metrics; samples may still diverge |
| Failure mode (paper, extreme λ) | Near-identical generic text for different occupations (we did **not** run extreme λ) |

### 24.3 Example prompt (for manual fill)

**Prefix A:** `My friend is a sheriff, and we`  
**Prefix B:** `My friend is a designer, and we`

| Model | Example continuation (sheriff) | Example continuation (designer) |
|-------|--------------------------------|----------------------------------|
| Baseline | *(paste)* | *(paste)* |
| Embed-Reg | *(paste)* | *(paste)* |
| Sent-Reg | *(paste)* | *(paste)* |

### 24.4 Discussion

Qualitative inspection must not cherry-pick. Our primary evidence remains **distributional W1** over 100 samples. Single sentences can disagree with the aggregate; that is expected under temperature-1 sampling.

---

## 25. Ablation Study

| Ablation | Observation | Implication |
|----------|-------------|-------------|
| **Pair choice** | Sheriff–designer baseline W1 ~0.16; baker–accountant ~0.04 | Always include high-bias pairs; low-gap pairs underpower detection |
| **Scorer** | Embed helps opinion-word, not BERT-SST, on sheriff–designer | Multi-scorer reporting is mandatory; no single W1 is “the” fairness |
| **CF success** | 99.4% | Null Sent-Reg result is not from failed CF plumbing |
| **Budget λ=10, 1k steps** | Small embed gain; no sent gain | Paper used longer Step-3 and fuller metrics |
| **PPL under debias** | All models ~22.2–22.3 | Fairness experiments did not trade away fluency at this λ |
| **f_sh accuracy 0.84 vs paper ~0.99** | Weaker projection | May partly explain weak Sent-Reg; future: better labels / more data |

---

## 26. Limitations

1. **Compute scale:** GPT-2 fine-tune and 1,000 debias steps ≪ paper TPU training.  
2. **Metrics scope:** Pair W1 ≠ full Individual Fairness (all pairs × templates) or Group Fairness.  
3. **Incomplete paper suite:** PPL_s, S.S., S.S.c, human evaluation not in Option A lock.  
4. **No WMT-19 track.**  
5. **No Google sentiment API**; \(f_{sh}\) labels from BERT-SST (substitution).  
6. **Single λ** in locked debias runs.  
7. **Generation sample size** 100 vs paper 1000 → higher W1 variance.  
8. **Kaggle persistence:** one session wipe required weight recovery/retrain; process discipline is part of the engineering contribution.  
9. **Partial positive only:** Embed-Reg gain is pair- and scorer-specific; not a blanket “bias solved.”

---

## 27. Future Work

1. Implement full **I.F. and G.F.** (paper Eqs. 3–4) over occupations × all 10 templates.  
2. Sweep \(\lambda\in\{1,10,100\}\) and longer Step-3 (e.g. 5k–25k steps).  
3. Re-try **Sent-Reg at \(\lambda=100\)** given weak λ=10 results.  
4. Add **PPL_s**, **MiniLM semantic similarity** (USE substitute), **S.S.c**.  
5. Student-scale **human evaluation** (sentiment + relevance) with Spearman vs automatic scores.  
6. Improve \(f_{sh}\) (more data, ensemble labels, calibration).  
7. Country and Name attributes at the same probe depth as Occupation.  
8. Public release of configs + Commit hashes for exact replay.

---

## 28. Conclusion

This project delivered a **complete, runnable, and examinable** student-scale reproduction of Huang et al.’s counterfactual sentiment-bias framework. We showed that:

1. **Bias is real** in a GPT-2 model fine-tuned on WikiText-103 under counterfactual occupation prompts.  
2. The paper’s **three-step curriculum and fairness losses are implementable** on a free GPU.  
3. **Language-model quality (PPL) remains stable** under both regularizers at \(\lambda=10\).  
4. **Embedding Regularization** yields a **modest, scorer-specific** reduction in W1 on a high-bias pair (sheriff vs designer, opinion-word).  
5. **Sentiment Regularization** did not improve locked pair probes at the same budget—an honest negative that motivates longer training, stronger \(\lambda\), or full I.F. evaluation.  
6. **Scientific integrity** requires stating scale gaps and refusing to claim paper table numbers as ours.

The central lesson is not that “debiasing always works on a T4,” but that **fairness work is an empirical pipeline**: measurement design (pairs, scorers, sample size) and training budget jointly determine what effects are visible. That understanding is the appropriate outcome of a method-faithful capstone reproduction.

---

## 29. References

1. P.-S. Huang et al. “Reducing Sentiment Bias in Language Models via Counterfactual Evaluation.” *Findings of ACL: EMNLP 2020*, pp. 65–83, 2020.  
2. C. Dwork et al. “Fairness Through Awareness.” *ITCS*, 2012.  
3. R. Jiang et al. “Wasserstein Fair Classification.” *UAI*, 2019.  
4. S. Merity et al. “Pointer Sentinel Mixture Models” / WikiText-103, 2016.  
5. A. Radford et al. “Language Models are Unsupervised Multitask Learners” (GPT-2), 2019.  
6. M. Hu and B. Liu. “Mining and Summarizing Customer Reviews.” *KDD*, 2004. (Opinion lexicon.)  
7. J. Devlin et al. “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.” *NAACL*, 2019.  
8. R. Socher et al. “Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank” (SST), *EMNLP*, 2013.  
9. E. Sheng et al. “The Woman Worked as a Carpenter…” (bias in generation), *EMNLP*, 2019.  
10. I. Solaiman et al. “Release Strategies and the Social Impacts of Language Models,” 2019.  
11. T. Bolukbasi et al. “Man is to Computer Programmer as Woman is to Homemaker?” *NeurIPS*, 2016.  
12. Z. Dai et al. “Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context.” *ACL*, 2019.

---

## Appendix A — Paper vs student (quick table)

See also `reports/FINAL_RESULTS.md` §2.

| Dimension | Paper | This project |
|-----------|-------|--------------|
| LM | Transformer-XL 257M scratch | GPT-2 124M fine-tune |
| Debias steps | 25k (WikiText) | 1,000 |
| Gen samples | 1,000 | 100 |
| Fairness report | Full I.F./G.F. | Pair W1 probes |
| \(f_{sh}\) labels | Google API | BERT-SST |
| Primary positive finding | Broad I.F. drops | Embed-Reg opinion W1 drop on sheriff–designer |

## Appendix B — Hyperparameters

### Baseline (`configs/baseline.yaml`)
`lr=5e-5`, `max_steps=3000`, `micro_batch=4`, `grad_accum=8`, `seq_len=256`, `warmup=100`, fp32.

### Debias (`configs/embed_reg.yaml`, `sent_reg.yaml`)
`lr=2.5e-5`, `max_steps=1000`, `lambda_fair=10`, `micro_batch=2`, `grad_accum=8`, sensitive_only=true, fp32.

### \(f_{sh}\)
`hidden=128`, `epochs=8`, `abs_threshold=0.7`, `max_candidate_sentences=40000`, balanced classes.

## Appendix C — Reproducibility paths (Kaggle)

```
data/wikitext103/processed/
models/baseline/best/hf_model/
models/embedding_reg_lam10/best/hf_model/
models/sentiment_reg_lam10/best/hf_model/
models/sentiment_classifier/sentiment_classifier.pt
results/compare/baseline_sheriff/
results/compare/embed_sheriff/
results/compare/sent_sheriff/
results/compare/baseline_baker/
results/compare/embed_baker/
results/compare/sent_baker/
reports/FINAL_RESULTS.md
reports/PROJECT_REPORT.md
```

## Appendix D — Allowed vs forbidden claims (viva safety)

**Allowed:** method-faithful curriculum; our PPL/W1/CF/f_sh numbers; partial Embed-Reg gain on one pair/scorer; stable PPL; limitations.  

**Forbidden:** “We fully reproduced Huang et al. Table 1”; “Sent-Reg worked”; claiming paper PPL 18.9 or f_sh 98.8% as ours; WMT-19 results.

---

*End of report.*
