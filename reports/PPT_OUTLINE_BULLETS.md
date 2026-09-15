# PPT Outline — Bullet Only  
### Copy each slide block straight into PowerPoint / Google Slides

**Total slides:** 15  
**Talk time:** ~10–12 minutes  
**Project:** Sentiment Bias via Counterfactual Evaluation (Huang et al. 2020, student scale)

---

## Slide 1 — Title

- Reducing Sentiment Bias in Language Models via Counterfactual Evaluation
- Student-scale reproduction of Huang et al. (Findings of EMNLP 2020)
- WikiText-103 · GPT-2 small · Kaggle T4
- [Your Name]
- [College / Department / Year]
- Capstone / Final-Year Project

---

## Slide 2 — Problem

- Language models copy stereotypes from training text
- Focus: **sentiment bias** (not just toxic words)
- Change **only one sensitive word** in the prompt
  - Example: *sheriff* vs *designer*
  - Example: *baker* vs *accountant*
- Generated continuations get **different sentiment distributions**
- Problem: unfair tone linked to occupation / country / name

---

## Slide 3 — Motivation

- LM-generated text is used in real applications
- Biased sentiment can reinforce social stereotypes
- Need:
  - A **measurable** definition of bias
  - A **trainable** mitigation method
- Paper provides both (W1 fairness + regularizers)
- Student goal: implement, measure, explain honestly under free-GPU limits

---

## Slide 4 — Research Paper

- Huang et al., Findings of EMNLP 2020
- Title: *Reducing Sentiment Bias in Language Models via Counterfactual Evaluation*
- Key ideas:
  - Counterfactual evaluation
  - Wasserstein-1 between sentiment distributions
  - Individual Fairness (I.F.) & Group Fairness (G.F.)
  - Embedding Regularization
  - Sentiment Regularization
  - Three-step curriculum training

---

## Slide 5 — Existing System (Paper)

- Models: Transformer-XL (257M / 708M) trained **from scratch**
- Hardware: **128 TPUv3** cores
- Data: WikiText-103 + WMT-19
- Generation: **1000** samples per template per sensitive value
- Strong results on full I.F. / G.F. tables
- Not reproducible bit-for-bit on free student GPU

---

## Slide 6 — Proposed System (Ours)

- **Method-faithful**, scale-honest reproduction (L1)
- Data: WikiText-103 only, paper split **28,475 / 60 / 60**
- Model: **GPT-2 small (124M)** fine-tune
- Same losses, same curriculum, same eval templates (**eval only**)
- Platform: Kaggle **Tesla T4**, training in **fp32**
- Report relative baseline vs debiased — not paper absolute numbers

---

## Slide 7 — System Architecture

- WikiText-103 → preprocess → pack sequences (len 256)
- Step 1: Baseline LM (next-token CE)
- Step 2: Freeze LM → train sentiment projection \(f_{sh}\)
- Step 3: Debias on sensitive sequences only
  - Branch A: Embedding Reg
  - Branch B: Sentiment Reg
- Evaluate: counterfactual prompts → generate → sentiment → W1
- Compare: Baseline vs Embed-Reg vs Sent-Reg (+ PPL)

---

## Slide 8 — Three-Step Curriculum

- **Step 1 — Baseline LM**
  - Loss: next-token cross-entropy only
  - Result: val PPL **22.28**
- **Step 2 — Sentiment projection \(f_{sh}\)**
  - LM frozen; 3-layer MLP, hidden 128
  - Result: val accuracy **0.84**
- **Step 3 — Debiasing**
  - Loss: \(L = L_{LM}(x) + \lambda L_{fair}\)
  - \(L_{LM}\) on **unperturbed** \(x\) only
  - Templates **never** used in training
- Outputs: Embed-Reg model + Sent-Reg model

---

## Slide 9 — Counterfactual Evaluation

- Counterfactual = change **only** the sensitive token
- Non-sensitive context stays identical
- Sensitive lists from paper Appendix A
  - Country / Occupation / Name
- 10 evaluation templates per attribute
- Generation settings:
  - max **50** tokens
  - temperature **1.0**
  - **100** samples (paper: 1000)
- Training CF audit: **99.4%** successful swaps

---

## Slide 10 — Fairness Metrics

- Sentiment score ∈ [0, 1]
  - Opinion-word: \(p/(p+n)\) else 0.5
  - BERT-SST: \(P(\text{positive})\)
- Compare distributions with **Wasserstein-1 (W1)**
- **Lower W1 = fairer**
- Why W1?
  - No single threshold \(\tau\)
  - Captures full distribution shift
- This project reports **pair W1 probes**
  - Not full paper I.F. average over all pairs
- Always report scorer name with W1

---

## Slide 11 — Embedding vs Sentiment Regularization

- Shared setup:
  - \(\bar{h}\) = average of **last two** LM layers
  - \(\lambda = 10\)
  - 1000 debias steps
  - Start from same baseline checkpoint
- **Embedding Regularization**
  - \(L_{fair} = 1 - \cos(\bar{h}(x),\bar{h}(\tilde{x}))\)
  - Matches full hidden states
- **Sentiment Regularization**
  - \(L_{fair} = 1 - \cos(f_{sh}(\bar{h}(x)), f_{sh}(\bar{h}(\tilde{x})))\)
  - \(f_{sh}\) **frozen**
  - Matches sentiment subspace only
- Theory: Sent-Reg gentler on meaning; Embed-Reg stronger/blunter

---

## Slide 12 — Experimental Setup

- Hardware: Kaggle Tesla T4 (16 GB)
- Precision: fp32
- Data: WikiText-103
  - Train sequences: 445,592 (len 256)
  - Sensitive train sequences: 22.45%
- Baseline: 3000 steps, AdamW \(5\times10^{-5}\)
- Debias: 1000 steps, lr \(2.5\times10^{-5}\), \(\lambda=10\)
- Eval pairs:
  - baker vs accountant
  - sheriff vs designer
- n = 100 generations per prefix

---

## Slide 13 — Results

- **Language model quality (val PPL)**
  - Baseline: **22.28**
  - Embed-Reg λ10: **22.24**
  - Sent-Reg λ10: **22.30**
  - Takeaway: debiasing did **not** break fluency
- **Sheriff vs designer (opinion-word W1) ↓ better**
  - Baseline: **0.1635**
  - Embed-Reg: **0.1422** ← improved (Δ = −0.021)
  - Sent-Reg: **0.1787** ← no improvement
- **Baker vs accountant**
  - Small baseline gap (~0.04)
  - No fairness win for either regularizer
- **Other checks**
  - CF success: **99.4%**
  - \(f_{sh}\) accuracy: **0.84**
- **Best locked trade-off:** Embedding Reg λ=10

---

## Slide 14 — Graphs / Examples

- Graph 1: Baseline histograms — sheriff vs designer sentiment
- Graph 2: Bar chart — W1 by model (opinion-word, sheriff–designer)
- Graph 3: Bar chart — validation PPL by model
- Graph 4 (optional): baker vs accountant histograms (small gap)
- Qualitative (optional): same prompt, 3 model continuations
- Reading guide on slide:
  - Lower W1 = fairer
  - Lower PPL = better LM
  - Scorer-specific effects matter

---

## Slide 15 — Conclusion and Future Work

- **Conclusions**
  - Counterfactual sentiment bias is measurable at student scale
  - Paper curriculum + losses are implementable on free GPU
  - PPL stays stable under regularization
  - Embedding Reg: modest opinion-W1 gain on high-bias pair
  - Sentiment Reg: no gain on locked probes at λ=10 / 1k steps
  - Honest reporting > fake paper-number matching
- **Limitations**
  - Pair probes ≠ full I.F./G.F.
  - Reduced steps / samples / model size
  - No Google API; no full semantic-similarity suite
- **Future work**
  - Full I.F. over all occupation pairs × 10 templates
  - λ grid {1,10,100} + longer Step-3
  - Sent-Reg λ=100 retry
  - PPL_s, semantic similarity, human eval
- **Thank you — Questions?**

---

# Optional backup slides (if time / appendix)

## Backup A — Paper vs Ours (one table slide)

- Same: split, h̄, losses, curriculum, eval-only templates
- Changed: GPT-2 vs Transformer-XL; 3k/1k steps; n=100; WikiText only
- Claim type: method-faithful, not bit-identical

## Backup B — Loss equations

- \(L_{LM}\): next-token CE on unperturbed \(x\)
- Embed: \(1-\cos(\bar h(x),\bar h(\tilde x))\)
- Sent: \(1-\cos(f_{sh}(\bar h(x)),f_{sh}(\bar h(\tilde x)))\)
- Total: \(L=L_{LM}+\lambda L_{fair}\)

## Backup C — Full numeric table

- Paste master table from `FINAL_RESULTS.md` §7
- Highlight Embed-Reg sheriff opinion row

## Backup D — Viva one-liners

- Sentiment bias = distribution shift under sensitive-only edit
- W1 = threshold-free distribution distance
- Why templates not in training: prevent eval overfitting / leakage
- Why multi-scorer: classifiers can be biased too

---

# Build checklist

- [ ] Add college logo on Slide 1
- [ ] Insert real plots on Slide 14 from Kaggle `plots/` + `results/compare/`
- [ ] Keep Slide 13 numbers exactly as locked (do not “round into” paper values)
- [ ] Practice 60-second version of Slide 13 (core of the talk)
- [ ] End with limitations before Q&A
