# Final Experimental Results

**Project:** Reducing Sentiment Bias in Language Models via Counterfactual Evaluation  
**Paper:** Huang et al., Findings of EMNLP 2020  
**Setting:** STUDENT / COMPUTE-LIMITED (Option A — Kaggle, L1 method-faithful)  
**Decision:** Option A — lock results; no further heavy training  
**Date recorded:** 2026-09-15 (user local)

---

## 1. Executive summary

We implemented the paper’s **three-step curriculum** at student scale on WikiText-103 with GPT-2 small:

1. Baseline LM fine-tune  
2. Sentiment projection \(f_{sh}\) on frozen LM hidden states  
3. Debiasing with Embedding Regularization and Sentiment Regularization (\(\lambda=10\), 1000 steps)

**What worked**

- Full pipeline runs on free Kaggle GPU  
- Counterfactual token swaps succeed at **99.4%**  
- Clear **baseline counterfactual sentiment bias** (especially sheriff vs designer)  
- Debiasing **preserves language-model quality** (val PPL ≈ 22.3 for all models)  
- **Embedding Reg + opinion-word** reduces W1 on sheriff vs designer (**0.1635 → 0.1422**)

**What did not work (this setting)**

- **Sentiment Reg \(\lambda=10\)** did not reduce W1 on tested pairs  
- **Baker vs accountant** showed no fairness gain (and some regressions)  
- We did **not** reproduce paper absolute I.F./G.F. tables (different model, budget, sample counts)

These are real experimental outcomes, not failures of the project idea.

---

## 2. Paper vs our implementation

| Item | Original paper | Our implementation | Same idea? |
|------|----------------|--------------------|------------|
| Primary corpus | WikiText-103 (+ WMT-19) | WikiText-103 only | Yes (primary track) |
| Article split | 28,475 / 60 / 60 | **28,475 / 60 / 60** | Yes |
| LM | Transformer-XL 257M from scratch | GPT-2 small 124M **fine-tune** | Family differs; objective same |
| Step-1 steps | 250k | **3000** | Reduced |
| Seq length | 512 | **256** | Reduced |
| \(h̄\) | mean last 2 layers | **mean last 2 layers** | Yes |
| Fairness loss | \(1-\cos\) embed or via \(f_{sh}\) | **Same formulas** | Yes |
| \(L = L_{LM}(x)+\lambda L_{fair}\) | Yes; \(L_{LM}\) on unperturbed \(x\) | **Yes** | Yes |
| Templates in training | Never | **Never** | Yes |
| \(\lambda\) (WikiText) | {1,10,100} | **10** (main); grid optional | Partial |
| Step-3 steps | 25k | **1000** | Reduced |
| Gen samples / prefix | 1000 | **100** | Reduced |
| Sentiment labels for \(f_{sh}\) | Google Cloud API | **BERT-SST** (substitution) | Labelled sub |
| Eval scorers | Google + BERT-SST + opinion | **BERT-SST + opinion** | Partial |
| Semantic similarity | USE, thr 0.4 | Not completed in locked run | Pending / future |
| Full I.F. (all pairs) | Yes | **Pair probes only** | Partial |

---

## 3. Dataset (Phase 2)

| Split | Articles | Sequences (len 256) | % with sensitive token |
|-------|----------|---------------------|-------------------------|
| Train | 28,475 | 445,592 | 22.45% (100,052 seqs) |
| Val | 60 | 811 | 24.66% |
| Test | 60 | 911 | 21.3% |

- Observed articles in HF raw dump: **29,566** (paper states 28,591; we report observed).  
- Tokenizer: **gpt2**, vocab **50,257**.  
- Sensitive lists: paper Appendix A (10 countries, 29 occupations, 17+17 names).

---

## 4. Models trained

| Model | Path (Kaggle) | Steps | best val PPL |
|-------|---------------|-------|--------------|
| Baseline | `models/baseline/best` | 3000 | **22.2753** |
| Embed-Reg \(\lambda=10\) | `models/embedding_reg_lam10/best` | 1000 | **22.2381** |
| Sent-Reg \(\lambda=10\) | `models/sentiment_reg_lam10/best` | 1000 | **22.2984** |
| \(f_{sh}\) | `models/sentiment_classifier/sentiment_classifier.pt` | 8 epochs | val acc **0.8413** |

**PPL takeaway:** Debiasing did **not** destroy LM quality (all ≈ 22.2–22.3).

### \(f_{sh}\) metrics (val)

| Metric | Value |
|--------|-------|
| Accuracy | 0.8413 |
| Macro F1 | 0.8413 |
| Confusion | [[1427, 288], [242, 1383]] |
| Architecture | 3-layer MLP, hidden 128, \(d_{in}=768\) |
| Labels | BERT-SST, \(\|2p-1\|>0.7\), balanced |

Paper reported ~98.8% with Google API + ~370k sentences — **not comparable**; we report **0.841**.

---

## 5. Counterfactual mechanism audit

| Check | Result |
|-------|--------|
| CF single-token swap success | **497 / 500 = 99.4%** |
| Conclusion | Fairness loss is **not** starved by failed CF construction |

---

## 6. Fairness probes (pair W1)

Protocol: occupation templates, max 50 new tokens, temperature 1.0, **n=100** samples/prefix.  
Scorers: opinion-word \(p/(p+n)\); DistilBERT-SST \(P(+)\).  
**Lower W1 = fairer.**

### 6.1 Baker vs accountant (template 4)

| Model | opinion W1 | bert W1 |
|-------|------------|---------|
| Baseline | **0.0377** | **0.0473** |
| Embed-Reg \(\lambda=10\) | 0.0923 | 0.0472 |
| Sent-Reg \(\lambda=10\) | 0.0482 | 0.0980 |

**Δ vs baseline (negative = better)**

| Model | opinion Δ | bert Δ |
|-------|-----------|--------|
| Embed-Reg | +0.0546 | −0.0001 |
| Sent-Reg | +0.0105 | +0.0507 |

**Verdict:** No fairness gain on this pair at this budget. Baseline gap already small (~0.04).

### 6.2 Sheriff vs designer (template 4) — primary positive signal

| Model | opinion W1 | bert W1 |
|-------|------------|---------|
| Baseline | **0.1635** | **0.1793** |
| Embed-Reg \(\lambda=10\) | **0.1422** | 0.1803 |
| Sent-Reg \(\lambda=10\) | 0.1787 | 0.2289 |

**Δ vs baseline**

| Model | opinion Δ | bert Δ |
|-------|-----------|--------|
| **Embed-Reg** | **−0.0213** | +0.0010 |
| Sent-Reg | +0.0152 | +0.0496 |

**Verdict:**  
- **Embedding regularization** improves **opinion-word** fairness on this high-bias pair.  
- Effect does **not** transfer cleanly to BERT-SST.  
- **Sentiment regularization** does not improve either scorer here.

### 6.3 Earlier baseline-only runs (context)

| Setting | opinion W1 | bert W1 |
|---------|------------|---------|
| Baker/accountant n=20 smoke | 0.0425 | — |
| Baker/accountant n=100 | 0.0593 | 0.0638 |
| Baker/accountant mean over t4,5,8 | 0.0498 | 0.0730 |
| Sheriff/designer mean over t4,5,8 | 0.1552 | 0.1641 |

Baseline bias is **real and pair-dependent**.

---

## 7. Master results table (locked run)

| Model | Attribute pair | \(\lambda\) | Scorer | W1 ↓ | Val PPL ↓ | Notes |
|-------|----------------|-------------|--------|------|-----------|-------|
| Baseline | baker–accountant | — | opinion | 0.0377 | 22.28 | Small gap |
| Baseline | baker–accountant | — | bert_sst | 0.0473 | 22.28 | |
| Embed-Reg | baker–accountant | 10 | opinion | 0.0923 | 22.24 | Worse |
| Embed-Reg | baker–accountant | 10 | bert_sst | 0.0472 | 22.24 | Flat |
| Sent-Reg | baker–accountant | 10 | opinion | 0.0482 | 22.30 | Slightly worse |
| Sent-Reg | baker–accountant | 10 | bert_sst | 0.0980 | 22.30 | Worse |
| Baseline | sheriff–designer | — | opinion | 0.1635 | 22.28 | Strong bias |
| Baseline | sheriff–designer | — | bert_sst | 0.1793 | 22.28 | |
| **Embed-Reg** | **sheriff–designer** | **10** | **opinion** | **0.1422** | **22.24** | **Best fairness on this probe** |
| Embed-Reg | sheriff–designer | 10 | bert_sst | 0.1803 | 22.24 | Flat |
| Sent-Reg | sheriff–designer | 10 | opinion | 0.1787 | 22.30 | Worse |
| Sent-Reg | sheriff–designer | 10 | bert_sst | 0.2289 | 22.30 | Worse |

### Model selection (honest)

| Criterion | Choice |
|-----------|--------|
| Best fairness on locked probes | **Embed-Reg \(\lambda=10\)** (sheriff/designer, opinion-word) |
| Best / stable PPL | **Embed-Reg** (22.24) slightly edges baseline (22.28) |
| Best semantic / full I.F. | **Not measured** in locked run |
| Best overall trade-off (what we have) | **Embed-Reg \(\lambda=10\)** — only model with a clear W1 drop on a high-bias pair without hurting PPL |
| Sent-Reg at \(\lambda=10\) / 1k steps | **Not recommended** from these probes alone |

---

## 8. Limitations (must appear in report)

1. **Scale:** Not Transformer-XL from scratch; not paper step counts or 1000 generations.  
2. **Metrics:** Pair W1 probes ≠ full Individual Fairness (Eq. 3) over all pairs/templates.  
3. **Group Fairness, S.S., S.S.c, PPL_s:** Not completed in the locked Option A freeze.  
4. **WMT-19:** Not trained (compute + no document boundaries in public crawl).  
5. **Google sentiment API:** Not used; BERT-SST labels for \(f_{sh}\) (D12b circularity risk → opinion-word is the independent check).  
6. **Single \(\lambda\), short Step-3:** Only \(\lambda=10\), 1000 steps.  
7. **Scorer dependence:** Embed helps opinion-word more than BERT-SST.  
8. **Session persistence:** One Kaggle wipe lost weights; results below are from recovered retrain + user-pasted metrics.

---

## 9. Conclusions

1. Counterfactual sentiment bias is **measurable** in a student-scale GPT-2 fine-tuned on WikiText-103.  
2. The paper’s **curriculum and losses are implementable** on free GPU hardware.  
3. **PPL remains stable** under both regularizers at \(\lambda=10\).  
4. **Embedding regularization** shows a **modest, scorer-specific** fairness gain on a high-bias occupation pair.  
5. **Sentiment regularization** did not show gains under the locked budget — consistent with needing stronger \(\lambda\), longer training, better \(f_{sh}\), or full I.F. evaluation.  
6. **Fairness claims must be pair- and metric-specific**; one pair is not the paper’s full story.

---

## 10. Artefacts to keep (Kaggle Commit)

```
data/wikitext103/
models/baseline/best/
models/embedding_reg_lam10/best/
models/sentiment_reg_lam10/best/
models/sentiment_classifier/sentiment_classifier.pt
results/compare/
results/baseline/
results/FINAL_RESULTS.md   # this file (from git)
```
