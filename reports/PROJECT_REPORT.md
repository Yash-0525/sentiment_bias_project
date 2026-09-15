# Reducing Sentiment Bias in Language Models via Counterfactual Evaluation  
### A Student-Scale Reproduction of Huang et al. (Findings of EMNLP 2020)

---

## 1. Title

**Reducing Sentiment Bias in Language Models via Counterfactual Evaluation: A Method-Faithful Reproduction at Student Computational Scale**

## 2. Abstract

Large language models can assign different sentiment distributions to generated text when only a sensitive attribute (e.g., occupation) changes in the prompt. Huang et al. (EMNLP Findings 2020) quantify this *counterfactual sentiment bias* with Wasserstein-1-based Individual and Group Fairness and reduce it with embedding and sentiment regularization in a three-step curriculum.

We reproduce that **methodology** on WikiText-103 using GPT-2 small (124M) on a single free Kaggle GPU. We train a baseline LM (val PPL 22.28), a sentiment projection \(f_{sh}\) (val acc 0.84), and debiased models with embedding and sentiment regularization (\(\lambda=10\), 1000 steps). Counterfactual training pairs succeed at 99.4%. Baseline bias is clear on sheriff vs designer (opinion W1 0.16). Embedding regularization lowers that W1 to 0.14 while preserving PPL; sentiment regularization at the same budget does not improve the tested pairs. We report pair-level probes rather than full paper I.F. tables and state all scale substitutions explicitly.

## 3. Introduction

Neural language models trained on web-scale text internalize social stereotypes. One operationalization is **sentiment bias under counterfactual intervention**: replace a sensitive token (e.g., *baker* → *accountant*) and observe whether the distribution of sentiment scores of continuations shifts.

Huang et al. propose:

- Evaluation via many template prefixes and sampled continuations  
- Wasserstein-1 distance between sentiment distributions  
- Aggregate **Individual Fairness (I.F.)** and **Group Fairness (G.F.)**  
- Two training regularizers on hidden states  
- A three-step curriculum so the LM first learns language, then a sentiment subspace, then fairness

This project implements that pipeline at a scale a final-year student can run, debug, and defend.

## 4. Problem Statement

**Given** an autoregressive LM and a sensitive attribute with values \(\mathcal{A}\),  
**measure** whether \(P_S(x)\) and \(P_S(\tilde{x})\) differ when \(x\) and \(\tilde{x}\) differ only in the sensitive token,  
**and reduce** that discrepancy without destroying fluency (perplexity).

## 5. Motivation

- Generated text is used in assistants, stories, and tools; biased sentiment can harm users.  
- Average sentiment difference is threshold-dependent; W1 integrates over thresholds.  
- Industry-scale replications are inaccessible to students; a **method-faithful** small run still teaches the science.

## 6. Objectives

1. Explain and implement counterfactual sentiment bias evaluation.  
2. Train baseline GPT-2 on WikiText-103 (paper split).  
3. Demonstrate non-trivial baseline bias.  
4. Train \(f_{sh}\) and both regularizers.  
5. Compare baseline vs Embed-Reg vs Sent-Reg on W1 and PPL.  
6. Document paper vs student gaps honestly.

## 7. Literature Review (brief)

- **Fairness in ML:** demographic parity, individual fairness (Dwork et al.), Wasserstein fairness (Jiang et al.).  
- **Bias in LMs:** stereotyping in embeddings and generation (e.g., Bolukbasi; Sheng et al.; Solaiman et al.).  
- **Base paper:** Huang et al., Findings of EMNLP 2020 — counterfactual sentiment + regularization.

## 8. Base Paper

Huang, P.-S., et al. (2020). *Reducing Sentiment Bias in Language Models via Counterfactual Evaluation.* Findings of EMNLP 2020, 65–83.

Core contributions we follow: counterfactual evaluation protocol, I.F./G.F. via W1, embedding vs sentiment regularization, three-step curriculum.

## 9. Proposed Methodology (ours = paper method at reduced scale)

1. **Data:** WikiText-103, article split 28,475 / 60 / 60.  
2. **Baseline LM:** Fine-tune GPT-2 small, next-token CE only.  
3. **Bias eval:** Templates × sensitive values → sample continuations → score sentiment → W1.  
4. **\(f_{sh}\):** Freeze LM; label strong pos/neg sentences; MLP on mean-pooled \(h̄\).  
5. **Debias:** On sensitive sequences only,  
   \(L = L_{LM}(x) + \lambda\, d(r(x), r(\tilde{x}))\)  
   with \(r = h̄\) (embed) or \(r = f_{sh}(h̄)\) (sentiment).  
6. **Constraints:** No evaluation templates in training; \(L_{LM}\) on unperturbed \(x\).

## 10. System Architecture

```
WikiText-103 → preprocess → GPT-2 LM
    → Step1 baseline
    → Step2 f_sh (frozen LM)
    → Step3 Embed-Reg / Sent-Reg
    → counterfactual generation → sentiment scores
    → W1 pair probes (+ PPL)
    → baseline vs debiased comparison
```

## 11–12. Dataset and preprocessing

See `results/FINAL_RESULTS.md` §3. HuggingFace `Salesforce/wikitext` / `wikitext-103-raw-v1`, GPT-2 BPE, sequences of length 256, sensitive-token flags for Step 3.

## 13. Baseline language model

- GPT-2 small (12 layers, \(d=768\), 124.4M params)  
- AdamW lr \(5\times10^{-5}\), effective batch 32, 3000 steps, fp32  
- **best val PPL = 22.2753**

Paper WikiText Transformer-XL val PPL ≈ 17; we do **not** claim parity.

## 14. Counterfactual evaluation

Sensitive token lists and 10 templates per attribute from Appendix A.  
Counterfactual = change only the sensitive surface form (a/an may adjust for occupations).  
Generation: max 50 tokens, temperature 1.0, n=100 (paper n=1000).

## 15–16. Sentiment classifiers and \(f_{sh}\)

- **Eval:** opinion-word (paper formula); BERT-SST / DistilBERT-SST (paper classifier ii substitute).  
- **No Google API.**  
- **\(f_{sh}\):** 3× Linear MLP, hidden 128; input mean-pooled \(h̄\); labels BERT-SST \(\|2p-1\|>0.7\); val acc **0.841**.

## 17–18. Embedding / Sentiment regularization and curriculum

Implemented as in the paper (§4). Student Step-3: lr \(2.5\times10^{-5}\), \(\lambda=10\), 1000 steps, sensitive-only data. CF success rate 99.4%.

## 19–20. Fairness and performance metrics

- **Used:** pair W1, validation PPL.  
- **Paper full suite not all completed:** full I.F., G.F., PPL_s, S.S., S.S.c, human eval.

## 21. Experimental setup

- **Platform:** Kaggle, Tesla T4, fp32  
- **Persistence:** Files Only + Commit (critical)  
- **Code:** https://github.com/Yash-0525/sentiment_bias_project branch `arena/01a0a078-sentiment-bias-project`

## 22–23. Results and graphs

See `results/FINAL_RESULTS.md` §§6–7.

**Headline numbers**

| Model | Val PPL | Sheriff–designer opinion W1 |
|-------|---------|------------------------------|
| Baseline | 22.28 | 0.1635 |
| Embed-Reg λ=10 | **22.24** | **0.1422** |
| Sent-Reg λ=10 | 22.30 | 0.1787 |

Suggested plots (from existing Kaggle `plots/baseline_bias` + compare folders):

1. Baseline baker vs accountant sentiment histograms  
2. Baseline sheriff vs designer histograms  
3. Bar chart: W1 by model (sheriff/designer, opinion)  
4. Bar chart: val PPL by model  

## 24. Qualitative analysis

*(Fill with 2–3 decoded continuations from `results/compare/*/generations.csv` on Kaggle — same prompt, three models, fixed seed if available.)*

Discuss: baseline may polarize sheriff vs designer; embed-reg may soften tone differences; excessive λ (not run) can collapse meaning (paper App. C.6).

## 25. Ablation study

| Ablation | Finding |
|----------|---------|
| Pair choice | Sheriff/designer much more biased than baker/accountant |
| Scorer | Opinion vs BERT disagree on whether embed helps |
| CF rate | 99.4% — not the failure mode |
| λ=10 / 1k steps | Enough for small embed gain on one pair; not for sent-reg |

## 26. Limitations

See FINAL_RESULTS §8. Main: scale, pair probes ≠ full I.F., no USE/S.S., no WMT-19, API substitution, short debias schedule.

## 27. Future work

1. Full I.F./G.F. over all occupation pairs × 10 templates  
2. \(\lambda \in \{1,10,100\}\) and longer Step-3  
3. PPL_s, S.S. (MiniLM), S.S.c  
4. Sent-Reg \(\lambda=100\)  
5. Human ratings (student-scale)  
6. Stronger \(f_{sh}\) or multi-checkpoint label ensembling  

## 28. Conclusion

We delivered a **working, demonstrable, method-faithful** student reproduction of Huang et al.’s counterfactual sentiment-bias pipeline. Bias is detectable; PPL is preserved under regularization; embedding regularization yields a **modest, scorer-specific** W1 reduction on a high-bias occupation pair. Sentiment regularization did not improve the locked probes. Absolute paper numbers are not claimed. The main scientific lesson is the **fairness–evaluation design trade-off**: what you measure (which pairs, which scorers, how many samples) changes the story as much as the regularizer.

## 29. References

1. Huang et al. (2020). Reducing Sentiment Bias… Findings of EMNLP.  
2. Dwork et al. (2012). Fairness through awareness.  
3. Jiang et al. (2019). Wasserstein Fair Classification.  
4. Merity et al. (2016). WikiText-103.  
5. Radford et al. (2019). GPT-2.  
6. Hu & Liu (2004). Opinion lexicon.  
7. Devlin et al. (2019). BERT; SST-2 fine-tunes as used in tooling.

---

*Numerical results in this report are taken only from our runs documented in `results/FINAL_RESULTS.md`. Paper numbers appear only as reference, never as our results.*
