# Viva Q&A (student-friendly)

Answers match **our** implementation and **our** numbers.

---

### 1. What is sentiment bias?
When a language model’s **generated text** is systematically more positive or negative depending on a sensitive word in the prompt (occupation, country, name), even if everything else is the same.

### 2. What is counterfactual evaluation?
Build a second prompt that differs **only** in the sensitive attribute value, generate many continuations from both, and compare sentiment **distributions**.

### 3. Why is changing only the sensitive attribute important?
If you change other words too, you can’t tell whether the sentiment shift came from the sensitive attribute or from the new context.

### 4. What is Wasserstein-1 distance?
A distance between two distributions: how much “probability mass” you must move (and how far) to turn one into the other. For 1D sentiment scores it equals the average absolute difference of sorted samples (also `scipy.stats.wasserstein_distance`).

### 5. Why W1 instead of average sentiment difference?
Average difference ignores shape (e.g. different spreads). Many fairness definitions need a threshold \(\tau\); averaging disparity over \(\tau\) yields W1 (strong demographic parity). No single threshold choice.

### 6. What is Individual Fairness (I.F.)?
In the paper: average W1 over templates and pairs of sensitive values—similar individuals (almost identical prompts) should get similar sentiment distributions. **Lower is better.**

### 7. What is Group Fairness (G.F.)?
Average W1 between each subgroup’s sentiment distribution and the overall distribution. **Lower is better.** We did not fully compute G.F. in the locked run.

### 8. Embedding vs sentiment regularization?
- **Embed:** match raw hidden states \(h̄(x)\) and \(h̄(\tilde{x})\) with cosine distance.  
- **Sent:** match \(f_{sh}(h̄(x))\) and \(f_{sh}(h̄(\tilde{x}))\) with \(f_{sh}\) **frozen**.  
Sent aims to equalize **sentiment subspace** only; embed equalizes broader features.

### 9. Why the final two hidden layers?
Paper: later layers encode higher-level semantics (incl. sentiment); averaging too many layers makes \(h̄(x)\approx h̄(\tilde{x})\) too easily and weakens the regularizer.

### 10. Why cosine distance?
Scale-invariant similarity; paper uses \(d=1-\cos\). Bounded, stable with λ multipliers (we train fp32 because λ can amplify cosine noise in low precision).

### 11. Why is λ important?
It trades off language modeling loss vs fairness loss. Larger λ → stronger push toward fairness (and risk of semantic collapse if too large).

### 12. What if λ is too high?
Model may ignore the sensitive token: I.F. looks perfect but outputs become generic/irrelevant (paper App. C.6). Always monitor PPL and semantic similarity.

### 13. Why can excessive regularization hurt semantic relevance?
Perfect fairness is trivial if the model never uses the sensitive word. Then generations match but don’t answer the prompt.

### 14. What is perplexity?
\(\mathrm{PPL}=\exp(\text{average NLL per token})\). Lower → model predicts the corpus better. Ours: baseline **22.28**, embed **22.24**, sent **22.30**.

### 15. Why PPL on sensitive subsets (PPL_s)?
Sensitive tokens are rare; overall PPL can hide degradation on the biased contexts. We logged overall val PPL; PPL_s is future work.

### 16. What is semantic similarity?
Fraction of generations whose embedding cosine with the prefix exceeds a threshold (paper: USE, 0.4). We did not lock S.S. in Option A freeze.

### 17. Why multiple sentiment classifiers?
Classifiers can themselves be biased. Paper uses Google API, BERT-SST, and opinion words. We use BERT-SST + opinion words; opinion is the more independent check when \(f_{sh}\) was trained with BERT-SST labels.

### 18. Why are evaluation templates not used in training?
Paper rule: avoids overfitting fairness only on the eval templates and leaking the test protocol into training. We only CF-edit real WikiText sequences with sensitive tokens.

### 19. Why counterfactual pairs?
They isolate the sensitive attribute. Training encourages similar representations (or similar sentiment projections) for \(x\) and \(\tilde{x}\).

### 20. Limitations of this project?
Student scale (GPT-2, 1000 debias steps, n=100); pair probes ≠ full I.F.; no WMT-19; no Google API; no full S.S./human eval; Kaggle disk/session issues once wiped weights.

### 21. What would you change with more compute?
Full I.F./G.F.; λ grid {1,10,100}; longer Step-3; 1000 samples; larger model closer to paper; WMT track if article boundaries available; USE or MiniLM S.S.; human ratings.

### 22. Extend beyond sentiment?
Same curriculum with another specification \(f\) (toxicity, regard, topic) instead of sentiment—paper frames it as specification-general.

### 23. Ethical concerns?
- Debiasing can erase legitimate context (e.g. real historical suffering tied to a country name).  
- Choosing which attributes and templates is a value judgment.  
- Reporting only “good” pairs is unethical—we report baker null and sheriff partial.

### 24. Fairness vs model performance?
Fairness (W1) and performance (PPL, S.S.) can trade off. We require **both**: our embed-reg slightly improved one W1 probe **and** kept PPL flat.

### 25. Main contribution of the research paper?
A practical framework to **quantify** counterfactual sentiment bias with W1-based I.F./G.F. and **reduce** it with representation regularizers without fully sacrificing LM quality—plus evidence of the fairness–semantics trade-off.

---

### Extra questions they often ask

**Q. Your Sent-Reg failed—did you fail the project?**  
A. No. A valid experiment can reject a setting. CF works, bias exists, PPL holds, embed-reg shows a partial gain. Sent-Reg needs more budget or full I.F. to judge fairly.

**Q. Why GPT-2 not Transformer-XL?**  
A. Paper-scale scratch training needs TPU pods. GPT-2 fine-tune is precedented in the paper’s own GPT-2 evaluation appendix. Objective and fairness math stay the same.

**Q. Is W1 0.14 “good”?**  
A. It’s **better than 0.16** on that pair, not “fair.” Absolute fairness needs a domain ε; we emphasize **relative** improvement and limitations.

**Q. Why opinion-word helped but BERT didn’t for embed-reg?**  
A. Different score geometries and biases. Paper also finds scores aren’t comparable across classifiers; trends matter more than absolute W1.
