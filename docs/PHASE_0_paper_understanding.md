# PHASE 0 — Understanding the Research Paper

**Paper:** *Reducing Sentiment Bias in Language Models via Counterfactual Evaluation*
**Authors:** Po-Sen Huang, Huan Zhang, Ray Jiang, Robert Stanforth, Johannes Welbl, Jack W. Rae,
Vishal Maini, Dani Yogatama, Pushmeet Kohli (DeepMind / UCLA / UCL)
**Venue:** Findings of EMNLP 2020, pages 65–83
**Source file:** `uploads/2020.findings-emnlp.7.pdf` (19 pages, extracted text in `paper/paper_full.txt`)

Every fact below was read from the uploaded PDF. Where I give a number, the section/page it
comes from is noted.

---

## 1. The 18 Questions

### 1.1 What problem does the paper solve?
Language models learn to imitate their training corpus. When the corpus contains social
stereotypes, the model reproduces them in *generated* text. The paper targets one measurable
form of this: **the sentiment (positivity/negativity) of generated text changes systematically
depending on which sensitive attribute value you put in the prompt.**

Two-part goal (§1, p.65):
- **Quantify** it — propose metrics (Individual Fairness, Group Fairness).
- **Reduce** it — propose two regularizers on the model's hidden representations.

### 1.2 What is sentiment bias?
Not "the model outputs offensive words". It is narrower and measurable:
**the distribution of sentiment scores of generated continuations differs when only the
sensitive attribute value is changed and nothing else.**

Paper's motivating example (Figure 1, p.65): prompt `My friend is a/an <occupation>, and we...`
Generate 1,000 continuations each for "baker" and "accountant", score them with the Google
Cloud sentiment API. Result: "baker" → more positive, "accountant" → more negative.
Reported Wasserstein-1 distance between the two distributions: **0.13** (stated in Fig. 2 caption).

### 1.3 What is counterfactual evaluation?
A *counterfactual* input is a minimal edit of the original: change **only** the sensitive part.

Paper's formal definition (§3, p.67):
- Sensitive attribute `A`, value set `𝒜` (e.g. `𝒜 = {female, male}`)
- `φ(a)` = the set of sensitive tokens for value `a`
  - e.g. `φ(male) = {he, his, him, husband, Paul}`
- Counterfactual: `x̃ = cf(x, a, ã)` — replace **all** occurrences of every token in `φ(a)`
  with the corresponding token in `φ(ã)`, **leave every other token unchanged**.
- Then compare the sentiment distribution of `LM(x)` vs `LM(x̃)`.

### 1.4 What is a sensitive attribute?
A category of things we want the model to be fair across. The paper uses three:
**Country**, **Occupation**, **Name** (§5.2, p.70).
Formally: `A` is the attribute, `𝒜` is its value set, `A = a` is the random variable taking value `a`.

### 1.5 What are sensitive attribute values?
The members of `𝒜`, each with its own token set `φ(a)`. From Appendix A (p.76–77), verbatim:

| Attribute | # values | Subgroup rule for Group Fairness |
|---|---|---|
| Country | 10 | **each country is its own subgroup** (K = 10) |
| Occupation | 29 | **each occupation is its own subgroup** (K = 29) |
| Name | 34 (17 male + 17 female) | **all male names = one subgroup, all female = one subgroup** (K = 2) |

(Exact token lists are implemented in Phase 3.)

### 1.6 What is a counterfactual input?
Concretely (§4, p.68): given prefix `x_{1:i}` where the **last** token `x_i ∈ φ(a)`, build
`x̃_{1:i} = (x_{1:i-1}, x̃_i)` where `x̃_i ∈ φ(ã)`, `ã ≠ a`.

```
Many tourists visit France for ...
Many tourists visit Italy  for ...
        ^ only this differs
```

### 1.7 What is a sentiment distribution?
- `fs` = a pretrained sentiment classifier with output in **[0, 1]**
- `LM(x)` = a sentence sampled from the LM conditioned on prefix `x`
- `S(x) = fs(LM(x))` = a random variable (the sentiment score of a generated sentence)
- **`P_S(x)` = the distribution of `S(x)`**, estimated by sampling many continuations
  (1,000 per prefix in the paper) and collecting their scores.

It is a *distribution*, not a single number. This matters for the metric choice below.

### 1.8 Why Wasserstein-1 distance?
Paper's reasoning chain (§3, p.67–68), which is elegant:

1. Classic fairness for binary classification = **demographic parity**:
   `p(ŷ=1 | A=a) = p(ŷ=1 | A=ã)`, i.e. equal positive rates.
2. Measured by **demographic disparity**: `|p(S(x) > τ) − p(S(x̃) > τ)|`.
3. But this needs a decision threshold `τ`. `τ` is often user-defined or unknown at training time.
4. Fix: average the disparity over **all** thresholds:
   `E_{τ~U[0,1]} |p(S(x) > τ) − p(S(x̃) > τ)|`
5. **This quantity is exactly the Wasserstein-1 distance** between `P_S(x)` and `P_S(x̃)`. (Eq. 1)
6. This is "Strong Demographic Parity" (Jiang et al. 2019).

So the paper's Equation 1 is literally:

```
W1(P_S(x), P_S(x̃)) = E_{τ~U[0,1]} | p(S(x) > τ) − p(S(x̃) > τ) |
```

**Intuition for a presentation:** W1 is "how much probability mass must you move, and how far,
to turn one distribution into the other." Two advantages the paper states:
- **No threshold needed** (it integrates over all of them).
- **No assumption about distribution shape** — "using Wasserstein-1 distance to compare two
  distributions does not require assumptions on their shape (e.g., symmetry)" (§3, p.68).
- Because scores live in [0,1], **W1 ∈ [0,1]** and is directly interpretable.

Fairness definition (Eq. 2): model is counterfactually fair for sentiment if
`W1(P_S(x), P_S(cf(x,a,ã))) < ε` for every `a, ã`.

**Verified numerically (this project):** the paper's Figure 2 examples reproduce exactly.
Two point masses at 0.555 and 0.445 → W1 = **0.110** (paper labels it 0.1).
Two point masses at 0.505 and 0.494 → W1 = **0.011** (paper labels it 0.01).
The paper's labels are rounded to 1 decimal place; the exact values are 2×|Δ|/2 = |Δ| for point masses.

### 1.9 Individual Fairness (I.F.)
Paper Eq. 3 (p.68):

```
              2                M
I.F.  =  ─────────────    Σ    Σ        W1( P_S(x_m),  P_S(x̃_m) )
         M · |𝒜|(|𝒜|−1)  m=1  a,ã∈𝒜
```

- Inner sum runs over the `|𝒜|(|𝒜|−1)/2` **unordered** pairs of distinct values.
- Outer sum runs over the `M` templates (M = 10).
- The `2 / (M·|𝒜|(|𝒜|−1))` prefactor makes it a **plain average of the W1 values**.
  (Verified: for |𝒜|=3, M=2, the formula equals the arithmetic mean of all 6 W1 terms.)
- **Lower I.F. = better.**
- Concept: "similar individuals should be treated similarly" (Dwork et al. 2012) — two
  sentences that differ only in the sensitive token should get the same sentiment distribution.

### 1.10 Group Fairness (G.F.)
Paper Eq. 4 (p.68):

```
              1
G.F.  =  ─────────  Σ      W1( P_S^a ,  P_S^* )
           |𝒜|    a∈𝒜
```

- Split all evaluation sentences into **K = |𝒜| disjoint subgroups**; a sentence belongs to
  subgroup `a` if it contains a token from `φ(a)`.
- `P_S^a` = sentiment distribution of all generated sentences from subgroup `a`.
- `P_S^*` = sentiment distribution over the **entire evaluation set**.
- Average the per-subgroup W1 distances.
- **Lower G.F. = better.**
- Difference from I.F.: I.F. compares **counterfactual pairs**; G.F. compares each
  **subgroup against the global distribution** (so it also catches "one group is off even if
  pairs look balanced").

### 1.11 Embedding Regularization
Method 1 (§4, p.69). Directly push hidden representations of `x` and `x̃` together.

```
d( h(x), h(x̃) ) :=  1 −  ⟨ h̄(x), h(x̃) ⟩ / ( ‖h̄(x)‖ · ‖h̄(x̃)‖ )      (cosine distance)

h̄(x)  =  average of h^(L−1)(x) and h^(L)(x)      ← last TWO hidden layers
```

Why exactly two layers? The paper gives two reasons, verbatim:
1. *"we want to capture high-level semantics (e.g., sentiments) and embedding in later layers
   represents higher level semantics"* (Tenney et al. 2019).
2. *"we find that averaging too many layers can make the difference between h̄(x_{1:i}) and
   h̄(x̃_{1:i}) very small, reducing the effectiveness of regularization."*

**Strength:** specification-agnostic — it works for any bias measure, not just sentiment.
**Weakness (paper states this):** it can be *too strong*. Forcing hidden states to match makes
the model **ignore the sensitive token entirely**. Perfectly fair, but semantically dead.
Demonstrated in Appendix C.6 with λ=1000.

### 1.12 Sentiment Regularization
Method 2 (§4, p.69). Instead of matching raw hidden states, first **project** them through a
sentiment classifier `f_sh`, then match in that projected space:

```
L_fairness  =  d( f_sh(h(x)),  f_sh(h(x̃)) )        (again cosine distance)
```

- `f_sh`'s output can be **multi-dimensional** (e.g. a hidden layer of the sentiment
  classifier), so cosine similarity still applies.
- *"Applying the classifier f_sh can be seen as a projection from h(x) to a subspace that
  ideally only contains sentiment-related information."*
- **Strength:** avoids over-strong regularization — only the *sentiment* part must match, so
  the model can keep using the sensitive token for meaning.
- **Weakness:** effectiveness is bounded by the quality of `f_sh`.
  (Footnote 3: *"We use a sentiment classifier as a proxy... The classifier itself might not be
  perfect and might exhibit some biases; for this reason we compare several alternatives."*)

### 1.13 Three-step curriculum training
Paper §4, p.69 ("Implementation: Three-step curriculum training"):

| Step | What happens | Output |
|---|---|---|
| **1** | Train LM with plain **cross-entropy next-token loss**. *"a good validation perplexity ensures a relatively good hidden feature space has been learned."* | Baseline LM |
| **2** | **Freeze** the LM. Label corpus sentences with the **Google Cloud sentiment API**. Keep sentences with **\|score\| > 0.7**. Train `f_sh` (3-layer MLP, hidden 128) on **positive-vs-negative only** — neutral excluded, because *"we empirically found that training only on positive and negative sentiment data works better."* | Sentiment projection `f_sh` |
| **3** | **Fix `f_sh`.** Continue training the LM on the **subset of the training set containing any sensitive token**, with `L = L_LM + λ·L_fairness`. | Debiased LM |

Loss in step 3 (p.70):
```
L(x) = L_LM(x) + λ · L_fairness( h(x_{1:i}), h(x̃_{1:i}) )
```
where `L_LM` is standard next-token cross-entropy on the **unperturbed** input `x`.

**Critical constraint (p.70, verbatim):** *"Note that we do not use any template at any step
of training."*

### 1.14 Perplexity
`PPL = exp(average cross-entropy loss per token)`. It measures how well the model predicts
held-out text — low = good language model. The paper reports **two** versions (§5.3, p.71):
- **PPL** — whole test set (overall LM quality).
- **PPL_s** — subset of the test set containing **at least one sensitive token**.
  Justification, verbatim: *"Since the sensitive tokens only exist in a small fraction of test
  data, the subset perplexity PPL_s examines the language model performance specifically in
  contexts containing sensitive tokens."*

Footnote 6 is important for your report: they trained all models to convergence, and to rule
out "the debiased model just trained longer" they **also trained baseline models for the same
extra number of iterations** — *"We found performance differences to be insignificant, both in
terms of perplexity as well as fairness metrics."* → We must do the same (see Phase 5/10 notes).

### 1.15 Semantic Similarity
Two metrics (§5.3, p.71):
- **S.S.** — cosine similarity between the **Universal Sentence Encoder** embeddings of the
  *prefix* and the *generated continuation*. A continuation counts as semantically similar if
  `cos ≥ 0.4`. **S.S. = the fraction of continuations above threshold.**
- **S.S.c** — the **fraction of generated continuations that mention the sensitive attribute
  token**. Second proxy for whether the model still "uses" the sensitive concept.

**Higher = better for both.** The 0.4 threshold is justified with examples in Appendix C.7
(Table 9, p.82): e.g. prefix *"My friend is a baker, and we"* → continuation *"'ve baked a cake
& know it comes from scratch! Lets market a bakeshop!"* scores 0.402 (relevant), while
*"are all kind of crazy about the juicier things in life."* scores 0.121 (not relevant).

### 1.16 Why is there a fairness ↔ semantic-relevance trade-off?
Mechanism (paper §4, p.69 and Appendix C.6, p.81):

- Perfect fairness is trivially achievable by making the model **ignore the sensitive token** —
  then `P_S(x) = P_S(x̃)` exactly and I.F. = 0.
- But then the generated text no longer reflects the context ("sheriff" vs "designer").
- Appendix C.6, Table 8: an **embedding-reg model with λ=1000** produces *almost identical*
  outputs for "sheriff" and "designer" under the same random seed:
  > *"back for a hiring and replication at the SureStart April 23-21 team dealership in
  > South Los Angeles. As assistant, I made a good error of judgment this fall..."*
  — irrelevant to both occupations. Its scores: **S.S. = 4.9, S.S.c = 1.1** (vs baseline
  Occupation S.S. = 17.9, S.S.c = 9.9 in Table 1).
- The paper's conclusion: *"this model achieves very low semantic similarity scores... The
  example shows one extreme for trading off between fairness and performance, and also
  demonstrates the importance of using a semantic relevance metric to evaluate debiased models."*

**The paper's headline finding** (§5.4, p.72): sentiment regularization dominates embedding
regularization on this frontier — *"with similar semantic similarity scores, the sentiment
regularization based models achieve better individual fairness scores than embedding
regularization based models."*

### 1.17 Why multiple sentiment classifiers?
Three are used (§5.3, p.71):

| # | Classifier | Reported accuracy on SST val |
|---|---|---|
| i | Google Cloud sentiment API (score in **[-1, 1]**) | not reported |
| ii | BERT fine-tuned on SST | **92.7%** |
| iii | Opinion-word classifier (Hu & Liu 2004): `p/(p+n)`, 0.5 if none | **69.6%** |

Two stated reasons:
1. **The sentiment classifiers may themselves be biased.** *"We include this simple classifier
   as the Google Cloud sentiment API and the BERT-based classifier may themselves contain bias,
   which has been shown for many sentiment analysis systems (Kiritchenko & Mohammad, 2018).
   The opinion-word-based method, while being less accurate..., is less prone to giving biased
   judgments, as it does not contain sensitive tokens or learned associations: it only relies
   on opinion words."*
2. **Avoid single-measure artifacts.** *"since we also use the Google Cloud sentiment API to
   create the sentiment labels of the training data for learning f_sh, the BERT-based and
   opinion-word-based sentiment classifiers provide additional measures of sentiment, helping
   to avoid findings specific to one sentiment classification system in particular."*

Also note: the paper explicitly warns scores are on **different scales across classifiers**, so
*"the fairness scores are different across sentiment classifiers, [but] we can observe the
overall trends"* (§C.2, p.80). → **Never compare an I.F. number across different classifiers.**

### 1.18 What must the final experimental comparison demonstrate?
Five claims, all of which we must be able to show (or honestly fail to show) at student scale:

1. **Bias exists.** Baseline model: changing only the sensitive token changes the sentiment
   distribution (non-trivial I.F./G.F.).
2. **Both regularizers reduce it.** I.F. and G.F. drop vs baseline, for Country, Occupation, Name.
3. **λ matters, monotonically.** *"A larger regularization parameter λ typically reduces the
   bias further."*
4. **LM quality survives.** PPL and PPL_s stay roughly unchanged.
5. **Sent-Reg > Embed-Reg on the frontier.** At equal S.S., Sent-Reg has lower I.F.
   And at extreme λ, Embed-Reg collapses semantically (C.6).

**Real paper numbers we must NOT claim to reproduce** (Table 1, Occupation attribute):

| Model | WMT-19 PPL | PPL_s | S.S. | S.S.c | WikiText-103 PPL | PPL_s | S.S. | S.S.c |
|---|---|---|---|---|---|---|---|---|
| Baseline | 17.9 | 18.0 | 17.9 | 9.9 | 18.9 | 21.4 | 40.3 | 24.3 |
| Emb-Reg λ=1 | 17.6 | 17.6 | 12.8 | 5.6 | 18.4 | 20.9 | 24.4 | 3.7 |
| Emb-Reg λ=10 | 17.8 | 17.9 | 7.3 | 2.2 | 18.5 | 20.8 | 24.0 | 3.1 |
| Emb-Reg λ=100 | 18.5 | 18.5 | 5.9 | 1.8 | 18.4 | 20.8 | 23.7 | 3.9 |
| Sent-Reg λ=1 | – | – | – | – | 18.4 | 21.0 | 32.4 | 11.9 |
| Sent-Reg λ=10 | 17.6 | 17.7 | 14.5 | 6.4 | 18.4 | 20.9 | 28.2 | 8.9 |
| Sent-Reg λ=100 | 17.7 | 17.7 | 10.8 | 4.5 | 18.4 | 21.0 | 22.6 | 3.4 |
| Sent-Reg λ=1000 | 17.9 | 17.9 | 8.4 | 2.4 | 18.4 | 21.0 | 22.8 | 2.0 |

---

## 2. End-to-End Architecture

### 2.1 The pipeline you asked for

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  DATASET                                                                     │
│  WikiText-103 (28,475 train / 60 val / 60 test articles)                     │
│  [WMT-19 in the paper — dropped at student scale, see Phase 2]               │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│  PREPROCESSING            src/data_preprocessing.py                          │
│  • split articles → train/val/test                                           │
│  • tokenize (GPT-2 BPE, no padding inside sequences)                         │
│  • chunk into fixed-length sequences                                         │
│  • sanity checks: doc counts, token counts, vocab size, examples             │
│  • SENSITIVE-TOKEN DETECTION → tag every sequence containing a φ(a) token    │
│    (only these sequences are used in the debiasing step)                     │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│  LANGUAGE MODEL           src/model.py                                       │
│  decoder-only Transformer (paper: Transformer-XL; ours: GPT-2 backbone)      │
│  MUST expose:  h(x) = [h^(1) ... h^(L)]  ← per-layer hidden states           │
│                logits → next-token prediction → perplexity                   │
│                sampling/generation                                           │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                ↓
╔══════════════════════════════════════════════════════════════════════════════╗
║  STEP 1 — NORMAL LM TRAINING              src/train_baseline.py              ║
║  loss = cross-entropy(next-token prediction)                                 ║
║  log: train loss, val loss, val PPL every N steps                            ║
║  → models/baseline/          ← THE BASELINE MODEL                            ║
╚═══════════════════════════════════════════╤══════════════════════════════════╝
                                            ↓
        ┌───────────────────────────────────┴────────────────────────────┐
        │                                                                │
        ↓                                                                ↓
╔═══════════════════════════════════════════╗   ╔══════════════════════════════════════════╗
║  STEP 2 — SENTIMENT PROJECTION            ║   ║  PHASE 6 — BASELINE BIAS MEASUREMENT     ║
║  src/sentiment_classifier.py              ║   ║  src/counterfactual.py + evaluation.py   ║
║  • FREEZE the LM                          ║   ║  • fill 10 templates × each φ(a) value   ║
║  • extract h from frozen LM               ║   ║  • generate N continuations each         ║
║  • label sentences with sentiment API     ║   ║  • score with 3 sentiment classifiers    ║
║  • keep |score| > 0.7, drop neutral       ║   ║  • save generations CSV                  ║
║  • train 3-layer MLP (hidden 128) = f_sh  ║   ║  • plot distributions (baker vs          ║
║  → models/sentiment_classifier.pt         ║   ║    accountant) → BIAS IS DEMONSTRATED    ║
╚═══════════════════════════╤═══════════════╝   ╚══════════════════════════════════════════╝
                            ↓
╔══════════════════════════════════════════════════════════════════════════════╗
║  STEP 3 — DEBIASING (f_sh FIXED)           src/debias.py                     ║
║                                                                              ║
║  training data = ONLY sequences containing a sensitive token                 ║
║  for each such prefix x_{1:i} with x_i ∈ φ(a):                               ║
║      build x̃_{1:i} = (x_{1:i-1}, x̃_i),  x̃_i ∈ φ(ã)                          ║
║                                                                              ║
║     ┌────────────────────────────┐   ┌────────────────────────────┐          ║
║     │ A. EMBEDDING REGULARIZATION│   │ B. SENTIMENT REGULARIZATION│          ║
║     │  h̄ = avg(h^(L-1), h^(L))   │   │  z  = f_sh(h)              │          ║
║     │  L_fair = 1 − cos(h̄, h̄̃)    │   │  L_fair = 1 − cos(z, z̃)    │          ║
║     │  λ ∈ {1, 10, 100}          │   │  λ ∈ {1, 10, 100}          │          ║
║     └─────────────┬──────────────┘   └──────────────┬─────────────┘          ║
║                   └──────────────┬──────────────────┘                        ║
║                                  ↓                                           ║
║         L(x) = L_LM(x) + λ · L_fairness(h(x_{1:i}), h(x̃_{1:i}))              ║
║                                  ↓                                           ║
║      → models/embedding_reg/lam{1,10,100}/                                   ║
║      → models/sentiment_reg/lam{1,10,100}/                                   ║
║      ✗ NO EVALUATION TEMPLATE IS EVER USED IN TRAINING                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
                            ↓
╔══════════════════════════════════════════════════════════════════════════════╗
║  COUNTERFACTUAL EVALUATION                                                   ║
║  for attribute in {Country, Occupation, Name}:                               ║
║    for template in 10 templates:                                             ║
║      for value in φ:                                                         ║
║        prefix = template.fill(value)                                         ║
║        continuations = model.generate(prefix, N samples, 50 tokens, T=1.0)   ║
║        scores = f_s(continuations)          ← 3 classifiers                  ║
║        → P_S(prefix)                                                         ║
║      for each unordered pair (a, ã):  W1(P_S(x), P_S(x̃))                     ║
║        → Individual Fairness (Eq. 3)                                         ║
║    group every prefix by subgroup → W1(P_S^a, P_S^*) → Group Fairness (Eq.4) ║
╚═══════════════════════════════════════════╤══════════════════════════════════╝
                                            ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│  PERFORMANCE METRICS                                                         │
│  • PPL      — whole test set                                                 │
│  • PPL_s    — test subset containing ≥1 sensitive token                      │
│  • S.S.     — fraction of continuations with USE cosine(prefix, cont) ≥ 0.4  │
│  • S.S.c    — fraction of continuations mentioning the sensitive token       │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│  BASELINE  vs  EMBED-REG(λ)  vs  SENT-REG(λ)     src/visualization.py        │
│  → master results CSV, 10 publication figures, qualitative examples,         │
│    ablations, paper-vs-ours comparison, report, slides, viva Q&A             │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 The three-step curriculum, isolated (Figure 3 of the paper)

```
                    ┌──────────── LANGUAGE MODEL (trainable) ────────────┐
                    │                                                    │
   Documents ──────►│  x  ──► h(x)  ─────┬──────────────► L_LM(x)        │
   (original        │                    │              (cross-entropy)  │
    sentences       │                    │                               │
    containing a    │                    ▼                               │
    sensitive       │             ┌─────────────┐                        │
    token)          │             │  h̄(x)       │                        │
                    │             │ =avg(L-1,L) │──┐                     │
                    │             └─────────────┘  │   cosine            │
                    │                              ├── distance ──► L_fairness
                    │             ┌─────────────┐  │                     │
   Same doc with    │             │  h̄(x̃)      │──┘                     │
   sensitive token  │  x̃ ──► h(x̃)─┤             │                        │
   SWAPPED     ────►│             └─────────────┘                        │
                    │                    │                               │
                    │                    │  (for Sentiment Reg only)     │
                    │                    ▼                               │
                    │             ┌─────────────┐                        │
                    │             │  f_sh  (❄)   │  frozen MLP            │
                    │             └─────────────┘                        │
                    └────────────────────┬───────────────────────────────┘
                                         ↓
                        L(x) = L_LM(x) + λ · L_fairness
```

---

## 3. Paper's Actual Configuration (verbatim, for Phase 25 comparison)

| Item | Paper setting | Where |
|---|---|---|
| Architecture | **Transformer-XL**, decoder-only | §5.1, p.70 |
| WikiText-103 model | **18 layers, d_model 1024, 8 heads, 257M params**, val PPL **17.06** | App. B, p.76–77 |
| WMT-19 model | **48 layers, d_model 1024, 708M params**, val PPL **17.46** | App. B, p.77 |
| Step-1 optimizer | Adam, lr **2.5e-4** | App. B, p.77 |
| Sequence length | **512**; state kept for latest 512 tokens | App. B, p.77 |
| Step-1 batch size | **256** (WMT-19), **512** (WikiText-103) | App. B, p.77 |
| Step-1 steps | **5e5** (WMT-19), **2.5e5** (WikiText-103) | App. B, p.77 |
| Step-1 hardware | **128 Google Cloud TPUv3 cores** | App. B, p.77 |
| `f_sh` | **3-layer MLP, hidden size 128** | App. B, p.77 |
| `f_sh` data selection | **\|Google Cloud score\| > 0.7**, positive-vs-negative only | App. B, p.77 |
| `f_sh` train size | **28,957,245** sentences (WMT-19); **369,594** (WikiText-103) | App. B, p.77 |
| `f_sh` accuracy | **98.8%** (WikiText-103), **98.7%** (WMT-19) | App. B, p.77 |
| `f_sh` training cost | **1 × V100 GPU, 14–21 hrs** | App. B, p.77 |
| Step-3 lr | **2.5e-5** (10× lower than step 1) | App. B, p.77 |
| Step-3 steps | **5e4** (WMT-19), **2.5e4** (WikiText-103) | App. B, p.77 |
| Step-3 batch size | **16** (WMT-19), **32** (WikiText-103) | App. B, p.77 |
| Step-3 hardware / time | **16 TPUv3 cores**, **3–15 hrs** | App. B, p.77 |
| λ (embed-reg) | **{1, 10, 100}** for both datasets | §5.1, p.70 |
| λ (sent-reg) | **{10, 100, 1000}** WMT-19; **{1, 10, 100}** WikiText-103 | §5.1, p.70 |
| Generation | **1,000 sentences per template per sensitive value**, max **50 tokens**, **temperature 1.0** | App. B, p.77 |
| Templates | **M = 10 per attribute** | §5.2, p.70 |
| S.S. threshold | **0.4** cosine, Universal Sentence Encoder | §5.3, p.71; App. C.7 |
| Human eval | **19 annotators**, 50–100 sentences each, **2 annotators/sentence** | §5.5, p.72 |
| Inter-annotator | Cohen's κ = **0.47** (sentiment), **0.45** (relevance) | App. D, p.81 |
| Human↔auto Spearman | Google API **0.75**, BERT **0.79**, opinion-word **0.67**; S.S. **0.72**, S.S.c **0.63** | §5.5, p.73 |
| Human-scored I.F. | designer/accountant: **0.333 → 0.056**; Libya/Iceland: **0.291 → 0.155** | §5.5, p.73 |
| Extra baseline | **GPT-2 1.5B** evaluated for comparison (App. C.4) | App. C.4, p.80 |
| Distinct words | `argmax_w p(w\|a)/p(w\|ã)` — top-10 per pair | App. C.8, p.81 |

**Scale gap in one line:** the paper's Step 1 used **128 TPUv3 cores for 250k–500k steps**.
A Colab T4 is roughly 1/1000th of that. This is addressed head-on in Phase 2 and Phase 5.
