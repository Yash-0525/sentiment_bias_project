# Presentation Slides (12–15)  
### What to put on the slide + what to say

---

## Slide 1 — Title

**On slide**

- Title: Reducing Sentiment Bias in LMs via Counterfactual Evaluation  
- Subtitle: Student-scale reproduction of Huang et al. (EMNLP Findings 2020)  
- Your name, college, year  
- Kaggle / GPT-2 / WikiText-103  

**Say:**  
“This project measures and tries to reduce a specific bias: when only one sensitive word in a prompt changes, the sentiment of the model’s generated text can shift. We follow Huang et al. 2020 at a scale I can actually train.”

---

## Slide 2 — Problem

**On slide**

- Prompt: “My friend is a/an **baker**, and we…” vs **accountant** / **sheriff** vs **designer**  
- Same model → different sentiment distributions of continuations  
- Figure idea: two histograms, shifted  

**Say:**  
“This is not about rude words. It’s about systematic positivity or negativity depending on occupation, country, or name—even when the rest of the sentence is identical.”

---

## Slide 3 — Motivation

**On slide**

- LMs used in real products  
- Stereotypes in training data → stereotypes in generation  
- Need **measurable** fairness + **trainable** fix  

**Say:**  
“If we can’t measure it, we can’t fix it. The paper gives a clear metric and two regularizers. My goal was to implement that pipeline so I can explain every step in a viva.”

---

## Slide 4 — Research paper

**On slide**

- Huang et al., Findings of EMNLP 2020  
- Counterfactual evaluation  
- W1 → Individual Fairness & Group Fairness  
- Embedding Reg + Sentiment Reg  
- Three-step curriculum  

**Say:**  
“I treat the paper as the primary source. I copy the math and protocol; I reduce model size and steps for hardware, and I label every substitution.”

---

## Slide 5 — Existing system (paper)

**On slide**

- Transformer-XL 257M–708M from scratch  
- 128 TPUv3  
- WikiText-103 + WMT-19  
- 1000 samples per prefix  

**Say:**  
“The original training budget is roughly three orders of magnitude beyond a free T4. A fake ‘full reproduction’ would be dishonest.”

---

## Slide 6 — Proposed system (ours)

**On slide**

- GPT-2 small fine-tune on WikiText-103  
- Paper split 28,475 / 60 / 60  
- Same losses, same curriculum, same templates for **eval only**  
- Kaggle T4, fp32  

**Say:**  
“We keep the scientific comparison: baseline and debiased models share the same start checkpoint and evaluation protocol; only the fairness term differs.”

---

## Slide 7 — Architecture

**On slide**

```
Data → Baseline LM → f_sh → Debias (Embed | Sent)
         ↓
   Counterfactual gen → Sentiment → W1 → Compare
```

**Say:**  
“Walk top to bottom. Templates never enter training. Debiasing uses only corpus sequences that already contain sensitive tokens.”

---

## Slide 8 — Three-step curriculum

**On slide**

| Step | What | Output |
|------|------|--------|
| 1 | LM CE only | Baseline PPL 22.28 |
| 2 | Freeze LM; train f_sh | Acc 0.84 |
| 3 | L_LM + λ L_fair | Embed & Sent models |

**Say:**  
“Step 1 learns language. Step 2 learns a sentiment subspace. Step 3 pushes original and counterfactual representations together in that space or in raw hidden space.”

---

## Slide 9 — Counterfactual evaluation

**On slide**

- Change **only** sensitive token  
- CF success in training data: **99.4%**  
- Sample 100 continuations, 50 tokens, T=1.0  
- Score with opinion-word and BERT-SST  

**Say:**  
“If non-sensitive context changed, the comparison would be invalid. We verified swaps are single-token and highly reliable.”

---

## Slide 10 — Fairness metrics

**On slide**

- W1 = earth-mover distance between sentiment distributions  
- Lower W1 → fairer  
- We report **pair W1 probes** (not full I.F. table)  
- Why W1: no single threshold \(\tau\)  

**Say:**  
“Full I.F. averages many pairs and templates. Due to compute I locked pair probes—especially sheriff vs designer, which shows large baseline bias.”

---

## Slide 11 — Embedding vs Sentiment Reg

**On slide**

- Embed: \(1-\cos(h̄(x), h̄(\tilde{x}))\)  
- Sent: \(1-\cos(f_{sh}(h̄(x)), f_{sh}(h̄(\tilde{x})))\), f_sh frozen  
- λ = 10, 1000 steps  

**Say:**  
“Embedding matches full hidden states—can be strong but blunt. Sentiment matches only the projected sentiment subspace—in theory gentler on meaning.”

---

## Slide 12 — Experimental setup

**On slide**

- Kaggle T4, fp32  
- WikiText-103, GPT-2, seq 256  
- Baseline 3000 steps; debias 1000  
- n=100 generations  

**Say:**  
“Every number I show is from this setup. I never paste paper table values as mine.”

---

## Slide 13 — Results

**On slide**

| Model | PPL | Sheriff–designer opinion W1 |
|-------|-----|------------------------------|
| Baseline | 22.28 | 0.164 |
| Embed λ10 | 22.24 | **0.142** |
| Sent λ10 | 22.30 | 0.179 |

Also: baker pair — no gain; CF 99.4%; f_sh 0.84  

**Say:**  
“PPL stays flat—we didn’t break the LM. Embed-reg reduces W1 on the high-bias pair with the opinion-word scorer. Sent-reg at this budget does not help these pairs. Baker remains inconclusive.”

---

## Slide 14 — Graphs / examples

**On slide**

- Histograms: baseline sheriff vs designer  
- Bar: W1 by model  
- Optional: one qualitative triple of continuations  

**Say:**  
“Point at the shift in histograms for baseline bias. Point at the bar drop for embed-reg. Be honest that bert_sst doesn’t show the same drop.”

---

## Slide 15 — Conclusion and future work

**On slide**

- Delivered working curriculum + metrics  
- Bias measurable; PPL preserved; partial embed gain  
- Future: full I.F., λ grid, S.S., longer Step-3, human eval  

**Say:**  
“Success for a capstone is a correct, honest pipeline—not a fake leaderboard match to DeepMind’s TPU run. I’m happy to discuss limitations and next experiments.”

---

## Timing guide (10–12 min talk)

| Slides | Minutes |
|--------|---------|
| 1–3 | 2 |
| 4–8 | 3 |
| 9–12 | 3 |
| 13–15 | 3 |
| Buffer / demo | 1–2 |
