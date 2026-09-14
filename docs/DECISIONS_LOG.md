# DECISIONS LOG — read this before every phase

Every decision made during the project is recorded here so later phases stay consistent
with earlier ones. **Do not contradict an entry without adding a new entry that supersedes it.**

| # | Phase | Decision | Rationale | Status |
|---|---|---|---|---|
| D01 | 0 | Primary source = `uploads/2020.findings-emnlp.7.pdf`, extracted to `paper/paper_full.txt` | User instruction: paper is PRIMARY SOURCE | LOCKED |
| D02 | 0 | **User: use exactly the paper's datasets, add nothing new.** Primary = **WikiText-103** (paper-exact, 28,475/60/60). WMT-19 = documented partial track only. | Both verified live 2026-09-14. See `docs/DATASET_VERIFICATION.md`. WMT-19 has **no document boundaries** in the 2019 release, so the paper's article-level split is unreconstructable even with unlimited compute. | **LOCKED** |
| D03 | 0 | Environment = **Kaggle PRIMARY (Option A)**; Colab is backup only. `src/paths.py` auto-detects `colab` / `kaggle` / `local`. | User chose Option A 2026-09-14. Kaggle: ~30 GPU-h/wk published quota. Persistence must be opted in (D19). | **LOCKED** |
| D21 | 1 | Fidelity level = **L1 method-faithful** (not L0 bit-identical). WikiText-103 exact; GPT-2 small fine-tune; paper-exact losses/metrics/curriculum; 100 gen samples; BERT-SST + opinion-word; MiniLM for S.S. | Free single-GPU cannot run Transformer-XL-from-scratch + WMT-19. Central claim is relative baseline vs regularizers. | **LOCKED** |
| D04 | 0 | LM = **fine-tune GPT-2 small (124M, 12 layers, d=768)** on WikiText-103 as Step 1 | User selected. 12 layers ⇒ `h̄ = mean(h^(11), h^(12))`, exactly the paper's last-two-layer rule. ~2 h on T4. | **LOCKED** |
| D05 | 0 | `h̄(x) = mean(h^(L−1)(x), h^(L)(x))` — last **two** hidden layers, exactly as paper | Paper §4, p.69 | LOCKED |
| D06 | 0 | `L_fairness = 1 − cosine(h̄(x), h̄(x̃))` for Embed-Reg; `1 − cosine(f_sh(h), f_sh(h̃))` for Sent-Reg | Paper §4, p.69 | LOCKED |
| D07 | 0 | Step-3 loss `L(x) = L_LM(x) + λ·L_fairness`, `L_LM` computed on the **unperturbed** input | Paper p.70 | LOCKED |
| D08 | 0 | **Evaluation templates are never used in any training step** | Paper p.70, verbatim | LOCKED |
| D09 | 0 | λ grid: Embed-Reg **{1, 10, 100}**, Sent-Reg **{1, 10, 100}** (paper's WikiText-103 grid) | Paper §5.1, p.70 — we follow WikiText-103 since that is our corpus | LOCKED |
| D10 | 0 | Samples per template per sensitive value: **100** (paper: 1000). 50 for smoke tests. | 1000 × 30 prefixes × 10 templates × 3 attributes = 900k generations/model ≈ hours on T4. 100 keeps W1 stable enough and runs in ~30 min/model. | **LOCKED** (free-tier budget) |
| D11 | 0 | Generation: **max 50 tokens, temperature 1.0** — unchanged from paper | Paper App. B, p.77 | LOCKED |
| D12 | 0 | Sentiment classifiers: (1) **primary = BERT fine-tuned on SST** (maps to the paper's classifier ii, which the paper reports at 92.7% SST-val accuracy), (2) **opinion-word p/(p+n), 0.5 if none** (paper's classifier iii, 69.6%). Google Cloud API **not used**. | User selected local-only. Replaces the paper's classifier i. **Must be labelled a substitution.** We will report our own SST-val accuracy, never claim the paper's 92.7%. | **LOCKED** |
| D12b | 7 | `f_sh` training labels come from the **primary BERT classifier**, not the Google API | The paper labelled with the Google API. Consequence: primary-classifier results are partly circular, so **the opinion-word classifier is the independent held-out measure.** State this in the report. | **LOCKED** |
| D13 | 0 | Semantic similarity: **sentence-transformers (all-MiniLM-L6-v2)** replacing Universal Sentence Encoder. Threshold **0.4** kept; also report **mean cosine** (threshold-free) because the 0.4 cut-off was calibrated for USE, not MiniLM. | USE is TensorFlow/TF-Hub, ~1 GB, awkward in a PyTorch Colab. MiniLM is 80 MB and PyTorch-native. **Must be labelled a substitution.** | **LOCKED** |
| D14 | 0 | W1 implemented two ways and cross-checked: hand-rolled CDF-gap formula + `scipy.stats.wasserstein_distance`, must agree to <1e-12 | Already verified in the workspace: 300 random pairs, max diff **5.55e-17** | DONE |
| D15 | 0 | Report the paper's numbers only as "paper reference"; never as our results | User instruction: never fabricate | LOCKED |
| D16 | 0 | Train an **extended-iteration baseline** (same extra steps as the debiasing run, no fairness loss) as an extra control | Paper footnote 6, p.71 did exactly this to rule out "trained longer" as a confounder | PLANNED (Phase 10) |
| D17 | 0 | Compute budget = **free Colab T4, ~15–30 GPU-h/week**. Total plan sized to ≈ **12–18 GPU-hours.** | User selected. Forces D04 and D10. Every training script must checkpoint to Drive so a 12 h disconnect costs a step, never a model. | **LOCKED** |
| D19 | 1 | **Kaggle is a first-class target**, not an afterthought. Added `paths.bootstrap()`, `paths.persistence_status()`, `paths.checkpoint_dir()`, `save_state()/load_state()`, and `notebooks/kaggle_bootstrap.ipynb`. | Kaggle's `/kaggle/working` is **not** persistent by default — Kaggle staff confirm interactive-session output is discarded unless you set **Session options → Persistence → Files Only** or finish with **Save & Run All (Commit)**. `check_environment` check 10 reports this as **WARN on Kaggle** (the switch is invisible from inside the VM) and **FAIL on Colab/local** (fixable immediately). | **LOCKED** |
| D20 | 1 | **The project trains in fp32.** No bf16, no fp16 autocast. | Kaggle's free GPUs are P100 (sm_70) and T4 (sm_75) — **neither supports bf16**. Independent of that: the debiasing loss is `λ·(1 − cos(·,·))` with λ up to 100, so a small relative error in the cosine becomes a large error in the gradient. `check_environment` check 11 records the GPU's capability so this choice stays explicit. | **LOCKED** |
| D18 | 1 | Library versions validated in-workspace: torch 2.14, transformers **5.17.0**, datasets 5.0.1, sentence-transformers 6.0.1, scipy 1.17.1. `check_environment.py` enforces minima, not exact pins. | transformers 5.x differs from 4.x in places. Check 5 in `check_environment.py` executes the exact `output_hidden_states` → `h̄ = (h[-2]+h[-1])/2` path Phase 10 depends on, so a breaking API change fails loudly here instead of silently corrupting results later. | **LOCKED** |

---

## Compute reality — the honest version

**What the paper spent on Step 1 (baseline LM):**
- 128 Google Cloud TPUv3 cores
- 250,000 steps × batch 512 × seq 512 ≈ 65 billion token-passes (WikiText-103)
- On a 257M-parameter model

**What you have (Colab free tier, verified Sept 2026):**
- 1 × NVIDIA T4, 16 GB VRAM, ~12–13 GB system RAM
- Session cap ≈ **12 hours**, disconnects after ≈ **90 min idle**
- Roughly **15–30 GPU-hours per week**, dynamic and not guaranteed
- GPU assignment is not guaranteed at all on the free tier

**Consequence:** training a 257M Transformer-XL from scratch on 100M tokens is off by roughly
**three orders of magnitude** in compute. A from-scratch tiny model (4 layers, d=256) on a small
WikiText-103 slice would *run*, but it would produce incoherent text, the sentiment signal in its
generations would be near-noise, and the whole experiment would measure noise instead of bias.
That is worse than a smaller faithful experiment — it produces a project that cannot demonstrate
anything.

**The substitution (D04), stated plainly:**

| | Original paper | Our student version | Changed? |
|---|---|---|---|
| Corpus | WikiText-103 + WMT-19 | **WikiText-103 only** (subset) | YES |
| Architecture family | Transformer-XL, from scratch | **GPT-2 (decoder Transformer), fine-tuned** | YES |
| Params | 257M / 708M | **124M** | YES |
| Step-1 objective | cross-entropy next-token | **cross-entropy next-token** | NO |
| Step-1 hardware | 128 TPUv3 cores | **1 T4** | YES |
| Step-1 steps | 250k | **~2–3k** | YES |
| `h̄` definition | avg of last two layers | **avg of last two layers** | NO |
| Fairness loss | 1 − cosine | **1 − cosine** | NO |
| Curriculum | 3 steps, same order | **3 steps, same order** | NO |
| Templates in training | never | **never** | NO |
| I.F. / G.F. definitions | Eq. 3 / Eq. 4 | **Eq. 3 / Eq. 4 verbatim** | NO |
| Generation | 1000 samples, 50 tok, T=1.0 | **100 samples, 50 tok, T=1.0** | samples only |

**Does this damage the scientific comparison? No — and here is the argument to make in your viva.**

The paper's central claim is *relative*: **"regularization reduces I.F./G.F. compared to the
baseline, at comparable PPL and S.S."** For that claim to hold in our reproduction, we need:

1. The baseline and every debiased model start from **the same checkpoint**. ✅ (all fork from the
   Step-1 baseline)
2. They receive the **same debiasing budget** (same steps, same data subset, same lr), differing
   **only** in the fairness term and λ. ✅
3. They are scored by the **same** sentiment classifiers, templates, generation settings. ✅
4. The model is good enough that generated text carries a real sentiment signal. ✅ (this is
   exactly why we do NOT train from scratch)

If (1)–(4) hold, any I.F. difference is attributable to the regularizer. What we **cannot** claim:
- that our absolute I.F. ≈ 0.04 matches the paper's 0.04 (different model, corpus slice, and
  sentiment classifier → different score scale; the paper itself notes fairness scores are not
  comparable across classifiers, §C.2 p.80)
- that our PPL ≈ 18.9 (a 124M GPT-2 fine-tuned on a slice will not hit the 257M model's val PPL 17.06)
- anything about WMT-19 at all

**Bonus:** the paper *itself* evaluates **GPT-2 1.5B** with these exact metrics (App. C.4,
Figs 16–17) and finds it *worse* than their trained baselines. So using a GPT-2-family model as
the subject of this metric suite is precedented by the paper, not a deviation from it.

---

| D22 | 2 | Phase 2 seq_len default = **256** (paper 512). Tokenizer = **gpt2**. Full article split **28,475/60/60** kept. Sensitive lists loaded from Appendix A into `src/sensitive_attributes.py` (also covers Phase 3 lists). | GPT-2 context and T4 VRAM; packing at 512 doubles tokens/step. Split rule unchanged. | **LOCKED** |
| D23 | 2 | Phase 3 sensitive lists shipped early inside Phase 2 so packed sequences can be flagged for Step-3 debiasing subset. Templates (Phase 4) still evaluation-only and **not** written into any training file. | Paper p.70 | **LOCKED** |
| D24 | 5 | Baseline train defaults: GPT-2 small, lr **5e-5**, micro_batch **4**, accum **8** (eff 32), **3000** steps, eval/save every **500**, fp32, resume on. Smoke: 50 steps. | Kaggle T4 ~1–3 h full; paper 250k scratch steps impossible. Relative comparisons still valid (D21). | **LOCKED** |

---

## Project status checklist

```
PHASE  0  Paper understanding .................... DONE
PHASE  1  Environment + project setup ............ DONE (code ready; user Kaggle run in progress / verify)
PHASE  2  Dataset ................................ DONE (Kaggle: 28475/60/60, 445592 train seqs, verified)
PHASE  3  Sensitive attributes ................... DONE (Appendix A lists; detection live in Phase 2 stats)
PHASE  4  Sentence templates ..................... DONE (730 prompts; baker/accountant pair verified)
PHASE  5  Baseline language model ................ IN PROGRESS (code ready; user trains on Kaggle GPU)
PHASE  5  Baseline language model ................ PENDING
PHASE  6  Baseline bias evaluation ............... PENDING
PHASE  7  Sentiment classifiers .................. PENDING
PHASE  8  Sentiment projection classifier ........ PENDING
PHASE  9  Counterfactual pairs for training ...... PENDING
PHASE 10  Embedding regularization ............... PENDING
PHASE 11  Sentiment regularization ............... PENDING
PHASE 12  Three-step curriculum check ............ PENDING
PHASE 13  Fairness metrics (W1 / I.F. / G.F.) .... PENDING
PHASE 14  PPL + PPL_s ............................ PENDING
PHASE 15  Semantic similarity .................... PENDING
PHASE 16  Final evaluation ....................... PENDING
PHASE 17  Visualizations ......................... PENDING
PHASE 18  Qualitative analysis ................... PENDING
PHASE 19  Ablations .............................. PENDING
PHASE 20  Human evaluation ....................... PENDING
PHASE 21  Reproducibility ........................ PENDING
PHASE 22  Error handling (continuous) ............ ACTIVE
PHASE 23  Scientific validation checklist ........ PENDING
PHASE 24  Final results table .................... PENDING
PHASE 25  Paper-vs-ours comparison ............... PENDING
PHASE 26  Final report ........................... PENDING
PHASE 27  Presentation ........................... PENDING
PHASE 28  Viva preparation ....................... PENDING

ERRORS: none logged
```
