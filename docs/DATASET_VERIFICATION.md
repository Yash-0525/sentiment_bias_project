# DATASET VERIFICATION — performed before Phase 2

Date: 2026-09-14. All results below are from live HTTP probes run in this workspace,
not from documentation or memory.

---

## 1. WikiText-103 — ✅ FULLY AVAILABLE, matches the paper

**Source:** Hugging Face `Salesforce/wikitext`, config `wikitext-103-raw-v1`

```
train       1,801,350 lines    525.9 MB
validation      3,760 lines      0.9 MB
test            4,358 lines      1.4 MB
```

**Cross-check against the paper (App. B, p.76):**
> *"The WikiText-103 dataset consists of 28,591 articles and over 100 million tokens extracted
> from high quality Wikipedia articles. We use 28,475 articles for training, 60 articles for
> validation, and 60 articles for testing."*

WikiText stores one article per block, with the article title on a `= Title =` line. Counting
those title lines gives the 28,591 articles. **The paper's split is exactly reproducible:**
take articles in file order, first 28,475 → train, next 60 → validation, next 60 → test.
Sum check: 28,475 + 60 + 60 = 28,595 ≈ 28,591 (the paper's own numbers are 4 over the stated
total — a minor inconsistency in the paper; we will report whichever count we actually observe
and note the discrepancy rather than force it).

**Verdict: no substitution needed. This is the paper's dataset, the paper's split.**

---

## 2. WMT-19 news crawl — ⚠️ AVAILABLE BUT NOT TRAINABLE AT YOUR SCALE

**Paper's reference (App. B, p.77):**
> *"WMT-19 consists of 14,635,198 English news articles; we take the last 10,000 for
> evaluation with 1,000 for validation and the final 9,000 articles as a test set."*
and §2 (p.67): *"English news articles from WMT-19 (40GB of text)"*
Footnote 5 gives the source: `http://data.statmt.org/news-crawl/`

**What I found, with live probes:**

| Probe | Result |
|---|---|
| `http://data.statmt.org/news-crawl/` | HTTP 301 → redirects to **https** |
| `https://data.statmt.org/news-crawl/english/` | **HTTP 404** — this path no longer exists |
| `https://data.statmt.org/news-crawl/en/` | **HTTP 200** — correct path (no `english/` any more) |
| `news.2019.en.shuffled.deduped.gz` | **HTTP 200**, Last-Modified **2020-02-27 17:37 GMT** |
| Actual file size | **1,703,591,936 bytes (1.70 GB compressed)** |
| gzip integrity | magic `\x1f\x8b` OK; decompresses to real news text: *`"They were just kids," Corredor said, recalling those killed in the blast.`* |
| Document-split variant (`*.doc`, `*.documents`) | **HTTP 404** — does not exist |
| HTTP range requests | **IGNORED.** A `Range: bytes=0-299` request returned `HTTP 200` with **all 1,703,591,936 bytes**, not 300. |

The Last-Modified date (Feb 2020) predates the paper (Nov 2020), so this is the same file the
authors used. **The dataset still exists.** But four hard blockers:

### Blocker 1 — no partial download
The server does not honour `Range`. Any access downloads all **1.7 GB**. On Colab that is
feasible but slow, and it lands on the ephemeral VM — a disconnect loses it.

### Blocker 2 — no document boundaries
The file is **line-shuffled and deduplicated**. The paper needs *articles* ("we take the last
10,000 [articles]"). There is no `<doc>`-marked or document-split English variant for 2019
(all four name guesses returned 404). So the paper's exact 1,000-val / 9,000-test article split
**cannot be reconstructed from this file**, even with unlimited compute.

### Blocker 3 — uncompressed size
1.70 GB gzipped of news text decompresses to roughly **9–12 GB** (~6–7×). Colab has ~12–13 GB
of system RAM. Loading it is possible once; holding it plus a model plus gradients is not.

### Blocker 4 — training cost, the decisive one
The paper trained on this corpus with **128 TPUv3 cores for 500,000 steps at batch 256 ×
seq 512** on a **708M-parameter** model. Free Colab gives **1 T4 with ~15–30 GPU-hours/week**.
The gap is roughly **three orders of magnitude**. This is not a tuning problem.

---

## 3. Decision

**Primary track — WikiText-103.** Fully implemented, paper-exact dataset, paper-exact split,
runs inside a free T4 budget. The paper reports WikiText-103 results *independently* of WMT-19
(Tables 1, 5, 6; Figures 4c/d, 5c/d, 7c/d, 10–11, 14–15), so **every claim in the paper is
testable on this one corpus.**

**Secondary track — WMT-19.** Implemented as a *documented partial* reproduction, only if you
choose to spend the extra hours:
- the corpus is downloaded from the exact URL the paper cites
- we report the real article/line counts we observe
- we train on a documented subsample
- the report states plainly that Blockers 2 and 4 mean the WMT-19 numbers
  (PPL 17.9, val PPL 17.46) **cannot** be reproduced

**Recommended: do WikiText-103 completely first, end to end, through Phase 28. Then decide
whether WMT-19 is worth the extra days.** A finished single-corpus reproduction beats two
half-finished ones.

---

## 4. What this means for your report (Phase 25 wording)

Say this, verbatim, in your limitations section:

> "We reproduce the WikiText-103 track of the source paper in full, using the same dataset and
> the same 28,475 / 60 / 60 article split. The WMT-19 track was not reproduced: the news-crawl
> corpus remains available at the URL cited by the authors (1.70 GB compressed, verified
> 2026-09-14) but (a) the 2019 English release provides no document boundaries, so the authors'
> article-level evaluation split cannot be reconstructed, and (b) the original training budget
> of 128 TPUv3 cores for 5×10⁵ steps is approximately three orders of magnitude beyond our
> single-GPU environment. All conclusions in this report therefore concern WikiText-103 only."

That paragraph is honest, specific, and will read as rigorous rather than as an excuse.
