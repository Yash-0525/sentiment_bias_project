# Option A lock — 2026-09-15

User chose **Option A**: stop heavy training; finalize documentation from measured results.

## Locked claims (allowed)

1. Pipeline implements Huang et al. curriculum at student scale.  
2. WikiText-103 paper article split reproduced.  
3. Baseline val PPL ≈ **22.28**.  
4. \(f_{sh}\) val acc ≈ **0.84**.  
5. CF pair construction success **99.4%**.  
6. Baseline counterfactual bias exists (sheriff–designer opinion W1 **0.1635**).  
7. Embed-Reg λ=10 reduces sheriff–designer **opinion** W1 to **0.1422**; PPL **22.24**.  
8. Sent-Reg λ=10 does not improve locked pair probes.  
9. Baker–accountant probes show no fairness win at this budget.

## Forbidden claims

1. Full reproduction of paper Table 1 I.F./G.F. numbers.  
2. “Sentiment regularization worked” without new positive evidence.  
3. Matching paper PPL ≈ 18 or f_sh acc 98.8%.  
4. WMT-19 results.  
5. Bit-identical Transformer-XL run.

## Next student actions (no new science required)

1. Kaggle: **Save Version → Commit** all models + `results/compare/`.  
2. Download compare CSVs / PNGs for the report appendix.  
3. Use `reports/PROJECT_REPORT.md`, `PRESENTATION_SLIDES.md`, `VIVA_QA.md`.  
4. Optional later: full I.F. code (Phase 13) as future work—not required for Option A freeze.
