# Project summary — Store Sales (Favorita) forecasting

**Final Kaggle public LB: 0.43646 RMSLE.** Built from scratch, one *validated* step at a
time — no copied notebooks. (Leaderboard top ≈ 0.376; public-notebook cluster ≈ 0.378.)

## Results — every number earned and explained

| Stage | Local | Kaggle LB | Note |
|---|---|---|---|
| Per-series mean (all history) | 0.69453 | — | baseline floor (~2× off) |
| Recent 14-day mean | 0.51911 | — | recency fixes the uptrend |
| Level × weekday shape | 0.49871 | — | best naive (decomposition) |
| **iter1** global LightGBM | fold0 0.428 | 0.48509 | first ML; calibration submission |
| **iter2** tuned + holidays/store-meta/oil | fold0 0.418 | 0.51756 | **WORSE** — overfit the wrong season |
| **iter3** simple + promotions | Jul 0.422 | 0.43893 | the breakthrough |
| **iter4** + future-promo | — | 0.44188 | helped local Aug, didn't transfer |
| **iter5** ensemble (final) | ≤ single all seasons | **0.43646** | best; the reserved 5th |

## The final pipeline (iter5)
- **One global LightGBM** across all 1,782 (store, family) series; target = `log1p(sales)`
  (RMSLE lives in log space), predictions `expm1`'d and clipped at 0.
- **Features:** store & family identity; calendar (day-of-week, month, payday); `onpromotion`;
  **leakage-safe recency** (lag-16 + rolling means shifted ≥16 days, so every feature is legal
  for all 16 forecast days); **promotion intensity** (trailing 7/28-day sums + store-day total +
  next-7-day window — `onpromotion` is known for the test, so future-promo is fair game).
- **Recent data only** (2016+); older years *hurt* (trend / regime shift).
- **Ensemble:** average (in log space) of 4 decorrelated models — 3 seeds of the promo model
  + 1 with future-promo for diversity.

## Validation — the heart of the project
- Time-based **16-day holdout** matching the test shape (28,512 rows); never random splits.
- **Local did NOT track the LB.** iter2 improved locally but *regressed* on Kaggle (0.485→0.518).
- **Diagnosis:** we validated on June/July windows, but the test is **late August** — wrong
  season — and the extra capacity overfit those easy interior windows.
- **Fix — multi-season agreement:** a change must help (or hold) across July *and* recent-August
  holdouts; reject anything that helps one season and hurts another. This guard correctly
  rejected the exact features that sank iter2, and correctly kept promotions + the ensemble.
- Even so, the local→LB offset is unstable. We use local for *ranking*, theory for *direction*,
  and spend submissions sparingly.

## Biggest mistake
**iteration 2.** Tuned hyperparameters and selected features on non-representative interior
folds, trusted the local gain, and shipped it — LB went *backwards* (0.485 → 0.518). Caught only
because iter1 was spent as a calibration submission. **Lesson: a validation set that doesn't
match the test is worse than none — it manufactures false confidence.**

## Concepts this project actually taught
RMSLE & `log1p`; temporal leakage and horizon-safe features; recursive vs direct multi-step
forecasting (recursive lost to compounding errors; per-horizon ≈ direct-safe but 16× the cost);
global vs local models; recency beats "more data" under a trend; **feature importance ≠
generalization value** (oil ranked high, hurt most); **overfitting the validation set**;
ensembling for variance reduction.

## If I had two more weeks
- Per-family (or family-group) models — families behave very differently.
- Explicit zero/sparse-series handling (a two-stage "will it sell?" + "how much?").
- A regime-matched, season-aligned CV that actually tracks the LB.
- Richer promotion features and per-family-validated holiday effects.
