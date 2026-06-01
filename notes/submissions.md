# Submission log

Budget is tight (final submission should be the best), so we submit only when the
**local holdout** says we genuinely improved. The leaderboard is for calibration,
not tuning. Local holdout = last 16 days of train (cutoff 2017-07-31). LB top ≈ 0.376.

| Iter | Date | Model | Local RMSLE | Kaggle LB | File |
|---|---|---|---|---|---|
| 1 | 2026-06-01 | Global LightGBM, `direct_safe` (lag≥16) + promo/calendar/store/family | 0.42751 | _pending_ | `submissions/iter1_lgbm_direct_safe.csv` |

### Iteration 1 — purpose: calibrate
First real ML model and our first submission. Goal isn't a great score yet — it's to
confirm our local holdout **tracks** the leaderboard. If LB ≈ 0.43, our whole "trust
the harness" strategy is validated and we can engineer features with confidence. If
they diverge badly, the validation scheme is broken and we fix that before anything else.

**Next after calibration:** holidays (transferred-day trap), oil, store metadata,
earthquake/payday flags, richer recency → push below 0.427.
