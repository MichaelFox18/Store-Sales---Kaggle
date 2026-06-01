# Submission log

Budget is tight (final submission should be the best), so we submit only when the
**local holdout** says we genuinely improved. The leaderboard is for calibration,
not tuning. Local holdout = last 16 days of train (cutoff 2017-07-31). LB top ≈ 0.376.

| Iter | Date | Model | Local RMSLE | Kaggle LB | File |
|---|---|---|---|---|---|
| 1 | 2026-06-01 | Global LightGBM, `direct_safe` (lag≥16) + promo/calendar/store/family | 0.42751 | **0.48509** | `submissions/iter1_lgbm_direct_safe.csv` |
| 2 | 2026-06-01 | + holidays + store-meta, tuned (lr0.03 / ~600 trees / reg); oil & recency2 dropped | 0.41765 | **0.51756 ⚠ WORSE** | `submissions/iter2_lgbm_holidays_storemeta_tuned.csv` |
| 3 | 2026-06-01 | iter1-simple + promotion features (trail7/28 + store-day intensity); 2016+ | Jul-holdout 0.42154 | **0.43893 ✓ BEST** | `submissions/iter3_lgbm_promo.csv` |
| 4 | 2026-06-01 | iter3 + future-promo (promo_lead7); holidays/store-meta dropped (failed guard) | Jul 0.4223, Aug −0.027 | 0.44188 (≈iter3) | `submissions/iter4_lgbm_promo_lead.csv` |
| 5 | 2026-06-01 | FINAL ensemble: avg(3×iter3 seeds + iter4) in log space | ≤ single on ALL seasons (Aug2015 −0.015) | **0.43646 ✓ BEST** | `submissions/iter5_ensemble_iter3x3_iter4.csv` |

**Final: iter5 ensemble, LB 0.43646** — best of all five, and the one we reserved. The
multi-season-validated ensemble delivered the predicted variance-reduction gain over iter3.

### Iteration 1 — purpose: calibrate
First real ML model and our first submission. Goal isn't a great score yet — it's to
confirm our local holdout **tracks** the leaderboard. If LB ≈ 0.43, our whole "trust
the harness" strategy is validated and we can engineer features with confidence. If
they diverge badly, the validation scheme is broken and we fix that before anything else.

**Result: local 0.42751 → LB 0.48509 — harness optimistic by ~0.057.** Not broken
(leakage-safe; still beat the naive models directionally), but our single window
(Jul31–Aug15) was easier than the real test (Aug16–31). Two likely contributors:
(1) single-window variance, and (2) a continuing uptrend leaving the shift-16 level
stale for late-test days → under-prediction, which RMSLE punishes extra. Fix before
tuning features: **rolling-origin CV** (avg several windows) for an honest number.

**Next after calibration:** holidays (transferred-day trap), oil, store metadata,
earthquake/payday flags, richer recency → push below 0.427.

### Iteration 2 — features + tuning: WORSE on LB ⚠ (validation crisis)
Local fold0 0.41765 (BETTER than iter1) but **LB 0.51756 (WORSE than iter1's 0.48509)**.
The local→LB offset jumped +0.057 → +0.100 and the direction inverted. **Our local
validation does NOT track the leaderboard.** We tuned HP + selected features on rolling
June/July folds (interior, easy, wrong season); that overfit, and the late-August test
got worse. Simpler iter1 generalizes better.
**Rules from here:** (1) don't trust interior-fold gains; (2) build a season-aligned
validation (late-Aug, prior years) that tracks the LB; (3) prefer simpler/robust models;
(4) iter1 remains our best LB (0.48509) — revert toward it, add only sound signals.
