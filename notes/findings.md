# Findings log

Running record of what we tried, what it scored on the **local 16-day holdout**
(cutoff 2017-07-31 — mimics the Kaggle test shape: 28,512 rows, 16 days), and
what we learned. Local RMSLE is our scoreboard; Kaggle submissions are spent
only on *validated* gains. Leaderboard reference: top 0.376, public-notebook
cluster 0.378.

## Results

| # | Model | Local RMSLE | Note |
|---|-------|-------------|------|
| 0 | Per-series arithmetic mean (all history) | **0.69453** | baseline floor; "typically off by ~2x" |
| – | Per-series geometric mean `expm1(mean(log1p))` | 1.34483 | WORSE — negative result, see below |
| 1 | Per-series mean, recent **14-day** window | **0.51911** | recency fixes trend; −0.175 vs floor (≈ off by 1.68×) |
| 2 | Recent mean × **day-of-week** (best 84d) | 0.52054 | tie with #1 — weekday signal ≈ freshness lost; see lesson |
| 3 | Fresh **14d level × weekday shape** (full-history factor) | **0.49871** | best naive; decomposition beats A and B (≈ off by 1.65×) |
| 4 | **Global LightGBM — direct_safe** (lag≥16) | **0.42751** | first ML model; −0.071 vs naive; **recommended** |
| 4 | ┗ per_horizon (16 models) | 0.42742 | lowest by 0.0001 (noise) but 16× the cost |
| 4 | ┗ recursive (lag-1 fed back) | 0.43088 | compounding errors → reliably worst |

## Lessons

### Geometric mean lost — trend is why (the big one)
The geometric mean is the RMSLE-optimal constant *in-sample*, but we're
forecasting a trending future, not re-predicting the past.
- Daily mean sales: full history **356.7** → Jul'17 **488.7** → holdout Aug'17 **467.1**
  (~30–37% upward trend). Any all-history mean under-predicts August.
- Geometric mean prediction (264.7) is even lower than arithmetic (356.7): zeros
  and skew drag the log-average down (Jensen's inequality, geo ≤ arith always).
- It under-predicts **81%** of nonzero actuals vs **68%** for arithmetic. RMSLE
  punishes under-prediction more, so the doubly-low estimate gets hammered.
- **Takeaway: recency matters.** Next step: average *recent* data, then add
  day-of-week seasonality. Measure each separately.

### Recency is the biggest lever so far — and there's a sweet spot
Swept the averaging window on the holdout:
7d 0.524 · **14d 0.519** · 21d 0.520 · 28d 0.522 · 56d 0.528 · 84d 0.534 · 140d 0.570 · all 0.695.
- 0.695 → 0.519: our biggest jump, exactly the trend fix the diagnostic predicted.
- U-shaped curve = **bias–variance tradeoff** made visible: too short (7d) = noisy
  (high variance, few samples); too long (140d/all) = stale level (high bias from the
  uptrend). Sweet spot ~2–4 weeks.
- "Fresher is always better" is FALSE. 14–28d is nearly flat (0.519–0.522), so the
  window choice is robust, not overfit to the holdout.

### Day-of-week is real signal, but cell-averaging can't bank it
At a FIXED window, per-weekday beats flat (56d: 0.521 vs 0.528; 84d: 0.521 vs 0.534) —
weekday matters. BUT splitting each series into 7 weekday cells starves them, forcing a
longer/staler window; freshness lost ≈ weekday gained, so DoW's best (0.521@84d) ≈ flat's
best (0.519@14d). Sweet spot shifted 14d → ~56-84d, as predicted (thin cells need more data).
- **Conclusion:** raw averaging can't have BOTH freshness and granularity. To combine
  weekday + recency you must pool the weekday effect across series → a global model.
  This is the motivation for LightGBM. Naive baselines are tapped out at ~0.52.

### Decomposition works — ratios are trend-robust (best naive model)
prediction = level (fresh 14d) × weekday_factor (long window). Sweeping factor window:
28d 0.528 · 56d 0.511 · 84d 0.505 · **140d 0.499** · all 0.499.
- 0.519 → 0.499, beating both A and B. Robust: every factor window ≥56d beats Step A.
- LONGER factor window is BETTER; "all" (4.5 yrs) ties best — the OPPOSITE of using all
  history for the level (0.695). Same data, opposite effect: a ratio cancels a uniform
  trend, so the weekday SHAPE is stable over all history while the LEVEL must stay fresh.
  This is the fresh-level + pooled-shape decoupling that cell-averaging couldn't do.
- **Naive ceiling ≈ 0.499** (~off by 1.65×). Remaining signal — promotions (big here),
  holidays, oil, cross-series learning — needs a model → LightGBM (Option D).

### First global model: 0.499 → 0.427, plus a horizon-strategy bake-off
One LightGBM (store/family/promo/calendar + leakage-safe recency), three ways to
cover the 16-day horizon, same params/features/holdout:
- per_horizon 0.42742 · direct_safe 0.42751 · recursive 0.43088.
- **recursive is reliably WORST** — compounding errors cost more than fresh lag-1 buys.
- **per_horizon ≈ direct_safe** (0.0001 apart = noise) but per_horizon costs 16× the
  training. Fresh short lags barely help: the recent LEVEL (rmean) is ~flat over 16 days,
  and promo (known for the test window) carries the within-window swings.
- **Decision: direct_safe** — tied-best accuracy, 1/16 the cost, simplest to iterate on.
- The 0.499→0.427 jump came from cross-series pooling + promotions — the signals
  cell-averaging structurally couldn't use. Now ~1.53× off vs leaderboard's ~1.46×.
- Caveat: single holdout, so only "recursive worst" and "direct_safe≈per_horizon" are
  robust; the 0.0001 ordering is noise. Confirm with rolling-origin CV later.

### Process
Built validation BEFORE features (Phase 2 first). It immediately caught a
plausible, math-backed idea that would have ~doubled our error. Measure, don't
trust theory.
