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

### Calibration: local is optimistic, and the TEST WINDOW is the reason
iter1 LightGBM: local single-holdout 0.428 → Kaggle LB **0.485** (+0.057). Ran rolling-
origin CV (6×16d) to diagnose: folds = 0.428/0.405/0.390/0.401/0.415/0.444, mean 0.413.
- The rolling MEAN (0.413) is MORE optimistic than fold 0 (0.428) — older mid-2017
  windows are just easier. So the gap is NOT variance; the real test (Aug16-31) is
  genuinely the hardest window (the true extrapolation frontier, entirely past training).
- **Lesson:** for forecasting the next window, the most-recent holdout (fold 0) is the
  honest metric; averaging older windows is rosy. More folds ≠ more honest here.
- The fold0→LB offset (~0.06) >> fold noise (std 0.018): a real, roughly-stable
  "extrapolation tax". Local still tracks RELATIVE gains → tune locally, expect
  LB ≈ recent-local + ~0.06. Likely causes: shift-16 level goes stale under the
  uptrend (under-prediction, RMSLE punishes it) + unmodeled holidays/events.
- **Primary metric from now on = fold 0 (most recent 16d).** Next: iteration 2 features.

### Iteration 2 (12 features at once) — a wash, and a discipline lesson
Added holidays + store-meta + oil + trend together (23 feats vs 11). Rolling CV:
fold 0 (primary) 0.428 → **0.446 (+0.018 WORSE)**; mean 0.413 → 0.416 (+0.003).
- Batching broke "one group at a time": can't attribute, and the extra capacity
  overfit the forward-most fold — the one that matters for the test.
- Importances: store_nbr/family/dow/month/promo dominate (as in iter1). **Oil is
  suspiciously high** — a single macro series, likely a time/regime proxy that overfits
  forward. **Holidays + store-meta are LOW** importance, didn't earn their slots.
- Fix: ablate each group alone (seeded for reproducibility), select on the 2 most
  recent folds, keep only what improves fold 0.

### Ablation: importance ≠ generalization value
Each group added alone on BASE (seeded), marginal on fold 0:
+oil **+0.012 (worst)** · +recency2 +0.0046 · +holidays **−0.0029 (best)** · +storemeta −0.0017.
- OIL hurts despite the highest importance of the new features — it's a time/regime
  proxy that overfits forward (high split-count = heavy USE, not generalization). Dropped.
- recency2 (rmean56 + growth) also hurts fold 0 — added overfitting, not signal. Dropped.
- holidays + store-meta help modestly despite LOW importance. Kept.
- **Lesson: feature importance measures how much a tree USES a feature, not whether it
  GENERALIZES.** Judge features by held-out marginal effect on recent folds.
- iter2 feature set = BASE + holidays + store-meta. Feature tweaks alone are only ~0.005,
  so test bigger levers (full history, tweedie objective) before spending a submission.

### Bigger levers: more data and tweedie — both flat-to-worse
combo (BASE + holidays + store-meta), folds 0/1:
- @2016 L2: fold0 **0.42378** (best; beats iter1 0.42751)
- @2013 L2: fold0 0.46403 — **full history HURTS** (old lower-sales regime dilutes the
  recent level; recency wins again, echoing Step A).
- tweedie (default power 1.5): 0.4265 @2016 — no better than L2-on-log1p.
- Lesson: "more data" is not free under a trend/regime shift; recent-only wins. Default
  tweedie ≠ improvement here.
- iter2 best so far = combo @2016 L2 (fold0 0.42378), only ~0.004 better than iter1.
  Next lever before submitting: HP tuning (more trees + lower LR + early stopping).

### HP tuning: lower LR + more trees + early stopping — the real iter2 gain
combo features, folds 0/1 (early-stopped tree count):
- lr0.05/63: 0.43059 (early stop spent 16d on internal val → worse than fixed-150 combo)
- **lr0.03/63/reg: fold0 0.41765** (~557–688 trees) ← chosen (best fold 0)
- lr0.02/127/reg: 0.41816 (best avg 0.41273)
- **iter2 = combo + lr0.03 / ~600 trees / reg_lambda=1.** fold0 0.41765 vs iter1 0.42751
  (−0.0099). submit_iter2.py refits on all 2016+ data through Aug15 at fixed 600 trees.
  Expected LB ≈ 0.418 + ~0.06 tax ≈ 0.475–0.48.

### ⚠ Local does NOT track the LB — the big lesson
iter2 was BETTER locally (fold0 0.41765 < iter1 0.42751) but WORSE on Kaggle
(**0.51756 > 0.48509**). The offset jumped +0.057 → +0.100 and the direction inverted.
- Cause: we TUNED HP + SELECTED features on rolling June/July folds — interior, easy, and
  the WRONG SEASON. More trees + more features overfit them; the late-August test got worse.
  Simpler iter1 generalized better. We optimized the wrong objective.
- README's nightmare realized: "if improving locally makes Kaggle worse, the validation is
  broken — fix it here." STOP trusting interior-fold gains.
- Fix hypothesis: the test is LATE AUGUST; validate on Aug 16–31 of PRIOR YEARS
  (season-aligned). That should track the LB and expose iter2's overfit. → cv_august.py.
- iter1 (simpler, 0.48509) is still our best LB. Bias toward simplicity + robust signals.

### Process
Built validation BEFORE features (Phase 2 first). It immediately caught a
plausible, math-backed idea that would have ~doubled our error. Measure, don't
trust theory.
