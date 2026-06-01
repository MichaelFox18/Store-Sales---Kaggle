# Store Sales — Time Series Forecasting (Corporación Favorita)

A learning project built around the Kaggle "Store Sales – Time Series Forecasting" competition. The goal is **not** to paste in a winning notebook and submit — it's to build a working forecasting pipeline from scratch, one concept at a time, using Claude Code as a tutor rather than a code vending machine.

If you only want a leaderboard score, this repo is the slow way to get one. If you want to actually understand time-series forecasting, read on.

---

## The problem in one paragraph

Predict daily **unit sales** for every (store, product-family) pair at Favorita's Ecuadorian grocery stores, for a 16-day window into the future. You're scored on **RMSLE** (Root Mean Squared Logarithmic Error), which punishes under-prediction more than over-prediction and cares about *ratios*, not absolute errors — so a model that's "off by 10 units" on a high-volume item matters less than the same miss on a low-volume one. That metric choice shapes almost every decision you'll make.

---

## The data

Drop the raw CSVs in `data/raw/`. Here's what's actually in them (verified against the files, not just the competition page):

| File | Rows | Columns | Notes |
|---|---|---|---|
| `train.csv` | ~3,000,888 | `id, date, store_nbr, family, sales, onpromotion` | 2013-01-01 → 2017-08-15. `sales` is the target (can be fractional). |
| `test.csv` | 28,512 | `id, date, store_nbr, family, onpromotion` | 2017-08-16 → 2017-08-31 (16 days). No `sales` — that's what you predict. |
| `stores.csv` | 54 | `store_nbr, city, state, type, cluster` | Store metadata. `cluster` groups similar stores. |
| `oil.csv` | 1,218 | `date, dcoilwtico` | Daily oil price. **Has missing values** (e.g. weekends, holidays) — you'll need to handle them. Ecuador's economy is oil-sensitive. |
| `transactions.csv` | 83,488 | `date, store_nbr, transactions` | Daily transaction counts per store. Available for train dates only — think about whether/how you can use it for the test window. |
| `sample_submission.csv` | 28,512 | `id, sales` | The exact format your submission must match. |
| `holidays_events.csv` | — | `date, type, locale, locale_name, description, transferred` | **Not currently in this repo — download it from Kaggle.** Read the `transferred` / `Transfer` / `Bridge` / `Work Day` logic carefully; it's a trap. |

**Shape facts worth memorizing:** 54 stores × 33 product families = 1,782 series. The test set is those 1,782 series × 16 days = 28,512 predictions.

### Domain quirks that are secretly features
- **Paydays:** public-sector wages land on the 15th and the last day of each month. Sales spike around then.
- **Earthquake:** a magnitude-7.8 quake hit on **2016-04-16**. Sales of water and staples spiked for weeks. Don't let this anomaly silently poison your trend estimates.
- **Holidays:** transferred holidays behave like normal days; the *Transfer* row is the real celebration date. Bridges and Work Days come in pairs.
- **Sparsity / new items:** some (store, family) series are mostly zeros or start partway through the timeline. A single global model can handle this badly if you're not careful.

---

## How to use this repo (the part that matters)

This is a **tutored build**, not autocomplete. The difference is entirely in how you prompt. Two rules:

> **Rule 1 — Understand before you generate.** Before asking Claude Code to write a block of code, ask it to explain the approach and the tradeoffs. Then *you* decide. Only then generate.

> **Rule 2 — Predict before you peek.** Before running any cell or script, write down what you expect to happen. Compare. The gap between expectation and reality is where the learning is.

### Prompt patterns that teach (copy these into Claude Code)

- *"Explain the three or four ways people typically handle missing oil prices for a daily forecast, with the tradeoffs of each. Don't write code yet — I want to choose."*
- *"I'm about to merge `oil.csv` into the training frame on `date`. Walk me through what could go wrong (NaNs, misaligned dates, leakage) before we do it."*
- *"Here's my validation split `[paste code]`. Critique it specifically for time-series leakage. Where am I letting the future leak into the past?"*
- *"Give me the smallest possible baseline that produces a valid submission. I want to understand the plumbing before the modeling."*
- *"Quiz me: ask me 5 questions about why RMSLE changes how I should transform the target. Wait for my answers before telling me if I'm right."*

### Anti-patterns (these defeat the purpose)
- ❌ "Write the whole solution and get me a good score." (You'll learn nothing and won't be able to debug it.)
- ❌ Accepting code you can't explain line-by-line. If you can't explain it, ask, don't merge.
- ❌ Skipping the baseline because it feels too simple.

A `CLAUDE.md` file is included to set these expectations for Claude Code automatically every session.

---

## The roadmap

Work through these in order. Each phase has a **goal**, **tasks**, a **checkpoint** (how you know you're done), and **reflection questions** to answer in `notes/`. Don't move on until you can answer the reflection questions in your own words.

### Phase 0 — Plumbing & a valid submission
- **Goal:** produce a submission file that Kaggle accepts, using the dumbest possible model.
- **Tasks:** load the data, build a baseline (e.g. predict 0, or predict each series' historical mean, or the last observed value). Write `sample_submission`-formatted output.
- **Checkpoint:** `submissions/baseline.csv` has 28,512 rows + header and the right `id`s.
- **Reflect:** What RMSLE does "predict the mean" get? Why is even this non-trivial to beat? What does a zero-prediction baseline tell you about the metric?

### Phase 1 — EDA
- **Goal:** understand the series before modeling them.
- **Tasks:** plot total sales over time; look at weekly/yearly seasonality; inspect the earthquake window; compare families and stores; quantify zeros/sparsity; check the oil-price gaps.
- **Checkpoint:** you can describe the dominant seasonal patterns and name three series that behave very differently from each other.
- **Reflect:** Is this data additive or multiplicative seasonal? Which would log-transforming the target imply, and does that match the metric?

### Phase 2 — Validation strategy (do this *before* feature engineering)
- **Goal:** a validation scheme that mimics the real task — predicting a future window you haven't seen.
- **Tasks:** implement a **time-based holdout** (e.g. last 16 days of train as a fake test set), then optionally **forward-chaining / rolling-origin** cross-validation. Never shuffle rows randomly.
- **Checkpoint:** your local validation RMSLE tracks your leaderboard RMSLE in the same direction. If improving locally makes Kaggle worse, your validation is broken — fix it here, not later.
- **Reflect:** Why is a random `train_test_split` actively harmful here? What's the exact mechanism by which it leaks?

### Phase 3 — Features
- **Goal:** turn raw tables into signal.
- **Tasks (build incrementally, measuring each):** calendar features (day-of-week, month, payday flags); lag features (sales 7/14/365 days ago); rolling statistics (trailing means/std); `onpromotion`; merged oil price; holidays (handle `transferred` correctly); earthquake flag; store metadata.
- **Checkpoint:** each feature group is added one at a time and you've recorded its effect on validation RMSLE.
- **Reflect:** Which lag features are *unavailable* at prediction time for the 16-day horizon, and how do you handle that (recursive vs. direct forecasting)? This is the single most important conceptual question in the project.

### Phase 4 — Modeling
- **Goal:** beat your baseline meaningfully and understand *why*.
- **Tasks:** try a few model classes and compare — e.g. a per-series statistical model, a single global gradient-boosted tree (LightGBM/XGBoost) over all series, and optionally a simple neural approach. Tune with your Phase-2 validation.
- **Checkpoint:** a clear, logged comparison table of models × validation RMSLE.
- **Reflect:** Why does one global tree model often beat 1,782 individual models? What does it borrow across series?

### Phase 5 — Iterate, ensemble, document
- **Goal:** squeeze and consolidate.
- **Tasks:** error analysis (which families/stores hurt most?), targeted features, optional blending, write up findings.
- **Checkpoint:** `notes/findings.md` explains your best model to a future-you who's forgotten everything.
- **Reflect:** If you had two more weeks, what would you try next and why?

---

## Suggested repo structure

```
.
├── README.md
├── CLAUDE.md              # tutor instructions for Claude Code
├── .gitignore
├── requirements.txt
├── data/
│   ├── raw/               # the Kaggle CSVs (gitignored — see below)
│   └── processed/         # feature frames you generate (gitignored)
├── notebooks/             # EDA and exploration
├── src/
│   ├── load.py            # data loading + joins
│   ├── features.py        # feature engineering
│   ├── validation.py      # time-based splits / CV
│   ├── models.py          # model definitions
│   └── submit.py          # writes a sample_submission-formatted file
├── submissions/           # output CSVs (gitignore the big ones)
└── notes/                 # your reflections per phase — the real deliverable
```

---

## Setup

```bash
# 1. Environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Data
#    Download all CSVs from the Kaggle competition page (including
#    holidays_events.csv) and place them in data/raw/.
#    Kaggle CLI alternative:
#    kaggle competitions download -c store-sales-time-series-forecasting
```

A reasonable starting `requirements.txt`: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `lightgbm`, `jupyter`. Add as you go.

---

## Git / GitHub notes (read before your first push)

- **Do not commit the raw data.** `train.csv` is ~117 MB, and GitHub **rejects any file over 100 MB**. The push will fail. The CSVs are gitignored by default.
- If you genuinely need to version a large file, that's what **Git LFS** is for — but for a Kaggle dataset, just gitignore it and document the download step (done above).
- **Commit your reasoning, not just code.** Your `notes/` reflections and a clear commit history ("add payday flag, +0.003 RMSLE local") are what make this a portfolio piece.
- Consider one **branch per experiment** so you can compare and roll back cleanly.

A starter `.gitignore`:

```gitignore
data/raw/
data/processed/
submissions/*.csv
.venv/
__pycache__/
*.ipynb_checkpoints
.DS_Store
```

---

## Concept cheat-sheet (the things this project is really teaching)

- **RMSLE & `log1p`:** because the metric is logarithmic, training on `log1p(sales)` and predicting `expm1` back is usually the right move. Understand *why* before you do it.
- **Temporal leakage:** the cardinal sin of time-series ML. Every split, every feature, every join must respect the arrow of time.
- **Recursive vs. direct multi-step forecasting:** how you produce 16 days of predictions when your best features are lags depends on this choice.
- **Global vs. local models:** one model across all series vs. one per series — a fundamental tradeoff in hierarchical forecasting.
- **Baselines as a ruler:** you can't claim a model is "good" without a naive baseline to measure against.

---

## Resources

- Kaggle's **Time Series** course — walks through a first submission for this exact competition.
- The competition's **Discussion** and **Code** tabs — but treat them as references to understand *after* you've attempted something, not as answers to copy.

---

*Definition of done for this project: you can explain your final pipeline, your validation scheme, and your biggest mistake — out loud, without notes.*
