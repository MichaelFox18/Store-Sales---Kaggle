# `src/` — pipeline scripts

Run everything from the project root with the venv, e.g.:
`./.venv/Scripts/python.exe src/model_lgbm.py`

Scripts import the harness via `from validation import ...`, which works because
the running script's own folder (`src/`) is on the path.

| File | Phase | What it does | Best local RMSLE |
|---|---|---|---|
| `validation.py` | 2 | The harness: `rmsle()` + `time_train_holdout_split()` (last-16-day holdout). Imported by everything; its `__main__` scores the mean baseline. | 0.69453 |
| `baseline_mean.py` | 0 | Per-series historical mean → first valid 28,512-row submission file. | 0.695 |
| `baselines.py` | 0–2 | Arithmetic vs geometric per-series mean (geometric lost — trend). | 1.345 (geo) |
| `diagnose_geomean.py` | 2 | Diagnostic: *why* the geometric mean lost (uptrend + under-prediction). | — |
| `baseline_recent.py` | 3 | Step A: recent-window mean (recency fixes the trend bias). | 0.519 (14d) |
| `baseline_dow.py` | 3 | Step B: split by day-of-week (freshness vs granularity tradeoff). | 0.521 |
| `baseline_decomp.py` | 3 | Step C: fresh level × stable weekday shape — **best naive**. | 0.499 |
| `model_lgbm.py` | 4 | Global LightGBM; 3-way horizon bake-off (direct_safe / recursive / per_horizon). | 0.42742 |
| `submit.py` | 0 | Retrains the chosen model on all data through 2017-08-15 and writes the iteration submission file. | — |

Full narrative + lessons: [`../notes/findings.md`](../notes/findings.md).
Submission log: [`../notes/submissions.md`](../notes/submissions.md).
