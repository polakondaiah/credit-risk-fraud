# Credit Risk / Fraud Detection (SQL + ML)

## Description
A fraud/credit-risk modeling pipeline that uses a real SQL database for feature engineering and Python for modeling. A synthetic transaction stream (200k rows, 0.44% fraud) is loaded into SQLite; SQL builds merchant-level aggregates, hour buckets, and window features (LAG, ROW_NUMBER, rolling z-score supplement). A Python stage does a **temporal train/test split with a leakage-free, smoothed merchant-risk encoding fit on training data only** (see "A real leakage bug, found and fixed" below), trains logistic regression and random forest models with class-imbalance handling, evaluates with precision/recall/ROC-AUC/PR-AUC plus a business-cost-weighted threshold, calibration, and a bootstrap CI (not accuracy, and not a single point estimate), and logs feature importance. The project shows end-to-end SQL fluency, handling of highly imbalanced data, catching and fixing a realistic data-leakage bug, and production-grade evaluation (cost-weighted decisions, calibration, uncertainty) rather than just a metrics table.

## Dataset
Synthetic transaction data (Kaggle Credit Card Fraud–like, 0.44% fraud) — 200,000 rows, 887 frauds.
Generated via `generate_data.py` (lognormal amounts, 7 merchant categories, hour/age effects, fraud biased to `online_electronics`, high amounts, night hours). Sorted by day/hour for realistic windows.
Saved to `data/transactions.csv` → loaded into `data/fraud.db` (SQLite, 21 MB).

> Real Kaggle dataset (284k rows, 0.17% fraud) is drop-in replacement: download `creditcard.csv`, rename columns, rerun `src/build_db.py`.

## SQL Feature Engineering — Real SQL, not pandas
File `sql/features.sql` executes inside SQLite:

1. **Merchant aggregates** (`merchant_stats`):
```sql
SELECT merchant, COUNT(*), AVG(amount), SUM(is_fraud), CAST(SUM(is_fraud) AS REAL)/COUNT(*) AS fraud_rate
FROM transactions GROUP BY merchant
```
→ `online_electronics` highest fraud_rate 1.43% vs ~0.26% others — validates synthetic logic.

2. **Hour aggregates** (`hour_stats`): fraud rate per hour bucket.

3. **Window features** (`transaction_features`):
```sql
LAG(day) OVER (PARTITION BY merchant ORDER BY day, txn_id) AS prev_day_merchant,
ROW_NUMBER() OVER (PARTITION BY merchant ...) AS merchant_txn_seq,
CASE WHEN hour BETWEEN 0 AND 5 THEN 1 ELSE 0 END AS is_night,
JOIN merchant_stats USING (merchant)  -- merchant_fraud_rate prior
```
Rolling amount z-score (20-txn window) computed in Python after SQL step (SQLite lacks window STDDEV) — then bulk-written back. Keeps SQL evidence honest: all aggregations/window mechanics are SQL-native.

> **Note:** the `merchant_fraud_rate` column produced by this SQL step (via the `merchant_stats` full-table aggregate) is full-sample and therefore **not** what the model actually trains on — it's kept here as a real, demonstrable SQL aggregation, but `src/features.py` recomputes a leakage-free, train-only smoothed version before modeling. See "A real leakage bug, found and fixed" below.

Validation in SQL footer prints row counts + fraud rate by merchant.

Build: `python3 src/build_db.py` (creates `fraud.db`, runs `features.sql`, adds rolling z).

## Modeling (Python / scikit-learn)

### A real leakage bug, found and fixed
`sql/features.sql`'s `merchant_fraud_rate` was originally computed as a `GROUP BY merchant` aggregate over the **entire** transactions table — including each row's own fraud label and every future transaction at that merchant. A model trained on that column is partly trained on its own test-set labels: classic target leakage, and a realistic mistake (this exact pattern — a merchant/account-level "risk score" built from the full dataset before splitting — is one of the most common leakage bugs in real fraud pipelines).

**Fix (`src/features.py`):** split **temporally** first (train on days ≤ the 75th-percentile day, test on the days after — the realistic production setup, since fraud patterns drift over time, not a random shuffle), then fit `merchant_fraud_rate` as a **smoothed target encoding from the training partition only**: `(fraud_count + α·global_rate) / (count + α)` with α=10 acting as a prior sample size, so low-volume merchants get pulled toward the global rate instead of trusting a noisy small-sample estimate. Test-set (and unseen) merchants look up the train-fitted rate. `tests/test_features_and_evaluate.py::test_merchant_fraud_rate_unaffected_by_test_labels` is a regression test against this exact bug reappearing.

`src/model.py` trains on the leakage-free features: `amount, age, hour, is_night, is_high_amount, merchant_fraud_rate (train-fit, smoothed), amount_z_merchant`.

### Results (temporal split: train days ≤273, test days >273 — 150,237 train / 49,763 test rows)
| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | Notes |
|---|---|---|---|---|---|---|
| LogReg (no weight) | 0.000 | 0.000 | 0.000 | 0.542 | 0.005 | Predicts all negative — accuracy 99.6% but useless |
| LogReg (balanced) | 0.006 | 0.627 | 0.013 | 0.651 | 0.011 | Class-weighted fixes imbalance |
| LogReg thr=0.2 | 0.004 | 1.000 | 0.008 | 0.651 | 0.011 | Threshold ↓ → recall 100%, FP 49,559 |
| RF balanced | 0.012 | 0.328 | 0.023 | 0.637 | 0.009 | Best precision of the four, still very low |

> **Accuracy is deliberately not reported as a headline number** — on 0.44% fraud, 99.5%+ accuracy is trivial (predict all 0). RF feature importance: `merchant_fraud_rate` 35.6% > `amount_z_merchant` 18.7% > `amount` 18.4%.

### Business-cost-weighted threshold (`evaluate.py::expected_value_curve`)
Precision/recall alone don't say *which* threshold to deploy. Under a simple two-cost model (flagging a transaction costs $5 in review friction; missing a fraud costs $200), the best active threshold on the RF model (t=0.57) has **net_value = -$38,495 on the test set — it does not beat simply flagging nothing** (net_value=$0 by construction). This is the honest, important finding: at this precision level (~1%), the cost of reviewing ~70 false positives per true positive caught outweighs the fraud losses avoided, under these cost assumptions. Reporting only precision/recall/AUC would have hidden this; the point of adding a cost-weighted view is that it can't be. Full curve: `results/expected_value_curve.csv`.

### Calibration (`evaluate.py::calibration_report`)
Brier score: RF 0.112, LogReg (balanced) 0.216 — RF's probabilities are meaningfully better-calibrated (lower is better), even though its precision/recall aren't dramatically different, which matters if the score is ever used for anything beyond a single fixed threshold (e.g. risk-tiering, or feeding into the cost-weighted analysis above).

### Bootstrap CI on PR-AUC (`evaluate.py::bootstrap_pr_auc_ci`)
RF PR-AUC = 0.0094, 95% stratified-bootstrap CI **[0.0076, 0.0130]** (500 resamples) — a tight-looking point estimate on ~50k rows with only ~200 positives still carries real sampling uncertainty; stating the interval is more honest than the point estimate alone.

## Production Considerations
- The cost-weighted analysis above is the concrete version of "threshold tuning" — trading FP (customer friction) vs FN (loss) in dollar terms, not just precision/recall abstractly. Real deployment would need per-merchant or per-amount-bucket cost estimates rather than the flat $5/$200 used here.
- Drift: monitor `merchant_fraud_rate` shift + `amount_z_merchant` distribution over time; the temporal split above already simulates train/test drift once — production would re-fit the smoothed encoding on a rolling window, not once.
- Given PR-AUC this low, the realistic next step before deployment is better features (device/IP signals, velocity checks, graph-based merchant/cardholder features), not just re-tuning thresholds on the current feature set.

## Tests (`tests/test_features_and_evaluate.py`)
7 pytest cases: the leakage regression test above; smoothing correctly pulls a noisy low-volume merchant rate toward the global rate while barely moving a high-volume one; unseen merchants fall back to the global rate; the cost-weighted threshold search correctly prefers "do nothing" when review cost dominates fraud cost on uninformative scores; calibration report shape; bootstrap CI contains its own point estimate. Run: `python3 -m pytest tests/ -v`.

## Reproduce
```bash
python3 generate_data.py          # 200k synthetic
python3 src/build_db.py           # SQLite + SQL features (merchant_stats table is EDA-only now; see leakage note above)
python3 src/model.py              # temporal split, cost-weighted threshold, calibration, bootstrap CI
python3 -m pytest tests/ -v
sqlite3 data/fraud.db < sql/features.sql  # run SQL standalone
sqlite3 data/fraud.db "SELECT * FROM merchant_stats ORDER BY fraud_rate DESC"
```

## Structure
```
data/transactions.csv  data/fraud.db
sql/features.sql  src/build_db.py  src/features.py  src/evaluate.py  src/model.py
tests/test_features_and_evaluate.py
results/{model_comparison.csv,feature_importance.csv,expected_value_curve.csv,cost_and_calibration.json}
```
