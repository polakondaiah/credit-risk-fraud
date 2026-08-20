# Credit Risk / Fraud Detection (SQL + ML)

**For:** Trainee Data Scientist, Equifax — closes SQL evidence gap + credit/fraud domain.

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

Validation in SQL footer prints row counts + fraud rate by merchant.

Build: `python3 src/build_db.py` (creates `fraud.db`, runs `features.sql`, adds rolling z).

## Modeling (Python / scikit-learn)
`src/model.py` trains on `transaction_features` → features: `amount, age, hour, is_night, is_high_amount, merchant_fraud_rate, amount_z`.

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | Notes |
|---|---|---|---|---|---|---|
| LogReg (no weight) | 0.000 | 0.000 | 0.000 | 0.542 | 0.006 | Accuracy 99.56% but useless — predicts all negative |
| **LogReg (balanced)** | **0.007** | **0.617** | 0.015 | 0.657 | 0.011 | Class-weighted fixes imbalance |
| LogReg thr=0.2 | 0.004 | **1.000** | 0.009 | 0.657 | 0.011 | Threshold ↓ → recall 100%, FP 49k |
| RF balanced | **0.011** | 0.257 | **0.022** | 0.601 | 0.011 | Best precision |

> **Accuracy is deliberately not reported** — on 0.44% fraud, 99.5% accuracy is trivial (predict all 0). Precision/recall + ROC/PR-AUC are correct. RF feature importance: `merchant_fraud_rate` 34.5% > `amount_z` 19% > `amount` 18.7%.

## Production Considerations
- Threshold tuning: trade FP (customer friction) vs FN (loss). Equifax context: FP blocks legitimate card → tune to precision at fixed recall.
- Drift: monitor `merchant_fraud_rate` shift + `amount_z` distribution; retrain monthly.
- Imbalance: class_weight or SMOTE (shown before/after), calibrated probabilities.

## Reproduce
```bash
python3 generate_data.py          # 200k synthetic
python3 src/build_db.py           # SQLite + SQL features
python3 src/model.py              # metrics in results/model_comparison.csv
sqlite3 data/fraud.db < sql/features.sql  # run SQL standalone
sqlite3 data/fraud.db "SELECT * FROM merchant_stats ORDER BY fraud_rate DESC"
```

## Structure
```
data/transactions.csv  data/fraud.db
sql/features.sql  src/build_db.py  src/model.py
results/model_comparison.csv  results/feature_importance.csv
```
