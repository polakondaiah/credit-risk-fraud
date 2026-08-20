"""Fraud modeling: logistic vs RF, imbalance handling, precision/recall."""
import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score, confusion_matrix
import json

ROOT = Path(__file__).parent.parent
DB = ROOT / "data" / "fraud.db"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True, parents=True)

def load_features():
    con = sqlite3.connect(DB)
    # handle NaN amount_z -> 0
    df = pd.read_sql("""
        SELECT amount, age, hour, is_night, is_high_amount,
               merchant_fraud_rate,
               COALESCE(amount_z_merchant, 0) AS amount_z,
               is_fraud
        FROM transaction_features
    """, con)
    con.close()
    print(f"[load] {len(df):,} rows {df.is_fraud.mean():.4%} fraud")
    return df

def evaluate(name, y_true, y_pred, y_prob):
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    roc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true))>1 else 0.5
    pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_true))>1 else 0
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"{name:20s} prec={prec:.3f} rec={rec:.3f} f1={f1:.3f} roc={roc:.3f} pr_auc={pr_auc:.3f}  TP={tp} FP={fp} FN={fn}")
    return {"model": name, "precision": float(prec), "recall": float(rec), "f1": float(f1), "roc_auc": float(roc), "pr_auc": float(pr_auc), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}

def main():
    df = load_features()
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    print(f"[split] train {len(X_train):,} test {len(X_test):,}")

    results = []

    # Baseline: logistic, no weighting (accuracy is meaningless — we report precision/recall)
    lr = LogisticRegression(max_iter=500)
    lr.fit(X_train, y_train)
    prob = lr.predict_proba(X_test)[:,1]
    pred = (prob > 0.5).astype(int)
    results.append(evaluate("LogReg (no weight)", y_test, pred, prob))

    # Logistic with class_weight balanced
    lr_w = LogisticRegression(max_iter=500, class_weight="balanced")
    lr_w.fit(X_train, y_train)
    prob = lr_w.predict_proba(X_test)[:,1]
    pred = (prob > 0.5).astype(int)
    results.append(evaluate("LogReg (balanced)", y_test, pred, prob))

    # Threshold tuned for recall (0.2) — show tradeoff
    pred_tuned = (lr_w.predict_proba(X_test)[:,1] > 0.2).astype(int)
    results.append(evaluate("LogReg bal thr=0.2", y_test, pred_tuned, lr_w.predict_proba(X_test)[:,1]))

    # Random Forest balanced
    rf = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    prob = rf.predict_proba(X_test)[:,1]
    pred = (prob > 0.5).astype(int)
    results.append(evaluate("RF balanced", y_test, pred, prob))

    # Save
    pd.DataFrame(results).to_csv(OUT / "model_comparison.csv", index=False)
    with open(OUT/"metrics.json","w") as f:
        json.dump(results, f, indent=2)
    # feature importance from RF
    imp = pd.DataFrame({"feature": X.columns, "importance": rf.feature_importances_}).sort_values("importance", ascending=False)
    imp.to_csv(OUT/"feature_importance.csv", index=False)
    print(imp.to_string(index=False))
    print(f"[done] saved to {OUT}")

if __name__ == "__main__":
    main()
