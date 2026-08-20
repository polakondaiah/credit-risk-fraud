"""Fraud modeling: logistic vs RF, leakage-free features, temporal split,
imbalance handling, cost-weighted threshold selection, calibration."""
from pathlib import Path
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score, confusion_matrix

from features import build_train_test
from evaluate import expected_value_curve, best_threshold_by_value, calibration_report, bootstrap_pr_auc_ci

ROOT = Path(__file__).parent.parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True, parents=True)


def evaluate(name, y_true, y_pred, y_prob):
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    roc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"{name:20s} prec={prec:.3f} rec={rec:.3f} f1={f1:.3f} roc={roc:.3f} pr_auc={pr_auc:.3f}  TP={tp} FP={fp} FN={fn}")
    return {"model": name, "precision": float(prec), "recall": float(rec), "f1": float(f1),
            "roc_auc": float(roc), "pr_auc": float(pr_auc), "tp": int(tp), "fp": int(fp),
            "fn": int(fn), "tn": int(tn)}


def main():
    train, test, feature_cols, global_rate = build_train_test()
    print(f"[split] train {len(train):,} (days <= {train['day'].max()}), "
          f"test {len(test):,} (days > {train['day'].max()}) -- temporal, not random")
    print(f"[fraud rate] train {train['is_fraud'].mean():.4%}, test {test['is_fraud'].mean():.4%}, "
          f"global (train) {global_rate:.4%}")

    X_train, y_train = train[feature_cols], train["is_fraud"].astype(int)
    X_test, y_test = test[feature_cols], test["is_fraud"].astype(int)

    results = []

    lr = LogisticRegression(max_iter=500)
    lr.fit(X_train, y_train)
    prob = lr.predict_proba(X_test)[:, 1]
    pred = (prob > 0.5).astype(int)
    results.append(evaluate("LogReg (no weight)", y_test, pred, prob))

    lr_w = LogisticRegression(max_iter=500, class_weight="balanced")
    lr_w.fit(X_train, y_train)
    prob_lr_w = lr_w.predict_proba(X_test)[:, 1]
    pred = (prob_lr_w > 0.5).astype(int)
    results.append(evaluate("LogReg (balanced)", y_test, pred, prob_lr_w))

    pred_tuned = (prob_lr_w > 0.2).astype(int)
    results.append(evaluate("LogReg bal thr=0.2", y_test, pred_tuned, prob_lr_w))

    rf = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced",
                                 random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    prob_rf = rf.predict_proba(X_test)[:, 1]
    pred = (prob_rf > 0.5).astype(int)
    results.append(evaluate("RF balanced", y_test, pred, prob_rf))

    import pandas as pd
    pd.DataFrame(results).to_csv(OUT / "model_comparison.csv", index=False)
    with open(OUT / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    imp = pd.DataFrame({"feature": X_train.columns, "importance": rf.feature_importances_}) \
        .sort_values("importance", ascending=False)
    imp.to_csv(OUT / "feature_importance.csv", index=False)
    print(imp.to_string(index=False))

    # Business-cost-weighted threshold (on the RF model, the best PR-AUC of the four)
    ev_curve = expected_value_curve(y_test, prob_rf)
    ev_curve.to_csv(OUT / "expected_value_curve.csv", index=False)
    best = best_threshold_by_value(ev_curve)
    verdict = "beats" if best["beats_do_nothing"] else "does NOT beat"
    print(f"\n[cost-weighted] best active threshold={best['threshold']:.3f} -> "
          f"net_value=${best['net_value']:,.0f} (TP={best['tp']}, FP={best['fp']}, FN={best['fn']}) "
          f"-- {verdict} the 'flag nothing' baseline (net_value=$0)")

    # Calibration (RF vs LogReg-balanced)
    calib_rf = calibration_report(y_test, prob_rf)
    calib_lr = calibration_report(y_test, prob_lr_w)
    print(f"\n[calibration] Brier score -- RF: {calib_rf['brier_score']:.4f}, "
          f"LogReg(balanced): {calib_lr['brier_score']:.4f} (lower is better-calibrated)")

    # Bootstrap CI on PR-AUC for the best model
    boot = bootstrap_pr_auc_ci(y_test, prob_rf)
    print(f"[bootstrap] RF PR-AUC {boot['point_estimate']:.4f}, "
          f"95% CI [{boot['ci_low']:.4f}, {boot['ci_high']:.4f}]")

    with open(OUT / "cost_and_calibration.json", "w") as f:
        json.dump({
            "best_threshold_by_value": best,
            "calibration_rf": calib_rf,
            "calibration_logreg_balanced": calib_lr,
            "bootstrap_pr_auc_rf": boot,
            "cost_assumptions": {"cost_review": 5.0, "cost_missed_fraud": 200.0},
        }, f, indent=2)

    print(f"[done] saved to {OUT}")


if __name__ == "__main__":
    main()
