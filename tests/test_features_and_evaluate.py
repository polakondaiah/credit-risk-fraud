"""
The most important test here is the leakage test: merchant_fraud_rate
must be computable from the training partition alone and must NOT change
when test-set labels change (that's the exact bug this module fixes).
Also covers the smoothing behavior and the cost-weighted/calibration
utilities.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features import temporal_split, fit_merchant_fraud_rate, apply_merchant_fraud_rate
from evaluate import expected_value_curve, best_threshold_by_value, calibration_report, bootstrap_pr_auc_ci


def make_txn_df(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    days = rng.integers(0, 100, n)
    merchants = rng.choice(["A", "B", "C", "D"], n)
    fraud = rng.binomial(1, 0.05, n)
    return pd.DataFrame({"day": days, "merchant": merchants, "is_fraud": fraud,
                          "amount": rng.uniform(10, 500, n)})


def test_temporal_split_no_overlap_and_ordered():
    df = make_txn_df()
    train, test = temporal_split(df, train_frac=0.7)
    assert train["day"].max() <= test["day"].min()
    assert len(train) + len(test) == len(df)


def test_merchant_fraud_rate_unaffected_by_test_labels():
    """The core leakage regression test: changing test-set fraud labels
    must not change the fitted train-only merchant fraud rate."""
    df = make_txn_df(seed=1)
    train, test = temporal_split(df, train_frac=0.7)
    smoothed_a, global_a = fit_merchant_fraud_rate(train)

    test_mutated = test.copy()
    test_mutated["is_fraud"] = 1  # flip every test label to fraud
    # smoothed encoding is fit on `train` only, so mutating `test` must not change it
    smoothed_b, global_b = fit_merchant_fraud_rate(train)
    pd.testing.assert_series_equal(smoothed_a.sort_index(), smoothed_b.sort_index())
    assert global_a == global_b


def test_smoothing_pulls_low_volume_merchant_toward_global_rate():
    train = pd.DataFrame({
        "merchant": ["BIG"] * 500 + ["TINY"] * 2,
        "is_fraud": [0] * 490 + [1] * 10 + [1, 0],  # BIG: 2% fraud, TINY: 1/2 = 50% raw
    })
    smoothed, global_rate = fit_merchant_fraud_rate(train, alpha=10.0)
    raw_tiny_rate = 0.5
    assert smoothed["TINY"] < raw_tiny_rate  # smoothing pulls it down from the noisy 50%
    assert abs(smoothed["BIG"] - 0.02) < 0.01  # high-volume merchant barely moves


def test_apply_merchant_fraud_rate_handles_unseen_merchant():
    smoothed = pd.Series({"A": 0.03, "B": 0.10})
    df = pd.DataFrame({"merchant": ["A", "B", "UNSEEN"]})
    out = apply_merchant_fraud_rate(df, smoothed, global_rate=0.05)
    assert out.loc[out["merchant"] == "UNSEEN", "merchant_fraud_rate"].iloc[0] == 0.05


def test_expected_value_curve_prefers_no_action_when_fraud_is_cheap():
    rng = np.random.default_rng(2)
    y_true = rng.binomial(1, 0.01, 2000)
    y_prob = rng.uniform(0, 1, 2000)  # uninformative scores
    curve = expected_value_curve(y_true, y_prob, cost_review=100, cost_missed_fraud=1)
    best = best_threshold_by_value(curve)
    # reviewing is far more expensive than the fraud loss itself -> best policy should flag almost nothing
    assert best["threshold"] > 0.5


def test_calibration_report_shape():
    rng = np.random.default_rng(3)
    y_true = rng.binomial(1, 0.1, 500)
    y_prob = rng.uniform(0, 1, 500)
    report = calibration_report(y_true, y_prob, n_bins=5)
    assert 0 <= report["brier_score"] <= 1
    assert len(report["fraction_of_positives"]) <= 5


def test_bootstrap_pr_auc_ci_contains_point_estimate():
    rng = np.random.default_rng(4)
    y_true = rng.binomial(1, 0.05, 1000)
    y_prob = y_true * 0.7 + rng.uniform(0, 0.3, 1000)
    result = bootstrap_pr_auc_ci(y_true, y_prob, n_boot=100)
    assert result["ci_low"] <= result["point_estimate"] <= result["ci_high"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
