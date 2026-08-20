"""
Leakage-free feature engineering, fixing a real bug in the original
pipeline: `sql/features.sql` computed `merchant_fraud_rate` as an
aggregate over the *entire* transactions table -- including each row's
own fraud label and every future transaction at that merchant. A model
trained on that column is partly trained on the test set's own labels
(classic target leakage), which inflates every downstream metric.

Fix: split temporally (train on early days, test on later days -- the
realistic production setup, since fraud patterns drift over time), then
compute the merchant fraud-rate prior from the **training partition
only**, with additive (Laplace-style) smoothing toward the global train
fraud rate so merchants with few training transactions don't get a noisy,
overfit per-merchant rate. Test-set merchants (including ones unseen in
training) get the smoothed rate looked up from the train-fitted table, or
the global rate if the merchant never appeared in training at all.
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
DB = ROOT / "data" / "fraud.db"


def load_raw_features(db_path: Path = DB) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    df = pd.read_sql(
        "SELECT txn_id, day, hour, amount, merchant, age, is_night, "
        "is_high_amount, amount_z_merchant, is_fraud FROM transaction_features",
        con,
    )
    con.close()
    return df


def temporal_split(df: pd.DataFrame, train_frac: float = 0.75) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_day = df["day"].quantile(train_frac)
    train = df[df["day"] <= split_day].copy()
    test = df[df["day"] > split_day].copy()
    return train, test


def fit_merchant_fraud_rate(train: pd.DataFrame, alpha: float = 10.0) -> tuple[pd.Series, float]:
    """Smoothed target encoding, fit on train only.
    smoothed_rate = (fraud_count + alpha * global_rate) / (count + alpha)
    alpha acts as a prior sample size -- larger alpha pulls low-volume
    merchants harder toward the global rate instead of trusting a noisy
    small-sample estimate.
    """
    global_rate = train["is_fraud"].mean()
    grp = train.groupby("merchant")["is_fraud"].agg(["sum", "count"])
    smoothed = (grp["sum"] + alpha * global_rate) / (grp["count"] + alpha)
    return smoothed, global_rate


def apply_merchant_fraud_rate(df: pd.DataFrame, smoothed: pd.Series, global_rate: float) -> pd.DataFrame:
    out = df.copy()
    out["merchant_fraud_rate"] = out["merchant"].map(smoothed).fillna(global_rate)
    return out


def build_train_test(db_path: Path = DB, train_frac: float = 0.75, alpha: float = 10.0):
    raw = load_raw_features(db_path)
    train_raw, test_raw = temporal_split(raw, train_frac)
    smoothed, global_rate = fit_merchant_fraud_rate(train_raw, alpha)
    train = apply_merchant_fraud_rate(train_raw, smoothed, global_rate)
    test = apply_merchant_fraud_rate(test_raw, smoothed, global_rate)
    feature_cols = ["amount", "age", "hour", "is_night", "is_high_amount",
                     "merchant_fraud_rate", "amount_z_merchant"]
    return train, test, feature_cols, global_rate
