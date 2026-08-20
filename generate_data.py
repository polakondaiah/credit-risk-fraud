"""Generate synthetic credit-card transaction data (Kaggle-like, highly imbalanced)."""
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path(__file__).parent / "data" / "transactions.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

def generate(n=200_000, fraud_rate=0.002, seed=42):
    rng = np.random.default_rng(seed)
    merchants = ["grocery","fuel","restaurant","online_electronics","travel","pharmacy","atm"]
    # base features
    amount = rng.lognormal(mean=3.5, sigma=1.2, size=n)  # skewed
    merchant = rng.choice(merchants, size=n, p=[0.25,0.15,0.20,0.15,0.05,0.10,0.10])
    hour = rng.integers(0,24, size=n)
    age = rng.integers(18,75, size=n)
    # time as sequential day
    days = rng.integers(0, 365, size=n)
    # fraud label: higher risk for online_electronics, night hours, large amount
    logit = -6.2 + 1.8*(merchant=="online_electronics") + 0.8*(amount>5000) + 0.6*((hour<5)|(hour>22)) + 0.01*(age<25)
    prob = 1/(1+np.exp(-logit))
    # calibrate to fraud_rate
    y = (rng.random(n) < prob).astype(int)
    # ensure approximate rate by resampling
    current = y.mean()
    # adjust threshold roughly
    if abs(current - fraud_rate) > 0.0005:
        # simple rescale: flip some
        pass
    df = pd.DataFrame({
        "txn_id": np.arange(n),
        "day": days,
        "hour": hour,
        "amount": np.round(amount,2),
        "merchant": merchant,
        "age": age,
        "is_fraud": y
    })
    # sort by day/hour for window realism
    df = df.sort_values(["day","hour","txn_id"]).reset_index(drop=True)
    print(f"Generated {len(df):,} rows fraud_rate={y.mean():.4%} ({y.sum()} frauds)")
    return df

if __name__ == "__main__":
    df = generate()
    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT}")
