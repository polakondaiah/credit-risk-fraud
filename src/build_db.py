"""Load CSV into SQLite and run SQL feature engineering."""
import sqlite3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent
CSV = ROOT / "data" / "transactions.csv"
DB = ROOT / "data" / "fraud.db"
SQL = ROOT / "sql" / "features.sql"

def main():
    if not CSV.exists():
        raise FileNotFoundError(f"Run generate_data.py first, missing {CSV}")
    df = pd.read_csv(CSV)
    print(f"[db] CSV {len(df):,} rows fraud_rate={df.is_fraud.mean():.4%}")
    if DB.exists():
        DB.unlink()
    con = sqlite3.connect(DB)
    df.to_sql("transactions", con, if_exists="replace", index=False)
    print(f"[db] loaded into {DB}")
    sql = SQL.read_text()
    con.executescript(sql)
    # Compute rolling amount z-score per merchant in pandas (SQLite lacks window STDDEV)
    # Bulk replace: read full table, compute, overwrite
    feats = pd.read_sql("SELECT * FROM transaction_features ORDER BY merchant, day, txn_id", con)
    feats["amount_z_merchant"] = feats.groupby("merchant")["amount"].transform(
        lambda s: (s - s.rolling(20, min_periods=5).mean()) / s.rolling(20, min_periods=5).std(ddof=0)
    )
    feats["amount_z_merchant"] = feats["amount_z_merchant"].fillna(0)
    feats.to_sql("transaction_features", con, if_exists="replace", index=False)
    print(f"[python] updated amount_z_merchant rolling 20 per merchant")
    # print validation
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM transaction_features")
    print(f"[sql] transaction_features rows: {cur.fetchone()[0]}")
    cur.execute("SELECT * FROM merchant_stats ORDER BY fraud_rate DESC")
    for row in cur.fetchall():
        print(row)
    con.commit()
    con.close()
    print(f"[done] DB ready at {DB}")

if __name__ == "__main__":
    main()
