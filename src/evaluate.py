"""
Evaluation beyond precision/recall/ROC-AUC: a business-cost-weighted
threshold search (precision/recall alone don't say *which* threshold to
deploy), a calibration check (a model can rank-order well but still have
badly miscalibrated probabilities, which matters if the score is used for
anything beyond a fixed threshold), and a bootstrap CI on PR-AUC (a single
point estimate on ~50k test rows with ~220 positives has real sampling
uncertainty worth stating).
"""
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss


def expected_value_curve(y_true, y_prob, cost_review: float = 5.0, cost_missed_fraud: float = 200.0,
                          thresholds=None) -> pd.DataFrame:
    """
    Net expected value per threshold, under a simple two-cost model:
    - Flagging a transaction (true or false positive) costs cost_review
      (manual review / customer friction).
    - Missing a fraud (false negative) costs cost_missed_fraud (the fraud
      loss itself).
    - A true negative costs nothing; a true positive costs cost_review but
      avoids cost_missed_fraud (so its net value is +cost_missed_fraud -
      cost_review vs. doing nothing).
    This is a simplification (real cost structures vary by merchant/amount)
    but it turns an abstract precision/recall tradeoff into a concrete
    "which threshold minimizes expected dollar loss" answer.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 50)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    rows = []
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        cost = fp * cost_review + tp * cost_review + fn * cost_missed_fraud
        benefit = tp * cost_missed_fraud  # fraud loss avoided by catching it
        net_value = benefit - cost
        rows.append({"threshold": float(t), "tp": tp, "fp": fp, "fn": fn,
                      "cost": float(cost), "net_value": float(net_value)})
    df = pd.DataFrame(rows)
    return df


def best_threshold_by_value(ev_curve: pd.DataFrame) -> dict:
    """Best active threshold, explicitly compared against the "flag
    nothing" baseline (net_value=0 by construction) -- with a model this
    imprecise, "do nothing" can legitimately beat every active threshold
    tested, and that's the honest answer to report, not something to
    average away by only showing the best active row."""
    best = ev_curve.loc[ev_curve["net_value"].idxmax()].to_dict()
    best["beats_do_nothing"] = bool(best["net_value"] > 0)
    return best


def calibration_report(y_true, y_prob, n_bins: int = 10) -> dict:
    brier = brier_score_loss(y_true, y_prob)
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    return {
        "brier_score": float(brier),
        "n_bins": n_bins,
        "fraction_of_positives": frac_pos.tolist(),
        "mean_predicted_value": mean_pred.tolist(),
    }


def bootstrap_pr_auc_ci(y_true, y_prob, n_boot: int = 500, ci: float = 0.95, seed: int = 42) -> dict:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    scores = np.empty(n_boot)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    for b in range(n_boot):
        # stratified bootstrap: resample within each class to guarantee
        # both classes are present even with a very small positive count
        boot_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        boot_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([boot_pos, boot_neg])
        scores[b] = average_precision_score(y_true[idx], y_prob[idx])
    lo_pct, hi_pct = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    return {
        "point_estimate": float(average_precision_score(y_true, y_prob)),
        "ci_low": float(np.percentile(scores, lo_pct)),
        "ci_high": float(np.percentile(scores, hi_pct)),
        "n_boot": n_boot,
        "ci_level": ci,
    }
