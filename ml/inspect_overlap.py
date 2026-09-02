"""Hard-negative overlap inspector.

This is the honesty proof for the whole project.

Detecting an injected abuse ring is trivial if the generator made rings look
obviously different from everything else. The claim RazorShield actually makes
is that it separates coordinated abuse from legitimate clusters that look
structurally identical on any single axis.

This script measures that. For each cluster-level structural feature it computes
the single-feature AUC between abuse rings and hard negatives. An AUC near 1.0
means that one feature alone gives the answer away, and the dataset is too easy.
An AUC near 0.5 means the feature carries no standalone signal at all.

What we want: no feature above the ceiling, and a healthy number of features
sitting in the genuinely ambiguous band. That forces the detector to earn its
result from combinations, which is the product thesis.

Run:
    python -m ml.inspect_overlap
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config as C

# A single feature may not exceed this AUC, or the problem is giveaway-easy.
MAX_SINGLE_FEATURE_AUC = 0.95
# At least this many features must sit in the ambiguous band.
AMBIGUOUS_BAND = (0.20, 0.80)
MIN_AMBIGUOUS_FEATURES = 5


def load() -> tuple[pd.DataFrame, ...]:
    tx = pd.read_parquet(C.EVENTS_DIR / "transactions.parquet")
    cu = pd.read_parquet(C.EVENTS_DIR / "customers.parquet")
    rt = pd.read_parquet(C.EVENTS_DIR / "returns.parquet")
    rf = pd.read_parquet(C.EVENTS_DIR / "refunds.parquet")
    lb = pd.read_parquet(C.LABELS_DIR / "transaction_labels.parquet")
    cl = pd.read_parquet(C.LABELS_DIR / "clusters.parquet")
    return tx, cu, rt, rf, lb, cl


def cluster_features(frames: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Structural features per cluster.

    This uses cluster membership, which is label information. That is fine here:
    this is dataset diagnostics, not modelling. Nothing in this module feeds the
    feature pipeline.
    """
    if frames is None:
        tx, cu, rt, rf, lb, cl = load()
    else:
        tx, cu = frames["transactions"], frames["customers"]
        rt, rf = frames["returns"], frames["refunds"]
        lb, cl = frames["transaction_labels"], frames["clusters"]

    tx = tx.merge(lb[["transaction_id", "cluster_id"]], on="transaction_id")
    tx = tx[tx["cluster_id"].notna()]
    tx = tx.merge(cu, on="customer_id", how="left")

    returns_by_txn = set(rt["transaction_id"])
    refunds = rf.groupby("transaction_id")["amount"].sum()

    rows = []
    for cid, g in tx.groupby("cluster_id"):
        captured = g[g["payment_status"] == "captured"]
        n_acc = g["customer_id"].nunique()
        n_dev = g["device_id"].nunique()
        n_addr = g["shipping_address_id"].nunique()
        n_ip = g["ip_address"].nunique()

        coupons = g["coupon_code"].dropna()
        coupon_reuse = (
            float(coupons.value_counts().iloc[0] / len(g)) if len(coupons) else 0.0
        )

        signups = g.drop_duplicates("customer_id")["signup_ts"]
        signup_span = float((signups.max() - signups.min()).total_seconds() / 86400)

        active_days = max(
            float((g["timestamp"].max() - g["timestamp"].min()).total_seconds() / 86400),
            0.5,
        )

        captured_value = float(captured["amount"].sum()) or 1.0
        refund_value = float(refunds.reindex(captured["transaction_id"]).fillna(0).sum())

        rows.append(
            {
                "cluster_id": cid,
                "n_accounts": n_acc,
                "accounts_per_device": n_acc / n_dev,
                "accounts_per_address": n_acc / n_addr,
                "accounts_per_ip": n_acc / n_ip,
                "return_rate": (
                    float(captured["transaction_id"].isin(returns_by_txn).mean())
                    if len(captured) else 0.0
                ),
                "refund_value_ratio": refund_value / captured_value,
                "coupon_reuse": coupon_reuse,
                "signup_span_days": signup_span,
                "failed_payment_rate": float((g["payment_status"] == "failed").mean()),
                "txn_velocity": len(g) / n_acc / active_days,
            }
        )

    feats = pd.DataFrame(rows).merge(
        cl[["cluster_id", "cluster_kind", "profile", "is_abusive"]], on="cluster_id"
    )
    return feats


FEATURE_COLS = [
    "n_accounts",
    "accounts_per_device",
    "accounts_per_address",
    "accounts_per_ip",
    "return_rate",
    "refund_value_ratio",
    "coupon_reuse",
    "signup_span_days",
    "failed_payment_rate",
    "txn_velocity",
]


def single_feature_aucs(feats: pd.DataFrame) -> pd.DataFrame:
    y = feats["is_abusive"].to_numpy()
    rows = []
    rings = feats[feats["is_abusive"] == 1]
    hns = feats[feats["is_abusive"] == 0]
    for col in FEATURE_COLS:
        auc = float(roc_auc_score(y, feats[col].to_numpy()))
        # Report the separating power in either direction.
        rows.append(
            {
                "feature": col,
                "auc": round(auc, 3),
                "separating_power": round(abs(auc - 0.5) * 2, 3),
                "ring_p10": round(float(np.percentile(rings[col], 10)), 3),
                "ring_p90": round(float(np.percentile(rings[col], 90)), 3),
                "hn_p10": round(float(np.percentile(hns[col], 10)), 3),
                "hn_p90": round(float(np.percentile(hns[col], 90)), 3),
            }
        )
    out = pd.DataFrame(rows).sort_values("separating_power", ascending=False)
    return out.reset_index(drop=True)


def evaluate(frames: dict[str, pd.DataFrame] | None = None) -> dict:
    feats = cluster_features(frames)
    aucs = single_feature_aucs(feats)
    lo, hi = AMBIGUOUS_BAND
    ambiguous = aucs[(aucs["auc"] > lo) & (aucs["auc"] < hi)]
    worst = aucs.iloc[0]
    return {
        "n_rings": int((feats["is_abusive"] == 1).sum()),
        "n_hard_negatives": int((feats["is_abusive"] == 0).sum()),
        "max_separating_power": float(worst["separating_power"]),
        "most_separating_feature": str(worst["feature"]),
        "n_ambiguous_features": int(len(ambiguous)),
        "passes": bool(
            worst["separating_power"] <= (MAX_SINGLE_FEATURE_AUC - 0.5) * 2
            and len(ambiguous) >= MIN_AMBIGUOUS_FEATURES
        ),
        "table": aucs,
        "features": feats,
    }


def main() -> None:
    res = evaluate()
    print("\nHARD-NEGATIVE OVERLAP REPORT")
    print("=" * 78)
    print(
        f"{res['n_rings']} abuse rings vs {res['n_hard_negatives']} "
        f"legitimate lookalike clusters\n"
    )
    print(
        res["table"].to_string(
            index=False,
            columns=[
                "feature", "auc", "separating_power",
                "ring_p10", "ring_p90", "hn_p10", "hn_p90",
            ],
        )
    )
    print("\nper-profile cluster counts:")
    print(
        res["features"]
        .groupby(["cluster_kind", "profile"])
        .size()
        .to_string()
    )
    print("\n" + "=" * 78)
    print(
        f"most separating single feature : {res['most_separating_feature']} "
        f"(power {res['max_separating_power']:.3f})"
    )
    print(f"features in the ambiguous band : {res['n_ambiguous_features']}/{len(FEATURE_COLS)}")
    print("VERDICT:", "PASS - no single feature gives it away" if res["passes"] else "FAIL - dataset is too easy")

    summary = {k: v for k, v in res.items() if k not in ("table", "features")}
    (C.LABELS_DIR / "overlap_report.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
