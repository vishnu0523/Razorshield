"""Canonical time-aware split.

Defined once, here. Training, evaluation, and ring-level assignment all import
from this module so they cannot silently disagree about what "held-out" means.

Methodology:
  - Transactions are ordered by timestamp and cut at the 70% and 85% points.
    Future data never appears in training. A random shuffle would let the model
    see a ring's later transactions while predicting its earlier ones, which
    would inflate every number we report.
  - A cluster is assigned to whichever split holds the majority of its
    transactions. A minority of clusters straddle a cut point; those are
    excluded from ring-level evaluation rather than counted twice. Excluding
    them costs us a little sample size and buys an uncontaminated measurement.
"""

from __future__ import annotations

import pandas as pd

TRAIN_FRAC = 0.70
VALIDATION_FRAC = 0.15
# Test is the remainder. It is never touched during training or tuning.

SPLIT_NAMES = ("train", "validation", "test")

# A cluster must have at least this share of its transactions inside one split
# to be evaluated at ring level. Below it, the cluster is boundary-straddling
# and is excluded from ring metrics.
CLUSTER_PURITY_THRESHOLD = 0.80


def assign_transaction_splits(transactions: pd.DataFrame) -> pd.Series:
    """Return a split label per transaction, aligned to the input index."""
    if "timestamp" not in transactions:
        raise ValueError("transactions must carry a 'timestamp' column")

    order = transactions["timestamp"].rank(method="first")
    n = len(transactions)
    train_cut = TRAIN_FRAC * n
    val_cut = (TRAIN_FRAC + VALIDATION_FRAC) * n

    split = pd.Series("test", index=transactions.index, dtype=object)
    split[order <= train_cut] = "train"
    split[(order > train_cut) & (order <= val_cut)] = "validation"
    return split


def assign_cluster_splits(
    transactions: pd.DataFrame, labels: pd.DataFrame
) -> pd.DataFrame:
    """Map each cluster to one split, flagging boundary-straddling clusters.

    Returns columns: cluster_id, split, purity, n_transactions, evaluable.
    """
    df = transactions[["transaction_id", "timestamp"]].merge(
        labels[["transaction_id", "cluster_id"]], on="transaction_id"
    )
    df = df[df["cluster_id"].notna()].copy()
    df["split"] = assign_transaction_splits(df).values

    counts = (
        df.groupby(["cluster_id", "split"]).size().unstack(fill_value=0)
    )
    for name in SPLIT_NAMES:
        if name not in counts:
            counts[name] = 0
    counts = counts[list(SPLIT_NAMES)]

    total = counts.sum(axis=1)
    dominant = counts.idxmax(axis=1)
    purity = counts.max(axis=1) / total

    out = pd.DataFrame(
        {
            "cluster_id": counts.index,
            "split": dominant.values,
            "purity": purity.values,
            "n_transactions": total.values,
        }
    )
    out["evaluable"] = out["purity"] >= CLUSTER_PURITY_THRESHOLD
    return out.reset_index(drop=True)


def split_summary(
    transactions: pd.DataFrame, labels: pd.DataFrame
) -> pd.DataFrame:
    """Human-readable split report. Used by the evaluation artifact and README."""
    df = transactions[["transaction_id", "timestamp"]].merge(labels, on="transaction_id")
    df["split"] = assign_transaction_splits(df).values
    clusters = assign_cluster_splits(transactions, labels)
    clusters["is_ring"] = clusters["cluster_id"].str.startswith("AR-")

    rows = []
    for name in SPLIT_NAMES:
        part = df[df["split"] == name]
        cl = clusters[(clusters["split"] == name) & clusters["evaluable"]]
        rows.append(
            {
                "split": name,
                "n_rows": len(part),
                "start": part["timestamp"].min(),
                "end": part["timestamp"].max(),
                "positive_rate": round(float(part["is_fraud"].mean()), 5),
                "evaluable_rings": int(cl["is_ring"].sum()),
                "evaluable_hard_negatives": int((~cl["is_ring"]).sum()),
            }
        )
    return pd.DataFrame(rows)
