"""Held-out evaluation.

Writes artifacts/metrics.json, which the API serves verbatim. Nothing in the
dashboard is hardcoded; if a number appears on screen it came from this file.

Three deliberate choices:

1.  The test split is scored exactly once, with the threshold already fixed on
    validation. No tuning happens here.

2.  Transaction-level and ring-level metrics are computed and reported
    separately. They measure different problems and merging them would be
    misleading.

3.  Ring-level numbers in this phase come from a deliberately naive detector:
    group accounts that share a device, score the group by mean transaction
    risk. It uses no graph structure at all. It exists so that Phase 7's graph
    detector has a documented baseline to beat, rather than producing an
    impressive-looking number with nothing to compare it against.

Run:
    python -m ml.evaluate
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from . import config as C
from .features import canonical_transactions, feature_matrix
from .financial import Assumptions, assumptions_payload, compute_impact
from .rings import detect as detect_rings
from .splits import assign_cluster_splits, split_summary
from .train import ARTIFACTS, LEAKAGE_ALARM_PR_AUC, load_labelled

# Naive ring baseline rules, fixed rather than tuned. Documented so Phase 7's
# improvement is attributable to the graph and not to threshold shopping.
NAIVE_MIN_ACCOUNTS = 3
NAIVE_OVERLAP_FOR_CREDIT = 0.50

CAVEATS = [
    "Dataset is synthetic. Metrics measure recovery of injected patterns under "
    "the generative assumptions documented in ml/config.py, not real-world "
    "fraud performance.",
    "Transaction features are point-in-time correct and verified by test. Graph "
    "and ring detection run retrospectively over a trailing window, not in the "
    "online scoring path.",
    "The decision threshold was selected on the validation split. The held-out "
    "test split was scored once and never used for tuning.",
    "Ring-level metrics come from the graph detector. The naive shared-device "
    "baseline, which uses no graph structure, is reported alongside it so the "
    "graph's contribution is attributable rather than asserted. Both are scored "
    "on the same held-out clusters with the same credit rule.",
    "Ring-level metrics rest on a small sample and should be read as indicative.",
]


def _classifier_block(
    name: str, y: np.ndarray, scores: np.ndarray, threshold: float
) -> dict:
    pred = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "model_name": name,
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "pr_auc": round(float(average_precision_score(y, scores)), 4),
        "roc_auc": round(float(roc_auc_score(y, scores)), 4),
        "accuracy": round(float((pred == y).mean()), 4),
        "decision_threshold": round(float(threshold), 4),
        "confusion_matrix": {
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
        },
        "support": int(len(y)),
        "positive_rate": round(float(y.mean()), 5),
    }


def score_cluster_detection(
    flagged_accounts: set[str],
    membership: pd.DataFrame,
    evaluable_clusters: set[str],
) -> dict:
    """Confusion matrix over true clusters, given whichever accounts were flagged.

    Both the naive baseline and the graph detector are scored through this one
    function, on the same clusters, with the same credit rule. Comparing two
    detectors on different denominators would make the comparison meaningless.
    """
    truth = membership[membership["cluster_id"].isin(evaluable_clusters)]
    tp = fp = tn = fn = 0
    for cid, group in truth.groupby("cluster_id"):
        members = set(group["customer_id"])
        detected = (
            len(members & flagged_accounts) / len(members) >= NAIVE_OVERLAP_FOR_CREDIT
        )
        is_ring = str(cid).startswith("AR-")
        if is_ring and detected:
            tp += 1
        elif is_ring:
            fn += 1
        elif detected:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "n_true_rings": tp + fn,
        "n_true_rings_detected": tp,
        "n_hard_negative_clusters": fp + tn,
        "n_hard_negatives_flagged": fp,
    }


def naive_flagged_accounts(
    test_df: pd.DataFrame, transactions: pd.DataFrame, threshold: float
) -> set[str]:
    """Group accounts that share a device; score each group by mean risk.

    No graph. No structural features. Just co-presence on a device, which is the
    obvious thing a team would try first and roughly what a rules engine does.
    """
    tx = canonical_transactions(transactions)
    scored = test_df[["transaction_id", "risk", "cluster_id"]].merge(
        tx[["transaction_id", "customer_id", "device_id"]],
        on="transaction_id",
    )

    g = nx.Graph()
    for cust, dev in zip(scored["customer_id"], scored["device_id"]):
        g.add_edge(f"c:{cust}", f"d:{dev}")

    account_risk = scored.groupby("customer_id")["risk"].mean()

    flagged_accounts: set[str] = set()
    n_components = 0
    for component in nx.connected_components(g):
        accounts = [n[2:] for n in component if n.startswith("c:")]
        if len(accounts) < NAIVE_MIN_ACCOUNTS:
            continue
        n_components += 1
        if float(account_risk.reindex(accounts).fillna(0).mean()) >= threshold:
            flagged_accounts.update(accounts)

    return flagged_accounts


def evaluate() -> dict:
    meta = json.loads((ARTIFACTS / "training_meta.json").read_text())
    df = load_labelled()  # already carries is_fraud and cluster_id
    labels = pd.read_parquet(C.LABELS_DIR / "transaction_labels.parquet")

    transactions = pd.read_parquet(C.EVENTS_DIR / "transactions.parquet")
    test_df = df[df["split"] == "test"].copy()
    X_test = feature_matrix(test_df).to_numpy(dtype=float)
    y_test = test_df["is_fraud"].to_numpy()

    blocks = {}
    for key, model_name in (
        ("transaction_model", meta["primary_model"]),
        ("baseline_model", meta["baseline_model"]),
    ):
        pipe = joblib.load(ARTIFACTS / f"model_{model_name}.joblib")
        scores = pipe.predict_proba(X_test)[:, 1]
        threshold = meta["models"][model_name]["threshold"]
        blocks[key] = _classifier_block(model_name, y_test, scores, threshold)
        if key == "transaction_model":
            test_df["risk"] = scores
            primary_threshold = threshold

    if blocks["transaction_model"]["pr_auc"] > LEAKAGE_ALARM_PR_AUC:
        raise RuntimeError(
            f"Held-out PR-AUC is {blocks['transaction_model']['pr_auc']}, above the "
            f"{LEAKAGE_ALARM_PR_AUC} alarm. On synthetic data this means leakage, "
            "not success. Investigate before shipping any number."
        )

    cluster_split = assign_cluster_splits(transactions, labels)
    evaluable_clusters = set(
        cluster_split[(cluster_split["split"] == "test") & cluster_split["evaluable"]][
            "cluster_id"
        ]
    )
    membership = (
        transactions[["transaction_id", "customer_id"]]
        .merge(labels, on="transaction_id")
        .dropna(subset=["cluster_id"])[["customer_id", "cluster_id"]]
        .drop_duplicates()
    )

    # Floor: shared-device grouping scored by mean transaction risk. No graph.
    baseline_block = score_cluster_detection(
        naive_flagged_accounts(test_df, transactions, primary_threshold),
        membership,
        evaluable_clusters,
    )

    # The graph detector, scored through the same function on the same clusters.
    detection = detect_rings()
    graph_flagged = {
        acc
        for ring in detection["rings"]
        for acc in ring["accounts"]
    }
    ring_block = score_cluster_detection(
        graph_flagged, membership, evaluable_clusters
    )

    amounts = test_df["amount"].to_numpy(dtype=float)
    flagged = test_df["risk"].to_numpy() >= primary_threshold
    impact = compute_impact(y_test, flagged, amounts, Assumptions())

    summary = split_summary(transactions, labels)
    splits = {}
    for _, row in summary.iterrows():
        splits[row["split"]] = {
            "n_rows": int(row["n_rows"]),
            "start": pd.Timestamp(row["start"]).isoformat(),
            "end": pd.Timestamp(row["end"]).isoformat(),
            "positive_rate": float(row["positive_rate"]),
        }

    manifest = json.loads((C.LABELS_DIR / "manifest.json").read_text())
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "is_placeholder": False,
        "dataset": {
            "seed": manifest["seed"],
            "n_transactions": manifest["n_transactions"],
            "n_customers": manifest["n_customers"],
            "n_rings_injected": manifest["n_rings_injected"],
            "n_hard_negative_clusters": manifest["n_hard_negative_clusters"],
            "split_method": "time_aware",
            "splits": splits,
        },
        "transaction_model": blocks["transaction_model"],
        "baseline_model": blocks["baseline_model"],
        "ring_model": ring_block,
        "ring_baseline_model": baseline_block,
        "financial": {
            "currency": "INR",
            "exposure_detected": impact.exposure_detected,
            "prevented_loss": impact.prevented_loss,
            "false_positive_cost": impact.false_positive_cost,
            "net_protected_value": impact.net_protected_value,
            "assumptions": assumptions_payload(),
        },
        "caveats": CAVEATS,
    }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "metrics.json").write_text(json.dumps(payload, indent=2))

    # The held-out decisions themselves, so the financial engine can recompute
    # under different assumptions without retraining anything. Recomputing from
    # a stored summary would let the arithmetic drift from the decisions it
    # claims to describe.
    (ARTIFACTS / "decisions.json").write_text(
        json.dumps(
            {
                "split": "test",
                "threshold": round(float(primary_threshold), 6),
                "is_fraud": [int(v) for v in y_test],
                "flagged": [bool(v) for v in flagged],
                "amount": [round(float(a), 2) for a in amounts],
            }
        )
    )
    return payload


def main() -> None:
    p = evaluate()
    t, b, r, f = (
        p["transaction_model"], p["baseline_model"], p["ring_model"], p["financial"]
    )
    print("\nHELD-OUT TEST RESULTS")
    print("=" * 62)
    print(f"positive rate           {t['positive_rate']:.4f}  (random PR-AUC floor)")
    print()
    print(f"{'':24s} {'PR-AUC':>8s} {'prec':>8s} {'recall':>8s} {'F1':>8s}")
    for name, blk in (("gradient boosting", t), ("logistic baseline", b)):
        print(
            f"{name:24s} {blk['pr_auc']:8.4f} {blk['precision']:8.4f} "
            f"{blk['recall']:8.4f} {blk['f1']:8.4f}"
        )
    cm = t["confusion_matrix"]
    print(f"\nconfusion (primary)     tp {cm['tp']}  fp {cm['fp']}  "
          f"fn {cm['fn']}  tn {cm['tn']}")
    b = p["ring_baseline_model"]
    print(
        f"\n{'ring detection':24s} {'prec':>8s} {'recall':>8s} "
        f"{'rings':>9s} {'lookalike':>9s}"
    )
    for name, blk in (("graph detector", r), ("naive floor (no graph)", b)):
        print(
            f"{name:24s} {blk['precision']:8.3f} {blk['recall']:8.3f} "
            f"{blk['n_true_rings_detected']:4d}/{blk['n_true_rings']:<4d} "
            f"{blk['n_hard_negatives_flagged']:4d}/{blk['n_hard_negative_clusters']:<4d}"
        )
    print(f"\nprevented loss          INR {f['prevented_loss']:,.0f}")
    print(f"false-positive cost     INR {f['false_positive_cost']:,.0f}")
    print(f"NET PROTECTED VALUE     INR {f['net_protected_value']:,.0f}")
    print(f"\nwritten: {ARTIFACTS / 'metrics.json'}")


if __name__ == "__main__":
    main()
