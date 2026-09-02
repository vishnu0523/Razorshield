"""Graph diagnostics.

Answers two questions that decide whether Phase 7 is worth building at all.

1.  RECOVERY. When accounts genuinely belong together, does the graph put them
    in one component? A detector cannot score a group the graph never formed,
    so recovery is a ceiling on ring-level recall. Measured for abuse rings and
    for legitimate lookalikes alike.

2.  ADDED SIGNAL. Do the component-level features separate rings from lookalikes
    better than the one-hop entity counts the row model already has? If they do
    not, the graph is decoration and we should say so rather than ship it.

Run:
    python -m ml.inspect_graph
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config as C
from .graph import FEATURE_COLUMNS, build_graph, component_features, load_events

# A cluster counts as recovered when this share of its accounts land together.
RECOVERY_THRESHOLD = 0.70
# The graph must recover at least this share of abuse rings, or ring-level
# recall is capped below anything worth demoing.
MIN_RING_RECOVERY = 0.80


def cluster_membership() -> pd.DataFrame:
    """True cluster per customer. Diagnostics only; never used to build."""
    tx = pd.read_parquet(C.EVENTS_DIR / "transactions.parquet")
    labels = pd.read_parquet(C.LABELS_DIR / "transaction_labels.parquet")
    merged = tx[["transaction_id", "customer_id"]].merge(labels, on="transaction_id")
    merged = merged[merged["cluster_id"].notna()]
    return merged[["customer_id", "cluster_id"]].drop_duplicates()


def recovery_report(feats: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """For each true cluster, how much of it landed in a single component."""
    account_to_component: dict[str, str] = {}
    for _, row in feats.iterrows():
        for acc in row["accounts"]:
            account_to_component[acc] = row["component_id"]

    rows = []
    for cid, group in membership.groupby("cluster_id"):
        accounts = list(group["customer_id"])
        placed = [account_to_component.get(a) for a in accounts]
        found = [p for p in placed if p is not None]
        if found:
            counts = pd.Series(found).value_counts()
            best, best_n = counts.index[0], int(counts.iloc[0])
        else:
            best, best_n = None, 0

        rows.append(
            {
                "cluster_id": cid,
                "kind": "ring" if str(cid).startswith("AR-") else "lookalike",
                "n_accounts": len(accounts),
                "best_component": best,
                "recovered": best_n / len(accounts),
                "fragmented": len(set(found)) if found else 0,
            }
        )
    return pd.DataFrame(rows)


def added_signal(feats: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """Single-feature separating power of component features, rings vs lookalikes.

    Components are labelled by the true cluster holding most of their accounts.
    Components dominated by neither (mixed or incidental) are excluded.
    """
    lookup = dict(zip(membership["customer_id"], membership["cluster_id"]))

    labels = []
    for _, row in feats.iterrows():
        clusters = [lookup.get(a) for a in row["accounts"]]
        named = [c for c in clusters if c is not None]
        if not named or len(named) / len(clusters) < 0.5:
            labels.append(None)
            continue
        dominant = pd.Series(named).value_counts().index[0]
        labels.append("ring" if str(dominant).startswith("AR-") else "lookalike")

    df = feats.assign(kind=labels)
    df = df[df["kind"].notna()]
    y = (df["kind"] == "ring").astype(int).to_numpy()

    rows = []
    for col in FEATURE_COLUMNS:
        v = df[col].to_numpy(dtype=float)
        if np.ptp(v) == 0:
            continue
        auc = float(roc_auc_score(y, v))
        rows.append(
            {
                "feature": col,
                "auc": round(auc, 3),
                "power": round(abs(auc - 0.5) * 2, 3),
            }
        )
    out = pd.DataFrame(rows).sort_values("power", ascending=False).reset_index(drop=True)
    out.attrs["n_rings"] = int(y.sum())
    out.attrs["n_lookalikes"] = int((1 - y).sum())
    return out


def evaluate() -> dict:
    ev = load_events()
    build = build_graph(ev["transactions"])
    feats = component_features(
        build, ev["transactions"], ev["customers"], ev["returns"], ev["refunds"]
    )
    membership = cluster_membership()

    rec = recovery_report(feats, membership)
    rings = rec[rec["kind"] == "ring"]
    looks = rec[rec["kind"] == "lookalike"]
    ring_recovery = float((rings["recovered"] >= RECOVERY_THRESHOLD).mean())

    signal = added_signal(feats, membership)

    return {
        "n_components": int(len(feats)),
        "ring_recovery_rate": round(ring_recovery, 4),
        "ring_median_recovered": round(float(rings["recovered"].median()), 4),
        "lookalike_recovery_rate": round(
            float((looks["recovered"] >= RECOVERY_THRESHOLD).mean()), 4
        ),
        "passes": ring_recovery >= MIN_RING_RECOVERY,
        "recovery": rec,
        "signal": signal,
    }


def main() -> None:
    res = evaluate()
    rec, signal = res["recovery"], res["signal"]

    print("\nCLUSTER RECOVERY")
    print("=" * 70)
    print(
        rec.groupby("kind")
        .agg(
            clusters=("cluster_id", "size"),
            median_recovered=("recovered", "median"),
            fully_recovered=("recovered", lambda s: float((s >= 0.999).mean())),
            recovered_70pct=("recovered", lambda s: float((s >= RECOVERY_THRESHOLD).mean())),
            median_fragments=("fragmented", "median"),
        )
        .round(3)
        .to_string()
    )

    print("\n\nCOMPONENT FEATURE SIGNAL (abuse rings vs legitimate lookalikes)")
    print("=" * 70)
    print(
        f"{signal.attrs['n_rings']} ring components vs "
        f"{signal.attrs['n_lookalikes']} lookalike components\n"
    )
    print(signal.head(12).to_string(index=False))

    print("\n" + "=" * 70)
    print(f"components formed        : {res['n_components']:,}")
    print(f"rings recovered (>=70%)  : {res['ring_recovery_rate']:.1%}")
    print(f"VERDICT: {'PASS' if res['passes'] else 'FAIL - graph cannot see most rings'}")

    summary = {k: v for k, v in res.items() if k not in ("recovery", "signal")}
    summary["top_features"] = signal.head(6).to_dict("records")
    (C.LABELS_DIR / "graph_report.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
