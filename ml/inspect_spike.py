"""Spike detector evaluation.

The interesting question is not "does it find spikes" -- any threshold finds
spikes -- but whether it can tell a coordinated abuse burst apart from a
legitimate one.

The dataset contains flash-sale cohorts for exactly this reason: 30 to 60
accounts registering and ordering inside a single day, all redeeming the same
promo code, with elevated payment failure from gateway strain. Structurally that
is a promo farm. Commercially it is a marketing success. A detector that cannot
tell them apart will page an analyst every time the merchant runs a campaign.

Run:
    python -m ml.inspect_spike
"""

from __future__ import annotations

import json

import pandas as pd

from . import config as C
from .spike import Spike, detect_spikes, load_events, load_risk_scores

LEGITIMATE_BURST_PROFILE = "flash_sale_cohort"


def cluster_windows() -> pd.DataFrame:
    return pd.read_parquet(C.LABELS_DIR / "clusters.parquet")


def _overlaps(spike: Spike, windows: pd.DataFrame) -> bool:
    if windows.empty:
        return False
    return bool(
        (
            (windows["window_start"] <= spike.window_end)
            & (windows["window_end"] >= spike.window_start)
        ).any()
    )


def transaction_attribution() -> pd.DataFrame:
    """Which cluster kind each transaction belongs to, if any.

    Window overlap is NOT a usable measure here. Ninety-eight per cent of the
    timeline sits inside some ring's active window, so "the alert overlapped a
    ring" is what random alerting scores. What matters is whether the spike is
    ATTRIBUTABLE to abuse: whether the transactions driving it actually came
    from a ring rather than merely coinciding with one.
    """
    tx = pd.read_parquet(C.EVENTS_DIR / "transactions.parquet")
    labels = pd.read_parquet(C.LABELS_DIR / "transaction_labels.parquet")
    clusters = pd.read_parquet(C.LABELS_DIR / "clusters.parquet")

    kind = clusters.set_index("cluster_id")
    merged = tx[["transaction_id", "timestamp"]].merge(labels, on="transaction_id")
    merged["cluster_kind"] = (
        merged["cluster_id"].map(kind["cluster_kind"]).fillna("normal")
    )
    merged["profile"] = merged["cluster_id"].map(kind["profile"]).fillna("normal")
    return merged.set_index("timestamp").sort_index()


def classify(spikes: list[Spike], clusters: pd.DataFrame) -> pd.DataFrame:
    """Attribute each alert to whatever actually drove the transactions in it."""
    attribution = transaction_attribution()

    rows = []
    for s in spikes:
        window = attribution.loc[s.window_start : s.window_end]
        n = max(len(window), 1)
        ring_share = float((window["cluster_kind"] == "ABUSE_RING").mean())
        flash_share = float(
            (window["profile"] == LEGITIMATE_BURST_PROFILE).mean()
        )

        # The alert is attributed to whichever population dominates the window,
        # with a plurality bar so a handful of transactions cannot claim it.
        if ring_share >= 0.5 and ring_share > flash_share:
            verdict = "driven_by_abuse"
        elif flash_share >= 0.5:
            verdict = "driven_by_flash_sale"
        else:
            verdict = "unattributed"

        rows.append(
            {
                "spike_id": s.spike_id,
                "metric": s.metric,
                "z_score": s.z_score,
                "verdict": verdict,
                "ring_share": round(ring_share, 3),
                "flash_share": round(flash_share, 3),
                "n_window_transactions": n,
                "exposure": s.estimated_exposure,
            }
        )
    return pd.DataFrame(rows)


def base_rate() -> float:
    """Share of all transactions belonging to abuse rings.

    This is what an alert would score by chance. Any claim about the detector
    has to be read against it.
    """
    attribution = transaction_attribution()
    return float((attribution["cluster_kind"] == "ABUSE_RING").mean())


def ring_window_recall(spikes: list[Spike], clusters: pd.DataFrame) -> float:
    """Share of abuse rings whose active window produced at least one alert."""
    rings = clusters[clusters["cluster_kind"] == "ABUSE_RING"]
    if rings.empty:
        return 0.0
    hit = 0
    for _, ring in rings.iterrows():
        for s in spikes:
            if (
                ring["window_start"] <= s.window_end
                and ring["window_end"] >= s.window_start
            ):
                hit += 1
                break
    return hit / len(rings)


def evaluate(corroborate: bool = True) -> dict:
    ev = load_events()
    risk = load_risk_scores()
    spikes = detect_spikes(
        ev["transactions"], ev["customers"], ev["refunds"],
        risk=risk, corroborate=corroborate,
    )
    clusters = cluster_windows()

    table = classify(spikes, clusters)
    if table.empty:
        table = pd.DataFrame(
            columns=["spike_id", "metric", "z_score", "verdict",
                     "ring_share", "flash_share", "n_window_transactions", "exposure"]
        )
    counts = table["verdict"].value_counts().to_dict()
    base = base_rate()
    mean_ring_share = float(table["ring_share"].mean()) if len(table) else 0.0

    return {
        "n_spikes": len(spikes),
        "driven_by_abuse": counts.get("driven_by_abuse", 0),
        "driven_by_flash_sale": counts.get("driven_by_flash_sale", 0),
        "unattributed": counts.get("unattributed", 0),
        "ring_window_recall": round(ring_window_recall(spikes, clusters), 4),
        "abuse_base_rate": round(base, 4),
        "mean_ring_share_in_alerts": round(mean_ring_share, 4),
        "lift_over_chance": round(mean_ring_share / base, 2) if base else 0.0,
        "n_flash_sale_cohorts": int(
            (clusters["profile"] == LEGITIMATE_BURST_PROFILE).sum()
        ),
        "table": table,
    }


def main() -> None:
    naive = evaluate(corroborate=False)
    res = evaluate(corroborate=True)
    table = res["table"]

    print("\nWITHOUT CORROBORATION (volume alone)")
    print("=" * 66)
    print(
        f"  {naive['n_spikes']} alerts: {naive['driven_by_abuse']} abuse, "
        f"{naive['driven_by_flash_sale']} flash sale, "
        f"{naive['unattributed']} unattributed  |  "
        f"lift {naive['lift_over_chance']}x"
    )

    print("\nSPIKE DETECTOR")
    print("=" * 66)
    print(f"alerts raised              : {res['n_spikes']}")
    print(f"  driven by abuse          : {res['driven_by_abuse']}")
    print(f"  driven by a flash sale   : {res['driven_by_flash_sale']}")
    print(f"  no dominant driver       : {res['unattributed']}")
    print()
    print(f"abuse share of an alert win: {res['mean_ring_share_in_alerts']:.1%}")
    print(f"abuse share by chance      : {res['abuse_base_rate']:.1%}")
    print(f"lift over chance           : {res['lift_over_chance']}x")
    print(f"rings producing an alert   : {res['ring_window_recall']:.1%}")

    print("\nby metric:")
    print(
        table.pivot_table(
            index="metric", columns="verdict", values="spike_id", aggfunc="count"
        )
        .fillna(0)
        .astype(int)
        .to_string()
    )

    summary = {k: v for k, v in res.items() if k != "table"}
    (C.LABELS_DIR / "spike_report.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
