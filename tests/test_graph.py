"""Phase 6 gates: does the graph see what it needs to see?

Two tests carry the weight.

test_ring_recovery_is_high, because a detector cannot score a group the graph
never formed. Recovery is a hard ceiling on ring-level recall.

test_hub_suppression_is_load_bearing, because the hub threshold is the single
most consequential design decision in this module, and a threshold nobody has
tested is a threshold nobody should trust. It asserts that removing suppression
collapses the graph -- proving the choice does real work rather than being
defensive boilerplate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml import config as C
from ml.graph import (
    CUSTOMER_PREFIX,
    FEATURE_COLUMNS,
    HUB_THRESHOLDS,
    build_graph,
    component_features,
    components,
)
from ml.generate import Generator
from ml.inspect_graph import (
    MIN_RING_RECOVERY,
    RECOVERY_THRESHOLD,
    added_signal,
    recovery_report,
)


@pytest.fixture(scope="module")
def frames() -> dict[str, pd.DataFrame]:
    return Generator(seed=C.SEED).run()


@pytest.fixture(scope="module")
def build(frames):
    return build_graph(frames["transactions"])


@pytest.fixture(scope="module")
def feats(build, frames) -> pd.DataFrame:
    return component_features(
        build,
        frames["transactions"],
        frames["customers"],
        frames["returns"],
        frames["refunds"],
    )


@pytest.fixture(scope="module")
def membership(frames) -> pd.DataFrame:
    tx, labels = frames["transactions"], frames["transaction_labels"]
    merged = tx[["transaction_id", "customer_id"]].merge(labels, on="transaction_id")
    merged = merged[merged["cluster_id"].notna()]
    return merged[["customer_id", "cluster_id"]].drop_duplicates()


# --------------------------------------------------------------------------- #
# Recovery
# --------------------------------------------------------------------------- #


def test_ring_recovery_is_high(feats, membership) -> None:
    """A ring the graph splits apart can never be detected as one thing."""
    rec = recovery_report(feats, membership)
    rings = rec[rec["kind"] == "ring"]
    rate = float((rings["recovered"] >= RECOVERY_THRESHOLD).mean())
    assert rate >= MIN_RING_RECOVERY, (
        f"only {rate:.1%} of rings land in a single component; ring recall is "
        "capped below that no matter how good the scorer is"
    )


def test_lookalikes_also_form_components(feats, membership) -> None:
    """Legitimate clusters must survive too, or they cannot be false positives.

    A graph that fragments the office network and keeps the ring would flatter
    our precision by making the hard cases disappear.
    """
    rec = recovery_report(feats, membership)
    looks = rec[(rec["kind"] == "lookalike") & (rec["n_accounts"] >= 3)]
    rate = float((looks["recovered"] >= RECOVERY_THRESHOLD).mean())
    assert rate >= 0.80, (
        f"only {rate:.1%} of multi-account lookalikes form components; the hard "
        "negatives are being fragmented rather than confronted"
    )


# --------------------------------------------------------------------------- #
# Hub suppression
# --------------------------------------------------------------------------- #


def test_public_coupons_are_suppressed_and_private_ones_kept(build) -> None:
    """Linking on a code used by hundreds is not evidence of a relationship."""
    assert build.suppressed["coupon"] > 0, "no public coupon was suppressed"
    kept = {
        n[2:] for n in build.graph.nodes
        if build.graph.nodes[n].get("type") == "coupon"
    }
    assert any(c.startswith("PROMO-") for c in kept), (
        "cluster-specific promo codes were suppressed; they are the real signal"
    )
    assert not any(c.startswith("SAVE") for c in kept), (
        "a public pool coupon survived suppression"
    )


def test_hub_suppression_is_load_bearing(frames, monkeypatch) -> None:
    """Without suppression the merchant collapses into one component.

    This is what justifies the thresholds existing at all. If the graph looked
    the same either way, the suppression would be superstition.
    """
    monkeypatch.setitem(HUB_THRESHOLDS, "coupon", 10**9)
    unsuppressed = build_graph(frames["transactions"])
    largest = max(
        sum(1 for n in comp if n.startswith(f"{CUSTOMER_PREFIX}:"))
        for comp in components(unsuppressed, min_accounts=1)
    )
    n_customers = frames["transactions"]["customer_id"].nunique()
    assert largest > 0.5 * n_customers, (
        "expected an unsuppressed graph to collapse; if it does not, the hub "
        "thresholds may be unnecessary"
    )


def test_suppressed_graph_has_no_giant_component(build, frames) -> None:
    largest = max(
        sum(1 for n in comp if n.startswith(f"{CUSTOMER_PREFIX}:"))
        for comp in components(build, min_accounts=1)
    )
    n_customers = frames["transactions"]["customer_id"].nunique()
    assert largest < 0.05 * n_customers, (
        f"largest component holds {largest} of {n_customers} accounts"
    )


# --------------------------------------------------------------------------- #
# Structural properties
# --------------------------------------------------------------------------- #


def test_accounts_belong_to_exactly_one_component(feats) -> None:
    seen: set[str] = set()
    for accounts in feats["accounts"]:
        overlap = seen & set(accounts)
        assert not overlap, f"accounts in two components: {sorted(overlap)[:3]}"
        seen.update(accounts)


def test_min_accounts_filter_is_respected(feats) -> None:
    assert (feats["n_accounts"] >= 3).all()


def test_graph_is_deterministic(frames) -> None:
    a = build_graph(frames["transactions"])
    b = build_graph(frames["transactions"])
    assert a.graph.number_of_nodes() == b.graph.number_of_nodes()
    assert a.graph.number_of_edges() == b.graph.number_of_edges()
    assert a.suppressed == b.suppressed


def test_component_features_are_finite(feats) -> None:
    X = feats[FEATURE_COLUMNS].to_numpy(dtype=float)
    assert np.isfinite(X).all()


def test_no_forbidden_column_among_component_features() -> None:
    assert not (set(FEATURE_COLUMNS) & C.FORBIDDEN_FEATURE_COLUMNS)


def test_graph_never_reads_labels(monkeypatch, tmp_path, frames) -> None:
    monkeypatch.setattr(C, "LABELS_DIR", tmp_path / "gone")
    assert build_graph(frames["transactions"]).graph.number_of_nodes() > 0


# --------------------------------------------------------------------------- #
# Does the graph earn its place?
# --------------------------------------------------------------------------- #


def test_a_component_feature_beats_every_one_hop_count(feats, membership) -> None:
    """The graph must contribute something the row model cannot express.

    bridging_entity_ratio -- the share of a component's entities that link more
    than one account -- is a property of the component, not of any row. If it
    carried no signal, the graph would be constituting groups but telling us
    nothing new about them.
    """
    signal = added_signal(feats, membership).set_index("feature")
    assert "bridging_entity_ratio" in signal.index
    assert signal.loc["bridging_entity_ratio", "power"] > 0.30, (
        "the component-only feature carries no signal; the graph would be "
        "aggregation with no structural contribution"
    )


def test_no_component_feature_is_a_giveaway(feats, membership) -> None:
    """Same honesty bar as Phase 2: nothing may separate rings on its own."""
    signal = added_signal(feats, membership)
    worst = signal.iloc[0]
    assert worst["power"] <= 0.90, (
        f"'{worst['feature']}' separates rings from lookalikes too cleanly "
        f"(power {worst['power']})"
    )
