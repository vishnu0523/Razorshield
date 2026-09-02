"""Phase 2 gates: dataset validity.

The important ones are test_no_label_column_leaks_into_events and
test_hard_negatives_are_actually_hard. Everything else is hygiene.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ml import config as C
from ml.generate import Generator
from ml.inspect_overlap import FEATURE_COLS, evaluate
from ml.splits import assign_cluster_splits, split_summary

EVENT_FRAMES = ("customers", "transactions", "returns", "refunds")


@pytest.fixture(scope="module")
def frames() -> dict[str, pd.DataFrame]:
    return Generator(seed=C.SEED).run()


# --------------------------------------------------------------------------- #
# Leakage barrier
# --------------------------------------------------------------------------- #


def test_no_label_column_leaks_into_events(frames) -> None:
    """The feature builder only ever reads these frames. Nothing labelled here."""
    for name in EVENT_FRAMES:
        cols = set(frames[name].columns)
        leaked = cols & C.FORBIDDEN_FEATURE_COLUMNS
        assert not leaked, f"{name} exposes label columns: {sorted(leaked)}"


def test_labels_live_in_a_separate_directory() -> None:
    """Physical separation, not a naming convention someone has to remember."""
    assert C.EVENTS_DIR != C.LABELS_DIR
    assert C.LABELS_DIR.name == "labels"


def test_transactions_carry_no_cluster_membership(frames) -> None:
    """Cluster id is the ring label. It must not be reachable from the event log."""
    assert "cluster_id" not in frames["transactions"].columns


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #


def test_same_seed_is_byte_identical() -> None:
    a = Generator(seed=7).run()["transactions"]
    b = Generator(seed=7).run()["transactions"]
    assert a.equals(b)


def test_different_seed_differs() -> None:
    a = Generator(seed=7).run()["transactions"]
    b = Generator(seed=8).run()["transactions"]
    assert not a.equals(b)


# --------------------------------------------------------------------------- #
# Scale and realism
# --------------------------------------------------------------------------- #


def test_transaction_count_in_target_range(frames) -> None:
    assert 10_000 <= len(frames["transactions"]) <= 20_000


def test_fraud_rate_is_plausible(frames) -> None:
    """A merchant with a real abuse problem, not a coin flip and not a rounding error."""
    rate = frames["transaction_labels"]["is_fraud"].mean()
    assert 0.03 <= rate <= 0.12, f"fraud rate {rate:.4f} is not plausible"


def test_classes_are_imbalanced(frames) -> None:
    """Justifies refusing to headline accuracy."""
    assert frames["transaction_labels"]["is_fraud"].mean() < 0.20


def test_all_ring_profiles_present(frames) -> None:
    got = set(frames["clusters"].query("cluster_kind == 'ABUSE_RING'")["profile"])
    assert got == set(C.ABUSE_PROFILES), f"missing ring profiles: {set(C.ABUSE_PROFILES) - got}"


def test_all_hard_negative_profiles_present(frames) -> None:
    got = set(frames["clusters"].query("cluster_kind == 'HARD_NEGATIVE'")["profile"])
    assert got == set(C.HARD_NEGATIVE_PROFILES)


def test_enough_rings_for_ring_level_metrics(frames) -> None:
    n = (frames["clusters"]["cluster_kind"] == "ABUSE_RING").sum()
    assert n >= 60, "too few rings; ring-level precision would be noise"


# --------------------------------------------------------------------------- #
# Referential integrity
# --------------------------------------------------------------------------- #


def test_returns_and_refunds_reference_real_transactions(frames) -> None:
    txn_ids = set(frames["transactions"]["transaction_id"])
    assert set(frames["returns"]["transaction_id"]) <= txn_ids
    assert set(frames["refunds"]["transaction_id"]) <= txn_ids


def test_failed_payments_produce_no_refunds(frames) -> None:
    failed = set(
        frames["transactions"].query("payment_status == 'failed'")["transaction_id"]
    )
    assert not (set(frames["refunds"]["transaction_id"]) & failed)
    assert not (set(frames["returns"]["transaction_id"]) & failed)


def test_refunds_never_precede_their_transaction(frames) -> None:
    merged = frames["refunds"].merge(
        frames["transactions"][["transaction_id", "timestamp"]],
        on="transaction_id",
        suffixes=("_refund", "_txn"),
    )
    assert (merged["timestamp_refund"] >= merged["timestamp_txn"]).all()


def test_refund_never_exceeds_order_value(frames) -> None:
    merged = frames["refunds"].merge(
        frames["transactions"][["transaction_id", "amount"]],
        on="transaction_id",
        suffixes=("_refund", "_txn"),
    )
    assert (merged["amount_refund"] <= merged["amount_txn"] + 0.01).all()


def test_every_customer_signed_up_before_transacting(frames) -> None:
    merged = frames["transactions"].merge(frames["customers"], on="customer_id")
    assert (merged["timestamp"] >= merged["signup_ts"]).all()


def test_labels_align_with_transactions(frames) -> None:
    assert list(frames["transaction_labels"]["transaction_id"]) == list(
        frames["transactions"]["transaction_id"]
    )


# --------------------------------------------------------------------------- #
# Split viability
# --------------------------------------------------------------------------- #


def test_split_is_time_ordered_and_disjoint(frames) -> None:
    summary = split_summary(frames["transactions"], frames["transaction_labels"])
    rows = {r["split"]: r for _, r in summary.iterrows()}
    assert rows["train"]["end"] <= rows["validation"]["start"]
    assert rows["validation"]["end"] <= rows["test"]["start"]


def test_held_out_split_has_enough_evaluable_rings(frames) -> None:
    """Without this, ring-level precision is a number computed from three samples."""
    summary = split_summary(frames["transactions"], frames["transaction_labels"])
    test_row = summary.set_index("split").loc["test"]
    assert test_row["evaluable_rings"] >= 10, (
        f"only {test_row['evaluable_rings']} evaluable rings in the held-out split"
    )
    assert test_row["evaluable_hard_negatives"] >= 10


def test_most_clusters_do_not_straddle_split_boundaries(frames) -> None:
    clusters = assign_cluster_splits(frames["transactions"], frames["transaction_labels"])
    assert clusters["evaluable"].mean() >= 0.70


# --------------------------------------------------------------------------- #
# The honesty gate
# --------------------------------------------------------------------------- #


def test_hard_negatives_are_actually_hard(frames) -> None:
    """No single structural feature may separate abuse rings from lookalikes.

    If this fails, the dataset is giving the answer away and every downstream
    metric is meaningless. This is the most important test in Phase 2.
    """
    res = evaluate(frames)
    table = res["table"].set_index("feature")

    assert res["max_separating_power"] <= 0.90, (
        f"'{res['most_separating_feature']}' separates rings from hard negatives "
        f"too cleanly (power {res['max_separating_power']:.3f}). The dataset is too easy."
    )
    assert res["n_ambiguous_features"] >= 5, (
        f"only {res['n_ambiguous_features']}/{len(FEATURE_COLS)} features are "
        "genuinely ambiguous; hard negatives are not doing their job"
    )

    # Size and IP sharing must carry essentially no standalone signal. A detector
    # that flags "many accounts on one IP" would accuse every office network.
    for weak in ("n_accounts", "accounts_per_ip"):
        assert table.loc[weak, "separating_power"] <= 0.35, (
            f"{weak} alone is too predictive; office networks would be flagged"
        )


def test_ring_and_hard_negative_ranges_overlap(frames) -> None:
    """Interquantile ranges must intersect, not merely sit near each other."""
    table = evaluate(frames)["table"].set_index("feature")
    overlapping = 0
    for feat in FEATURE_COLS:
        r = table.loc[feat]
        if r["ring_p10"] <= r["hn_p90"] and r["hn_p10"] <= r["ring_p90"]:
            overlapping += 1
    assert overlapping >= 8, f"only {overlapping}/{len(FEATURE_COLS)} features overlap"
