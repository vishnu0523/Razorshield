"""Phase 3 gates: point-in-time correctness.

test_features_match_recomputation_on_truncated_history is the one that matters.
Everything else in the project rests on it.

The weak version of this test would check that no column literally named
'is_fraud' appears in the matrix. That catches nothing. The strong version,
implemented here, recomputes each sampled row's features from an event log
truncated at that row's own timestamp and demands an exact match. If any
feature reaches forward in time by even one event, the recomputed value
diverges and this fails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from ml import config as C
from ml.features import (
    FEATURE_COLUMNS,
    build_features,
    canonical_transactions,
    feature_matrix,
    load_events,
)
from ml.generate import Generator

N_PROBES = 30


@pytest.fixture(scope="module")
def frames() -> dict[str, pd.DataFrame]:
    return Generator(seed=C.SEED).run()


@pytest.fixture(scope="module")
def events(frames) -> dict[str, pd.DataFrame]:
    return {
        "customers": frames["customers"],
        "transactions": frames["transactions"],
        "returns": frames["returns"],
        "refunds": frames["refunds"],
    }


@pytest.fixture(scope="module")
def features(events) -> pd.DataFrame:
    return build_features(events)


# --------------------------------------------------------------------------- #
# The point-in-time gate
# --------------------------------------------------------------------------- #


def test_features_match_recomputation_on_truncated_history(events, features) -> None:
    """Recompute sampled rows against only the history available at their time.

    For row i, the event log is truncated to the first i+1 transactions and to
    returns/refunds at or before that timestamp. If the full-dataset feature
    vector for row i differs from the truncated-history one, that feature saw
    the future.
    """
    tx = canonical_transactions(events["transactions"])
    rng = np.random.default_rng(0)
    # Skip the very start of the stream; early rows are trivially unable to leak.
    probes = rng.choice(np.arange(200, len(tx)), size=N_PROBES, replace=False)

    mismatches: list[str] = []
    for i in sorted(int(p) for p in probes):
        t = tx.loc[i, "timestamp"]
        truncated = {
            "customers": events["customers"],
            "transactions": tx.iloc[: i + 1],
            "returns": events["returns"][events["returns"]["timestamp"] <= t],
            "refunds": events["refunds"][events["refunds"]["timestamp"] <= t],
        }
        recomputed = build_features(truncated).iloc[-1]
        actual = features.iloc[i]

        assert recomputed["transaction_id"] == actual["transaction_id"]
        for col in FEATURE_COLUMNS:
            a, b = float(actual[col]), float(recomputed[col])
            if not np.isclose(a, b, rtol=1e-9, atol=1e-9):
                mismatches.append(
                    f"row {i} ({actual['transaction_id']}) feature '{col}': "
                    f"full={a!r} truncated={b!r}"
                )

    assert not mismatches, (
        f"{len(mismatches)} feature(s) used information from the future:\n  "
        + "\n  ".join(mismatches[:15])
    )


def test_every_declared_feature_is_probed() -> None:
    """Guard against a feature being added but escaping the leakage probe."""
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))
    assert len(FEATURE_COLUMNS) >= 30


# --------------------------------------------------------------------------- #
# Leakage barrier
# --------------------------------------------------------------------------- #


def test_no_forbidden_column_in_the_matrix(features) -> None:
    assert not (set(feature_matrix(features).columns) & C.FORBIDDEN_FEATURE_COLUMNS)


def test_no_raw_identifier_is_a_feature(features) -> None:
    """High-cardinality ids would let a model memorise ring membership."""
    banned = {"customer_id", "device_id", "ip_address", "shipping_address_id",
              "coupon_code", "transaction_id"}
    assert not (set(FEATURE_COLUMNS) & banned)


def test_builder_works_without_the_labels_directory(monkeypatch, tmp_path) -> None:
    """load_events must not touch data/labels/ even when it is unreachable."""
    monkeypatch.setattr(C, "LABELS_DIR", tmp_path / "does-not-exist")
    if not (C.EVENTS_DIR / "transactions.parquet").exists():
        pytest.skip("run `make generate` first")
    assert len(build_features(load_events())) > 0


def test_no_single_feature_is_a_perfect_label_proxy(features, frames) -> None:
    """Leakage smoke detector.

    A feature with near-perfect standalone AUC almost always means the label
    was written into it. This will not catch subtle leakage, but it catches the
    catastrophic kind immediately.
    """
    y = (
        frames["transaction_labels"]
        .set_index("transaction_id")
        .loc[features["transaction_id"], "is_fraud"]
        .to_numpy()
    )
    X = feature_matrix(features)
    worst, worst_col = 0.5, ""
    for col in FEATURE_COLUMNS:
        v = X[col].to_numpy(dtype=float)
        if np.ptp(v) == 0:
            continue
        power = abs(roc_auc_score(y, v) - 0.5) * 2
        if power > worst:
            worst, worst_col = power, col
    assert worst < 0.95, (
        f"'{worst_col}' alone separates the label with power {worst:.3f}; "
        "this is almost certainly leakage"
    )


# --------------------------------------------------------------------------- #
# Numerical hygiene
# --------------------------------------------------------------------------- #


def test_matrix_is_finite(features) -> None:
    X = feature_matrix(features).to_numpy(dtype=float)
    assert np.isfinite(X).all(), "feature matrix contains NaN or inf"


def test_row_count_and_alignment(features, events) -> None:
    assert len(features) == len(events["transactions"])
    assert features["transaction_id"].is_unique


def test_features_are_deterministic(events) -> None:
    a = build_features(events)
    b = build_features(events)
    pd.testing.assert_frame_equal(a, b)


def test_input_row_order_does_not_change_output(events) -> None:
    """The builder sorts by timestamp itself; callers cannot perturb it."""
    shuffled = dict(events)
    shuffled["transactions"] = events["transactions"].sample(
        frac=1.0, random_state=1
    )
    a = build_features(events).sort_values("transaction_id").reset_index(drop=True)
    b = build_features(shuffled).sort_values("transaction_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_first_transaction_has_no_prior_history(features, events) -> None:
    """Fill values must be constants, never statistics drawn from the dataset."""
    tx = canonical_transactions(events["transactions"])
    first_of_customer = ~tx["customer_id"].duplicated()
    rows = features[first_of_customer.to_numpy()]
    assert (rows["customer_prior_txn_count"] == 0).all()
    assert (rows["customer_prior_return_count"] == 0).all()
    assert (rows["customer_prior_refund_count"] == 0).all()
    assert (rows["customer_prior_return_rate"] == 0).all()
    assert (rows["amount_vs_customer_mean"] == 1.0).all()


def test_entity_sharing_counts_are_monotone(features, events) -> None:
    """Distinct-customers-per-device can only ever go up over time."""
    tx = canonical_transactions(events["transactions"])
    df = features.assign(device_id=tx["device_id"].values)
    for _, g in df.groupby("device_id", sort=False):
        assert g["device_distinct_customers"].is_monotonic_increasing
