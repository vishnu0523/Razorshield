"""Phase 4 gates: is the measurement honest?

The suspicious outcome for this project is not a bad score, it is a good one.
test_held_out_score_is_not_suspiciously_perfect exists because a near-ceiling
PR-AUC on synthetic data means leakage, and shipping that number would be worse
than shipping a mediocre one.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ml.evaluate import evaluate
from ml.financial import Assumptions, compute_impact
from ml.train import ARTIFACTS, LEAKAGE_ALARM_PR_AUC, train

pytestmark = pytest.mark.skipif(
    not (ARTIFACTS.parent / "data" / "labels" / "manifest.json").exists()
    and not (ARTIFACTS / "metrics.json").exists(),
    reason="run `make generate` first",
)


@pytest.fixture(scope="module")
def metrics() -> dict:
    path = ARTIFACTS / "metrics.json"
    if not path.exists():
        train()
        return evaluate()
    return json.loads(path.read_text())


# --------------------------------------------------------------------------- #
# Honesty of the score
# --------------------------------------------------------------------------- #


def test_held_out_score_is_not_suspiciously_perfect(metrics) -> None:
    """A near-perfect result on synthetic data is a bug report, not a win."""
    pr_auc = metrics["transaction_model"]["pr_auc"]
    assert pr_auc <= LEAKAGE_ALARM_PR_AUC, (
        f"held-out PR-AUC {pr_auc} is above the leakage alarm; investigate "
        "before reporting anything"
    )


def test_model_beats_the_random_floor(metrics) -> None:
    """For PR-AUC the random baseline is the positive rate, not 0.5."""
    floor = metrics["transaction_model"]["positive_rate"]
    assert metrics["transaction_model"]["pr_auc"] > floor * 2, (
        "the model barely beats guessing"
    )


def test_baseline_is_reported_alongside_the_primary_model(metrics) -> None:
    """Reporting only the strong model would hide whether complexity earned its keep."""
    assert metrics["baseline_model"]["model_name"] == "logistic_regression"
    assert metrics["transaction_model"]["model_name"] == "hist_gradient_boosting"
    assert metrics["baseline_model"]["pr_auc"] > 0


def test_metrics_are_not_flagged_placeholder(metrics) -> None:
    assert metrics["is_placeholder"] is False


def test_caveats_disclose_the_synthetic_limitation(metrics) -> None:
    joined = " ".join(metrics["caveats"]).lower()
    assert "synthetic" in joined
    assert "threshold" in joined and "validation" in joined


# --------------------------------------------------------------------------- #
# No tuning on the held-out split
# --------------------------------------------------------------------------- #


def test_threshold_was_selected_on_validation() -> None:
    meta = json.loads((ARTIFACTS / "training_meta.json").read_text())
    assert "validation" in meta["threshold_selection"]
    assert "test never used" in meta["threshold_selection"]
    for block in meta["models"].values():
        assert 0.0 < block["threshold"] < 1.0


def test_confusion_matrix_matches_reported_rates(metrics) -> None:
    """Guards against a metric being edited without its matrix following."""
    for key in ("transaction_model", "baseline_model"):
        m = metrics[key]
        cm = m["confusion_matrix"]
        total = cm["tp"] + cm["fp"] + cm["tn"] + cm["fn"]
        assert total == m["support"]
        precision = cm["tp"] / max(cm["tp"] + cm["fp"], 1)
        recall = cm["tp"] / max(cm["tp"] + cm["fn"], 1)
        assert np.isclose(precision, m["precision"], atol=1e-3)
        assert np.isclose(recall, m["recall"], atol=1e-3)


# --------------------------------------------------------------------------- #
# Ring-level metrics are separate and adequately sampled
# --------------------------------------------------------------------------- #


def test_ring_metrics_are_reported_separately(metrics) -> None:
    assert metrics["ring_model"] is not metrics["transaction_model"]
    assert "n_hard_negatives_flagged" in metrics["ring_model"]


def test_ring_evaluation_has_enough_samples(metrics) -> None:
    r = metrics["ring_model"]
    assert r["n_true_rings"] >= 10, "ring precision computed from too few rings"
    assert r["n_hard_negative_clusters"] >= 10, (
        "no legitimate lookalikes in the held-out split, so false accusations "
        "are unmeasurable"
    )


def test_ring_confusion_matrix_is_consistent(metrics) -> None:
    r = metrics["ring_model"]
    cm = r["confusion_matrix"]
    assert cm["tp"] + cm["fn"] == r["n_true_rings"]
    assert cm["fp"] + cm["tn"] == r["n_hard_negative_clusters"]
    assert cm["tp"] == r["n_true_rings_detected"]
    assert cm["fp"] == r["n_hard_negatives_flagged"]


# --------------------------------------------------------------------------- #
# Financial arithmetic
# --------------------------------------------------------------------------- #


def test_net_protected_value_is_prevented_minus_cost(metrics) -> None:
    f = metrics["financial"]
    assert np.isclose(
        f["net_protected_value"], f["prevented_loss"] - f["false_positive_cost"], atol=1.0
    )


def test_prevented_loss_never_exceeds_detected_exposure(metrics) -> None:
    """We apply a recovery rate rather than claiming everything detected was saved."""
    f = metrics["financial"]
    assert f["prevented_loss"] <= f["exposure_detected"] + 1.0


def test_assumptions_are_disclosed_and_adjustable(metrics) -> None:
    keys = {a["key"] for a in metrics["financial"]["assumptions"]}
    assert keys == {
        "recovery_rate", "cost_per_false_review", "cost_per_false_block_ratio"
    }
    assert all(a["adjustable"] for a in metrics["financial"]["assumptions"])
    assert all(a["description"] for a in metrics["financial"]["assumptions"])


def test_zero_recovery_rate_means_zero_prevented_loss() -> None:
    y = np.array([1, 1, 0, 0])
    flagged = np.array([1, 0, 1, 0])
    amounts = np.array([1000.0, 500.0, 2000.0, 300.0])
    impact = compute_impact(y, flagged, amounts, Assumptions(recovery_rate=0.0))
    assert impact.prevented_loss == 0.0
    assert impact.net_protected_value < 0


def test_impact_is_monotone_in_recovery_rate() -> None:
    y = np.array([1, 1, 0, 0])
    flagged = np.array([1, 1, 1, 0])
    amounts = np.array([1000.0, 500.0, 2000.0, 300.0])
    low = compute_impact(y, flagged, amounts, Assumptions(recovery_rate=0.3))
    high = compute_impact(y, flagged, amounts, Assumptions(recovery_rate=0.9))
    assert high.net_protected_value > low.net_protected_value


def test_false_positives_always_cost_something() -> None:
    y = np.array([0, 0])
    flagged = np.array([1, 1])
    amounts = np.array([100.0, 100.0])
    impact = compute_impact(y, flagged, amounts)
    assert impact.false_positive_cost > 0
    assert impact.net_protected_value < 0


def test_missed_fraud_is_reported_not_hidden() -> None:
    y = np.array([1, 1])
    flagged = np.array([1, 0])
    amounts = np.array([100.0, 900.0])
    impact = compute_impact(y, flagged, amounts)
    assert impact.missed_exposure == 900.0
    assert impact.n_false_negatives == 1
