"""Phase 7 gates: did the differentiator earn its place?

test_graph_detector_beats_the_naive_floor is the whole point of the project. If
it fails, the graph is decoration and the honest move is to say so rather than
quietly ship it.

test_evidence_weights_come_from_the_model matters almost as much. Evidence that
reads plausibly but was assembled after the decision is a story, not a reason,
and a merchant acting on it would be misled.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ml.rings import (
    ARTIFACTS,
    MIN_EVIDENCE_WEIGHT,
    RING_FEATURES,
    build_evidence,
    detect,
    exposure,
    risk_level,
)

pytestmark = pytest.mark.skipif(
    not (ARTIFACTS / "metrics.json").exists(),
    reason="run `make reproduce` first",
)


@pytest.fixture(scope="module")
def detection() -> dict:
    return detect()


@pytest.fixture(scope="module")
def metrics() -> dict:
    return json.loads((ARTIFACTS / "metrics.json").read_text())


# --------------------------------------------------------------------------- #
# The headline claim
# --------------------------------------------------------------------------- #


def test_graph_detector_beats_the_naive_floor(metrics) -> None:
    """The graph must catch rings that shared-device grouping misses."""
    graph = metrics["ring_model"]
    floor = metrics.get("ring_baseline_model")
    assert floor is not None, "the floor must be reported, not dropped"

    assert graph["n_true_rings"] == floor["n_true_rings"], (
        "the two detectors were scored on different clusters; the comparison "
        "would be meaningless"
    )
    assert graph["recall"] > floor["recall"], (
        f"graph recall {graph['recall']} does not beat the no-graph floor "
        f"{floor['recall']}; the differentiator did not earn its place"
    )


def test_graph_gains_are_not_bought_with_reckless_precision(metrics) -> None:
    """Catching more rings by accusing everyone is not an improvement."""
    graph = metrics["ring_model"]
    assert graph["precision"] >= 0.80, (
        f"ring precision {graph['precision']} is too low to act on"
    )
    assert graph["n_hard_negatives_flagged"] <= 2, (
        "too many legitimate clusters accused"
    )


def test_ring_results_are_not_suspiciously_perfect(metrics) -> None:
    """Same alarm as the transaction model, at cluster level."""
    graph = metrics["ring_model"]
    assert not (graph["precision"] == 1.0 and graph["recall"] == 1.0), (
        "flawless ring detection on synthetic data indicates the components "
        "encode the label; investigate before reporting"
    )


# --------------------------------------------------------------------------- #
# Weights are learned, threshold is not tuned on the held-out split
# --------------------------------------------------------------------------- #


def test_weights_are_learned_not_hand_written(detection) -> None:
    meta = detection["meta"]
    assert set(meta.coefficients) == set(RING_FEATURES)
    # Hand-picked weights are almost always round or uniform. Learned ones vary.
    values = np.array(list(meta.coefficients.values()))
    assert len(np.unique(np.round(np.abs(values), 2))) > 4
    assert np.any(values < 0), "no mitigating signal; a pure additive score"


def test_threshold_selected_by_cross_validation(detection) -> None:
    meta = detection["meta"]
    assert 0.0 < meta.threshold < 1.0
    assert meta.n_fit_components >= 50


def test_test_components_were_not_in_the_fitting_pool(detection) -> None:
    scored = detection["scored"]
    pool = int(scored["split"].isin(["train", "validation"]).sum())
    assert pool == detection["meta"].n_fit_components, (
        "held-out components leaked into the fitting pool"
    )


def test_model_is_deterministic() -> None:
    a, b = detect()["meta"], detect()["meta"]
    assert a.coefficients == b.coefficients
    assert a.threshold == b.threshold


# --------------------------------------------------------------------------- #
# Evidence fidelity
# --------------------------------------------------------------------------- #


def test_evidence_weights_come_from_the_model(detection) -> None:
    """Weights must be the model's own contributions, normalised."""
    pipe, meta, scored = detection["pipeline"], detection["meta"], detection["scored"]
    row = scored[scored["flagged"]].iloc[0]
    items = build_evidence(pipe, meta, row)

    weighted = [i for i in items if i["weight"] > 0]
    assert weighted, "no weighted evidence produced"
    assert abs(sum(i["weight"] for i in weighted) - 1.0) < 0.15, (
        "weights do not sum to the decision they claim to explain"
    )
    descending = [i["weight"] for i in weighted]
    assert descending == sorted(descending, reverse=True), (
        "evidence is not ordered by how much it moved the decision"
    )
    assert all(i["weight"] >= MIN_EVIDENCE_WEIGHT for i in weighted)


def test_every_evidence_item_matches_the_contract(detection) -> None:
    pipe, meta, scored = detection["pipeline"], detection["meta"], detection["scored"]
    required = {
        "code", "kind", "statement", "observed_value", "baseline_value",
        "unit", "weight",
    }
    for _, row in scored[scored["flagged"]].head(20).iterrows():
        for item in build_evidence(pipe, meta, row):
            assert set(item) == required, f"evidence shape drifted: {set(item)}"
            assert item["kind"] in {"FACT", "INFERENCE"}
            assert item["statement"].endswith(".")
            assert 0.0 <= item["weight"] <= 1.0


def test_evidence_never_leaks_the_label(detection) -> None:
    """A merchant-facing statement must not mention cluster ground truth."""
    pipe, meta, scored = detection["pipeline"], detection["meta"], detection["scored"]
    banned = ("cluster", "is_fraud", "AR-", "HN-", "label", "synthetic")
    for _, row in scored[scored["flagged"]].head(20).iterrows():
        for item in build_evidence(pipe, meta, row):
            lowered = item["statement"].lower()
            for word in banned:
                assert word.lower() not in lowered, f"evidence leaks '{word}'"


def test_structural_evidence_is_always_present(detection) -> None:
    pipe, meta, scored = detection["pipeline"], detection["meta"], detection["scored"]
    for _, row in scored[scored["flagged"]].head(10).iterrows():
        codes = [i["code"] for i in build_evidence(pipe, meta, row)]
        assert codes[0] == "CLUSTER_SHAPE"


# --------------------------------------------------------------------------- #
# Scoring arithmetic
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "score,expected",
    [
        (0.0, "LOW"), (39.9, "LOW"), (40.0, "MEDIUM"), (69.9, "MEDIUM"),
        (70.0, "HIGH"), (89.9, "HIGH"), (90.0, "CRITICAL"), (100.0, "CRITICAL"),
    ],
)
def test_risk_band_boundaries(score: float, expected: str) -> None:
    """The bands frozen in the contract at Phase 1, asserted at the edges."""
    assert risk_level(score) == expected


def test_exposure_is_refunds_plus_open_captured_value(detection) -> None:
    row = detection["scored"].iloc[0]
    assert exposure(row) == pytest.approx(
        row["refund_value"] + row["captured_value"], abs=0.02
    )


def test_scores_and_confidence_are_in_range(detection) -> None:
    scored = detection["scored"]
    assert scored["risk_score"].between(0, 100).all()
    assert scored["confidence"].between(0, 1).all()


def test_flagged_rings_carry_exposure_and_accounts(detection) -> None:
    for ring in detection["rings"][:20]:
        assert ring["financial_exposure"] >= 0
        assert ring["n_accounts"] >= 3
        assert len(ring["accounts"]) == ring["n_accounts"]
        assert ring["ring_id"].startswith("AR-")
