"""Phase 11 gates: policy boundaries and honest fusion reporting.

The policy tests are exhaustive at every edge because a boundary that is wrong
by one is a merchant blocking customers it should not, or letting through cases
it should have held. These are the tests most worth running in front of a judge.
"""

from __future__ import annotations

import json

import pytest

from backend.app.core.policy import (
    MAX_AUTO_INTERVENTION_AMOUNT,
    band_for,
    config,
    decide,
)
from ml.train import ARTIFACTS

SMALL = 100.0
LARGE = MAX_AUTO_INTERVENTION_AMOUNT + 1


# --------------------------------------------------------------------------- #
# Risk band boundaries, both sides of every edge
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "score,action",
    [
        (0.0, "ALLOW"), (39.0, "ALLOW"), (39.999, "ALLOW"),
        (40.0, "MONITOR"), (69.0, "MONITOR"), (69.999, "MONITOR"),
        (70.0, "VERIFY"), (89.0, "VERIFY"), (89.999, "VERIFY"),
        (90.0, "MANUAL_REVIEW"), (100.0, "MANUAL_REVIEW"),
    ],
)
def test_band_boundaries(score: float, action: str) -> None:
    assert decide(score, SMALL).recommended_action == action
    assert band_for(score)[0] == action


@pytest.mark.parametrize("score", [69, 70, 89, 90])
def test_the_four_specified_boundaries(score: int) -> None:
    """The exact scores named in the build brief."""
    expected = {69: "MONITOR", 70: "VERIFY", 89: "VERIFY", 90: "MANUAL_REVIEW"}
    assert decide(float(score), SMALL).recommended_action == expected[score]


def test_scores_outside_the_range_are_rejected() -> None:
    for bad in (-0.1, 100.1, 1000.0):
        with pytest.raises(ValueError, match="0-100"):
            decide(bad, SMALL)


# --------------------------------------------------------------------------- #
# The money cap
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "amount,capped",
    [(0.0, False), (4999.0, False), (4999.99, False), (5000.0, False),
     (5000.01, True), (5001.0, True), (100000.0, True)],
)
def test_amount_cap_boundary(amount: float, capped: bool) -> None:
    """Exactly at the limit is still within it. Only above triggers the cap."""
    decision = decide(75.0, amount)
    assert decision.amount_cap_applied is capped
    assert decision.requires_merchant_approval is capped


def test_cap_does_not_apply_to_non_intervening_actions() -> None:
    """MONITOR and ALLOW do not interrupt anyone, so no cap and no approval."""
    for score in (10.0, 50.0):
        decision = decide(score, 1_000_000.0)
        assert decision.amount_cap_applied is False
        assert decision.requires_merchant_approval is False


def test_manual_review_always_needs_a_person_regardless_of_amount() -> None:
    for amount in (0.0, 1.0, 4999.0, 5000.0, 500000.0):
        assert decide(95.0, amount).requires_merchant_approval is True


def test_negative_amounts_are_rejected() -> None:
    with pytest.raises(ValueError, match="negative"):
        decide(50.0, -1.0)


def test_high_confidence_does_not_bypass_the_cap() -> None:
    """Certainty is not authority to act unilaterally on a large amount."""
    decision = decide(100.0, 250_000.0)
    assert decision.requires_merchant_approval is True


# --------------------------------------------------------------------------- #
# Determinism and contract agreement
# --------------------------------------------------------------------------- #


def test_decisions_are_pure_and_repeatable() -> None:
    a, b = decide(73.4, 6200.0), decide(73.4, 6200.0)
    assert a == b


def test_config_bands_cover_the_range_without_gaps() -> None:
    bands = sorted(config()["bands"], key=lambda b: b["min_score"])
    assert bands[0]["min_score"] == 0
    assert bands[-1]["max_score"] == 100
    for lower, upper in zip(bands, bands[1:]):
        assert 0 < upper["min_score"] - lower["max_score"] < 0.01


def test_config_agrees_with_the_decision_function() -> None:
    """The advertised policy and the executed policy must be the same policy."""
    for band in config()["bands"]:
        for probe in (band["min_score"], band["max_score"]):
            assert decide(probe, SMALL).recommended_action == band["action"]


def test_cap_is_the_value_frozen_in_the_contract() -> None:
    assert MAX_AUTO_INTERVENTION_AMOUNT == 5000.0
    assert config()["max_auto_intervention_amount"] == 5000.0


# --------------------------------------------------------------------------- #
# Fusion: reported honestly, whatever the answer
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def fusion() -> dict:
    path = ARTIFACTS / "fusion.json"
    if not path.exists():
        pytest.skip("run `make fusion` first")
    return json.loads(path.read_text())


def test_fusion_is_compared_against_the_single_model(fusion) -> None:
    pr = fusion["held_out_pr_auc"]
    assert "transaction_model_alone" in pr
    assert pr["improvement"] == pytest.approx(
        pr["fused"] - pr["transaction_model_alone"], abs=1e-4
    )


def test_fusion_verdict_matches_its_own_numbers(fusion) -> None:
    """No claiming an improvement the measurement does not support."""
    pr = fusion["held_out_pr_auc"]
    assert fusion["fusion_helps"] == (pr["fused"] > pr["transaction_model_alone"])


def test_stacker_was_fit_on_validation_not_test(fusion) -> None:
    assert fusion["stacker_fit_on"] == "validation split"


def test_sub_score_correlation_is_recorded(fusion) -> None:
    """The diagnostic that explains the result has to be in the artifact.

    Reporting that fusion did not help without saying why would be a shrug.
    """
    corr = fusion["sub_score_correlation"]
    assert set(corr) == set(fusion["sub_scores"])
    off_diagonal = [
        corr[a][b] for a in corr for b in corr[a] if a != b
    ]
    assert max(off_diagonal) > 0.5, (
        "sub-scores were expected to be correlated; if they are now independent "
        "the fusion conclusion should be revisited"
    )


def test_refund_scorer_learned_its_weights(fusion) -> None:
    """A high return rate alone must not dominate: resellers return a lot."""
    coefs = fusion["refund_scorer_coefficients"]
    assert len(coefs) >= 6
    assert any(v < 0 for v in coefs.values()), "no mitigating signal"
