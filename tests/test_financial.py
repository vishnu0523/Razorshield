"""Phase 12 gates: are the money numbers auditable?

The point of adjustable assumptions is that a sceptic can disprove them. These
tests check the recomputation is a real sum over real decisions rather than a
summary being scaled, and that an assumption set exists under which RazorShield
looks bad. A panel that can only ever produce a flattering number is decoration.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from backend.app.core.financial import (
    Assumptions,
    DECISIONS_PATH,
    decisions_available,
    recompute,
)
from backend.app.main import app

client = TestClient(app)

pytestmark = pytest.mark.skipif(
    not decisions_available(), reason="run `make evaluate` first"
)


@pytest.fixture(scope="module")
def decisions() -> dict:
    return json.loads(DECISIONS_PATH.read_text())


# --------------------------------------------------------------------------- #
# The recomputation is real
# --------------------------------------------------------------------------- #


def test_decisions_are_stored_as_they_were_made(decisions) -> None:
    """Storing the decisions, not a summary, is what makes recompute honest."""
    n = len(decisions["is_fraud"])
    assert n > 100
    assert len(decisions["flagged"]) == n
    assert len(decisions["amount"]) == n
    assert decisions["split"] == "test"


def test_defaults_reproduce_the_published_metrics(decisions) -> None:
    """The dashboard's starting figures must equal the evaluation report's."""
    from backend.app.core.financial import ARTIFACTS

    published = json.loads((ARTIFACTS / "metrics.json").read_text())["financial"]
    fresh = recompute(Assumptions())
    for key in (
        "exposure_detected",
        "prevented_loss",
        "false_positive_cost",
        "net_protected_value",
    ):
        assert fresh[key] == pytest.approx(published[key], abs=1.0), key


def test_exposure_does_not_move_with_assumptions() -> None:
    """Detected exposure is an observation. Only the interpretation is tunable."""
    a = recompute(Assumptions(recovery_rate=0.1, cost_per_false_review=900))
    b = recompute(Assumptions(recovery_rate=0.9, cost_per_false_review=10))
    assert a["exposure_detected"] == b["exposure_detected"]


def test_recovery_rate_scales_prevented_loss() -> None:
    full = recompute(Assumptions(recovery_rate=1.0))
    half = recompute(Assumptions(recovery_rate=0.5))
    assert half["prevented_loss"] == pytest.approx(full["prevented_loss"] / 2, rel=1e-6)
    assert full["prevented_loss"] == pytest.approx(full["exposure_detected"], rel=1e-6)


def test_zero_recovery_means_nothing_was_saved() -> None:
    result = recompute(Assumptions(recovery_rate=0.0))
    assert result["prevented_loss"] == 0.0
    assert result["net_protected_value"] < 0


def test_net_is_always_the_stated_subtraction() -> None:
    for rate in (0.0, 0.35, 0.7, 1.0):
        r = recompute(Assumptions(recovery_rate=rate))
        assert r["net_protected_value"] == pytest.approx(
            r["prevented_loss"] - r["false_positive_cost"], abs=1.0
        )


def test_costs_are_monotone_in_their_assumptions() -> None:
    cheap = recompute(Assumptions(cost_per_false_review=10))
    dear = recompute(Assumptions(cost_per_false_review=900))
    assert dear["false_positive_cost"] > cheap["false_positive_cost"]
    assert dear["net_protected_value"] < cheap["net_protected_value"]


# --------------------------------------------------------------------------- #
# The panel can reach an unflattering answer
# --------------------------------------------------------------------------- #


def test_an_assumption_set_exists_where_we_lose_money() -> None:
    """If no setting makes RazorShield look bad, the panel proves nothing."""
    pessimistic = recompute(
        Assumptions(
            recovery_rate=0.7,
            cost_per_false_review=120,
            cost_per_false_block_ratio=0.6,
        )
    )
    assert pessimistic["net_protected_value"] < 0, (
        "no reachable assumption set produces a negative result; the panel is "
        "incapable of disagreeing with us"
    )


# --------------------------------------------------------------------------- #
# API surface
# --------------------------------------------------------------------------- #


def test_endpoint_returns_the_adjusted_assumptions() -> None:
    body = client.post(
        "/api/financial/recompute",
        json={
            "recovery_rate": 0.42,
            "cost_per_false_review": 333.0,
            "cost_per_false_block_ratio": 0.07,
        },
    ).json()
    values = {a["key"]: a["value"] for a in body["assumptions"]}
    assert values["recovery_rate"] == 0.42
    assert values["cost_per_false_review"] == 333.0
    assert values["cost_per_false_block_ratio"] == 0.07


def test_endpoint_rejects_impossible_assumptions() -> None:
    for payload in (
        {"recovery_rate": 1.5},
        {"recovery_rate": -0.1},
        {"cost_per_false_review": -5},
        {"cost_per_false_block_ratio": 2.0},
    ):
        assert client.post("/api/financial/recompute", json=payload).status_code == 422


def test_endpoint_matches_the_engine() -> None:
    """The API must not do its own arithmetic on the way out."""
    knobs = {
        "recovery_rate": 0.55,
        "cost_per_false_review": 250.0,
        "cost_per_false_block_ratio": 0.22,
    }
    served = client.post("/api/financial/recompute", json=knobs).json()
    direct = recompute(Assumptions(**knobs))
    assert served["net_protected_value"] == pytest.approx(
        direct["net_protected_value"], abs=0.01
    )
