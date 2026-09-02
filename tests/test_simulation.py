"""Phase 13 gates: is the demo honest and will it survive being run live?

test_demo_ring_proves_the_claim is the one that matters. The product asserts
that individually unremarkable transactions add up to a ring, so the demo ring
must actually satisfy that -- otherwise the most persuasive moment in the
presentation would be the least defensible.
"""

from __future__ import annotations

import json

import pytest

from ml.simulate import (
    COMPONENT_CRITICAL_AT,
    INDIVIDUALLY_UNREMARKABLE_BELOW,
    TOTAL_PHASES,
    build,
)
from ml.train import ARTIFACTS

pytestmark = pytest.mark.skipif(
    not (ARTIFACTS / "rings.json").exists(), reason="run `make rings` first"
)


@pytest.fixture(scope="module")
def script() -> dict:
    return build()


def test_demo_ring_proves_the_claim(script) -> None:
    """CRITICAL as a group, unremarkable as individual transactions."""
    ring = next(s["ring"] for s in script["steps"] if s["ring"])
    assert ring["summary"]["risk_score"] >= COMPONENT_CRITICAL_AT

    feed = [i for s in script["steps"] for i in s["feed_items"]]
    assert feed, "no transactions shown before the reveal"
    peak = max(i["risk_score"] for i in feed)
    assert peak < INDIVIDUALLY_UNREMARKABLE_BELOW, (
        f"a transaction scoring {peak} would have been stopped on its own, so "
        "the demo does not demonstrate what it claims"
    )


def test_unremarkable_orders_are_shown_before_the_reveal(script) -> None:
    """The contrast only lands if the audience watches them go past first."""
    reveal = next(s["phase"] for s in script["steps"] if s["ring"])
    early_feed = [
        i for s in script["steps"] if s["phase"] < reveal for i in s["feed_items"]
    ]
    assert len(early_feed) >= 8


def test_script_is_deterministic() -> None:
    a, b = build(), build()
    assert a["run_id"] == b["run_id"]
    assert json.dumps(a["steps"], sort_keys=True) == json.dumps(
        b["steps"], sort_keys=True
    )


def test_all_phases_present_and_ordered(script) -> None:
    phases = [s["phase"] for s in script["steps"]]
    assert phases == list(range(1, TOTAL_PHASES + 1))
    assert sum(s["is_final"] for s in script["steps"]) == 1
    assert script["steps"][-1]["is_final"]


def test_every_step_has_narrative_and_shape(script) -> None:
    required = {
        "phase", "phase_name", "narrative", "feed_items", "ring",
        "audit_entries", "financial_delta", "is_final",
    }
    for step in script["steps"]:
        assert set(step) == required
        assert step["narrative"].strip()


def test_the_reveal_carries_real_evidence(script) -> None:
    ring = next(s["ring"] for s in script["steps"] if s["ring"])
    assert ring["evidence"], "the reveal has no evidence behind it"
    assert ring["graph"]["nodes"], "no graph to show"
    assert ring["policy"]["recommended_action"] in {"VERIFY", "MANUAL_REVIEW"}


def test_final_value_applies_the_recovery_rate(script) -> None:
    """The demo must not claim the full exposure was saved."""
    from ml.financial import DEFAULT_RECOVERY_RATE

    final = script["steps"][-1]
    ring = next(s["ring"] for s in script["steps"] if s["ring"])
    exposure = ring["summary"]["financial_exposure"]
    assert final["financial_delta"] == pytest.approx(
        exposure * DEFAULT_RECOVERY_RATE, rel=1e-3
    )
    assert final["financial_delta"] < exposure


def test_merchant_approval_appears_before_the_result(script) -> None:
    """Nothing is actioned until a person agrees."""
    approval = next(
        s["phase"]
        for s in script["steps"]
        for e in s["audit_entries"]
        if e["actor"] == "merchant"
    )
    result = next(s["phase"] for s in script["steps"] if s["financial_delta"])
    assert approval < result
