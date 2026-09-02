"""Deterministic policy engine.

The single place in RazorShield where a risk score becomes an action. Nothing
else may decide, and in particular no language model may: the investigator in a
later phase writes prose about the evidence and its output is never consulted
here.

The bands and the automatic-intervention cap were frozen into the API contract
at Phase 1, before any model existed, precisely so they could not later drift to
flatter a demo. This module is the executable form of that contract.

Every boundary is tested at both sides: 69/70, 89/90, and the money cap at
4,999 / 5,000 / 5,001.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

PolicyAction = Literal["ALLOW", "MONITOR", "VERIFY", "MANUAL_REVIEW"]

POLICY_VERSION = "v1"

# (inclusive lower bound, action, label). Read top-down.
BANDS: tuple[tuple[float, PolicyAction, str], ...] = (
    (90.0, "MANUAL_REVIEW", "90-100"),
    (70.0, "VERIFY", "70-89"),
    (40.0, "MONITOR", "40-69"),
    (0.0, "ALLOW", "0-39"),
)

# Above this rupee value, no intervention happens without a person. The cap is
# on the money at stake, not on the confidence: being very sure is not a reason
# to act unilaterally on a large amount.
MAX_AUTO_INTERVENTION_AMOUNT = 5000.0

# Actions that actually interrupt a customer. MONITOR and ALLOW do not touch
# anyone, so the cap does not apply to them.
INTERVENING_ACTIONS: frozenset[str] = frozenset({"VERIFY", "MANUAL_REVIEW"})


@dataclass(frozen=True)
class Decision:
    recommended_action: PolicyAction
    requires_merchant_approval: bool
    reason: str
    policy_version: str
    risk_band: str
    amount_cap_applied: bool

    def to_dict(self) -> dict:
        return asdict(self)


def band_for(risk_score: float) -> tuple[PolicyAction, str]:
    if not 0.0 <= risk_score <= 100.0:
        raise ValueError(f"risk_score must be within 0-100, got {risk_score}")
    for lower, action, label in BANDS:
        if risk_score >= lower:
            return action, label
    raise AssertionError("bands do not cover the range")  # pragma: no cover


def decide(risk_score: float, amount: float) -> Decision:
    """Map a risk score and an amount at stake to a bounded action.

    Pure and total: same inputs, same output, no I/O, no model call.
    """
    if amount < 0:
        raise ValueError(f"amount must not be negative, got {amount}")

    action, band = band_for(risk_score)

    # Strictly greater than: an amount exactly at the limit is still within it.
    over_limit = amount > MAX_AUTO_INTERVENTION_AMOUNT
    cap_applied = over_limit and action in INTERVENING_ACTIONS
    requires_approval = action == "MANUAL_REVIEW" or cap_applied

    if cap_applied and action != "MANUAL_REVIEW":
        reason = (
            f"Risk score {risk_score:.0f} falls in the {band} band, but the "
            f"amount at stake of INR {amount:,.0f} exceeds the "
            f"INR {MAX_AUTO_INTERVENTION_AMOUNT:,.0f} automatic-intervention "
            "limit, so a person must approve before anything is actioned."
        )
    elif action == "MANUAL_REVIEW":
        reason = (
            f"Risk score {risk_score:.0f} falls in the {band} band, which "
            "always requires human review."
        )
    else:
        reason = f"Risk score {risk_score:.0f} falls in the {band} band."

    return Decision(
        recommended_action=action,
        requires_merchant_approval=requires_approval,
        reason=reason,
        policy_version=POLICY_VERSION,
        risk_band=band,
        amount_cap_applied=cap_applied,
    )


def config() -> dict:
    """Contract-shaped policy configuration for /api/policy/config."""
    ordered = sorted(BANDS, key=lambda b: b[0])
    bands = []
    for i, (lower, action, _) in enumerate(ordered):
        upper = ordered[i + 1][0] - 0.001 if i + 1 < len(ordered) else 100.0
        bands.append(
            {"min_score": lower, "max_score": round(upper, 3), "action": action}
        )
    return {
        "policy_version": POLICY_VERSION,
        "bands": bands,
        "max_auto_intervention_amount": MAX_AUTO_INTERVENTION_AMOUNT,
        "currency": "INR",
    }
