"""Bounded recovery workflows derived from risk artifacts.

This is an adjacent extension, not a second model. The detector has already
found a risky connected group and the policy engine has already bounded what can
happen. This module translates that into the next merchant-controlled recovery
step: review, verification, evidence preparation, or monitoring.
"""

from __future__ import annotations

from ml.financial import DEFAULT_RECOVERY_RATE


def _loss_type(features: dict[str, float]) -> str:
    if features.get("failed_payment_rate", 0.0) >= 0.50:
        return "payment_failure_abuse"
    if features.get("refund_value_ratio", 0.0) >= 0.25:
        return "refund_abuse"
    if features.get("return_rate", 0.0) >= 0.30:
        return "return_abuse"
    if features.get("signup_burst_ratio", 0.0) >= 0.50:
        return "promo_or_account_abuse"
    return "coordinated_abuse"


def _bounded_action(policy_action: str, loss_type: str) -> str:
    if policy_action == "MANUAL_REVIEW":
        return "MANUAL_REVIEW"
    if policy_action == "VERIFY":
        return "REQUEST_VERIFICATION"
    if loss_type in {"refund_abuse", "return_abuse"}:
        return "PREPARE_EVIDENCE"
    return "MONITOR_ONLY"


def _recommended_step(loss_type: str, bounded_action: str) -> str:
    if bounded_action == "MANUAL_REVIEW":
        return (
            "Queue the connected group for merchant review with refund, return "
            "and relationship evidence attached."
        )
    if bounded_action == "REQUEST_VERIFICATION":
        return (
            "Request lightweight customer verification before any refund, return "
            "or fulfilment intervention."
        )
    if bounded_action == "PREPARE_EVIDENCE":
        return (
            "Prepare the structured evidence packet for a refund or chargeback "
            "review; do not move money automatically."
        )
    if loss_type == "payment_failure_abuse":
        return (
            "Monitor the failed-payment pattern and escalate only if the same "
            "linked group keeps retrying."
        )
    return "Monitor the linked entities; no customer-facing action is recommended yet."


def _stopping_rule(policy: dict) -> str:
    if policy.get("requires_merchant_approval"):
        return (
            "Stop before any money movement or customer restriction; merchant "
            "approval is required."
        )
    return (
        "Stop if exposure exceeds INR 5,000, if confidence changes, or if a "
        "merchant dismisses the case."
    )


def workflow_from_ring(ring: dict) -> dict:
    summary = ring["summary"]
    policy = ring["policy"]
    features = ring.get("component_features", {})
    loss_type = _loss_type(features)
    bounded_action = _bounded_action(policy["recommended_action"], loss_type)

    return {
        "workflow_id": f"RCV-{summary['ring_id']}",
        "ring_id": summary["ring_id"],
        "loss_type": loss_type,
        "bounded_action": bounded_action,
        "recommended_step": _recommended_step(loss_type, bounded_action),
        "expected_protected_value": round(
            summary["financial_exposure"] * DEFAULT_RECOVERY_RATE, 2
        ),
        "requires_merchant_approval": bool(policy["requires_merchant_approval"]),
        "stopping_rule": _stopping_rule(policy),
        "audit_subject_id": summary["ring_id"],
    }


def workflows_from_rings(rings: list[dict], limit: int) -> list[dict]:
    ordered = sorted(
        rings,
        key=lambda r: (
            r["summary"]["risk_score"],
            r["summary"]["financial_exposure"],
        ),
        reverse=True,
    )
    return [workflow_from_ring(r) for r in ordered[:limit]]
