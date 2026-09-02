"""Scripted demo simulator.

Precomputed offline into a JSON script and replayed with timers. Real inference,
real evidence, real policy -- but on a fixed sequence that cannot vary between
runs. A demo that can fail live is a demo that will.

CHOOSING THE RING MATTERS MORE THAN THE ANIMATION.

The product's claim is that individually unremarkable transactions add up to a
ring. So the demo ring is selected to satisfy that claim rather than to look
impressive: every one of its transactions must sit below the VERIFY threshold on
its own, while the component scores CRITICAL together. If no such ring exists,
`build` says so and refuses to pick a flattering substitute.

Run:
    python -m ml.simulate
"""

from __future__ import annotations

import json
from datetime import timedelta

import numpy as np
import pandas as pd

from . import config as C
from .features import build_features, load_events
from .financial import DEFAULT_RECOVERY_RATE
from .rings import detect as detect_rings
from .train import ARTIFACTS, load_risk_scores

# A transaction below this score would not, on its own, trigger an intervention.
INDIVIDUALLY_UNREMARKABLE_BELOW = 70.0
# The component must land here for the contrast to be the story.
COMPONENT_CRITICAL_AT = 90.0

TOTAL_PHASES = 9


def _risk_level(score: float) -> str:
    if score >= 90:
        return "CRITICAL"
    if score >= 70:
        return "HIGH"
    if score >= 40:
        return "MEDIUM"
    return "LOW"


def _action(score: float) -> str:
    if score >= 90:
        return "MANUAL_REVIEW"
    if score >= 70:
        return "VERIFY"
    if score >= 40:
        return "MONITOR"
    return "ALLOW"


def _feed_item(row: pd.Series, score: float, ring_id: str | None) -> dict:
    return {
        "transaction_id": row["transaction_id"],
        "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
        "amount": round(float(row["amount"]), 2),
        "currency": "INR",
        "customer_id": row["customer_id"],
        "payment_method": row["payment_method"],
        "product_category": row["product_category"],
        "risk_score": round(float(score), 1),
        "risk_level": _risk_level(score),
        "policy_action": _action(score),
        "top_signals": [],
        "ring_id": ring_id,
    }


def _audit(seq: int, base, offset: int, **kwargs) -> dict:
    return {
        "entry_id": f"SIM-{seq:04d}",
        "timestamp": (base + timedelta(seconds=offset)).isoformat(),
        "policy": "v1",
        **kwargs,
    }


def choose_demo_ring(rings: list[dict], transaction_risk: pd.Series,
                     owner: pd.Series) -> tuple[dict | None, str]:
    """The ring that best demonstrates the claim, or nothing.

    Ranked by exposure among rings whose transactions are all individually
    unremarkable. Selecting on the claim rather than on the visuals is the
    difference between a demo and a sales deck.
    """
    account_of = owner.to_dict()
    candidates = []
    for ring in rings:
        if ring["summary"]["risk_score"] < COMPONENT_CRITICAL_AT:
            continue
        members = set(ring["accounts"])
        scores = [
            v * 100
            for tid, v in transaction_risk.items()
            if account_of.get(tid) in members
        ]
        if not scores:
            continue
        if max(scores) < INDIVIDUALLY_UNREMARKABLE_BELOW:
            candidates.append((ring["summary"]["financial_exposure"], max(scores), ring))

    if not candidates:
        return None, (
            "No ring has a CRITICAL component score while every one of its "
            "transactions stays below the VERIFY threshold. The demo would have "
            "to overstate the contrast, so it does not run."
        )
    candidates.sort(key=lambda c: -c[0])
    exposure, peak, ring = candidates[0]
    return ring, (
        f"{ring['ring_id']}: component scores "
        f"{ring['summary']['risk_score']:.0f}/100 while its hottest single "
        f"transaction reaches only {peak:.0f}."
    )


def build() -> dict:
    events = load_events()
    transactions = events["transactions"]
    risk = load_risk_scores()
    owner = transactions.set_index("transaction_id")["customer_id"]

    rings = json.loads((ARTIFACTS / "rings.json").read_text())["rings"]
    ring, rationale = choose_demo_ring(rings, risk, owner)
    if ring is None:
        raise RuntimeError(rationale)

    members = set(ring["accounts"])
    tx = transactions.assign(risk=transactions["transaction_id"].map(risk) * 100)
    ring_tx = tx[tx["customer_id"].isin(members)].sort_values("timestamp")
    normal_tx = tx[~tx["customer_id"].isin(members)].nsmallest(6, "risk")

    base = pd.Timestamp(ring["summary"]["detected_at"])
    summary = ring["summary"]
    policy = ring["policy"]
    exposure = summary["financial_exposure"]
    prevented = round(exposure * DEFAULT_RECOVERY_RATE, 2)

    early = ring_tx.head(4)
    middle = ring_tx.iloc[4:10] if len(ring_tx) > 4 else ring_tx.head(0)

    detail = {
        "summary": summary,
        "evidence": ring["evidence"],
        "graph": ring["graph"],
        "policy": policy,
        "explanation": ring["explanation"],
        "component_features": ring["component_features"],
    }

    steps = [
        {
            "phase": 1,
            "phase_name": "Normal activity",
            "narrative": (
                "Ordinary merchant traffic. Every order scores low and is "
                "allowed through without friction."
            ),
            "feed_items": [
                _feed_item(r, r["risk"], None) for _, r in normal_tx.iterrows()
            ],
            "ring": None,
            "audit_entries": [],
            "financial_delta": None,
            "is_final": False,
        },
        {
            "phase": 2,
            "phase_name": "Unremarkable orders",
            "narrative": (
                "New orders arrive. None of them would trigger an intervention "
                "on its own — a row-level model sees nothing worth stopping."
            ),
            "feed_items": [
                _feed_item(r, r["risk"], None) for _, r in early.iterrows()
            ],
            "ring": None,
            "audit_entries": [],
            "financial_delta": None,
            "is_final": False,
        },
        {
            "phase": 3,
            "phase_name": "Relationships accumulate",
            "narrative": (
                f"More orders from different accounts. They share "
                f"{summary['n_devices']} devices and "
                f"{summary['n_addresses']} delivery addresses, but no single "
                "order looks wrong."
            ),
            "feed_items": [
                _feed_item(r, r["risk"], None) for _, r in middle.iterrows()
            ],
            "ring": None,
            "audit_entries": [],
            "financial_delta": None,
            "is_final": False,
        },
        {
            "phase": 4,
            "phase_name": "Graph analysis",
            "narrative": (
                "The risk graph runs over the trailing window, linking accounts "
                "through the entities they share."
            ),
            "feed_items": [],
            "ring": None,
            "audit_entries": [
                _audit(1, base, 0, event="graph_analysis_completed", actor="system",
                       subject_id=ring["ring_id"],
                       input_summary=f"{summary['n_transactions']} orders in the window",
                       decision="component formed",
                       reason=f"{len(ring['graph']['nodes'])} entities linked",
                       result="component queued for scoring")
            ],
            "financial_delta": None,
            "is_final": False,
        },
        {
            "phase": 5,
            "phase_name": "Coordinated abuse detected",
            "narrative": (
                f"{summary['n_accounts']} accounts form one connected group "
                f"scoring {summary['risk_score']:.0f} out of 100. Exposure is "
                f"INR {exposure:,.0f}."
            ),
            "feed_items": [],
            "ring": detail,
            "audit_entries": [
                _audit(2, base, 2, event="component_scored", actor="model",
                       subject_id=ring["ring_id"],
                       input_summary="component feature vector",
                       decision=f"risk {summary['risk_score']:.0f}/100",
                       reason=ring["evidence"][0]["statement"],
                       result=summary["risk_level"]),
                _audit(3, base, 3, event="exposure_calculated", actor="system",
                       subject_id=ring["ring_id"],
                       input_summary="refunds issued plus captured value still open",
                       decision=f"INR {exposure:,.0f}",
                       reason="Upper bound; recovery rate applied downstream.",
                       result="exposure recorded"),
            ],
            "financial_delta": None,
            "is_final": False,
        },
        {
            "phase": 6,
            "phase_name": "Investigator explains",
            "narrative": ring["explanation"]["text"],
            "feed_items": [],
            "ring": detail,
            "audit_entries": [
                _audit(4, base, 5, event="explanation_generated",
                       actor="llm" if ring["explanation"]["source"] == "llm" else "system",
                       subject_id=ring["ring_id"],
                       input_summary=f"{len(ring['evidence'])} evidence items",
                       decision="summary written",
                       reason="Explains evidence; does not decide.",
                       result=ring["explanation"]["source"])
            ],
            "financial_delta": None,
            "is_final": False,
        },
        {
            "phase": 7,
            "phase_name": "Policy engine decides",
            "narrative": policy["reason"],
            "feed_items": [],
            "ring": detail,
            "audit_entries": [
                _audit(5, base, 6, event="policy_evaluated", actor="policy_engine",
                       subject_id=ring["ring_id"],
                       input_summary=f"risk {summary['risk_score']:.0f}, exposure INR {exposure:,.0f}",
                       decision=policy["recommended_action"],
                       reason=policy["reason"],
                       result="awaiting merchant approval"
                       if policy["requires_merchant_approval"] else "recommendation issued")
            ],
            "financial_delta": None,
            "is_final": False,
        },
        {
            "phase": 8,
            "phase_name": "Merchant approves",
            "narrative": (
                "The recommendation is held for a person. Nothing is actioned "
                "until the merchant agrees, and the decision is appended to the "
                "audit trail rather than overwriting anything."
            ),
            "feed_items": [],
            "ring": detail,
            "audit_entries": [
                _audit(6, base, 60, event="merchant_decision", actor="merchant",
                       subject_id=ring["ring_id"],
                       input_summary=f"recommended {policy['recommended_action']}",
                       decision="APPROVED",
                       reason="Shared devices and addresses confirmed.",
                       result="recorded")
            ],
            "financial_delta": None,
            "is_final": False,
        },
        {
            "phase": 9,
            "phase_name": "Net protected",
            "narrative": (
                f"INR {prevented:,.0f} protected from this group after applying "
                f"a {DEFAULT_RECOVERY_RATE:.0%} recovery rate. Not one of these "
                "orders would have been stopped by scoring transactions alone."
            ),
            "feed_items": [],
            "ring": detail,
            "audit_entries": [],
            "financial_delta": prevented,
            "is_final": True,
        },
    ]

    return {
        "run_id": f"SIM-{ring['ring_id']}",
        "total_phases": TOTAL_PHASES,
        "seed": C.SEED,
        "rationale": rationale,
        "steps": steps,
    }


def main() -> None:
    script = build()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "simulation.json").write_text(json.dumps(script, indent=2))

    print(f"demo ring selected: {script['rationale']}\n")
    for step in script["steps"]:
        marker = "*" if step["ring"] else " "
        print(
            f"  {marker} phase {step['phase']}  {step['phase_name']:28s} "
            f"{len(step['feed_items'])} feed, {len(step['audit_entries'])} audit"
        )
    print(f"\nwritten: {ARTIFACTS / 'simulation.json'}")


if __name__ == "__main__":
    main()
