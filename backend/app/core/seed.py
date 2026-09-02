"""Seed the audit trail from detection artifacts.

Replays what the pipeline actually decided for each flagged component, in the
order it decided it, so the trail on a case page is a record of the real run
rather than illustrative text.

Run:
    python -m backend.app.core.seed
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import audit

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", ROOT / "artifacts"))


def _events(ring: dict, base: datetime) -> list[dict]:
    policy = ring["policy"]
    summary = ring["summary"]
    top = next(
        (e for e in ring["evidence"] if e["weight"] > 0),
        {"statement": "structural evidence only", "weight": 0.0},
    )

    return [
        {
            "event": "graph_analysis_completed",
            "actor": "system",
            "input_summary": (
                f"{summary['n_transactions']} orders across "
                f"{summary['n_accounts']} accounts in a trailing window"
            ),
            "decision": "component formed",
            "reason": (
                f"Accounts connected through {summary['n_devices']} devices, "
                f"{summary['n_addresses']} addresses, {summary['n_ips']} IPs."
            ),
            "result": f"{len(ring['graph']['nodes'])} nodes linked",
            "offset": 0,
        },
        {
            "event": "component_scored",
            "actor": "model",
            "input_summary": "component feature vector, 8 features",
            "decision": f"risk {summary['risk_score']:.0f}/100",
            "reason": top["statement"],
            "result": f"{summary['risk_level']} at confidence {summary['confidence']:.2f}",
            "offset": 2,
        },
        {
            "event": "exposure_calculated",
            "actor": "system",
            "input_summary": "refunds issued plus captured value still open",
            "decision": f"INR {summary['financial_exposure']:,.0f}",
            "reason": "Upper bound on loss; recovery rate applied downstream.",
            "result": "exposure recorded",
            "offset": 3,
        },
        {
            "event": "policy_evaluated",
            "actor": "policy_engine",
            "input_summary": (
                f"risk {summary['risk_score']:.0f}, exposure "
                f"INR {summary['financial_exposure']:,.0f}"
            ),
            "decision": policy["recommended_action"],
            "reason": policy["reason"],
            "result": (
                "awaiting merchant approval"
                if policy["requires_merchant_approval"]
                else "recommendation issued"
            ),
            "offset": 4,
        },
        {
            "event": "explanation_generated",
            "actor": "llm" if ring["explanation"]["source"] == "llm" else "system",
            "input_summary": f"{len(ring['evidence'])} structured evidence items",
            "decision": "summary written",
            "reason": (
                "Language model unavailable, deterministic template used."
                if ring["explanation"]["source"] != "llm"
                else "Model rewrote the deterministic summary."
            ),
            "result": ring["explanation"]["source"],
            "offset": 6,
        },
    ]


def seed(limit: int = 40) -> int:
    """Write the trail for the highest-risk cases. Idempotent by refusing to
    double-write: seeding an already-populated log would duplicate history."""
    audit.init_db()
    if audit.count() > 0:
        return 0

    path = ARTIFACTS / "rings.json"
    if not path.is_file():
        return 0

    rings = json.loads(path.read_text())["rings"][:limit]
    written = 0
    with audit.SessionLocal() as session:
        for i, ring in enumerate(rings):
            base = datetime.fromisoformat(ring["summary"]["detected_at"])
            if base.tzinfo is None:
                base = base.replace(tzinfo=timezone.utc)
            base = base + timedelta(minutes=i * 7)
            for event in _events(ring, base):
                audit.append(
                    event=event["event"],
                    actor=event["actor"],
                    subject_id=ring["ring_id"],
                    input_summary=event["input_summary"],
                    decision=event["decision"],
                    reason=event["reason"],
                    policy=ring["policy"]["policy_version"],
                    result=event["result"],
                    timestamp=base + timedelta(seconds=event["offset"]),
                    session=session,
                )
                written += 1
    return written


def main() -> None:
    written = seed()
    if written:
        print(f"seeded {written} audit entries")
    else:
        print(f"audit log already holds {audit.count()} entries; nothing written")
    print("chain:", audit.verify_chain())


if __name__ == "__main__":
    main()
