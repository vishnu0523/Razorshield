"""Contract conformance.

This is the load-bearing test of Phase 1. It calls every endpoint on the live app
and validates the actual JSON against contracts/api.schema.json. If the backend
drifts from the contract, this fails before the frontend ever sees it.

Run it on every commit. It is fast and it is the reason frontend and backend can
be built in parallel without coordinating.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator, FormatChecker

from backend.app.main import app

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / "contracts" / "api.schema.json").read_text())

client = TestClient(app)


def subschema(endpoint: str) -> dict:
    """Endpoint schema merged with root $defs so internal $refs resolve."""
    assert endpoint in CONTRACT["endpoints"], f"{endpoint} missing from contract"
    return {"$defs": CONTRACT["$defs"], **CONTRACT["endpoints"][endpoint]}


def assert_conforms(endpoint: str, payload: object) -> None:
    validator = Draft202012Validator(
        subschema(endpoint), format_checker=FormatChecker()
    )
    errors = sorted(validator.iter_errors(payload), key=lambda e: list(e.path))
    if errors:
        detail = "\n".join(
            f"  at {'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}"
            for e in errors[:10]
        )
        pytest.fail(f"{endpoint} violates the contract:\n{detail}")


# --------------------------------------------------------------------------- #
# Every frozen endpoint must exist and conform
# --------------------------------------------------------------------------- #

GET_ENDPOINTS = [
    ("GET /api/health", "/api/health"),
    ("GET /api/metrics", "/api/metrics"),
    ("GET /api/feed", "/api/feed?limit=10"),
    ("GET /api/rings", "/api/rings"),
    ("GET /api/recovery/workflows", "/api/recovery/workflows?limit=5"),
    ("GET /api/spikes", "/api/spikes"),
    ("GET /api/audit", "/api/audit"),
    ("GET /api/policy/config", "/api/policy/config"),
    ("GET /api/simulate/step", "/api/simulate/step?phase=1"),
    ("GET /api/pipeline/log", "/api/pipeline/log"),
]


@pytest.mark.parametrize("endpoint,url", GET_ENDPOINTS, ids=[e for e, _ in GET_ENDPOINTS])
def test_get_endpoint_conforms(endpoint: str, url: str) -> None:
    resp = client.get(url)
    assert resp.status_code == 200, resp.text
    assert_conforms(endpoint, resp.json())


def test_health_reports_completed_build_phase() -> None:
    body = client.get("/api/health").json()
    assert body["phase"] == 16


def test_ring_detail_conforms() -> None:
    """Uses a real ring once artifacts exist, the placeholder before that."""
    listing = client.get("/api/rings?limit=1").json()
    ring_id = listing["rings"][0]["ring_id"] if listing["rings"] else "AR-PLACEHOLDER"
    resp = client.get(f"/api/rings/{ring_id}")
    assert resp.status_code == 200, resp.text
    assert_conforms("GET /api/rings/{ring_id}", resp.json())


def test_simulate_start_conforms() -> None:
    resp = client.post("/api/simulate/start")
    assert resp.status_code == 200, resp.text
    assert_conforms("POST /api/simulate/start", resp.json())


def test_financial_recompute_conforms() -> None:
    resp = client.post(
        "/api/financial/recompute",
        json={
            "recovery_rate": 0.55,
            "cost_per_false_review": 200.0,
            "cost_per_false_block_ratio": 0.22,
        },
    )
    assert resp.status_code == 200, resp.text
    assert_conforms("POST /api/financial/recompute", resp.json())


def test_contract_covers_every_route() -> None:
    """No endpoint may exist on the app without a contract entry."""
    documented = {e.split(" ", 1)[1] for e in CONTRACT["endpoints"]}
    live = {
        r.path
        for r in app.routes
        if getattr(r, "path", "").startswith("/api/")
    }
    # Contract paths use {ring_id}; FastAPI reports the same. Compare directly.
    undocumented = live - documented
    assert not undocumented, f"Routes missing from the contract: {sorted(undocumented)}"


# --------------------------------------------------------------------------- #
# Anti-fake-metric guards
# --------------------------------------------------------------------------- #


ARTIFACT_METRICS = ROOT / "artifacts" / "metrics.json"


def test_metrics_placeholder_flag_tracks_artifact_presence() -> None:
    """The API must never present an untrained state as a result, or vice versa."""
    body = client.get("/api/metrics").json()
    if ARTIFACT_METRICS.exists():
        assert body["is_placeholder"] is False
        assert not any("PLACEHOLDER" in c.upper() for c in body["caveats"])
    else:
        assert body["is_placeholder"] is True
        assert any("PLACEHOLDER" in c.upper() for c in body["caveats"])


def test_placeholder_metrics_are_all_zero() -> None:
    """A placeholder must never carry a plausible-looking number.

    Before training, every metric reads 0.00. There is no value in the
    untrained state that could be screenshotted and mistaken for a result.
    """
    if ARTIFACT_METRICS.exists():
        pytest.skip("artifacts present; real metrics are served")
    body = client.get("/api/metrics").json()
    for block in ("transaction_model", "baseline_model", "ring_model"):
        for key in ("precision", "recall", "f1"):
            assert body[block][key] == 0.0, f"{block}.{key} is not zero"
    assert body["financial"]["net_protected_value"] == 0.0


def test_served_metrics_are_byte_faithful_to_the_artifact() -> None:
    """The API is a read layer. It must not massage numbers on the way out."""
    if not ARTIFACT_METRICS.exists():
        pytest.skip("no artifacts yet")
    on_disk = json.loads(ARTIFACT_METRICS.read_text())
    served = client.get("/api/metrics").json()
    for block in ("transaction_model", "baseline_model", "ring_model"):
        assert served[block] == on_disk[block], f"{block} was altered in transit"
    assert served["financial"]["net_protected_value"] == (
        on_disk["financial"]["net_protected_value"]
    )


def test_recompute_echoes_adjusted_assumptions() -> None:
    """Assumptions are merchant-visible and merchant-adjustable by contract."""
    body = client.post(
        "/api/financial/recompute",
        json={"recovery_rate": 0.55, "cost_per_false_review": 200.0,
              "cost_per_false_block_ratio": 0.22},
    ).json()
    values = {a["key"]: a["value"] for a in body["assumptions"]}
    assert values["recovery_rate"] == 0.55
    assert values["cost_per_false_review"] == 200.0
    assert all(a["adjustable"] for a in body["assumptions"])


def test_unknown_ring_is_404_not_a_fabricated_ring() -> None:
    assert client.get("/api/rings/AR-DOES-NOT-EXIST").status_code == 404
    assert client.get("/api/rings/NOPE-1").status_code == 404


def test_recovery_workflows_are_bounded_and_merchant_controlled() -> None:
    body = client.get("/api/recovery/workflows?limit=10").json()
    assert "No Razorpay production action is executed" in body["caveat"]
    for workflow in body["workflows"]:
        assert workflow["expected_protected_value"] >= 0
        assert workflow["audit_subject_id"] == workflow["ring_id"]
        if workflow["bounded_action"] in {"MANUAL_REVIEW", "REQUEST_VERIFICATION"}:
            assert workflow["requires_merchant_approval"] is True


# --------------------------------------------------------------------------- #
# Policy bands are frozen from Phase 1 so Phase 11 cannot quietly move them
# --------------------------------------------------------------------------- #


def test_policy_bands_are_contiguous_and_complete() -> None:
    bands = client.get("/api/policy/config").json()["bands"]
    ordered = sorted(bands, key=lambda b: b["min_score"])
    assert ordered[0]["min_score"] == 0
    assert ordered[-1]["max_score"] == 100
    for lower, upper in zip(ordered, ordered[1:]):
        gap = upper["min_score"] - lower["max_score"]
        assert 0 < gap < 0.01, f"Band gap between {lower} and {upper}"
    assert [b["action"] for b in ordered] == [
        "ALLOW",
        "MONITOR",
        "VERIFY",
        "MANUAL_REVIEW",
    ]


def test_amount_cap_is_present_and_positive() -> None:
    cfg = client.get("/api/policy/config").json()
    assert cfg["max_auto_intervention_amount"] == 5000.0
    assert cfg["currency"] == "INR"
