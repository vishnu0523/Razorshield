"""Phase 14 and 15 gates: the model cannot decide, and live keys cannot load.

test_llm_failure_does_not_change_any_decision is the important one. The whole
safety architecture rests on the language model being outside the decision path,
and an architecture nobody tested is an architecture nobody should trust.
"""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient

from backend.app.adapters import llm, razorpay
from backend.app.main import app
from ml.train import ARTIFACTS

client = TestClient(app)


class Echo:
    """A provider that returns whatever it was told to."""

    name = "test"

    def __init__(self, text: str):
        self._text = text

    def available(self) -> bool:
        return True

    def complete(self, system: str, user: str) -> str:
        self.last_prompt = user
        return self._text


class Broken:
    name = "broken"

    def available(self) -> bool:
        return True

    def complete(self, system: str, user: str) -> str:
        raise ConnectionError("outage")


@pytest.fixture(scope="module")
def ring() -> dict:
    path = ARTIFACTS / "rings.json"
    if not path.exists():
        pytest.skip("run `make rings` first")
    return json.loads(path.read_text())["rings"][0]


DETERMINISTIC = "Seventeen accounts share devices and addresses."


# --------------------------------------------------------------------------- #
# The model is outside the decision path
# --------------------------------------------------------------------------- #


def test_llm_failure_does_not_change_any_decision(ring) -> None:
    """Kill the model; the case keeps its score, evidence and action."""
    ring_id = ring["ring_id"]
    before = client.get(f"/api/rings/{ring_id}").json()
    client.get(f"/api/rings/{ring_id}/explanation?simulate_failure=true")
    after = client.get(f"/api/rings/{ring_id}").json()

    assert after["summary"]["risk_score"] == before["summary"]["risk_score"]
    assert after["policy"] == before["policy"]
    assert after["evidence"] == before["evidence"]


def test_outage_degrades_to_deterministic_text(ring) -> None:
    explanation, log = llm.explain(ring, DETERMINISTIC, provider=Broken())
    assert explanation.source == "deterministic"
    assert explanation.text == DETERMINISTIC
    assert explanation.degraded is True
    assert "deterministic fallback used" in log


def test_no_configured_model_is_not_a_degraded_state(ring) -> None:
    """Running without credentials is a supported configuration, not a failure."""
    explanation, _ = llm.explain(ring, DETERMINISTIC, provider=llm.NullProvider())
    assert explanation.source == "deterministic"
    assert explanation.degraded is False


def test_endpoint_survives_a_simulated_outage(ring) -> None:
    body = client.get(
        f"/api/rings/{ring['ring_id']}/explanation?simulate_failure=true"
    ).json()
    assert body["degraded"] is True
    assert body["source"] == "deterministic"
    assert body["text"].strip()


# --------------------------------------------------------------------------- #
# Output validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        "You should block these accounts immediately as they are fraudsters.",
        "This proves the group is committing refund fraud.",
        "These accounts are fraud and must be suspended.",
        "We are 100% certain about this group of linked accounts here.",
    ],
)
def test_verdicts_and_overclaims_are_rejected(ring, text: str) -> None:
    """A model stating a verdict is claiming authority it does not have."""
    explanation, log = llm.explain(ring, DETERMINISTIC, provider=Echo(text))
    assert explanation.source == "deterministic"
    assert explanation.degraded is True
    assert "fallback" in log


def test_invented_figures_are_rejected(ring) -> None:
    """A plausible invented rupee figure is worse than no explanation."""
    text = (
        "This connected group of accounts shows unusual behaviour worth "
        "about 987654 rupees across the linked orders observed here."
    )
    explanation, log = llm.explain(ring, DETERMINISTIC, provider=Echo(text))
    assert explanation.source == "deterministic"
    assert "not in the evidence" in log


def test_a_clean_response_is_accepted(ring) -> None:
    n = ring["summary"]["n_accounts"]
    text = (
        f"A group of {n} accounts is connected through shared devices and "
        "delivery addresses. The pattern is only visible across the accounts "
        "together rather than in any single order."
    )
    explanation, _ = llm.explain(ring, DETERMINISTIC, provider=Echo(text))
    assert explanation.source == "llm"
    assert explanation.degraded is False


def test_empty_response_falls_back(ring) -> None:
    explanation, _ = llm.explain(ring, DETERMINISTIC, provider=Echo("  "))
    assert explanation.source == "deterministic"


def test_prompt_carries_no_identifiers_and_no_action(ring) -> None:
    """The model cannot endorse a verdict it was never shown."""
    prompt = llm.build_prompt(ring)
    for banned in ("CUST-", "TXN-", "ADDR-", "DEV-", "MANUAL_REVIEW", "VERIFY"):
        assert banned not in prompt, f"prompt leaked {banned}"


# --------------------------------------------------------------------------- #
# Razorpay: test mode only
# --------------------------------------------------------------------------- #


def test_live_credentials_are_rejected() -> None:
    with pytest.raises(razorpay.ProductionCredentialError):
        razorpay.assert_test_mode("rzp_" + "live_abc123")


def test_live_credentials_are_rejected_at_startup(monkeypatch) -> None:
    monkeypatch.setenv("RAZORPAY_KEY_ID", "rzp_" + "live_abc123")
    with pytest.raises(razorpay.ProductionCredentialError):
        with TestClient(app):
            pass


def test_non_razorpay_key_is_rejected() -> None:
    with pytest.raises(razorpay.ProductionCredentialError):
        razorpay.assert_test_mode("sk_test_something")


def test_test_key_and_absent_key_are_both_fine() -> None:
    razorpay.assert_test_mode("rzp_test_abc123")
    razorpay.assert_test_mode("")


def test_paise_are_converted_to_rupees() -> None:
    internal = razorpay.to_internal_payment(
        {"id": "pay_1", "amount": 129900, "status": "captured", "created_at": 1767225600}
    )
    assert internal["amount"] == 1299.0
    assert internal["payment_status"] == "captured"


def test_fields_the_processor_does_not_hold_are_left_absent() -> None:
    """Guessing them would misrepresent where the data comes from."""
    internal = razorpay.to_internal_payment({"id": "pay_1", "amount": 100})
    for field in ("shipping_address_id", "coupon_code", "product_category"):
        assert internal[field] is None


def test_webhook_signature_verification() -> None:
    body, secret = b'{"event":"payment.captured"}', "shhh"
    good = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    assert razorpay.verify_webhook_signature(body, good, secret) is True
    assert razorpay.verify_webhook_signature(body, "deadbeef", secret) is False
    assert razorpay.verify_webhook_signature(body, good, "wrong") is False
    assert razorpay.verify_webhook_signature(body, "", secret) is False


def test_unsupported_webhook_events_are_ignored() -> None:
    assert razorpay.handle_webhook({"event": "subscription.charged"}) is None


def test_supported_webhook_maps_to_an_internal_event() -> None:
    event = {
        "event": "payment.captured",
        "payload": {
            "payment": {
                "entity": {"id": "pay_9", "amount": 50000, "status": "captured"}
            }
        },
    }
    result = razorpay.handle_webhook(event)
    assert result["kind"] == "payment"
    assert result["data"]["amount"] == 500.0
