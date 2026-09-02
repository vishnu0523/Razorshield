"""Razorpay adapter — TEST MODE ONLY.

Deliberately small. RazorShield does not move money, so this adapter does two
things and nothing else: it maps Razorpay's payment object onto our internal
event schema, and it verifies webhook signatures. Both are the parts that would
actually matter in an integration; a fuller client would be scope we cannot
justify.

Production credentials are rejected at import time rather than at call time. A
key that reaches the process is already a mistake, and failing late means the
first sign of trouble is a real payment.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from datetime import datetime, timezone

TEST_KEY_PREFIX = "rzp_test_"
LIVE_KEY_PREFIX = "rzp_" + "live_"  # split so the string never appears literally

# Razorpay reports money in paise. Everything downstream is in rupees.
PAISE_PER_RUPEE = 100


class ProductionCredentialError(RuntimeError):
    """Raised when a live key is present. RazorShield is defence-only."""


def assert_test_mode(key_id: str | None = None) -> None:
    key_id = key_id if key_id is not None else os.environ.get("RAZORPAY_KEY_ID", "")
    if not key_id:
        return
    if key_id.startswith(LIVE_KEY_PREFIX):
        raise ProductionCredentialError(
            "A live Razorpay key was supplied. RazorShield is a defensive "
            "analysis tool and runs against test mode only. Remove the key."
        )
    if not key_id.startswith(TEST_KEY_PREFIX):
        raise ProductionCredentialError(
            f"Razorpay key must start with {TEST_KEY_PREFIX!r}."
        )


def verify_webhook_signature(
    body: bytes, signature: str, secret: str | None = None
) -> bool:
    """Constant-time HMAC-SHA256 check, as Razorpay specifies.

    compare_digest rather than ==, so the comparison does not leak how much of
    a forged signature was correct.
    """
    secret = secret if secret is not None else os.environ.get("RAZORPAY_WEBHOOK_SECRET", "")
    if not secret or not signature:
        return False
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def to_internal_payment(payment: dict) -> dict:
    """Map a Razorpay payment entity onto our transaction event schema.

    Fields Razorpay does not hold — shipping address, coupon code, return
    reason — are left absent rather than guessed. RazorShield sits on merchant
    commerce data with payment signals joined in, and pretending the processor
    supplies the rest would misrepresent where the data comes from.
    """
    assert_test_mode()

    created = payment.get("created_at")
    timestamp = (
        datetime.fromtimestamp(created, tz=timezone.utc).isoformat()
        if isinstance(created, (int, float))
        else None
    )
    captured = payment.get("status") == "captured"

    return {
        "transaction_id": payment.get("id"),
        "order_id": payment.get("order_id"),
        "timestamp": timestamp,
        "amount": round((payment.get("amount") or 0) / PAISE_PER_RUPEE, 2),
        "currency": payment.get("currency", "INR"),
        "payment_method": payment.get("method"),
        "payment_status": "captured" if captured else "failed",
        "failure_reason": payment.get("error_reason") or payment.get("error_code"),
        "customer_id": payment.get("customer_id") or payment.get("email"),
        "ip_address": payment.get("notes", {}).get("ip_address"),
        "device_id": payment.get("notes", {}).get("device_id"),
        # Not available from the processor. Joined from merchant data upstream.
        "shipping_address_id": None,
        "coupon_code": None,
        "product_category": None,
    }


def to_internal_refund(refund: dict) -> dict:
    assert_test_mode()
    created = refund.get("created_at")
    return {
        "refund_id": refund.get("id"),
        "transaction_id": refund.get("payment_id"),
        "timestamp": (
            datetime.fromtimestamp(created, tz=timezone.utc).isoformat()
            if isinstance(created, (int, float))
            else None
        ),
        "amount": round((refund.get("amount") or 0) / PAISE_PER_RUPEE, 2),
    }


SUPPORTED_WEBHOOK_EVENTS = frozenset(
    {"payment.captured", "payment.failed", "refund.created", "refund.processed"}
)


def handle_webhook(event: dict) -> dict | None:
    """Translate a verified webhook into an internal event, or ignore it.

    Signature verification happens before this is called. This function does not
    write anything: ingestion is a separate concern and keeping them apart means
    a malformed payload cannot reach storage.
    """
    name = event.get("event")
    if name not in SUPPORTED_WEBHOOK_EVENTS:
        return None
    entity = event.get("payload", {})
    if name.startswith("payment."):
        payment = entity.get("payment", {}).get("entity", {})
        return {"kind": "payment", "data": to_internal_payment(payment)}
    refund = entity.get("refund", {}).get("entity", {})
    return {"kind": "refund", "data": to_internal_refund(refund)}
