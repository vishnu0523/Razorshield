"""Generator configuration.

Every parameter that shapes the synthetic dataset lives here, in one auditable
place. A judge should be able to read this file and understand exactly what
assumptions our metrics rest on.

Design rule that governs the whole file: no single structural feature may
separate abuse rings from legitimate lookalike clusters. Ranges below are
deliberately overlapped. If a ring has 8 accounts on 2 devices, so does a
family household. The separation has to come from combinations, which is the
entire thesis of the product.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Events and labels are physically separated directories. The feature builder in
# Phase 3 is only ever pointed at EVENTS_DIR. This is a structural barrier
# against label leakage, not a convention someone has to remember.
DATA_DIR = ROOT / "data"
EVENTS_DIR = DATA_DIR / "events"
LABELS_DIR = DATA_DIR / "labels"

SEED = 42

# --------------------------------------------------------------------------- #
# Scale and time
# --------------------------------------------------------------------------- #

TARGET_TRANSACTIONS = 20_000
SPAN_DAYS = 120
START_DATE = "2026-01-06T00:00:00+00:00"

N_NORMAL_CUSTOMERS = 1_250
N_ABUSE_RINGS = 95
N_HARD_NEGATIVE_CLUSTERS = 112

# Shared entity pools. Normal customers draw IPs from a finite pool so that some
# incidental IP sharing exists in the background population. Without this, "two
# accounts share an IP" would be a perfect fraud signal, which is not true of
# the real world and would make the problem artificially easy.
N_IP_POOL = 1_200
N_COUPON_POOL = 40

CATEGORIES = ["electronics", "fashion", "beauty", "home", "grocery", "toys"]
CATEGORY_WEIGHTS = [0.18, 0.26, 0.14, 0.16, 0.16, 0.10]

# Lognormal amount parameters (mu, sigma) in log-INR, per category.
CATEGORY_AMOUNT = {
    "electronics": (8.6, 0.75),
    "fashion": (7.3, 0.65),
    "beauty": (6.9, 0.55),
    "home": (7.6, 0.70),
    "grocery": (6.6, 0.50),
    "toys": (7.0, 0.60),
}

PAYMENT_METHODS = ["upi", "card", "netbanking", "wallet", "cod"]
PAYMENT_METHOD_WEIGHTS = [0.44, 0.28, 0.10, 0.12, 0.06]

FAILURE_REASONS = [
    "insufficient_funds",
    "card_declined",
    "authentication_failed",
    "issuer_unavailable",
    "incorrect_cvv",
]

RETURN_REASONS = [
    "not_as_described",
    "damaged_on_arrival",
    "wrong_size",
    "changed_mind",
    "late_delivery",
    "missing_parts",
]

# --------------------------------------------------------------------------- #
# Normal population
# --------------------------------------------------------------------------- #

NORMAL_PROFILES = {
    "casual": {
        "weight": 0.58,
        "orders": (1, 7),
        "return_rate": (0.02, 0.10),
        "coupon_rate": (0.05, 0.20),
        "failure_rate": (0.03, 0.09),
        "devices": (1, 2),
    },
    "loyal": {
        "weight": 0.27,
        "orders": (8, 26),
        "return_rate": (0.03, 0.12),
        "coupon_rate": (0.10, 0.30),
        "failure_rate": (0.02, 0.07),
        "devices": (1, 3),
    },
    "bargain_hunter": {
        "weight": 0.15,
        "orders": (5, 18),
        # Heavy coupon use is normal behaviour, not fraud. A model that treats
        # coupon frequency alone as risk will fail on this cohort.
        "return_rate": (0.05, 0.18),
        "coupon_rate": (0.55, 0.90),
        "failure_rate": (0.04, 0.10),
        "devices": (1, 2),
    },
}

# --------------------------------------------------------------------------- #
# Hard negatives: legitimate clusters that structurally resemble abuse rings
#
# These are the reason our ring-level precision means anything. Detecting an
# injected ring is easy. Not accusing a family, a hostel, or a legitimate
# reseller is the actual problem.
# --------------------------------------------------------------------------- #

HARD_NEGATIVE_PROFILES = {
    "family_household": {
        "count": 34,
        "accounts": (3, 6),
        "devices": (1, 2),        # overlaps refund_ring device sharing
        "addresses": (1, 1),
        "ips": (1, 1),
        "orders_per_account": (2, 7),
        "return_rate": (0.03, 0.12),
        "coupon_rate": (0.10, 0.35),
        "failure_rate": (0.02, 0.08),
        "signup_span_days": (60, 700),   # the real discriminator vs promo_farm
        "active_span_days": (7, 25),
        "shared_coupon": False,
    },
    "office_network": {
        "count": 18,
        "accounts": (18, 40),     # exceeds most abuse rings on size alone
        "devices": (14, 34),
        "addresses": (10, 26),
        "ips": (1, 2),            # accounts-per-IP far worse than any ring
        "orders_per_account": (2, 4),
        "return_rate": (0.02, 0.10),
        "coupon_rate": (0.08, 0.30),
        "failure_rate": (0.02, 0.08),
        "signup_span_days": (120, 900),
        "active_span_days": (7, 25),
        "shared_coupon": False,
    },
    "hostel_cluster": {
        "count": 24,
        "accounts": (8, 20),      # squarely in abuse-ring size range
        "devices": (5, 14),
        "addresses": (1, 2),      # accounts-per-address like a return ring
        "ips": (1, 2),
        "orders_per_account": (2, 6),
        "return_rate": (0.04, 0.16),
        "coupon_rate": (0.50, 0.85),   # students hunt discounts
        "failure_rate": (0.05, 0.14),
        "signup_span_days": (14, 90),
        "active_span_days": (8, 28),
        "shared_coupon": True,    # same student-discount code, legitimately
    },
    "power_reseller": {
        "count": 18,
        "accounts": (1, 3),
        "devices": (1, 2),
        "addresses": (1, 1),
        "ips": (1, 1),
        "orders_per_account": (8, 22),
        "return_rate": (0.28, 0.48),   # overlaps return_fraud_ring directly
        "coupon_rate": (0.20, 0.50),
        "failure_rate": (0.03, 0.10),
        "signup_span_days": (0, 30),
        "active_span_days": (10, 30),
        "shared_coupon": False,
    },
    "flash_sale_cohort": {
        "count": 18,
        "accounts": (30, 60),
        "devices": (30, 65),
        "addresses": (28, 60),
        "ips": (25, 55),
        "orders_per_account": (1, 2),
        "return_rate": (0.04, 0.14),
        "coupon_rate": (0.85, 1.00),   # everyone uses the sale code
        "failure_rate": (0.08, 0.18),  # gateway strain, looks like card testing
        "signup_span_days": (0, 2),    # burst signup, like a promo farm
        "active_span_days": (0, 1),    # 1-day velocity spike
        "shared_coupon": True,
    },
}

# --------------------------------------------------------------------------- #
# Abuse rings
#
# Five distinct signatures so the detector cannot latch onto one template.
# Ring members also place ordinary orders, which are labelled 0. This makes the
# row-level problem realistically hard rather than trivially separable.
# --------------------------------------------------------------------------- #

ABUSE_PROFILES = {
    "refund_ring": {
        "weight": 0.26,
        "accounts": (5, 20),
        "devices": (2, 5),
        "addresses": (2, 4),
        "ips": (1, 3),
        "orders_per_account": (2, 4),
        "return_rate": (0.10, 0.25),
        "refund_rate": (0.45, 0.75),   # direct refunds without returns
        "coupon_rate": (0.20, 0.55),
        "failure_rate": (0.05, 0.14),
        "signup_span_days": (1, 21),
        "active_span_days": (3, 16),
        "shared_coupon": False,
        "abuse_share": (0.25, 0.52),
    },
    "coupon_ring": {
        "weight": 0.22,
        "accounts": (8, 25),
        "devices": (2, 6),
        "addresses": (2, 6),
        "ips": (1, 4),
        "orders_per_account": (2, 4),
        "return_rate": (0.06, 0.18),
        "refund_rate": (0.15, 0.40),
        "coupon_rate": (0.80, 1.00),
        "failure_rate": (0.04, 0.12),
        "signup_span_days": (1, 14),
        "active_span_days": (2, 12),
        "shared_coupon": True,
        "abuse_share": (0.28, 0.57),
    },
    "return_fraud_ring": {
        "weight": 0.20,
        "accounts": (4, 14),
        "devices": (1, 4),
        "addresses": (1, 3),
        "ips": (1, 3),
        "orders_per_account": (2, 5),
        "return_rate": (0.40, 0.70),   # overlaps power_reseller
        "refund_rate": (0.30, 0.55),
        "coupon_rate": (0.15, 0.45),
        "failure_rate": (0.04, 0.11),
        "signup_span_days": (2, 30),
        "active_span_days": (5, 20),
        "shared_coupon": False,
        "abuse_share": (0.25, 0.52),
    },
    "card_testing": {
        "weight": 0.14,
        "accounts": (3, 12),
        "devices": (1, 3),
        "addresses": (1, 2),
        "ips": (1, 2),
        "orders_per_account": (4, 10),
        "return_rate": (0.00, 0.04),
        "refund_rate": (0.02, 0.12),
        "coupon_rate": (0.00, 0.10),
        "failure_rate": (0.55, 0.85),  # overlaps flash_sale gateway strain
        "signup_span_days": (0, 4),
        "active_span_days": (0, 3),
        "shared_coupon": False,
        "abuse_share": (0.38, 0.67),
    },
    "promo_farm": {
        "weight": 0.18,
        "accounts": (10, 26),
        "devices": (3, 9),
        "addresses": (3, 9),
        "ips": (2, 6),
        "orders_per_account": (1, 3),
        "return_rate": (0.05, 0.20),
        "refund_rate": (0.25, 0.50),
        "coupon_rate": (0.75, 1.00),
        "failure_rate": (0.06, 0.16),
        "signup_span_days": (0, 5),    # burst, overlaps flash_sale_cohort
        "active_span_days": (1, 8),
        "shared_coupon": True,
        "abuse_share": (0.33, 0.62),
    },
}

# --------------------------------------------------------------------------- #
# Leakage guard
# --------------------------------------------------------------------------- #

# Columns the feature builder must never see. Enforced by tests in Phase 3.
FORBIDDEN_FEATURE_COLUMNS = {
    "is_fraud",
    "cluster_id",
    "cluster_kind",
    "is_abusive",
    "profile",
    "abuse_share",
}
