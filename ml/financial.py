"""Financial impact.

Accuracy is not the business metric. This module converts a confusion matrix
and the money attached to it into the number a merchant actually cares about.

Every assumption is named, defaulted, and adjustable. None is buried in a
constant. The dashboard exposes all three as sliders precisely so a judge can
disagree with our defaults and watch the answer change, rather than having to
take our arithmetic on faith.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------- #
# Assumptions
# --------------------------------------------------------------------------- #

# Not all detected exposure was genuinely preventable. Some fraud would have
# failed at the gateway anyway; some flagged activity would have resolved
# legitimately even without intervention. Claiming 100% would be dishonest.
DEFAULT_RECOVERY_RATE = 0.70

# Analyst time plus checkout friction when a legitimate transaction is escalated
# to human review. Flat per incident.
DEFAULT_COST_PER_FALSE_REVIEW = 120.0

# Contribution margin forfeited when a legitimate order is stopped and the
# customer abandons. Applied to the order value.
DEFAULT_COST_PER_FALSE_BLOCK_RATIO = 0.18


@dataclass(frozen=True)
class Assumptions:
    recovery_rate: float = DEFAULT_RECOVERY_RATE
    cost_per_false_review: float = DEFAULT_COST_PER_FALSE_REVIEW
    cost_per_false_block_ratio: float = DEFAULT_COST_PER_FALSE_BLOCK_RATIO


@dataclass(frozen=True)
class Impact:
    exposure_detected: float
    prevented_loss: float
    false_positive_cost: float
    net_protected_value: float
    missed_exposure: float
    n_true_positives: int
    n_false_positives: int
    n_false_negatives: int


def compute_impact(
    y_true: np.ndarray,
    flagged: np.ndarray,
    amounts: np.ndarray,
    assumptions: Assumptions | None = None,
) -> Impact:
    """Money terms for one set of decisions on one split.

    exposure_detected      value of fraudulent transactions correctly flagged
    prevented_loss         exposure_detected * recovery_rate
    false_positive_cost    per false alarm: review cost + forfeited margin
    net_protected_value    prevented_loss - false_positive_cost, may be negative
    missed_exposure        value of fraud that slipped through, reported for honesty
    """
    a = assumptions or Assumptions()
    y_true = np.asarray(y_true).astype(bool)
    flagged = np.asarray(flagged).astype(bool)
    amounts = np.asarray(amounts, dtype=float)

    tp = y_true & flagged
    fp = ~y_true & flagged
    fn = y_true & ~flagged

    exposure = float(amounts[tp].sum())
    prevented = exposure * a.recovery_rate
    fp_cost = float(
        fp.sum() * a.cost_per_false_review
        + amounts[fp].sum() * a.cost_per_false_block_ratio
    )

    return Impact(
        exposure_detected=round(exposure, 2),
        prevented_loss=round(prevented, 2),
        false_positive_cost=round(fp_cost, 2),
        net_protected_value=round(prevented - fp_cost, 2),
        missed_exposure=round(float(amounts[fn].sum()), 2),
        n_true_positives=int(tp.sum()),
        n_false_positives=int(fp.sum()),
        n_false_negatives=int(fn.sum()),
    )


def assumptions_payload(a: Assumptions | None = None) -> list[dict]:
    """Contract-shaped assumption list for /api/metrics and the dashboard."""
    a = a or Assumptions()
    return [
        {
            "key": "recovery_rate",
            "label": "Recovery rate",
            "value": a.recovery_rate,
            "unit": "ratio",
            "description": (
                "Share of detected exposure genuinely preventable. Some fraud "
                "would have failed anyway; some flagged activity would have "
                "resolved legitimately. We do not claim 100%."
            ),
            "adjustable": True,
            "min": 0.0,
            "max": 1.0,
        },
        {
            "key": "cost_per_false_review",
            "label": "Cost per false manual review",
            "value": a.cost_per_false_review,
            "unit": "INR",
            "description": (
                "Analyst time plus checkout friction when a legitimate "
                "transaction is escalated to human review."
            ),
            "adjustable": True,
            "min": 0.0,
            "max": 2000.0,
        },
        {
            "key": "cost_per_false_block_ratio",
            "label": "Lost-margin ratio on a false block",
            "value": a.cost_per_false_block_ratio,
            "unit": "ratio",
            "description": (
                "Contribution margin forfeited when a legitimate order is "
                "stopped and the customer abandons. Applied to order value."
            ),
            "adjustable": True,
            "min": 0.0,
            "max": 1.0,
        },
    ]
