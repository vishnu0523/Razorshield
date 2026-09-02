"""Financial impact, recomputable under merchant-chosen assumptions.

The arithmetic lives in `ml/financial.py` and is imported rather than copied:
the dashboard and the evaluation report must not be able to disagree about what
net protected value means.

What this module adds is recomputation. The held-out decisions are stored as
they were made, so changing an assumption re-runs the same sum over the same
decisions rather than scaling a summary figure. A judge who thinks our recovery
rate is optimistic can say so and watch the real number move.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.financial import Assumptions, assumptions_payload, compute_impact  # noqa: E402

ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", ROOT / "artifacts"))
DECISIONS_PATH = ARTIFACTS / "decisions.json"


def decisions_available() -> bool:
    return DECISIONS_PATH.is_file()


def recompute(assumptions: Assumptions) -> dict:
    """Re-run the impact sum over the stored held-out decisions."""
    raw = json.loads(DECISIONS_PATH.read_text())
    impact = compute_impact(
        np.array(raw["is_fraud"], dtype=int),
        np.array(raw["flagged"], dtype=bool),
        np.array(raw["amount"], dtype=float),
        assumptions,
    )
    return {
        "currency": "INR",
        "exposure_detected": impact.exposure_detected,
        "prevented_loss": impact.prevented_loss,
        "false_positive_cost": impact.false_positive_cost,
        "net_protected_value": impact.net_protected_value,
        "assumptions": assumptions_payload(assumptions),
    }
