"""Phase 8 gates: is the anomaly detector actionable, or does it page on marketing?

test_no_alert_fires_on_a_legitimate_flash_sale is the one that matters. A volume
detector that cannot tell a promo farm from a campaign will be muted by the
merchant within a week, and a muted detector protects nothing.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ml import config as C
from ml.spike import (
    ARTIFACTS,
    CORROBORATION_METRIC,
    MAD_FLOOR_COUNT,
    MAD_FLOOR_RATE,
    RATE_METRICS,
    Z_THRESHOLD,
    mad_floor,
    robust_z,
)

pytestmark = pytest.mark.skipif(
    not (ARTIFACTS / "spikes.json").exists(), reason="run `make spikes` first"
)


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads((C.LABELS_DIR / "spike_report.json").read_text())


@pytest.fixture(scope="module")
def spikes() -> dict:
    return json.loads((ARTIFACTS / "spikes.json").read_text())


# --------------------------------------------------------------------------- #
# Point-in-time baselines
# --------------------------------------------------------------------------- #


def test_a_spike_does_not_contaminate_its_own_baseline() -> None:
    """The window being judged is excluded from the statistic judging it."""
    series = pd.Series(
        np.r_[np.ones(200), [500.0]],
        name="transaction_rate",
        index=pd.date_range("2026-01-01", periods=201, freq="h"),
    )
    z = robust_z(series)
    assert z["baseline"].iloc[-1] == 1.0
    assert z["z"].iloc[-1] > Z_THRESHOLD


def test_median_baseline_survives_a_prior_spike() -> None:
    """Why median and MAD rather than mean and standard deviation.

    One large earlier spike inflates a mean-based baseline and its variance
    together, blinding the detector to the next one. The median barely moves.
    """
    values = np.ones(200)
    values[50] = 5000.0
    series = pd.Series(
        np.r_[values, [40.0]],
        name="transaction_rate",
        index=pd.date_range("2026-01-01", periods=201, freq="h"),
    )
    z = robust_z(series)
    assert z["baseline"].iloc[-1] == 1.0, "one outlier moved the median"
    assert z["z"].iloc[-1] > Z_THRESHOLD, "an outlier blinded the detector"


def test_flat_history_does_not_produce_infinite_scores() -> None:
    series = pd.Series(
        np.r_[np.ones(200), [2.0]],
        name="transaction_rate",
        index=pd.date_range("2026-01-01", periods=201, freq="h"),
    )
    assert np.isfinite(robust_z(series)["z"].iloc[-1])


def test_no_alert_before_enough_history() -> None:
    series = pd.Series(
        np.r_[np.ones(10), [500.0]],
        name="transaction_rate",
        index=pd.date_range("2026-01-01", periods=11, freq="h"),
    )
    assert pd.isna(robust_z(series)["baseline"].iloc[-1])


# --------------------------------------------------------------------------- #
# Scale-aware floors
# --------------------------------------------------------------------------- #


def test_rate_metrics_get_a_smaller_floor_than_counts() -> None:
    """A floor sized for a count is enormous relative to a proportion.

    With a single shared floor of 0.5, every rate metric's z-score was capped
    around 1.35 and could never cross the threshold. Regression test.
    """
    assert MAD_FLOOR_RATE < MAD_FLOOR_COUNT
    for metric in RATE_METRICS:
        assert mad_floor(metric) == MAD_FLOOR_RATE
    assert mad_floor("transaction_rate") == MAD_FLOOR_COUNT


def test_a_rate_metric_can_actually_cross_the_threshold() -> None:
    series = pd.Series(
        np.r_[np.full(200, 0.05), [0.95]],
        name="high_risk_rate",
        index=pd.date_range("2026-01-01", periods=201, freq="h"),
    )
    assert robust_z(series)["z"].iloc[-1] >= Z_THRESHOLD


# --------------------------------------------------------------------------- #
# Does it page on marketing?
# --------------------------------------------------------------------------- #


def test_no_alert_fires_on_a_legitimate_flash_sale(report) -> None:
    """A flash sale is structurally a promo farm and commercially a success.

    Confusing them would have the merchant muting the detector during exactly
    the periods that matter most to them.
    """
    assert report["driven_by_flash_sale"] == 0, (
        f"{report['driven_by_flash_sale']} alerts fired on the merchant's own "
        "campaigns"
    )


def test_alerts_are_meaningfully_better_than_chance(report) -> None:
    """98% of the timeline sits inside some ring's window, so overlap proves
    nothing. Attribution against the base rate is the only honest measure."""
    assert report["lift_over_chance"] >= 2.0, (
        f"lift {report['lift_over_chance']}x is close to random alerting"
    )
    assert report["mean_ring_share_in_alerts"] > report["abuse_base_rate"] * 2


def test_alert_volume_is_actionable(report) -> None:
    """Roughly one alert per two days. An analyst can work that queue."""
    assert report["n_spikes"] <= 200, "too noisy for anyone to act on"
    assert report["n_spikes"] >= 10, "so quiet it may be detecting nothing"


def test_corroboration_metric_is_monitored() -> None:
    assert CORROBORATION_METRIC in RATE_METRICS


# --------------------------------------------------------------------------- #
# Contract shape
# --------------------------------------------------------------------------- #


def test_every_spike_matches_the_contract(spikes) -> None:
    required = {
        "spike_id", "metric", "window_start", "window_end", "observed",
        "baseline", "z_score", "affected_transactions", "estimated_exposure",
        "contributing_signals",
    }
    assert spikes["total"] == len(spikes["spikes"])
    for s in spikes["spikes"]:
        assert set(s) == required
        assert s["z_score"] >= Z_THRESHOLD
        assert s["estimated_exposure"] >= 0
        assert s["window_end"] > s["window_start"]


def test_spikes_are_ordered_by_severity(spikes) -> None:
    scores = [s["z_score"] for s in spikes["spikes"]]
    assert scores == sorted(scores, reverse=True)
