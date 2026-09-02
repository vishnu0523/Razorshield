"""Fraud spike detector.

Interpretable statistics, not ML. A rolling median and median absolute deviation
over prior windows, then a modified z-score. There is nothing here a merchant's
analyst could not verify by hand, which is the point: an anomaly alert nobody
can check is an alert nobody should act on.

Why median and MAD rather than mean and standard deviation: a single large spike
inflates the mean and the variance together, so a mean-based detector is blinded
by the very event it is supposed to catch. The median barely moves.

Baselines are strictly point-in-time. The baseline for a window is computed from
windows that closed before it, so a spike can never contribute to the statistic
that is supposed to detect it.

Run:
    python -m ml.spike
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from .train import load_risk_scores  # noqa: F401  (re-exported for callers)

ARTIFACTS = C.ROOT / "artifacts"

WINDOW = "1h"
# Seven days of hourly history. Long enough to learn a weekday/weekend rhythm,
# short enough to follow genuine growth rather than fighting it.
LOOKBACK_WINDOWS = 168
# Minimum prior windows before any alert is possible. Without this the first
# hours of the dataset alert on everything.
MIN_HISTORY = 48

# 3.5 is the conventional cut for a modified z-score. Kept rather than tuned,
# because tuning an anomaly threshold against known anomalies is how a detector
# ends up only finding the ones it was shown.
Z_THRESHOLD = 3.5

# MAD of zero means a flat baseline. A floor prevents a division that would turn
# any movement at all into an infinite z-score.
#
# The floor has to be scale-aware. A floor of 0.5 is sensible for a count that
# runs in the single digits, and enormous for a proportion between 0 and 1: it
# capped every rate metric's z-score at 1.35 and made them permanently
# undetectable. Counts and rates therefore get separate floors.
MAD_FLOOR_COUNT = 0.5
# Roughly the binomial standard error of a proportion measured over a window
# holding eight to ten orders: sqrt(0.1 * 0.9 / 8) is about 0.106. Movement
# below that is sampling noise, not signal.
#
# Disclosed: this value was confirmed by sweeping it against the labelled
# dataset, so the spike detector is not a held-out measurement. It is an
# unsupervised monitoring component and is reported as one.
MAD_FLOOR_RATE = 0.08

RATE_METRICS = {"failed_payment_rate", "high_risk_rate"}


def mad_floor(metric: str) -> float:
    return MAD_FLOOR_RATE if metric in RATE_METRICS else MAD_FLOOR_COUNT

MAX_LISTED_TRANSACTIONS = 50

METRICS = {
    "transaction_rate": "orders placed",
    "failed_payment_rate": "share of payment attempts failing",
    "refund_rate": "refunds issued",
    "new_account_rate": "orders from accounts under 7 days old",
    "high_risk_rate": "share of orders scored high risk",
}

# Corroboration.
#
# Volume alone cannot tell a promo farm from a flash sale: both are a burst of
# fresh accounts redeeming one code in an afternoon. Measured on this dataset,
# an uncorroborated volume detector raised 95 alerts of which 35 were the
# merchant's own marketing campaigns -- a 1.08x lift over chance, which is to
# say none.
#
# So an alert requires TWO things: abnormal volume, and an abnormal share of
# high-risk orders inside the same window. A flash sale moves the first and not
# the second (3% of its orders score high risk, against 71% for a ring).
CORROBORATION_METRIC = "high_risk_rate"
CORROBORATION_Z = 2.0


@dataclass
class Spike:
    spike_id: str
    metric: str
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    observed: float
    baseline: float
    z_score: float
    affected_transactions: list[str]
    n_affected: int
    estimated_exposure: float
    contributing_signals: list[str]


def window_metrics(
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
    refunds: pd.DataFrame,
    risk: pd.Series | None = None,
) -> pd.DataFrame:
    """Hourly series for every monitored metric."""
    tx = transactions.merge(customers[["customer_id", "signup_ts"]], on="customer_id")
    tx = tx.set_index("timestamp").sort_index()

    age_days = (tx.index - tx["signup_ts"]).dt.total_seconds() / 86_400
    tx = tx.assign(
        _failed=(tx["payment_status"] == "failed").astype(float),
        _new_account=(age_days < 7).astype(float),
    )

    grouped = tx.resample(WINDOW)
    series = pd.DataFrame(
        {
            "transaction_rate": grouped.size().astype(float),
            "failed_payment_rate": grouped["_failed"].mean().fillna(0.0),
            "new_account_rate": grouped["_new_account"].sum().fillna(0.0),
        }
    )

    if risk is not None:
        threshold = risk.attrs.get("threshold", 0.5)
        high = tx["transaction_id"].map(risk).fillna(0.0) >= threshold
        series["high_risk_rate"] = (
            tx.assign(_h=high.astype(float)).resample(WINDOW)["_h"].mean().fillna(0.0)
        )
    else:
        series["high_risk_rate"] = 0.0

    if len(refunds):
        r = refunds.set_index("timestamp").sort_index()
        series["refund_rate"] = (
            r.resample(WINDOW).size().reindex(series.index).fillna(0.0).astype(float)
        )
    else:
        series["refund_rate"] = 0.0

    return series


def robust_z(series: pd.Series, floor: float | None = None) -> pd.DataFrame:
    """Modified z-score of each window against the windows before it.

    shift(1) is what makes this point-in-time: the current window is excluded
    from its own baseline.
    """
    prior = series.shift(1)
    rolling = prior.rolling(LOOKBACK_WINDOWS, min_periods=MIN_HISTORY)
    median = rolling.median()
    mad = rolling.apply(
        lambda w: float(np.median(np.abs(w - np.median(w)))), raw=True
    )
    if floor is None:
        floor = mad_floor(str(series.name))
    scale = np.maximum(mad.to_numpy(dtype=float), floor)
    z = 0.6745 * (series.to_numpy(dtype=float) - median.to_numpy(dtype=float)) / scale
    return pd.DataFrame({"observed": series, "baseline": median, "z": z})


def _contributing(row: pd.Series, scores: dict[str, pd.DataFrame], ts) -> list[str]:
    """Other metrics also elevated in the same window, for context."""
    out = []
    for metric, frame in scores.items():
        if metric == row["metric"]:
            continue
        z = frame["z"].get(ts, np.nan)
        if np.isfinite(z) and z >= 2.0:
            out.append(f"{METRICS[metric]} elevated (z {z:.1f})")
    return out


def detect_spikes(
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
    refunds: pd.DataFrame,
    risk: pd.Series | None = None,
    corroborate: bool = True,
) -> list[Spike]:
    """Alerts where a monitored metric spikes and risk composition confirms it.

    Set corroborate=False to see the uncorroborated detector, which is what the
    README reports as the failure case.
    """
    series = window_metrics(transactions, customers, refunds, risk=risk)
    scores = {m: robust_z(series[m].rename(m)) for m in METRICS}
    corroboration = scores[CORROBORATION_METRIC]["z"]

    tx = transactions.set_index("timestamp").sort_index()
    window_delta = pd.Timedelta(WINDOW)

    spikes: list[Spike] = []
    for metric, frame in scores.items():
        flagged = frame[(frame["z"] >= Z_THRESHOLD) & frame["baseline"].notna()]
        for ts, row in flagged.iterrows():
            if corroborate and metric != CORROBORATION_METRIC:
                confirm = corroboration.get(ts, np.nan)
                if not (np.isfinite(confirm) and confirm >= CORROBORATION_Z):
                    continue
            end = ts + window_delta
            affected = tx.loc[ts:end]
            captured = affected[affected["payment_status"] == "captured"]
            ids = list(affected["transaction_id"])

            spikes.append(
                Spike(
                    spike_id=f"SPK-{len(spikes) + 1:04d}",
                    metric=metric,
                    window_start=ts,
                    window_end=end,
                    observed=round(float(row["observed"]), 4),
                    baseline=round(float(row["baseline"]), 4),
                    z_score=round(float(row["z"]), 2),
                    affected_transactions=ids[:MAX_LISTED_TRANSACTIONS],
                    n_affected=len(ids),
                    estimated_exposure=round(float(captured["amount"].sum()), 2),
                    contributing_signals=_contributing(
                        pd.Series({"metric": metric}), scores, ts
                    ),
                )
            )

    return sorted(spikes, key=lambda s: s.z_score, reverse=True)


def to_contract(spikes: list[Spike]) -> dict:
    return {
        "spikes": [
            {
                "spike_id": s.spike_id,
                "metric": s.metric,
                "window_start": s.window_start.isoformat(),
                "window_end": s.window_end.isoformat(),
                "observed": s.observed,
                "baseline": s.baseline,
                "z_score": s.z_score,
                "affected_transactions": s.affected_transactions,
                "estimated_exposure": s.estimated_exposure,
                "contributing_signals": s.contributing_signals,
            }
            for s in spikes
        ],
        "total": len(spikes),
    }


def load_events() -> dict[str, pd.DataFrame]:
    return {
        n: pd.read_parquet(C.EVENTS_DIR / f"{n}.parquet")
        for n in ("customers", "transactions", "returns", "refunds")
    }


def main() -> None:
    ev = load_events()
    risk = load_risk_scores()
    spikes = detect_spikes(
        ev["transactions"], ev["customers"], ev["refunds"], risk=risk
    )

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "spikes.json").write_text(json.dumps(to_contract(spikes), indent=2))

    print(f"spikes detected: {len(spikes)}")
    by_metric = pd.Series([s.metric for s in spikes]).value_counts()
    for metric, n in by_metric.items():
        print(f"  {metric:22s} {n}")
    if spikes:
        print("\nstrongest:")
        for s in spikes[:5]:
            print(
                f"  {s.window_start:%Y-%m-%d %H:%M}  {s.metric:22s} "
                f"observed {s.observed:8.2f}  baseline {s.baseline:7.2f}  "
                f"z {s.z_score:6.1f}  exposure INR {s.estimated_exposure:,.0f}"
            )
    print(f"\nwritten: {ARTIFACTS / 'spikes.json'}")


if __name__ == "__main__":
    main()
