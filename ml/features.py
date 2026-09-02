"""Point-in-time feature builder.

Every feature for a transaction at time t is computed from events strictly
before t. Nothing here may see the future, and nothing here opens data/labels/.

Two subtleties that are easy to get wrong and that quietly inflate every
downstream metric:

1.  A return or refund attached to an old order happens *later* than that order.
    Prior-return features must be keyed on the return's own timestamp, not the
    transaction's. Counting a refund that has not happened yet is leakage even
    though the underlying order is in the past.

2.  Filling a missing prior statistic with a global mean leaks the whole dataset
    into row one. Every fill value here is a fixed constant.

The claim is enforced, not asserted: tests/test_features.py recomputes features
for sampled rows against a truncated event log and requires an exact match.

Run:
    python -m ml.features
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

EVENT_FRAMES = ("customers", "transactions", "returns", "refunds")

# Fill constants. Never a statistic derived from the data.
FILL_RATIO = 0.0
FILL_NEUTRAL = 1.0
FILL_LONG_GAP_SECONDS = 30 * 86_400


def load_events() -> dict[str, pd.DataFrame]:
    """Read the event log. This function must never point at LABELS_DIR."""
    return {
        name: pd.read_parquet(C.EVENTS_DIR / f"{name}.parquet")
        for name in EVENT_FRAMES
    }


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #


def _epoch(series: pd.Series) -> np.ndarray:
    return series.to_numpy(dtype="datetime64[s]").astype(np.int64)


def _window_count(
    df: pd.DataFrame, key: str, seconds: int, ts: np.ndarray
) -> np.ndarray:
    """Count of rows sharing `key` within the trailing window, inclusive of self.

    df must already be sorted by timestamp ascending, so each group's slice is
    also ascending and searchsorted is valid.
    """
    out = np.zeros(len(df), dtype=float)
    for idx in df.groupby(key, sort=False).indices.values():
        t = ts[idx]
        start = np.searchsorted(t, t - seconds, side="left")
        out[idx] = np.arange(len(t)) - start + 1
    return out


def _distinct_so_far(df: pd.DataFrame, entity: str, other: str) -> np.ndarray:
    """Distinct `other` values seen on `entity` up to and including this row."""
    first_seen = (~df.duplicated(subset=[entity, other])).astype(int)
    return first_seen.groupby(df[entity], sort=False).cumsum().to_numpy(dtype=float)


def canonical_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """The canonical stream order. Anything reasoning about position uses this.

    Transactions can share a timestamp, and an unstable sort would order those
    ties differently depending on how the caller happened to hand over the
    frame -- silently changing every cumulative feature. Breaking ties on
    transaction_id makes stream position deterministic.
    """
    return (
        transactions.sort_values(["timestamp", "transaction_id"], kind="mergesort")
        .reset_index(drop=True)
    )


def _canonical(events: pd.DataFrame) -> pd.DataFrame:
    """Stable, total ordering for an event frame, so ties never float."""
    tiebreak = next(
        (c for c in ("return_id", "refund_id", "transaction_id") if c in events),
        None,
    )
    by = ["timestamp"] + ([tiebreak] if tiebreak else [])
    return events.sort_values(by, kind="mergesort")


def _prior_count(
    txns: pd.DataFrame, events: pd.DataFrame, key: str
) -> np.ndarray:
    """Number of `events` for the same key that occurred strictly before the row.

    merge_asof with allow_exact_matches=False gives the running count as of the
    instant before t, which is exactly the point-in-time semantics we want.
    """
    if events.empty:
        return np.zeros(len(txns), dtype=float)

    ev = _canonical(events)[[key, "timestamp"]].copy()
    ev["_running"] = ev.groupby(key, sort=False).cumcount() + 1

    merged = pd.merge_asof(
        txns[["timestamp", key]].reset_index(drop=True),
        ev,
        on="timestamp",
        by=key,
        direction="backward",
        allow_exact_matches=False,
    )
    return merged["_running"].fillna(0).to_numpy(dtype=float)


def _prior_sum(
    txns: pd.DataFrame, events: pd.DataFrame, key: str, value_col: str
) -> np.ndarray:
    """Running sum of `value_col` for the key, strictly before the row."""
    if events.empty:
        return np.zeros(len(txns), dtype=float)

    ev = _canonical(events)[[key, "timestamp", value_col]].copy()
    ev["_running"] = ev.groupby(key, sort=False)[value_col].cumsum()

    merged = pd.merge_asof(
        txns[["timestamp", key]].reset_index(drop=True),
        ev[[key, "timestamp", "_running"]],
        on="timestamp",
        by=key,
        direction="backward",
        allow_exact_matches=False,
    )
    return merged["_running"].fillna(0).to_numpy(dtype=float)


def _prior_mean_std(
    df: pd.DataFrame, key: str, value_col: str
) -> tuple[np.ndarray, np.ndarray]:
    """Expanding mean and std over the same key, excluding the current row."""
    v = df[value_col].to_numpy(dtype=float)
    g = df.groupby(key, sort=False)[value_col]
    n = g.cumcount().to_numpy(dtype=float)  # count of prior rows
    csum = g.cumsum().to_numpy(dtype=float) - v
    csq = (
        df.assign(_sq=v**2).groupby(key, sort=False)["_sq"].cumsum().to_numpy(dtype=float)
        - v**2
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, csum / np.maximum(n, 1), np.nan)
        var = np.where(n > 1, (csq / np.maximum(n, 1)) - mean**2, np.nan)
    return mean, np.sqrt(np.clip(var, 0, None))


# --------------------------------------------------------------------------- #
# Feature construction
# --------------------------------------------------------------------------- #


def build_features(events: dict[str, pd.DataFrame]) -> pd.DataFrame:
    tx = canonical_transactions(events["transactions"])
    customers = events["customers"]
    returns = events["returns"]
    refunds = events["refunds"]

    # Attach the owning customer to return and refund events so prior-behaviour
    # features can be keyed on the customer. The join brings customer_id across;
    # the timestamp used is always the return's or refund's own.
    txn_owner = tx[["transaction_id", "customer_id", "amount"]]
    returns = returns.merge(txn_owner, on="transaction_id", how="left")
    refunds = refunds.merge(
        txn_owner.rename(columns={"amount": "order_amount"}),
        on="transaction_id",
        how="left",
    )

    tx = tx.merge(customers[["customer_id", "signup_ts"]], on="customer_id", how="left")
    ts = _epoch(tx["timestamp"])
    n = len(tx)

    f = pd.DataFrame(
        {"transaction_id": tx["transaction_id"], "timestamp": tx["timestamp"]}
    )

    # -- amount ------------------------------------------------------------- #
    amount = tx["amount"].to_numpy(dtype=float)
    f["amount"] = amount
    f["amount_log"] = np.log1p(amount)

    cust_mean, cust_std = _prior_mean_std(tx, "customer_id", "amount")
    f["amount_vs_customer_mean"] = np.where(
        np.isnan(cust_mean), FILL_NEUTRAL, amount / np.maximum(cust_mean, 1.0)
    )
    f["amount_zscore_customer"] = np.where(
        np.isnan(cust_std) | (cust_std < 1.0),
        FILL_RATIO,
        (amount - cust_mean) / np.maximum(cust_std, 1.0),
    )

    cat_mean, _ = _prior_mean_std(tx, "product_category", "amount")
    f["amount_vs_category_mean"] = np.where(
        np.isnan(cat_mean), FILL_NEUTRAL, amount / np.maximum(cat_mean, 1.0)
    )

    # -- account tenure and cadence ----------------------------------------- #
    age_days = (ts - _epoch(tx["signup_ts"])) / 86_400.0
    f["account_age_days"] = np.clip(age_days, 0, None)

    prior_txns = tx.groupby("customer_id", sort=False).cumcount().to_numpy(dtype=float)
    f["customer_prior_txn_count"] = prior_txns
    f["customer_purchase_frequency"] = prior_txns / np.maximum(f["account_age_days"], 1.0)

    prev_ts = tx.groupby("customer_id", sort=False)["timestamp"].shift()
    gap = (tx["timestamp"] - prev_ts).dt.total_seconds().to_numpy(dtype=float)
    gap = np.where(np.isnan(gap), FILL_LONG_GAP_SECONDS, gap)
    f["seconds_since_prev_txn_log"] = np.log1p(np.clip(gap, 0, None))

    f["customer_txns_last_24h"] = _window_count(tx, "customer_id", 86_400, ts)
    f["customer_txns_last_7d"] = _window_count(tx, "customer_id", 7 * 86_400, ts)

    # -- prior payment reliability ------------------------------------------ #
    failed = (tx["payment_status"] == "failed").astype(float)
    prior_failed = (
        failed.groupby(tx["customer_id"], sort=False).cumsum().to_numpy(dtype=float)
        - failed.to_numpy(dtype=float)
    )
    f["customer_prior_failed_rate"] = np.where(
        prior_txns > 0, prior_failed / np.maximum(prior_txns, 1.0), FILL_RATIO
    )
    f["customer_failed_last_24h"] = _window_count(
        tx.assign(_k=tx["customer_id"].where(failed.astype(bool), other=pd.NA)),
        "_k", 86_400, ts,
    ) * failed.to_numpy(dtype=float)

    # -- prior returns and refunds, keyed on the event's own timestamp ------- #
    prior_returns = _prior_count(tx, returns, "customer_id")
    prior_refunds = _prior_count(tx, refunds, "customer_id")
    prior_refund_value = _prior_sum(tx, refunds, "customer_id", "amount")
    prior_order_value = _prior_sum(
        tx.assign(_v=tx["amount"]),
        tx.assign(_v=tx["amount"])[["customer_id", "timestamp", "_v"]],
        "customer_id",
        "_v",
    )

    f["customer_prior_return_count"] = prior_returns
    f["customer_prior_return_rate"] = np.where(
        prior_txns > 0, prior_returns / np.maximum(prior_txns, 1.0), FILL_RATIO
    )
    f["customer_prior_refund_count"] = prior_refunds
    f["customer_prior_refund_rate"] = np.where(
        prior_txns > 0, prior_refunds / np.maximum(prior_txns, 1.0), FILL_RATIO
    )
    f["customer_prior_refund_value_ratio"] = np.where(
        prior_order_value > 0,
        prior_refund_value / np.maximum(prior_order_value, 1.0),
        FILL_RATIO,
    )

    # -- entity sharing, as of now ------------------------------------------ #
    for entity, prefix in (
        ("device_id", "device"),
        ("ip_address", "ip"),
        ("shipping_address_id", "address"),
    ):
        f[f"{prefix}_distinct_customers"] = _distinct_so_far(tx, entity, "customer_id")
        f[f"{prefix}_txns_so_far"] = (
            tx.groupby(entity, sort=False).cumcount().to_numpy(dtype=float) + 1
        )
    f["device_txns_last_24h"] = _window_count(tx, "device_id", 86_400, ts)
    f["address_txns_last_7d"] = _window_count(tx, "shipping_address_id", 7 * 86_400, ts)

    # -- coupons ------------------------------------------------------------ #
    has_coupon = tx["coupon_code"].notna()
    f["has_coupon"] = has_coupon.astype(float)
    coupon_uses = (
        tx.groupby("coupon_code", sort=False).cumcount().to_numpy(dtype=float) + 1
    )
    f["coupon_uses_so_far"] = np.where(has_coupon, coupon_uses, FILL_RATIO)
    coupon_customers = _distinct_so_far(tx, "coupon_code", "customer_id")
    f["coupon_distinct_customers"] = np.where(has_coupon, coupon_customers, FILL_RATIO)

    # -- timing and payment method ------------------------------------------ #
    hour = tx["timestamp"].dt.hour.to_numpy(dtype=float)
    f["hour_of_day"] = hour
    f["is_night"] = ((hour >= 0) & (hour < 6)).astype(float)
    f["day_of_week"] = tx["timestamp"].dt.dayofweek.to_numpy(dtype=float)

    for method in C.PAYMENT_METHODS:
        f[f"pm_{method}"] = (tx["payment_method"] == method).astype(float)

    # -- merchant-wide backdrop --------------------------------------------- #
    f["merchant_txns_last_1h"] = _window_count(
        tx.assign(_all=1), "_all", 3_600, ts
    )

    assert len(f) == n
    return f.reset_index(drop=True)


FEATURE_COLUMNS: list[str] = [
    "amount", "amount_log", "amount_vs_customer_mean", "amount_zscore_customer",
    "amount_vs_category_mean",
    "account_age_days", "customer_prior_txn_count", "customer_purchase_frequency",
    "seconds_since_prev_txn_log", "customer_txns_last_24h", "customer_txns_last_7d",
    "customer_prior_failed_rate", "customer_failed_last_24h",
    "customer_prior_return_count", "customer_prior_return_rate",
    "customer_prior_refund_count", "customer_prior_refund_rate",
    "customer_prior_refund_value_ratio",
    "device_distinct_customers", "device_txns_so_far", "device_txns_last_24h",
    "ip_distinct_customers", "ip_txns_so_far",
    "address_distinct_customers", "address_txns_so_far", "address_txns_last_7d",
    "has_coupon", "coupon_uses_so_far", "coupon_distinct_customers",
    "hour_of_day", "is_night", "day_of_week",
    *[f"pm_{m}" for m in C.PAYMENT_METHODS],
    "merchant_txns_last_1h",
]

# Structural guarantee: no label column can be a feature.
assert not (set(FEATURE_COLUMNS) & C.FORBIDDEN_FEATURE_COLUMNS)


def feature_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """The model-facing matrix. Identifiers and timestamps are deliberately absent."""
    return features[FEATURE_COLUMNS]


def main() -> None:
    events = load_events()
    f = build_features(events)
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = C.DATA_DIR / "features.parquet"
    f.to_parquet(out, index=False)

    X = feature_matrix(f)
    print(f"rows      : {len(f):,}")
    print(f"features  : {len(FEATURE_COLUMNS)}")
    print(f"non-finite: {int((~np.isfinite(X.to_numpy(dtype=float))).sum())}")
    print(f"written   : {out}")


if __name__ == "__main__":
    main()
