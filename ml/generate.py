"""Synthetic merchant dataset generator.

Emits a raw event log only. It does not compute any features, and it writes
labels to a separate directory that the feature builder never opens. That
physical separation is the leakage barrier.

Output:
    data/events/customers.parquet      customer_id, signup_ts
    data/events/transactions.parquet   the order + payment event log
    data/events/returns.parquet
    data/events/refunds.parquet
    data/labels/transaction_labels.parquet
    data/labels/clusters.parquet
    data/labels/manifest.json

Run:
    python -m ml.generate --seed 42
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from . import config as C


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _uni(rng: np.random.Generator, span: tuple[float, float]) -> float:
    lo, hi = span
    return float(rng.uniform(lo, hi))


def _int(rng: np.random.Generator, span: tuple[int, int]) -> int:
    lo, hi = span
    return int(rng.integers(lo, hi + 1))


def _pick_weighted(rng: np.random.Generator, mapping: dict) -> str:
    keys = list(mapping)
    weights = np.array([mapping[k]["weight"] for k in keys], dtype=float)
    return str(rng.choice(keys, p=weights / weights.sum()))


@dataclass
class Ledger:
    """Accumulates rows. Kept dumb on purpose; all logic lives in the builder."""

    customers: list[dict] = field(default_factory=list)
    transactions: list[dict] = field(default_factory=list)
    returns: list[dict] = field(default_factory=list)
    refunds: list[dict] = field(default_factory=list)
    clusters: list[dict] = field(default_factory=list)


class Generator:
    def __init__(self, seed: int = C.SEED, n_transactions: int = C.TARGET_TRANSACTIONS):
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.target_transactions = n_transactions
        self.start = datetime.fromisoformat(C.START_DATE)
        self.end = self.start + timedelta(days=C.SPAN_DAYS)
        self.ledger = Ledger()

        self._cust_n = 0
        self._txn_n = 0
        self._ret_n = 0
        self._ref_n = 0

        self.ip_pool = [self._make_ip(i) for i in range(C.N_IP_POOL)]
        self.coupon_pool = [f"SAVE{i:03d}" for i in range(C.N_COUPON_POOL)]

    # -- id minting ---------------------------------------------------------- #

    def _make_ip(self, i: int) -> str:
        return f"49.{(i // 65536) % 256}.{(i // 256) % 256}.{i % 256}"

    def _next_customer_id(self) -> str:
        self._cust_n += 1
        return f"CUST-{self._cust_n:06d}"

    def _next_txn_id(self) -> str:
        self._txn_n += 1
        return f"TXN-{self._txn_n:06d}"

    # -- primitives ---------------------------------------------------------- #

    def _ts_between(self, a: datetime, b: datetime) -> datetime:
        if b <= a:
            return a
        secs = int((b - a).total_seconds())
        return a + timedelta(seconds=int(self.rng.integers(0, secs)))

    def _amount(self, category: str) -> float:
        mu, sigma = C.CATEGORY_AMOUNT[category]
        return round(float(np.exp(self.rng.normal(mu, sigma))), 2)

    def _category(self) -> str:
        return str(self.rng.choice(C.CATEGORIES, p=C.CATEGORY_WEIGHTS))

    def _payment_method(self) -> str:
        return str(self.rng.choice(C.PAYMENT_METHODS, p=C.PAYMENT_METHOD_WEIGHTS))

    def _add_customer(self, signup_ts: datetime) -> str:
        cid = self._next_customer_id()
        self.ledger.customers.append(
            {"customer_id": cid, "signup_ts": signup_ts}
        )
        return cid

    def _emit_transaction(
        self,
        *,
        customer_id: str,
        ts: datetime,
        device_id: str,
        ip_address: str,
        address_id: str,
        coupon_code: str | None,
        failure_rate: float,
        return_rate: float,
        refund_rate: float,
        is_fraud: bool,
        cluster_id: str | None,
    ) -> None:
        category = self._category()
        amount = self._amount(category)
        failed = bool(self.rng.random() < failure_rate)
        status = "failed" if failed else "captured"
        reason = (
            str(self.rng.choice(C.FAILURE_REASONS)) if failed else None
        )
        txn_id = self._next_txn_id()

        self.ledger.transactions.append(
            {
                "transaction_id": txn_id,
                "customer_id": customer_id,
                "timestamp": ts,
                "amount": amount,
                "product_category": category,
                "payment_method": self._payment_method(),
                "device_id": device_id,
                "ip_address": ip_address,
                "shipping_address_id": address_id,
                "coupon_code": coupon_code,
                "payment_status": status,
                "failure_reason": reason,
            }
        )
        self._labels.append(
            {
                "transaction_id": txn_id,
                "is_fraud": int(is_fraud),
                "cluster_id": cluster_id,
            }
        )

        if failed:
            return  # a failed payment produces no return and no refund

        # Returns
        if self.rng.random() < return_rate:
            self._ret_n += 1
            ret_ts = ts + timedelta(days=float(self.rng.uniform(2, 21)))
            if ret_ts < self.end:
                self.ledger.returns.append(
                    {
                        "return_id": f"RET-{self._ret_n:06d}",
                        "transaction_id": txn_id,
                        "timestamp": ret_ts,
                        "reason": str(self.rng.choice(C.RETURN_REASONS)),
                        "status": str(
                            self.rng.choice(
                                ["completed", "in_transit", "rejected"],
                                p=[0.78, 0.15, 0.07],
                            )
                        ),
                    }
                )
                self._emit_refund(txn_id, amount, ret_ts, full=True)
                return

        # Direct refunds without a return leg
        if self.rng.random() < refund_rate:
            ref_ts = ts + timedelta(days=float(self.rng.uniform(0.5, 12)))
            if ref_ts < self.end:
                self._emit_refund(txn_id, amount, ref_ts, full=False)

    def _emit_refund(
        self, txn_id: str, amount: float, ts: datetime, *, full: bool
    ) -> None:
        self._ref_n += 1
        value = amount if full else round(amount * float(self.rng.uniform(0.4, 1.0)), 2)
        self.ledger.refunds.append(
            {
                "refund_id": f"REF-{self._ref_n:06d}",
                "transaction_id": txn_id,
                "timestamp": ts,
                "amount": value,
            }
        )

    # -- populations --------------------------------------------------------- #

    def _generate_normal(self) -> None:
        names = list(C.NORMAL_PROFILES)
        weights = np.array([C.NORMAL_PROFILES[n]["weight"] for n in names])
        weights = weights / weights.sum()

        for i in range(C.N_NORMAL_CUSTOMERS):
            p = C.NORMAL_PROFILES[str(self.rng.choice(names, p=weights))]
            signup = self.start - timedelta(days=float(self.rng.uniform(0, 900)))
            cid = self._add_customer(signup)

            devices = [
                f"DEV-N{i:05d}-{d}" for d in range(_int(self.rng, p["devices"]))
            ]
            address = f"ADDR-N{i:05d}"
            ip = str(self.rng.choice(self.ip_pool))

            return_rate = _uni(self.rng, p["return_rate"])
            coupon_rate = _uni(self.rng, p["coupon_rate"])
            failure_rate = _uni(self.rng, p["failure_rate"])

            for _ in range(_int(self.rng, p["orders"])):
                self._emit_transaction(
                    customer_id=cid,
                    ts=self._ts_between(self.start, self.end),
                    device_id=str(self.rng.choice(devices)),
                    ip_address=ip,
                    address_id=address,
                    coupon_code=(
                        str(self.rng.choice(self.coupon_pool))
                        if self.rng.random() < coupon_rate
                        else None
                    ),
                    failure_rate=failure_rate,
                    return_rate=return_rate,
                    refund_rate=0.03,
                    is_fraud=False,
                    cluster_id=None,
                )

    def _generate_cluster(
        self,
        *,
        cluster_id: str,
        kind: str,
        profile_name: str,
        spec: dict,
        is_abusive: bool,
    ) -> None:
        rng = self.rng
        n_accounts = _int(rng, spec["accounts"])
        n_devices = max(1, min(_int(rng, spec["devices"]), n_accounts))
        n_addresses = max(1, min(_int(rng, spec["addresses"]), n_accounts))
        n_ips = max(1, min(_int(rng, spec["ips"]), n_accounts))

        devices = [f"DEV-{cluster_id}-{i}" for i in range(n_devices)]
        addresses = [f"ADDR-{cluster_id}-{i}" for i in range(n_addresses)]
        ips = [self._make_ip(int(rng.integers(0, 2**20))) for _ in range(n_ips)]

        shared_coupon = (
            f"PROMO-{cluster_id}" if spec.get("shared_coupon") else None
        )

        return_rate = _uni(rng, spec["return_rate"])
        refund_rate = _uni(rng, spec.get("refund_rate", (0.02, 0.05)))
        coupon_rate = _uni(rng, spec["coupon_rate"])
        failure_rate = _uni(rng, spec["failure_rate"])
        abuse_share = (
            _uni(rng, spec["abuse_share"]) if is_abusive else 0.0
        )

        active_days = _uni(rng, spec["active_span_days"])
        latest_start = max(0.0, C.SPAN_DAYS - active_days - 1)
        window_start = self.start + timedelta(days=float(rng.uniform(0, latest_start)))
        window_end = window_start + timedelta(days=max(active_days, 0.25))

        signup_span = _uni(rng, spec["signup_span_days"])
        signup_anchor = window_start - timedelta(days=float(rng.uniform(1, 40)))

        for _ in range(n_accounts):
            signup = signup_anchor - timedelta(days=float(rng.uniform(0, signup_span)))
            cid = self._add_customer(signup)
            device = str(rng.choice(devices))
            address = str(rng.choice(addresses))
            ip = str(rng.choice(ips))

            for _ in range(_int(rng, spec["orders_per_account"])):
                if shared_coupon is not None and rng.random() < coupon_rate:
                    coupon = shared_coupon
                elif rng.random() < coupon_rate:
                    coupon = str(rng.choice(self.coupon_pool))
                else:
                    coupon = None

                self._emit_transaction(
                    customer_id=cid,
                    ts=self._ts_between(window_start, window_end),
                    device_id=device,
                    ip_address=ip,
                    address_id=address,
                    coupon_code=coupon,
                    failure_rate=failure_rate,
                    return_rate=return_rate,
                    refund_rate=refund_rate,
                    # Ring members also shop normally. Only a share of their
                    # activity is the abuse itself, which keeps the row-level
                    # problem realistically hard.
                    is_fraud=is_abusive and rng.random() < abuse_share,
                    cluster_id=cluster_id,
                )

        self.ledger.clusters.append(
            {
                "cluster_id": cluster_id,
                "cluster_kind": kind,
                "profile": profile_name,
                "is_abusive": int(is_abusive),
                "n_accounts": n_accounts,
                "n_devices": n_devices,
                "n_addresses": n_addresses,
                "n_ips": n_ips,
                "window_start": window_start,
                "window_end": window_end,
            }
        )

    def _generate_hard_negatives(self) -> None:
        idx = 0
        for name, spec in C.HARD_NEGATIVE_PROFILES.items():
            for _ in range(spec["count"]):
                idx += 1
                self._generate_cluster(
                    cluster_id=f"HN-{idx:03d}",
                    kind="HARD_NEGATIVE",
                    profile_name=name,
                    spec=spec,
                    is_abusive=False,
                )

    def _generate_abuse_rings(self) -> None:
        for i in range(1, C.N_ABUSE_RINGS + 1):
            name = _pick_weighted(self.rng, C.ABUSE_PROFILES)
            self._generate_cluster(
                cluster_id=f"AR-{i:03d}",
                kind="ABUSE_RING",
                profile_name=name,
                spec=C.ABUSE_PROFILES[name],
                is_abusive=True,
            )

    # -- entry point --------------------------------------------------------- #

    def run(self) -> dict[str, pd.DataFrame]:
        self._labels: list[dict] = []
        self._generate_abuse_rings()
        self._generate_hard_negatives()
        self._generate_normal()

        tx = pd.DataFrame(self.ledger.transactions)
        tx = tx.sort_values("timestamp").reset_index(drop=True)

        labels = pd.DataFrame(self._labels)
        labels = (
            labels.set_index("transaction_id")
            .loc[tx["transaction_id"]]
            .reset_index()
        )

        return {
            "customers": pd.DataFrame(self.ledger.customers).sort_values("signup_ts"),
            "transactions": tx,
            "returns": pd.DataFrame(self.ledger.returns),
            "refunds": pd.DataFrame(self.ledger.refunds),
            "transaction_labels": labels,
            "clusters": pd.DataFrame(self.ledger.clusters),
        }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def write(frames: dict[str, pd.DataFrame], seed: int) -> dict:
    C.EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    C.LABELS_DIR.mkdir(parents=True, exist_ok=True)

    for name in ("customers", "transactions", "returns", "refunds"):
        frames[name].to_parquet(C.EVENTS_DIR / f"{name}.parquet", index=False)
    for name in ("transaction_labels", "clusters"):
        frames[name].to_parquet(C.LABELS_DIR / f"{name}.parquet", index=False)

    tx, lab, cl = frames["transactions"], frames["transaction_labels"], frames["clusters"]
    manifest = {
        "seed": seed,
        "generated_at": datetime.now().astimezone().isoformat(),
        "span_days": C.SPAN_DAYS,
        "n_transactions": int(len(tx)),
        "n_customers": int(len(frames["customers"])),
        "n_returns": int(len(frames["returns"])),
        "n_refunds": int(len(frames["refunds"])),
        "n_rings_injected": int((cl["cluster_kind"] == "ABUSE_RING").sum()),
        "n_hard_negative_clusters": int((cl["cluster_kind"] == "HARD_NEGATIVE").sum()),
        "fraud_rate": round(float(lab["is_fraud"].mean()), 5),
        "failed_payment_rate": round(
            float((tx["payment_status"] == "failed").mean()), 5
        ),
        "start": tx["timestamp"].min().isoformat(),
        "end": tx["timestamp"].max().isoformat(),
    }
    (C.LABELS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the RazorShield dataset.")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--transactions", type=int, default=C.TARGET_TRANSACTIONS)
    args = ap.parse_args()

    frames = Generator(seed=args.seed, n_transactions=args.transactions).run()
    manifest = write(frames, args.seed)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
