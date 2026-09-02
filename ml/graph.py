"""Merchant risk graph.

Builds a graph linking customers to the entities they share -- devices, IPs,
shipping addresses, coupon codes -- so that coordinated behaviour becomes a
connected component rather than a set of unremarkable rows.

The central engineering problem: hub entities collapse the graph.

A public coupon code used by four hundred customers is not evidence that those
four hundred people know each other, and neither is a carrier-grade NAT address.
Linking through them merges the entire merchant into one component and destroys
the signal. Every real graph system suppresses hubs; this one does it with an
explicit, documented degree threshold per entity type, and the suppressed edges
are counted and reported rather than dropped silently.

The thresholds are deliberately generous. Suppressing aggressively would make
the problem easier by fragmenting exactly the legitimate clusters that are
supposed to be hard -- an office network on one IP, a flash-sale cohort on one
promo code -- and we want those to stay in the picture as candidate components.

This module reads events only. It never opens data/labels/.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

from . import config as C

# Entity link definitions: (transaction column, node prefix, node type).
LINKS = (
    ("device_id", "d", "device"),
    ("ip_address", "i", "ip"),
    ("shipping_address_id", "a", "address"),
    ("coupon_code", "p", "coupon"),
)

# Above these many distinct customers, an entity is treated as infrastructure
# rather than as evidence of a relationship, and its edges are not created.
#
# Chosen so that the legitimate lookalikes survive as components: an office
# network (18-40 accounts on one IP) and a flash-sale cohort (30-60 accounts on
# one promo code) both remain connected and can therefore still be wrongly
# accused. Public coupons, used by hundreds, fall out naturally.
HUB_THRESHOLDS = {
    "device": 25,
    "ip": 40,
    "address": 30,
    "coupon": 70,
}

CUSTOMER_PREFIX = "c"


@dataclass
class GraphBuild:
    graph: nx.Graph
    suppressed: dict[str, int]
    entity_counts: dict[str, int]


def _node(prefix: str, value: str) -> str:
    return f"{prefix}:{value}"


def build_graph(transactions: pd.DataFrame) -> GraphBuild:
    """Build the customer-entity graph over whatever window is passed in.

    Graph construction is retrospective by design. It runs over a trailing
    window of transactions, not inside the online scoring path, which is how
    batch ring detection works in production and what keeps the online features
    point-in-time correct.
    """
    g = nx.Graph()
    suppressed: dict[str, int] = {}
    entity_counts: dict[str, int] = {}

    customers = transactions["customer_id"].dropna().unique()
    g.add_nodes_from((_node(CUSTOMER_PREFIX, c), {"type": "customer"}) for c in customers)

    for column, prefix, node_type in LINKS:
        pairs = transactions[["customer_id", column]].dropna().drop_duplicates()
        fanout = pairs.groupby(column)["customer_id"].nunique()
        limit = HUB_THRESHOLDS[node_type]

        hubs = set(fanout[fanout > limit].index)
        suppressed[node_type] = len(hubs)
        entity_counts[node_type] = int(fanout.size)

        kept = pairs[~pairs[column].isin(hubs)]
        for cust, value in zip(kept["customer_id"], kept[column]):
            entity = _node(prefix, str(value))
            if entity not in g:
                g.add_node(entity, type=node_type)
            g.add_edge(_node(CUSTOMER_PREFIX, cust), entity, relation=f"uses_{node_type}")

    return GraphBuild(graph=g, suppressed=suppressed, entity_counts=entity_counts)


def components(build: GraphBuild, min_accounts: int = 2) -> list[set[str]]:
    """Connected components holding at least `min_accounts` customer nodes."""
    out = []
    for comp in nx.connected_components(build.graph):
        n_accounts = sum(1 for n in comp if n.startswith(f"{CUSTOMER_PREFIX}:"))
        if n_accounts >= min_accounts:
            out.append(comp)
    return sorted(out, key=len, reverse=True)


def _accounts(component: set[str]) -> list[str]:
    return [n[2:] for n in component if n.startswith(f"{CUSTOMER_PREFIX}:")]


def _typed(component: set[str], prefix: str) -> list[str]:
    return [n[2:] for n in component if n.startswith(f"{prefix}:")]


def component_features(
    build: GraphBuild,
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
    returns: pd.DataFrame,
    refunds: pd.DataFrame,
    min_accounts: int = 3,
) -> pd.DataFrame:
    """Structural and behavioural features per connected component.

    These are the inputs Phase 7 scores. Nothing here uses a label, and nothing
    here is a threshold or a decision -- this module only describes.
    """
    tx = transactions.set_index("customer_id", drop=False)
    signup = customers.set_index("customer_id")["signup_ts"]
    returned = set(returns["transaction_id"])
    refund_by_txn = refunds.groupby("transaction_id")["amount"].sum()

    rows = []
    for idx, comp in enumerate(components(build, min_accounts=min_accounts)):
        accounts = _accounts(comp)
        devices = _typed(comp, "d")
        ips = _typed(comp, "i")
        addresses = _typed(comp, "a")
        coupons = _typed(comp, "p")

        g = tx.loc[tx.index.intersection(accounts)]
        if g.empty:
            continue
        captured = g[g["payment_status"] == "captured"]

        n_acc = len(accounts)
        n_dev = max(len(devices), 1)
        n_ip = max(len(ips), 1)
        n_addr = max(len(addresses), 1)

        # Behavioural
        captured_value = float(captured["amount"].sum()) or 1.0
        refund_value = float(
            refund_by_txn.reindex(captured["transaction_id"]).fillna(0).sum()
        )
        return_rate = (
            float(captured["transaction_id"].isin(returned).mean())
            if len(captured) else 0.0
        )

        codes = g["coupon_code"].dropna()
        coupon_reuse = (
            float(codes.value_counts().iloc[0] / len(g)) if len(codes) else 0.0
        )

        signups = signup.reindex(accounts).dropna()
        if len(signups) > 1:
            signup_span = float(
                (signups.max() - signups.min()).total_seconds() / 86_400
            )
            # Share of accounts created within a day of the earliest one. A burst
            # of registrations is the signature that separates a promo farm from
            # a family, which accretes accounts over years.
            burst = float(
                ((signups - signups.min()).dt.total_seconds() <= 86_400).mean()
            )
        else:
            signup_span, burst = 0.0, 1.0

        active_days = max(
            float(
                (g["timestamp"].max() - g["timestamp"].min()).total_seconds() / 86_400
            ),
            0.5,
        )

        # Structural. The useful question is not how dense the component is --
        # a component built from shared entities is dense by construction -- but
        # ACROSS HOW MANY CHANNELS the accounts overlap.
        #
        # A family shares one address and one router but each member has their
        # own phone: it bridges on two channels. A ring shares devices AND
        # addresses AND a promo code: it bridges on four. That distinction is
        # not expressible in any single one-hop count, which is the specific
        # thing the graph contributes over the row model.
        sub = build.graph.subgraph(comp)
        entities = [n for n in comp if not n.startswith(f"{CUSTOMER_PREFIX}:")]
        bridging = [n for n in entities if sub.degree(n) > 1]
        bridging_types = {sub.nodes[n]["type"] for n in bridging}
        # Mean accounts per bridging entity: how heavily each shared thing is
        # reused, as opposed to how many shared things exist.
        bridge_load = (
            float(np.mean([sub.degree(n) for n in bridging])) if bridging else 0.0
        )

        rows.append(
            {
                "component_id": f"CMP-{idx:04d}",
                "n_accounts": n_acc,
                "n_devices": len(devices),
                "n_ips": len(ips),
                "n_addresses": len(addresses),
                "n_coupons": len(coupons),
                "n_transactions": int(len(g)),
                "accounts_per_device": n_acc / n_dev,
                "accounts_per_ip": n_acc / n_ip,
                "accounts_per_address": n_acc / n_addr,
                # The multi-hop signal one-hop row features cannot express:
                # what share of the component's accounts are reachable through
                # an entity that more than one account touches.
                "bridging_channels": len(bridging_types),
                "bridging_entity_ratio": len(bridging) / max(len(entities), 1),
                "bridge_load": bridge_load,
                "return_rate": return_rate,
                "refund_value_ratio": refund_value / captured_value,
                "coupon_reuse": coupon_reuse,
                "failed_payment_rate": float((g["payment_status"] == "failed").mean()),
                "signup_span_days": signup_span,
                "signup_burst_ratio": burst,
                "txn_velocity": len(g) / n_acc / active_days,
                "median_amount": float(g["amount"].median()),
                "total_value": float(g["amount"].sum()),
                "captured_value": captured_value,
                "refund_value": refund_value,
                "accounts": accounts,
            }
        )

    return pd.DataFrame(rows)


FEATURE_COLUMNS = [
    "n_accounts", "n_devices", "n_ips", "n_addresses", "n_coupons",
    "n_transactions", "accounts_per_device", "accounts_per_ip",
    "accounts_per_address", "bridging_channels", "bridging_entity_ratio",
    "bridge_load", "return_rate", "refund_value_ratio", "coupon_reuse",
    "failed_payment_rate", "signup_span_days", "signup_burst_ratio",
    "txn_velocity", "median_amount",
]

assert not (set(FEATURE_COLUMNS) & C.FORBIDDEN_FEATURE_COLUMNS)


def load_events() -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_parquet(C.EVENTS_DIR / f"{name}.parquet")
        for name in ("customers", "transactions", "returns", "refunds")
    }


def main() -> None:
    ev = load_events()
    build = build_graph(ev["transactions"])
    feats = component_features(
        build, ev["transactions"], ev["customers"], ev["returns"], ev["refunds"]
    )

    print(f"nodes {build.graph.number_of_nodes():,}  edges {build.graph.number_of_edges():,}")
    print("\nhub entities suppressed (linking through these is not evidence):")
    for kind, n in build.suppressed.items():
        total = build.entity_counts[kind]
        print(f"  {kind:9s} {n:5d} of {total:6,d}  (threshold {HUB_THRESHOLDS[kind]} accounts)")

    print(f"\ncomponents with 3+ accounts: {len(feats):,}")
    if len(feats):
        sizes = feats["n_accounts"]
        print(
            f"  account count  min {sizes.min()}  median {int(sizes.median())}  "
            f"p95 {int(np.percentile(sizes, 95))}  max {sizes.max()}"
        )


if __name__ == "__main__":
    main()
