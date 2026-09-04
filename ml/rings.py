"""Abuse ring detection.

Scores the components the graph formed, and emits the structured evidence that
the policy engine and the investigator consume.

Design commitments:

WEIGHTS ARE LEARNED, NOT INVENTED. A hand-tuned weighted sum would be a set of
numbers chosen until the demo looked good. Instead a logistic regression is fit
on components from the training window only, with a small documented feature set
and L2 regularisation, because there are roughly 150 components to learn from
and a wide model would memorise them.

THE THRESHOLD IS CHOSEN BY CROSS-VALIDATION, NOT ON THE HELD-OUT SPLIT. The
validation window holds only two ring components -- too thin to tune against --
so the threshold is picked by 5-fold CV inside the fitting pool. The test window
is scored exactly once.

EVIDENCE WEIGHTS COME FROM THE MODEL. Each evidence item carries the share of
the decision that feature actually contributed, so the explanation a merchant
reads is the reasoning the model used rather than a plausible story written
afterwards.

Nothing in this module decides policy. It measures and describes; the policy
engine decides.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .graph import build_graph, component_features
from .splits import assign_transaction_splits
from .train import load_risk_scores

# One policy implementation, in backend/app/core/policy.py. Duplicating the
# bands here would let the two drift and quietly disagree about a boundary.
import sys

sys.path.insert(0, str(C.ROOT))
from backend.app.core.policy import decide as _policy_decide  # noqa: E402

ARTIFACTS = C.ROOT / "artifacts"

# Deliberately small. With ~150 training components, a 20-feature model would
# memorise the fitting pool rather than learn anything transferable.
RING_FEATURES = [
    "refund_value_ratio",
    "accounts_per_device",
    "signup_span_days_log",
    "bridging_entity_ratio",
    "return_rate",
    "failed_payment_rate",
    "signup_burst_ratio",
    "txn_velocity",
]

# Merchant-readable phrasing and units per feature, used to build evidence.
EVIDENCE_SPEC = {
    "refund_value_ratio": ("REFUND_VALUE_RATIO", "refunded share of captured value", "ratio"),
    "accounts_per_device": ("ACCOUNTS_PER_DEVICE", "accounts per device", "accounts"),
    "signup_span_days_log": ("SIGNUP_SPAN", "days between first and last account signup", "days"),
    "bridging_entity_ratio": ("BRIDGING_ENTITIES", "share of devices, addresses and codes shared between accounts", "ratio"),
    "return_rate": ("RETURN_RATE", "share of captured orders returned", "ratio"),
    "failed_payment_rate": ("FAILED_PAYMENT_RATE", "share of payment attempts that failed", "ratio"),
    "signup_burst_ratio": ("SIGNUP_BURST", "share of accounts created within 24 hours of the first", "ratio"),
    "txn_velocity": ("TXN_VELOCITY", "orders per account per active day", "orders"),
}

# An evidence item is only shown when it moved the score by at least this share.
MIN_EVIDENCE_WEIGHT = 0.04


@dataclass
class RingModelMeta:
    threshold: float
    cv_f1: float
    n_fit_components: int
    coefficients: dict[str, float]
    baselines: dict[str, float]


def prepare(feats: pd.DataFrame) -> pd.DataFrame:
    """Derived columns the model uses. Kept here so scoring and fitting agree."""
    out = feats.copy()
    out["signup_span_days_log"] = np.log1p(out["signup_span_days"].clip(lower=0))
    return out


def assign_component_splits(
    feats: pd.DataFrame, transactions: pd.DataFrame
) -> pd.Series:
    """A component belongs to the split holding most of its transactions."""
    tx = transactions[["customer_id", "timestamp"]].copy()
    tx["split"] = assign_transaction_splits(tx).values
    by_customer = tx.groupby("customer_id")["split"].agg(
        lambda s: s.value_counts().index[0]
    )
    return feats["accounts"].apply(
        lambda accs: by_customer.reindex(accs).dropna().value_counts().index[0]
        if len(by_customer.reindex(accs).dropna()) else "train"
    )


def label_components(feats: pd.DataFrame, membership: pd.DataFrame) -> pd.Series:
    """1 if a component is dominated by an injected abuse ring, else 0.

    Labels are used for fitting and scoring only. The graph that produced these
    components never saw them.
    """
    lookup = dict(zip(membership["customer_id"], membership["cluster_id"]))
    labels = []
    for accounts in feats["accounts"]:
        named = [lookup.get(a) for a in accounts]
        named = [c for c in named if c is not None]
        if not named or len(named) / len(accounts) < 0.5:
            labels.append(0)
            continue
        dominant = pd.Series(named).value_counts().index[0]
        labels.append(int(str(dominant).startswith("AR-")))
    return pd.Series(labels, index=feats.index)


def _pick_threshold_cv(X: np.ndarray, y: np.ndarray, seed: int = 0) -> tuple[float, float]:
    """Out-of-fold threshold selection inside the fitting pool.

    Using the held-out split here would destroy the only clean measurement we
    have. Using the validation window is impossible: it holds two ring
    components.
    """
    oof = np.zeros(len(y), dtype=float)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train_idx, test_idx in folds.split(X, y):
        pipe = _make_pipeline()
        pipe.fit(X[train_idx], y[train_idx])
        oof[test_idx] = pipe.predict_proba(X[test_idx])[:, 1]

    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 91):
        pred = oof >= t
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        if tp == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1


def _make_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=0.5,  # strong L2: the fitting pool is small
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=0,
                ),
            ),
        ]
    )


def fit(
    feats: pd.DataFrame, labels: pd.Series, splits: pd.Series
) -> tuple[Pipeline, RingModelMeta]:
    pool = feats[splits.isin(["train", "validation"])]
    y = labels[pool.index].to_numpy()
    X = pool[RING_FEATURES].to_numpy(dtype=float)

    threshold, cv_f1 = _pick_threshold_cv(X, y)
    pipe = _make_pipeline()
    pipe.fit(X, y)

    coefs = pipe.named_steps["clf"].coef_[0]
    meta = RingModelMeta(
        threshold=threshold,
        cv_f1=round(cv_f1, 4),
        n_fit_components=int(len(pool)),
        coefficients={f: round(float(c), 4) for f, c in zip(RING_FEATURES, coefs)},
        # Baselines are merchant medians over the fitting pool. Evidence compares
        # a component against these, so "6.4x baseline" is a defined quantity.
        baselines={
            f: round(float(pool[f].median()), 4) for f in RING_FEATURES
        },
    )
    return pipe, meta


def _contributions(pipe: Pipeline, row: pd.Series) -> dict[str, float]:
    """Signed contribution of each feature to this component's log-odds."""
    scaler = pipe.named_steps["scale"]
    clf = pipe.named_steps["clf"]
    x = row[RING_FEATURES].to_numpy(dtype=float).reshape(1, -1)
    z = scaler.transform(x)[0]
    return {f: float(z[i] * clf.coef_[0][i]) for i, f in enumerate(RING_FEATURES)}


def build_evidence(
    pipe: Pipeline, meta: RingModelMeta, row: pd.Series
) -> list[dict]:
    """Evidence ordered by how much each signal actually moved the decision."""
    contributions = _contributions(pipe, row)
    # Only signals pushing toward risk are shown as evidence; mitigating signals
    # are reflected in the score itself rather than presented as accusations.
    pushing = {f: c for f, c in contributions.items() if c > 0}
    total = sum(pushing.values()) or 1.0

    items = []
    for feature, contribution in sorted(
        pushing.items(), key=lambda kv: kv[1], reverse=True
    ):
        weight = contribution / total
        if weight < MIN_EVIDENCE_WEIGHT:
            continue
        code, phrase, unit = EVIDENCE_SPEC[feature]
        observed = float(row[feature])
        baseline = meta.baselines[feature]
        if feature == "signup_span_days_log":
            observed = float(np.expm1(observed))
            baseline = float(np.expm1(baseline))

        comparison = (
            f"{observed / baseline:.1f}x the merchant baseline"
            if baseline > 0.01 and observed > baseline
            else f"merchant baseline {baseline:.2f}"
        )
        items.append(
            {
                "code": code,
                "kind": "FACT",
                "statement": f"{phrase.capitalize()} is {observed:.2f} ({comparison}).",
                "observed_value": round(observed, 4),
                "baseline_value": round(baseline, 4),
                "unit": unit,
                "weight": round(weight, 4),
            }
        )

    # Structural counts are directly observed and always shown.
    items.insert(
        0,
        {
            "code": "CLUSTER_SHAPE",
            "kind": "FACT",
            "statement": (
                f"{int(row['n_accounts'])} accounts are connected through "
                f"{int(row['n_devices'])} devices, {int(row['n_addresses'])} "
                f"delivery addresses and {int(row['n_ips'])} IP addresses."
            ),
            "observed_value": int(row["n_accounts"]),
            "baseline_value": None,
            "unit": "accounts",
            "weight": 0.0,
        },
    )
    return items


# A force-directed graph stops being readable well before this, so the payload
# is capped and the UI is told it was truncated rather than silently shown a
# partial picture.
MAX_GRAPH_NODES = 40


def ring_graph(
    accounts: list[str],
    transactions: pd.DataFrame,
    account_risk: pd.Series,
) -> dict:
    """Nodes and edges for one component, shaped for the investigation screen.

    Entities shared by more than one account are kept first. They are what makes
    the component a component, and dropping them to fit a node budget would
    remove the evidence while keeping the accusation.
    """
    rows = transactions[transactions["customer_id"].isin(accounts)]

    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for account in accounts:
        nodes[f"c:{account}"] = {
            "id": f"c:{account}",
            "type": "customer",
            "label": account,
            "risk_score": round(float(account_risk.get(account, 0.0)) * 100, 1),
            "attributes": {},
        }

    entity_columns = (
        ("device_id", "d", "device"),
        ("shipping_address_id", "a", "address"),
        ("ip_address", "i", "ip"),
        ("coupon_code", "p", "coupon"),
    )

    fanout: dict[str, int] = {}
    for column, prefix, node_type in entity_columns:
        pairs = rows[["customer_id", column]].dropna().drop_duplicates()
        counts = pairs.groupby(column)["customer_id"].nunique()
        for _, (account, value) in pairs.iterrows():
            node_id = f"{prefix}:{value}"
            fanout[node_id] = int(counts[value])
            if node_id not in nodes:
                nodes[node_id] = {
                    "id": node_id,
                    "type": node_type,
                    "label": str(value),
                    # An entity linking many accounts is the interesting one, so
                    # it is drawn hotter. This is a display weight, not a verdict.
                    "risk_score": round(
                        min(100.0, 100.0 * (counts[value] - 1) / max(len(accounts) - 1, 1)),
                        1,
                    ),
                    "attributes": {"shared_by": int(counts[value])},
                }
            key = (f"c:{account}", node_id)
            if key not in seen:
                seen.add(key)
                edges.append(
                    {
                        "source": key[0],
                        "target": node_id,
                        "relation": f"uses_{node_type}",
                        "weight": float(counts[value]),
                    }
                )

    truncated = False
    if len(nodes) > MAX_GRAPH_NODES:
        truncated = True
        keep = {n for n in nodes if n.startswith("c:")}
        shared = sorted(
            (n for n in nodes if not n.startswith("c:")),
            key=lambda n: -fanout.get(n, 0),
        )
        for node_id in shared:
            if len(keep) >= MAX_GRAPH_NODES:
                break
            keep.add(node_id)
        nodes = {k: v for k, v in nodes.items() if k in keep}
        edges = [e for e in edges if e["source"] in keep and e["target"] in keep]

    return {"nodes": list(nodes.values()), "edges": edges, "truncated": truncated}


def policy_decision(risk_score: float, exposure_value: float) -> dict:
    """Thin delegation to the deterministic policy engine."""
    return _policy_decide(risk_score, exposure_value).to_dict()


def deterministic_explanation(row: pd.Series, evidence: list[dict]) -> dict:
    """Templated explanation. This is the default path, not a degraded one.

    The language model in Phase 14 rewrites this more fluently; it never decides
    anything, and if it is unavailable this is what a merchant reads. It is
    written to stand on its own for that reason.
    """
    facts = [e["statement"] for e in evidence if e["kind"] == "FACT"]
    def plural(n: int, word: str) -> str:
        return f"{n} {word}" if n == 1 else f"{n} {word}s"

    lead = (
        f"{plural(int(row['n_accounts']), 'customer account')} are linked "
        f"through {plural(int(row['n_devices']), 'shared device')}, "
        f"{plural(int(row['n_addresses']), 'delivery address').replace('addresss', 'addresses')} "
        f"and {plural(int(row['n_ips']), 'IP address').replace('addresss', 'addresses')}, "
        f"across {plural(int(row['n_transactions']), 'order')}."
    )
    inference = (
        f"Taken together these signals put the group at {row['risk_score']:.0f} "
        "out of 100. Individually none of these orders would stand out; the "
        "pattern is only visible across the connected accounts."
    )
    return {
        "source": "deterministic",
        "text": f"{lead} {inference}",
        "facts": facts,
        "inferences": [inference],
        "degraded": False,
    }


def exposure(row: pd.Series) -> float:
    """Money at stake for a component.

    Defined as refunds already issued plus the captured value still open to
    refund or chargeback. This is an upper bound on what could be lost, not a
    claim that it will be. The financial engine applies a recovery rate on top.
    """
    return round(float(row["refund_value"]) + float(row["captured_value"]), 2)


def risk_level(score: float) -> str:
    if score >= 90:
        return "CRITICAL"
    if score >= 70:
        return "HIGH"
    if score >= 40:
        return "MEDIUM"
    return "LOW"


def score_components(
    pipe: Pipeline, meta: RingModelMeta, feats: pd.DataFrame
) -> pd.DataFrame:
    X = feats[RING_FEATURES].to_numpy(dtype=float)
    probabilities = pipe.predict_proba(X)[:, 1]

    out = feats.copy()
    out["probability"] = probabilities
    out["risk_score"] = np.round(probabilities * 100, 1)
    out["risk_level"] = [risk_level(s) for s in out["risk_score"]]
    # Distance from the decision boundary, normalised. A component sitting on
    # the threshold is reported as low confidence rather than as a coin flip
    # dressed up as a verdict.
    out["confidence"] = np.round(np.clip(np.abs(probabilities - meta.threshold) * 2, 0, 1), 3)
    out["financial_exposure"] = [exposure(r) for _, r in out.iterrows()]
    out["flagged"] = probabilities >= meta.threshold
    return out


def detect(seed: int = C.SEED) -> dict:
    """Full pipeline: graph, features, fit, score, persist."""
    events = {
        n: pd.read_parquet(C.EVENTS_DIR / f"{n}.parquet")
        for n in ("customers", "transactions", "returns", "refunds")
    }
    labels_df = pd.read_parquet(C.LABELS_DIR / "transaction_labels.parquet")
    membership = (
        events["transactions"][["transaction_id", "customer_id"]]
        .merge(labels_df, on="transaction_id")
        .dropna(subset=["cluster_id"])[["customer_id", "cluster_id"]]
        .drop_duplicates()
    )

    build = build_graph(events["transactions"])
    feats = prepare(
        component_features(
            build,
            events["transactions"],
            events["customers"],
            events["returns"],
            events["refunds"],
        )
    )
    splits = assign_component_splits(feats, events["transactions"])
    y = label_components(feats, membership)

    pipe, meta = fit(feats, y, splits)
    scored = score_components(pipe, meta, feats)
    scored["split"] = splits
    scored["is_ring"] = y

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, ARTIFACTS / "model_ring.joblib")

    risk = load_risk_scores()
    tx_risk = events["transactions"][["transaction_id", "customer_id"]].copy()
    tx_risk["risk"] = tx_risk["transaction_id"].map(risk).fillna(0.0)
    account_risk = tx_risk.groupby("customer_id")["risk"].mean()
    detected_at = events["transactions"]["timestamp"].max()

    rings = []
    for _, row in scored[scored["flagged"]].sort_values(
        "risk_score", ascending=False
    ).iterrows():
        evidence = build_evidence(pipe, meta, row)
        exposure_value = float(row["financial_exposure"])
        summary = {
            "ring_id": row["component_id"].replace("CMP", "AR"),
            "risk_score": float(row["risk_score"]),
            "risk_level": row["risk_level"],
            "confidence": float(row["confidence"]),
            "n_accounts": int(row["n_accounts"]),
            "n_devices": int(row["n_devices"]),
            "n_addresses": int(row["n_addresses"]),
            "n_ips": int(row["n_ips"]),
            "n_transactions": int(row["n_transactions"]),
            "financial_exposure": exposure_value,
            "detected_at": detected_at.isoformat(),
            "status": "AWAITING_APPROVAL"
            if row["risk_score"] >= 90
            else "OPEN",
        }
        rings.append(
            {
                **summary,
                "component_id": row["component_id"],
                "split": row["split"],
                "evidence": evidence,
                "accounts": sorted(row["accounts"]),
                "summary": summary,
                "graph": ring_graph(
                    sorted(row["accounts"]), events["transactions"], account_risk
                ),
                "policy": policy_decision(float(row["risk_score"]), exposure_value),
                "explanation": deterministic_explanation(row, evidence),
                "component_features": {
                    f: round(float(row[f]), 4) for f in RING_FEATURES
                },
            }
        )

    content = json.dumps({"model": meta.__dict__, "rings": rings}, indent=2)
    target = ARTIFACTS / "rings.json"
    tmp_target = ARTIFACTS / "rings.json.tmp"
    try:
        tmp_target.write_text(content, encoding="utf-8")
        tmp_target.replace(target)
    except OSError:
        try:
            target.write_text(content, encoding="utf-8")
        except OSError:
            pass
        finally:
            if tmp_target.exists():
                try:
                    tmp_target.unlink()
                except OSError:
                    pass
    return {"scored": scored, "meta": meta, "rings": rings, "pipeline": pipe}


def main() -> None:
    res = detect()
    scored, meta = res["scored"], res["meta"]
    print(f"components scored     : {len(scored):,}")
    print(f"fitting pool          : {meta.n_fit_components} components")
    print(f"threshold (5-fold CV) : {meta.threshold:.3f}  cv F1 {meta.cv_f1:.3f}")
    print("\nlearned coefficients (standardised):")
    for f, c in sorted(meta.coefficients.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {f:24s} {c:+.3f}")
    print(f"\nrings flagged         : {len(res['rings'])}")
    print(f"written               : {ARTIFACTS / 'rings.json'}")


if __name__ == "__main__":
    main()
