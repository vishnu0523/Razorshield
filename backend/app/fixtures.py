"""Contract-shaped placeholder data for Phase 1.

Every value here is a SHAPE, not a RESULT. Nothing in this file is a claim about
model performance. The `is_placeholder` flag on /api/metrics is True while these
are in use, and the dashboard is required to render a visible banner when it is.

Phases 4-12 replace each of these readers with real artifact loads. The rule for
the rest of the build: a fixture may only ever be deleted, never quietly promoted
into a demo number.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .schemas import (
    AuditEntry,
    AuditListResponse,
    ClassifierMetrics,
    ConfusionMatrix,
    DatasetInfo,
    EvidenceItem,
    Explanation,
    FeedItem,
    FeedResponse,
    FinancialAssumption,
    FinancialImpact,
    GraphEdge,
    GraphNode,
    MetricsResponse,
    PolicyBand,
    PolicyConfigResponse,
    PolicyDecision,
    RingDetail,
    RingGraph,
    RingLevelMetrics,
    RingListResponse,
    RingSummary,
    Signal,
    SpikeListResponse,
    SpikeWindow,
    SplitInfo,
    Splits,
)

T0 = datetime(2026, 8, 1, 9, 0, tzinfo=timezone.utc)

PLACEHOLDER_CAVEATS = [
    "PLACEHOLDER DATA - Phase 1 scaffold. No model has been trained yet.",
    "These numbers are contract shape only and must never be shown as results.",
]

REAL_CAVEATS_TEMPLATE = [
    "Dataset is synthetic. Metrics measure recovery of injected patterns under "
    "documented generative assumptions, not real-world fraud performance.",
    "Graph and ring detection run retrospectively over a trailing window, not "
    "in the online scoring path. Transaction features are point-in-time correct.",
    "Ring-level metrics are reported on a small sample. Treat them as indicative.",
]


def _cm(tp: int, fp: int, tn: int, fn: int) -> ConfusionMatrix:
    return ConfusionMatrix(tp=tp, fp=fp, tn=tn, fn=fn)


def _split(n: int, start: datetime, days: int, rate: float) -> SplitInfo:
    return SplitInfo(
        n_rows=n, start=start, end=start + timedelta(days=days), positive_rate=rate
    )


def placeholder_metrics() -> MetricsResponse:
    return MetricsResponse(
        generated_at=T0,
        is_placeholder=True,
        dataset=DatasetInfo(
            seed=0,
            n_transactions=0,
            n_customers=0,
            n_rings_injected=0,
            n_hard_negative_clusters=0,
            split_method="time_aware",
            splits=Splits(
                train=_split(0, T0, 0, 0.0),
                validation=_split(0, T0, 0, 0.0),
                test=_split(0, T0, 0, 0.0),
            ),
        ),
        transaction_model=ClassifierMetrics(
            model_name="not_trained",
            precision=0.0,
            recall=0.0,
            f1=0.0,
            pr_auc=0.0,
            roc_auc=None,
            accuracy=None,
            decision_threshold=0.5,
            confusion_matrix=_cm(0, 0, 0, 0),
            support=0,
            positive_rate=0.0,
        ),
        baseline_model=ClassifierMetrics(
            model_name="not_trained",
            precision=0.0,
            recall=0.0,
            f1=0.0,
            pr_auc=0.0,
            roc_auc=None,
            accuracy=None,
            decision_threshold=0.5,
            confusion_matrix=_cm(0, 0, 0, 0),
            support=0,
            positive_rate=0.0,
        ),
        ring_model=RingLevelMetrics(
            precision=0.0,
            recall=0.0,
            f1=0.0,
            confusion_matrix=_cm(0, 0, 0, 0),
            n_true_rings=0,
            n_true_rings_detected=0,
            n_hard_negative_clusters=0,
            n_hard_negatives_flagged=0,
        ),
        financial=placeholder_financial(),
        caveats=PLACEHOLDER_CAVEATS,
    )


def placeholder_financial() -> FinancialImpact:
    return FinancialImpact(
        currency="INR",
        exposure_detected=0.0,
        prevented_loss=0.0,
        false_positive_cost=0.0,
        net_protected_value=0.0,
        assumptions=DEFAULT_ASSUMPTIONS,
    )


DEFAULT_ASSUMPTIONS: list[FinancialAssumption] = [
    FinancialAssumption(
        key="recovery_rate",
        label="Recovery rate",
        value=0.70,
        unit="ratio",
        description=(
            "Share of detected exposure genuinely preventable. Some fraud would "
            "have failed anyway; some flagged activity would have resolved "
            "legitimately. We do not claim 100%."
        ),
        adjustable=True,
        min=0.0,
        max=1.0,
    ),
    FinancialAssumption(
        key="cost_per_false_review",
        label="Cost per false manual review",
        value=120.0,
        unit="INR",
        description=(
            "Analyst time plus checkout friction when a legitimate transaction "
            "is escalated to human review."
        ),
        adjustable=True,
        min=0.0,
        max=2000.0,
    ),
    FinancialAssumption(
        key="cost_per_false_block_ratio",
        label="Lost-margin ratio on a false block",
        value=0.18,
        unit="ratio",
        description=(
            "Contribution margin forfeited when a legitimate order is stopped. "
            "Applied to the order value."
        ),
        adjustable=True,
        min=0.0,
        max=1.0,
    ),
]


def placeholder_feed(limit: int = 4) -> FeedResponse:
    rows = [
        ("TXN-PLACEHOLDER-1", 899.0, 12.0, "LOW", "ALLOW", None),
        ("TXN-PLACEHOLDER-2", 1499.0, 24.0, "LOW", "ALLOW", None),
        ("TXN-PLACEHOLDER-3", 4999.0, 58.0, "MEDIUM", "MONITOR", None),
        ("TXN-PLACEHOLDER-4", 12499.0, 81.0, "HIGH", "VERIFY", "AR-PLACEHOLDER"),
    ]
    items = [
        FeedItem(
            transaction_id=tid,
            timestamp=T0 + timedelta(minutes=i * 3),
            amount=amt,
            currency="INR",
            customer_id=f"CUST-PLACEHOLDER-{i}",
            payment_method="upi",
            product_category="electronics",
            risk_score=score,
            risk_level=level,
            policy_action=action,
            top_signals=[Signal(name="placeholder_signal", value=0.0, contribution=0.0)],
            ring_id=ring,
        )
        for i, (tid, amt, score, level, action, ring) in enumerate(rows)
    ]
    return FeedResponse(items=items[:limit], total=len(items))


def _placeholder_ring_summary() -> RingSummary:
    return RingSummary(
        ring_id="AR-PLACEHOLDER",
        risk_score=0.0,
        risk_level="LOW",
        confidence=0.0,
        n_accounts=0,
        n_devices=0,
        n_addresses=0,
        n_ips=0,
        n_transactions=0,
        financial_exposure=0.0,
        detected_at=T0,
        status="OPEN",
    )


def placeholder_rings() -> RingListResponse:
    return RingListResponse(rings=[_placeholder_ring_summary()], total=1)


def placeholder_ring_detail(ring_id: str) -> RingDetail:
    summary = _placeholder_ring_summary()
    summary.ring_id = ring_id
    return RingDetail(
        summary=summary,
        evidence=[
            EvidenceItem(
                code="PLACEHOLDER",
                kind="FACT",
                statement="Phase 1 scaffold. Ring detection lands in Phase 7.",
                observed_value=0.0,
                baseline_value=None,
                unit="none",
                weight=0.0,
            )
        ],
        graph=RingGraph(
            nodes=[
                GraphNode(id="CUST-PLACEHOLDER-0", type="customer", label="Account", risk_score=0.0),
                GraphNode(id="DEV-PLACEHOLDER-0", type="device", label="Device", risk_score=0.0),
            ],
            edges=[
                GraphEdge(
                    source="CUST-PLACEHOLDER-0",
                    target="DEV-PLACEHOLDER-0",
                    relation="used_device",
                    weight=1.0,
                )
            ],
            truncated=False,
        ),
        policy=PolicyDecision(
            recommended_action="ALLOW",
            requires_merchant_approval=False,
            reason="Placeholder. Policy engine lands in Phase 11.",
            policy_version="v0-placeholder",
            risk_band="0-39",
            amount_cap_applied=False,
        ),
        explanation=Explanation(
            source="deterministic",
            text="Phase 1 scaffold. No investigation has run.",
            facts=[],
            inferences=[],
            degraded=False,
        ),
        component_features={},
    )


def placeholder_spikes() -> SpikeListResponse:
    return SpikeListResponse(
        spikes=[
            SpikeWindow(
                spike_id="SPK-PLACEHOLDER",
                metric="transaction_rate",
                window_start=T0,
                window_end=T0 + timedelta(hours=1),
                observed=0.0,
                baseline=0.0,
                z_score=0.0,
                affected_transactions=[],
                estimated_exposure=0.0,
                contributing_signals=[],
            )
        ],
        total=1,
    )


def placeholder_audit() -> AuditListResponse:
    return AuditListResponse(
        entries=[
            AuditEntry(
                entry_id="AUD-PLACEHOLDER-1",
                timestamp=T0,
                event="scaffold_initialised",
                actor="system",
                subject_id="PHASE-1",
                input_summary="Contract frozen, stub API online.",
                decision="none",
                reason="Phase 1 has no decisions to make.",
                policy="v0-placeholder",
                result="ok",
            )
        ],
        total=1,
    )


POLICY_CONFIG = PolicyConfigResponse(
    policy_version="v1",
    bands=[
        PolicyBand(min_score=0, max_score=39.999, action="ALLOW"),
        PolicyBand(min_score=40, max_score=69.999, action="MONITOR"),
        PolicyBand(min_score=70, max_score=89.999, action="VERIFY"),
        PolicyBand(min_score=90, max_score=100, action="MANUAL_REVIEW"),
    ],
    max_auto_intervention_amount=5000.0,
    currency="INR",
)
