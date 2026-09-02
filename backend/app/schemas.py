"""Pydantic mirrors of contracts/api.schema.json.

These are the runtime enforcement of the frozen contract. The JSON Schema is the
source of truth; tests/test_contract.py validates that what these models actually
serialise still conforms to it. If the two drift, the test fails loudly.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    """Base model: reject unknown fields so contract drift surfaces immediately."""

    model_config = ConfigDict(extra="forbid")


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class PolicyAction(str, Enum):
    ALLOW = "ALLOW"
    MONITOR = "MONITOR"
    VERIFY = "VERIFY"
    MANUAL_REVIEW = "MANUAL_REVIEW"


class EvidenceKind(str, Enum):
    FACT = "FACT"
    INFERENCE = "INFERENCE"


class NodeType(str, Enum):
    CUSTOMER = "customer"
    DEVICE = "device"
    IP = "ip"
    ADDRESS = "address"
    ORDER = "order"
    PAYMENT = "payment"
    COUPON = "coupon"
    REFUND = "refund"
    RETURN = "return"


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


class ConfusionMatrix(Strict):
    tp: int = Field(ge=0)
    fp: int = Field(ge=0)
    tn: int = Field(ge=0)
    fn: int = Field(ge=0)


class ClassifierMetrics(Strict):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_name: str
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)
    pr_auc: float = Field(ge=0, le=1)
    roc_auc: float | None = Field(default=None, ge=0, le=1)
    accuracy: float | None = Field(default=None, ge=0, le=1)
    decision_threshold: float = Field(ge=0, le=1)
    confusion_matrix: ConfusionMatrix
    support: int = Field(ge=0)
    positive_rate: float = Field(ge=0, le=1)


class RingLevelMetrics(Strict):
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)
    confusion_matrix: ConfusionMatrix
    n_true_rings: int = Field(ge=0)
    n_true_rings_detected: int = Field(ge=0)
    n_hard_negative_clusters: int = Field(ge=0)
    n_hard_negatives_flagged: int = Field(ge=0)


class FinancialAssumption(Strict):
    key: str
    label: str
    value: float
    unit: Literal["INR", "ratio", "count"]
    description: str
    adjustable: bool
    min: float | None = None
    max: float | None = None


class FinancialImpact(Strict):
    currency: Literal["INR"] = "INR"
    exposure_detected: float = Field(ge=0)
    prevented_loss: float = Field(ge=0)
    false_positive_cost: float = Field(ge=0)
    net_protected_value: float
    assumptions: list[FinancialAssumption] = Field(min_length=1)


class SplitInfo(Strict):
    n_rows: int = Field(ge=0)
    start: datetime
    end: datetime
    positive_rate: float = Field(ge=0, le=1)


class Splits(Strict):
    train: SplitInfo
    validation: SplitInfo
    test: SplitInfo


class DatasetInfo(Strict):
    seed: int
    n_transactions: int = Field(ge=0)
    n_customers: int = Field(ge=0)
    n_rings_injected: int = Field(ge=0)
    n_hard_negative_clusters: int = Field(ge=0)
    split_method: Literal["time_aware", "random"]
    splits: Splits


class MetricsResponse(Strict):
    generated_at: datetime
    is_placeholder: bool
    dataset: DatasetInfo
    transaction_model: ClassifierMetrics
    baseline_model: ClassifierMetrics
    ring_model: RingLevelMetrics
    ring_baseline_model: RingLevelMetrics | None = None
    financial: FinancialImpact
    caveats: list[str] = Field(min_length=1)


# --------------------------------------------------------------------------- #
# Feed
# --------------------------------------------------------------------------- #


class Signal(Strict):
    name: str
    value: float | str
    contribution: float


class FeedItem(Strict):
    transaction_id: str
    timestamp: datetime
    amount: float = Field(ge=0)
    currency: Literal["INR"] = "INR"
    customer_id: str
    payment_method: str
    product_category: str
    risk_score: float = Field(ge=0, le=100)
    risk_level: RiskLevel
    policy_action: PolicyAction
    top_signals: list[Signal] = Field(max_length=5)
    ring_id: str | None = None


class FeedResponse(Strict):
    items: list[FeedItem]
    total: int = Field(ge=0)


# --------------------------------------------------------------------------- #
# Rings
# --------------------------------------------------------------------------- #


class EvidenceItem(Strict):
    code: str
    kind: EvidenceKind
    statement: str
    observed_value: float | str
    baseline_value: float | str | None = None
    unit: str
    weight: float = Field(ge=0, le=1)


class GraphNode(Strict):
    id: str
    type: NodeType
    label: str
    risk_score: float = Field(ge=0, le=100)
    attributes: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(Strict):
    source: str
    target: str
    relation: str
    weight: float = Field(ge=0)


class RingGraph(Strict):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    truncated: bool = False


class Explanation(Strict):
    source: Literal["llm", "deterministic"]
    text: str
    facts: list[str]
    inferences: list[str]
    degraded: bool


class PolicyDecision(Strict):
    recommended_action: PolicyAction
    requires_merchant_approval: bool
    reason: str
    policy_version: str
    risk_band: str
    amount_cap_applied: bool


class RingSummary(Strict):
    ring_id: str
    risk_score: float = Field(ge=0, le=100)
    risk_level: RiskLevel
    confidence: float = Field(ge=0, le=1)
    n_accounts: int = Field(ge=0)
    n_devices: int = Field(ge=0)
    n_addresses: int = Field(ge=0)
    n_ips: int = Field(ge=0)
    n_transactions: int = Field(ge=0)
    financial_exposure: float = Field(ge=0)
    detected_at: datetime
    status: Literal["OPEN", "AWAITING_APPROVAL", "APPROVED", "DISMISSED"]


class RingDetail(Strict):
    summary: RingSummary
    evidence: list[EvidenceItem]
    graph: RingGraph
    policy: PolicyDecision
    explanation: Explanation
    component_features: dict[str, float]


class RingListResponse(Strict):
    rings: list[RingSummary]
    total: int = Field(ge=0)


class RecoveryWorkflow(Strict):
    workflow_id: str
    ring_id: str
    loss_type: Literal[
        "refund_abuse",
        "return_abuse",
        "payment_failure_abuse",
        "promo_or_account_abuse",
        "coordinated_abuse",
    ]
    bounded_action: Literal[
        "MONITOR_ONLY",
        "REQUEST_VERIFICATION",
        "PREPARE_EVIDENCE",
        "MANUAL_REVIEW",
    ]
    recommended_step: str
    expected_protected_value: float = Field(ge=0)
    requires_merchant_approval: bool
    stopping_rule: str
    audit_subject_id: str


class RecoveryWorkflowListResponse(Strict):
    workflows: list[RecoveryWorkflow]
    total: int = Field(ge=0)
    caveat: str


# --------------------------------------------------------------------------- #
# Spikes, audit, policy, simulation
# --------------------------------------------------------------------------- #


class SpikeWindow(Strict):
    spike_id: str
    metric: str
    window_start: datetime
    window_end: datetime
    observed: float
    baseline: float
    z_score: float
    affected_transactions: list[str]
    estimated_exposure: float = Field(ge=0)
    contributing_signals: list[str]


class SpikeListResponse(Strict):
    spikes: list[SpikeWindow]
    total: int = Field(ge=0)


class AuditEntry(Strict):
    entry_id: str
    timestamp: datetime
    event: str
    actor: Literal["system", "model", "policy_engine", "llm", "merchant"]
    subject_id: str
    input_summary: str
    decision: str
    reason: str
    policy: str
    result: str


class AuditListResponse(Strict):
    entries: list[AuditEntry]
    total: int = Field(ge=0)


class PolicyBand(Strict):
    min_score: float
    max_score: float
    action: PolicyAction


class PolicyConfigResponse(Strict):
    policy_version: str
    bands: list[PolicyBand] = Field(min_length=4)
    max_auto_intervention_amount: float = Field(ge=0)
    currency: Literal["INR"] = "INR"


class MerchantDecision(Strict):
    ring_id: str
    decision: Literal["APPROVED", "DISMISSED"]
    recorded_at: datetime
    audit_entry_id: str


class DecisionRequest(Strict):
    decision: Literal["APPROVED", "DISMISSED"]
    note: str = Field(default="", max_length=500)


class ChainVerification(Strict):
    valid: bool
    checked: int = Field(ge=0)
    broken_at: str | None = None
    detail: str


class HealthResponse(Strict):
    status: Literal["ok", "degraded"]
    version: str
    artifacts_loaded: bool
    llm_available: bool
    phase: int


class RecomputeRequest(Strict):
    """Merchant-adjustable assumptions. Sent from the dashboard sliders."""

    recovery_rate: float = Field(default=0.70, ge=0, le=1)
    cost_per_false_review: float = Field(default=120.0, ge=0)
    cost_per_false_block_ratio: float = Field(default=0.18, ge=0, le=1)


class SimulationStartResponse(Strict):
    run_id: str
    total_phases: Literal[9] = 9
    seed: int


class SimulationStep(Strict):
    phase: int = Field(ge=1, le=9)
    phase_name: str
    narrative: str
    feed_items: list[FeedItem]
    ring: RingDetail | None = None
    audit_entries: list[AuditEntry]
    financial_delta: float | None = None
    is_final: bool
