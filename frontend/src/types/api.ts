/**
 * RazorShield API types - mirror of contracts/api.schema.json v1.0.0
 *
 * Hand-maintained against the contract. The backend is validated against the
 * same schema by tests/test_contract.py, so if this file is correct the two
 * sides cannot silently diverge.
 *
 * The frontend is built against the frozen contract rather than inferred from
 * whichever artifact happens to be present locally.
 */

export type RiskLevel = "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
export type PolicyAction = "ALLOW" | "MONITOR" | "VERIFY" | "MANUAL_REVIEW";

/** FACT is observed in the event log. INFERENCE comes from a model and carries
 *  uncertainty. These must render differently in the UI. */
export type EvidenceKind = "FACT" | "INFERENCE";

export type NodeType =
  | "customer" | "device" | "ip" | "address"
  | "order" | "payment" | "coupon" | "refund" | "return";

export type RingStatus = "OPEN" | "AWAITING_APPROVAL" | "APPROVED" | "DISMISSED";

export interface ConfusionMatrix { tp: number; fp: number; tn: number; fn: number }

export interface ClassifierMetrics {
  model_name: string;
  precision: number;
  recall: number;
  f1: number;
  pr_auc: number;
  roc_auc: number | null;
  /** Display only. Never headline it; the classes are imbalanced. */
  accuracy: number | null;
  decision_threshold: number;
  confusion_matrix: ConfusionMatrix;
  support: number;
  positive_rate: number;
}

export interface RingLevelMetrics {
  precision: number;
  recall: number;
  f1: number;
  confusion_matrix: ConfusionMatrix;
  n_true_rings: number;
  n_true_rings_detected: number;
  /** Legitimate clusters that structurally resemble rings. */
  n_hard_negative_clusters: number;
  /** The honesty number. Surface it prominently. */
  n_hard_negatives_flagged: number;
}

export interface FinancialAssumption {
  key: string;
  label: string;
  value: number;
  unit: "INR" | "ratio" | "count";
  description: string;
  adjustable: boolean;
  min: number | null;
  max: number | null;
}

export interface FinancialImpact {
  currency: "INR";
  exposure_detected: number;
  prevented_loss: number;
  false_positive_cost: number;
  /** May be negative. Render negatives honestly. */
  net_protected_value: number;
  assumptions: FinancialAssumption[];
}

export interface SplitInfo {
  n_rows: number;
  start: string;
  end: string;
  positive_rate: number;
}

export interface DatasetInfo {
  seed: number;
  n_transactions: number;
  n_customers: number;
  n_rings_injected: number;
  n_hard_negative_clusters: number;
  split_method: "time_aware" | "random";
  splits: { train: SplitInfo; validation: SplitInfo; test: SplitInfo };
}

export interface MetricsResponse {
  generated_at: string;
  /** When true the dashboard MUST show a visible placeholder banner. */
  is_placeholder: boolean;
  dataset: DatasetInfo;
  transaction_model: ClassifierMetrics;
  baseline_model: ClassifierMetrics;
  ring_model: RingLevelMetrics;
  /** The no-graph floor, reported so the graph's contribution is attributable. */
  ring_baseline_model?: RingLevelMetrics | null;
  financial: FinancialImpact;
  caveats: string[];
}

export interface Signal { name: string; value: number | string; contribution: number }

export interface FeedItem {
  transaction_id: string;
  timestamp: string;
  amount: number;
  currency: "INR";
  customer_id: string;
  payment_method: string;
  product_category: string;
  risk_score: number;
  risk_level: RiskLevel;
  policy_action: PolicyAction;
  top_signals: Signal[];
  ring_id: string | null;
}

export interface FeedResponse { items: FeedItem[]; total: number }

export interface EvidenceItem {
  code: string;
  kind: EvidenceKind;
  statement: string;
  observed_value: number | string;
  baseline_value: number | string | null;
  unit: string;
  weight: number;
}

export interface GraphNode {
  id: string;
  type: NodeType;
  label: string;
  risk_score: number;
  attributes?: Record<string, unknown>;
}

export interface GraphEdge {
  source: string;
  target: string;
  relation: string;
  weight: number;
}

export interface RingGraph {
  nodes: GraphNode[];
  edges: GraphEdge[];
  truncated: boolean;
}

export interface Explanation {
  source: "llm" | "deterministic";
  text: string;
  facts: string[];
  inferences: string[];
  /** True when the LLM was unavailable. Surface this in the UI. */
  degraded: boolean;
}

export interface PolicyDecision {
  recommended_action: PolicyAction;
  requires_merchant_approval: boolean;
  reason: string;
  policy_version: string;
  risk_band: string;
  amount_cap_applied: boolean;
}

export interface RingSummary {
  ring_id: string;
  risk_score: number;
  risk_level: RiskLevel;
  confidence: number;
  n_accounts: number;
  n_devices: number;
  n_addresses: number;
  n_ips: number;
  n_transactions: number;
  financial_exposure: number;
  detected_at: string;
  status: RingStatus;
}

export interface RingDetail {
  summary: RingSummary;
  evidence: EvidenceItem[];
  graph: RingGraph;
  policy: PolicyDecision;
  explanation: Explanation;
  component_features: Record<string, number>;
}

export interface RingListResponse { rings: RingSummary[]; total: number }

export type RecoveryLossType =
  | "refund_abuse"
  | "return_abuse"
  | "payment_failure_abuse"
  | "promo_or_account_abuse"
  | "coordinated_abuse";

export type RecoveryBoundedAction =
  | "MONITOR_ONLY"
  | "REQUEST_VERIFICATION"
  | "PREPARE_EVIDENCE"
  | "MANUAL_REVIEW";

export interface RecoveryWorkflow {
  workflow_id: string;
  ring_id: string;
  loss_type: RecoveryLossType;
  bounded_action: RecoveryBoundedAction;
  recommended_step: string;
  expected_protected_value: number;
  requires_merchant_approval: boolean;
  stopping_rule: string;
  audit_subject_id: string;
}

export interface RecoveryWorkflowListResponse {
  workflows: RecoveryWorkflow[];
  total: number;
  caveat: string;
}

export interface SpikeWindow {
  spike_id: string;
  metric: string;
  window_start: string;
  window_end: string;
  observed: number;
  baseline: number;
  z_score: number;
  affected_transactions: string[];
  estimated_exposure: number;
  contributing_signals: string[];
}

export interface SpikeListResponse { spikes: SpikeWindow[]; total: number }

export interface AuditEntry {
  entry_id: string;
  timestamp: string;
  event: string;
  actor: "system" | "model" | "policy_engine" | "llm" | "merchant";
  subject_id: string;
  input_summary: string;
  decision: string;
  reason: string;
  policy: string;
  result: string;
}

export interface AuditListResponse { entries: AuditEntry[]; total: number }

export interface PolicyBand { min_score: number; max_score: number; action: PolicyAction }

export interface PolicyConfigResponse {
  policy_version: string;
  bands: PolicyBand[];
  max_auto_intervention_amount: number;
  currency: "INR";
}

export interface MerchantDecision {
  ring_id: string;
  decision: "APPROVED" | "DISMISSED";
  recorded_at: string;
  audit_entry_id: string;
}

export interface ChainVerification {
  valid: boolean;
  checked: number;
  broken_at: string | null;
  detail: string;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  version: string;
  artifacts_loaded: boolean;
  llm_available: boolean;
  phase: number;
}

export interface SimulationStartResponse {
  run_id: string;
  total_phases: 9;
  seed: number;
}

export interface SimulationStep {
  phase: number;
  phase_name: string;
  narrative: string;
  feed_items: FeedItem[];
  ring: RingDetail | null;
  audit_entries: AuditEntry[];
  financial_delta: number | null;
  is_final: boolean;
}

/** Risk band -> action, mirroring backend policy. UI display only; the backend
 *  policy engine remains the single source of truth for actual decisions. */
export const RISK_LEVEL_ORDER: RiskLevel[] = ["LOW", "MEDIUM", "HIGH", "CRITICAL"];

const API_BASE = import.meta.env?.VITE_API_BASE ?? "http://localhost:8000";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json() as Promise<T>;
}

export const api = {
  health: () => get<HealthResponse>("/api/health"),
  metrics: () => get<MetricsResponse>("/api/metrics"),
  feed: (limit = 50) => get<FeedResponse>(`/api/feed?limit=${limit}`),
  rings: () => get<RingListResponse>("/api/rings"),
  ring: (id: string) => get<RingDetail>(`/api/rings/${encodeURIComponent(id)}`),
  recoveryWorkflows: () => get<RecoveryWorkflowListResponse>("/api/recovery/workflows"),
  spikes: () => get<SpikeListResponse>("/api/spikes"),
  audit: (subjectId?: string) =>
    get<AuditListResponse>(
      subjectId ? `/api/audit?subject_id=${encodeURIComponent(subjectId)}` : "/api/audit",
    ),
  auditVerify: () => get<ChainVerification>("/api/audit/verify"),
  policyConfig: () => get<PolicyConfigResponse>("/api/policy/config"),
  simulateStep: (phase: number) =>
    get<SimulationStep>(`/api/simulate/step?phase=${phase}`),
};
