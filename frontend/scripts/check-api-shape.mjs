/**
 * Frontend contract consumption check.
 *
 * tests/test_contract.py proves the backend matches the schema. This proves the
 * dashboard only reads fields that actually arrive. Together they mean a field
 * cannot be renamed on one side and silently break the other.
 *
 * Usage: node scripts/check-api-shape.mjs [baseUrl]
 */

const base = process.argv[2] ?? "http://localhost:8000";

// Exactly the paths App.tsx, Ledger.tsx and Panels.tsx dereference.
const REQUIRED = {
  "/api/health": ["artifacts_loaded", "llm_available"],
  "/api/rings": ["total", "rings"],
  "/api/recovery/workflows": [
    "total", "caveat", "workflows",
  ],
  "/api/metrics": [
    "generated_at", "is_placeholder", "caveats",
    "financial.exposure_detected", "financial.prevented_loss",
    "financial.false_positive_cost", "financial.net_protected_value",
    "financial.assumptions",
    "transaction_model.model_name", "transaction_model.pr_auc",
    "transaction_model.precision", "transaction_model.recall",
    "transaction_model.f1", "transaction_model.accuracy",
    "transaction_model.positive_rate", "transaction_model.support",
    "transaction_model.confusion_matrix.tp", "transaction_model.confusion_matrix.fp",
    "transaction_model.confusion_matrix.fn", "transaction_model.confusion_matrix.tn",
    "baseline_model.model_name", "baseline_model.pr_auc",
    "ring_model.n_true_rings", "ring_model.n_true_rings_detected",
    "ring_model.n_hard_negative_clusters", "ring_model.n_hard_negatives_flagged",
    "ring_model.precision", "ring_model.recall",
    "dataset.seed", "dataset.n_transactions", "dataset.n_customers",
    "dataset.n_rings_injected", "dataset.n_hard_negative_clusters",
    "dataset.splits.train.n_rows", "dataset.splits.validation.n_rows",
    "dataset.splits.test.n_rows", "dataset.splits.test.positive_rate",
    "dataset.splits.test.end",
  ],
};

const ASSUMPTION_KEYS = [
  "recovery_rate", "cost_per_false_review", "cost_per_false_block_ratio",
];

const dig = (obj, path) =>
  path.split(".").reduce((o, k) => (o == null ? undefined : o[k]), obj);

// The investigation screen dereferences these on a single ring detail.
const RING_DETAIL_FIELDS = [
  "summary.ring_id", "summary.risk_score", "summary.risk_level",
  "summary.confidence", "summary.financial_exposure", "summary.n_accounts",
  "summary.n_devices", "summary.n_transactions",
  "evidence", "graph.nodes", "graph.edges", "graph.truncated",
  "policy.recommended_action", "policy.reason",
  "policy.requires_merchant_approval", "policy.amount_cap_applied",
  "policy.policy_version",
  "explanation.text", "explanation.source", "explanation.degraded",
];

let failures = 0;
for (const [path, fields] of Object.entries(REQUIRED)) {
  let body;
  try {
    const res = await fetch(`${base}${path}`);
    if (!res.ok) throw new Error(`status ${res.status}`);
    body = await res.json();
  } catch (err) {
    console.error(`  FAIL ${path} unreachable: ${err.message}`);
    failures++;
    continue;
  }
  for (const field of fields) {
    if (dig(body, field) === undefined) {
      console.error(`  FAIL ${path} is missing ${field}`);
      failures++;
    }
  }
  if (path === "/api/metrics") {
    const keys = body.financial.assumptions.map((a) => a.key);
    for (const k of ASSUMPTION_KEYS) {
      if (!keys.includes(k)) {
        console.error(`  FAIL assumption '${k}' absent; the ledger caption reads it`);
        failures++;
      }
    }
  }
}

// Ring detail: fetch the first ring and check the shape the case view reads.
try {
  const list = await (await fetch(`${base}/api/rings?limit=1`)).json();
  const first = list.rings?.[0];
  if (!first) {
    console.error("  FAIL /api/rings returned no rings to inspect");
    failures++;
  } else {
    const detail = await (
      await fetch(`${base}/api/rings/${encodeURIComponent(first.ring_id)}`)
    ).json();
    for (const field of RING_DETAIL_FIELDS) {
      if (dig(detail, field) === undefined) {
        console.error(`  FAIL /api/rings/{id} is missing ${field}`);
        failures++;
      }
    }
    for (const node of detail.graph?.nodes ?? []) {
      for (const key of ["id", "type", "label", "risk_score"]) {
        if (node[key] === undefined) {
          console.error(`  FAIL graph node missing ${key}`);
          failures++;
        }
      }
    }
  }
} catch (err) {
  console.error(`  FAIL ring detail unreachable: ${err.message}`);
  failures++;
}

// Recovery workflow: fetch one row and check the extension tab fields.
try {
  const body = await (await fetch(`${base}/api/recovery/workflows?limit=1`)).json();
  const first = body.workflows?.[0];
  if (first) {
    for (const field of [
      "workflow_id", "ring_id", "loss_type", "bounded_action",
      "recommended_step", "expected_protected_value",
      "requires_merchant_approval", "stopping_rule", "audit_subject_id",
    ]) {
      if (dig(first, field) === undefined) {
        console.error(`  FAIL /api/recovery/workflows row is missing ${field}`);
        failures++;
      }
    }
  }
} catch (err) {
  console.error(`  FAIL recovery workflows unreachable: ${err.message}`);
  failures++;
}

if (failures) {
  console.error(`\n${failures} field(s) the dashboard reads are missing.`);
  process.exit(1);
}
console.log("   OK - every field the dashboard reads is present in the live API");
