#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/4  artifacts exist and conform to the frozen contract =="
python3 -c "
import json
from jsonschema import Draft202012Validator, FormatChecker
c = json.load(open('contracts/api.schema.json'))
m = json.load(open('artifacts/metrics.json'))
schema = {'\$defs': c['\$defs'], **c['endpoints']['GET /api/metrics']}
errs = list(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(m))
assert not errs, errs[:3]
assert m['is_placeholder'] is False
print('   OK - artifacts/metrics.json validates against the contract')"

echo "== 2/4  held-out score is honest, not suspiciously perfect =="
python3 -c "
import json
from ml.train import LEAKAGE_ALARM_PR_AUC
m = json.load(open('artifacts/metrics.json'))
t, b = m['transaction_model'], m['baseline_model']
floor = t['positive_rate']
assert t['pr_auc'] <= LEAKAGE_ALARM_PR_AUC, f\"PR-AUC {t['pr_auc']} implies leakage\"
assert t['pr_auc'] > floor * 2, 'model barely beats guessing'
print(f\"   OK - PR-AUC {t['pr_auc']:.4f} vs random floor {floor:.4f} ({t['pr_auc']/floor:.1f}x lift)\")
print(f\"        baseline logistic {b['pr_auc']:.4f}, precision {t['precision']:.3f}, recall {t['recall']:.3f}\")"

echo "== 3/4  ring-level metrics are separate and adequately sampled =="
python3 -c "
import json
r = json.load(open('artifacts/metrics.json'))['ring_model']
assert r['n_true_rings'] >= 10 and r['n_hard_negative_clusters'] >= 10
print(f\"   OK - {r['n_true_rings_detected']}/{r['n_true_rings']} rings caught, \"
      f\"{r['n_hard_negatives_flagged']}/{r['n_hard_negative_clusters']} lookalikes wrongly accused\")"

echo "== 4/4  financial arithmetic reconciles =="
python3 -c "
import json
f = json.load(open('artifacts/metrics.json'))['financial']
assert abs(f['net_protected_value'] - (f['prevented_loss'] - f['false_positive_cost'])) < 1.0
assert f['prevented_loss'] <= f['exposure_detected'] + 1.0
assert len(f['assumptions']) == 3 and all(a['adjustable'] for a in f['assumptions'])
print(f\"   OK - net INR {f['net_protected_value']:,.0f} = prevented {f['prevented_loss']:,.0f} - FP cost {f['false_positive_cost']:,.0f}\")"

echo
echo "PHASE 4 VERIFIED"
