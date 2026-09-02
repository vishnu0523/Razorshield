#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/3  the demo ring actually demonstrates the claim =="
python3 -c "
from ml.simulate import build, INDIVIDUALLY_UNREMARKABLE_BELOW, COMPONENT_CRITICAL_AT
s = build()
ring = next(x['ring'] for x in s['steps'] if x['ring'])
feed = [i for x in s['steps'] for i in x['feed_items']]
peak = max(i['risk_score'] for i in feed)
assert ring['summary']['risk_score'] >= COMPONENT_CRITICAL_AT
assert peak < INDIVIDUALLY_UNREMARKABLE_BELOW, f'a transaction scored {peak}; it would have been caught alone'
print(f\"   OK - {s['rationale']}\")
print(f\"        {ring['summary']['n_accounts']} accounts, {ring['summary']['n_devices']} devices, \"
      f\"{ring['summary']['n_addresses']} addresses, INR {ring['summary']['financial_exposure']:,.0f} exposure\")"

echo "== 2/3  the script is deterministic and contract-shaped =="
python3 -c "
import json
from jsonschema import Draft202012Validator, FormatChecker
from ml.simulate import build
a, b = build(), build()
assert json.dumps(a['steps'], sort_keys=True) == json.dumps(b['steps'], sort_keys=True), 'script varies between runs'
c = json.load(open('contracts/api.schema.json'))
v = Draft202012Validator({'\$defs': c['\$defs'], **c['endpoints']['GET /api/simulate/step']}, format_checker=FormatChecker())
for step in a['steps']:
    errs = list(v.iter_errors(step))
    assert not errs, (step['phase'], errs[0].message)
print(f\"   OK - {len(a['steps'])} phases, identical across runs, all validate against the contract\")"

echo "== 3/3  approval precedes the result, and recovery rate is applied =="
python3 -c "
from ml.simulate import build
from ml.financial import DEFAULT_RECOVERY_RATE
s = build()
ring = next(x['ring'] for x in s['steps'] if x['ring'])
approval = next(x['phase'] for x in s['steps'] for e in x['audit_entries'] if e['actor']=='merchant')
result = next(x['phase'] for x in s['steps'] if x['financial_delta'])
assert approval < result, 'money is claimed before a person approved'
final = s['steps'][-1]['financial_delta']
assert final < ring['summary']['financial_exposure'], 'demo claims the full exposure was saved'
print(f\"   OK - approval at phase {approval}, result at phase {result}\")
print(f\"        INR {final:,.0f} claimed from INR {ring['summary']['financial_exposure']:,.0f} exposure at {DEFAULT_RECOVERY_RATE:.0%} recovery\")"

echo
echo "PHASE 13 VERIFIED"
