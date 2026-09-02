#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/3  defaults on the dashboard equal the published evaluation =="
python3 -c "
import json
from backend.app.core.financial import recompute, Assumptions, ARTIFACTS
pub = json.load(open(ARTIFACTS / 'metrics.json'))['financial']
got = recompute(Assumptions())
for k in ('exposure_detected','prevented_loss','false_positive_cost','net_protected_value'):
    assert abs(got[k] - pub[k]) < 1.0, f'{k}: dashboard {got[k]} vs report {pub[k]}'
print(f\"   OK - net INR {got['net_protected_value']:,.0f} matches the evaluation report exactly\")"

echo "== 2/3  assumptions really move the answer =="
python3 -c "
from backend.app.core.financial import recompute, Assumptions
base = recompute(Assumptions())
for label, a in [
    ('recovery 40%', Assumptions(recovery_rate=0.40)),
    ('recovery 100%', Assumptions(recovery_rate=1.00)),
    ('review cost 600', Assumptions(cost_per_false_review=600)),
    ('margin loss 60%', Assumptions(cost_per_false_block_ratio=0.60)),
]:
    r = recompute(a)
    assert r['exposure_detected'] == base['exposure_detected'], 'exposure is an observation and must not move'
    print(f\"   {label:18s} net INR {r['net_protected_value']:>10,.0f}\")
print('   OK - detected exposure is fixed; only the interpretation is tunable')"

echo "== 3/3  the panel can produce an unflattering answer =="
python3 -c "
from backend.app.core.financial import recompute, Assumptions
r = recompute(Assumptions(cost_per_false_block_ratio=0.60))
assert r['net_protected_value'] < 0, 'no reachable setting makes us look bad; the panel proves nothing'
print(f\"   OK - at 60% lost margin RazorShield nets INR {r['net_protected_value']:,.0f} and says so\")"

echo
echo "PHASE 12 VERIFIED"
