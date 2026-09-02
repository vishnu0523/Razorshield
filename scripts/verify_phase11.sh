#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/3  policy boundaries, both sides of every edge =="
python3 -c "
from backend.app.core.policy import decide, config, MAX_AUTO_INTERVENTION_AMOUNT
cases = [(39.999,'ALLOW'),(40,'MONITOR'),(69,'MONITOR'),(70,'VERIFY'),(89,'VERIFY'),(90,'MANUAL_REVIEW')]
for score, expected in cases:
    got = decide(float(score), 100.0).recommended_action
    assert got == expected, f'risk {score}: expected {expected}, got {got}'
for amount, capped in [(4999.0,False),(5000.0,False),(5001.0,True)]:
    d = decide(75.0, amount)
    assert d.amount_cap_applied is capped, f'amount {amount}: cap should be {capped}'
assert decide(100.0, 250000.0).requires_merchant_approval
for b in config()['bands']:
    for probe in (b['min_score'], b['max_score']):
        assert decide(probe, 100.0).recommended_action == b['action'], 'config disagrees with the engine'
print('   OK - 69/70/89/90 and 4999/5000/5001 all correct; advertised policy matches executed policy')
print(f'   OK - certainty does not bypass the INR {MAX_AUTO_INTERVENTION_AMOUNT:,.0f} cap')"

echo "== 2/3  one policy implementation, not two =="
python3 -c "
from backend.app.core.policy import decide
from ml.rings import policy_decision
for score, amount in [(69,100),(70,6000),(90,100),(75,5000),(75,5001)]:
    assert policy_decision(float(score), float(amount)) == decide(float(score), float(amount)).to_dict(), \
        f'ring policy diverges from the engine at {score}/{amount}'
print('   OK - ring detection delegates to the same engine; no second copy to drift')"

echo "== 3/3  fusion is reported honestly =="
python3 -c "
import json
f = json.load(open('artifacts/fusion.json'))
pr = f['held_out_pr_auc']
assert f['stacker_fit_on'] == 'validation split'
assert f['fusion_helps'] == (pr['fused'] > pr['transaction_model_alone']), 'verdict contradicts the numbers'
corr = f['sub_score_correlation']
worst = max(corr[a][b] for a in corr for b in corr[a] if a != b)
print(f\"   OK - fused {pr['fused']:.4f} vs transaction model alone {pr['transaction_model_alone']:.4f} ({pr['improvement']:+.4f})\")
print(f\"        verdict: fusion {'helps' if f['fusion_helps'] else 'does not help'}; peak sub-score correlation {worst:.2f}\")"

echo
echo "PHASE 11 VERIFIED"
