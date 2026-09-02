#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/3  spikes conform to the frozen contract =="
python3 -c "
import json
from jsonschema import Draft202012Validator, FormatChecker
c = json.load(open('contracts/api.schema.json'))
d = json.load(open('artifacts/spikes.json'))
schema = {'\$defs': c['\$defs'], **c['endpoints']['GET /api/spikes']}
errs = list(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(d))
assert not errs, errs[:2]
print(f\"   OK - {d['total']} alerts validate against the contract\")"

echo "== 2/3  alerts are attributable to abuse, not to marketing campaigns =="
python3 -c "
import json
r = json.load(open('data/labels/spike_report.json'))
assert r['driven_by_flash_sale'] == 0, f\"{r['driven_by_flash_sale']} alerts fired on legitimate flash sales\"
assert r['lift_over_chance'] >= 2.0, f\"lift {r['lift_over_chance']}x is barely above chance\"
print(f\"   OK - {r['n_spikes']} alerts: {r['driven_by_abuse']} abuse-driven, \"
      f\"{r['driven_by_flash_sale']} on flash sales, {r['unattributed']} unattributed\")
print(f\"        abuse share {r['mean_ring_share_in_alerts']:.1%} vs {r['abuse_base_rate']:.1%} by chance = {r['lift_over_chance']}x lift\")"

echo "== 3/3  baselines are point-in-time =="
python3 -c "
import numpy as np, pandas as pd
from ml.spike import robust_z
s = pd.Series(np.r_[np.ones(200), [500.0]], name='transaction_rate',
              index=pd.date_range('2026-01-01', periods=201, freq='h'))
z = robust_z(s)
assert z['baseline'].iloc[-1] == 1.0, 'the spike contaminated its own baseline'
assert z['z'].iloc[-1] > 100, 'an obvious spike was not detected'
print('   OK - a spike does not enter the statistic meant to detect it')"

echo
echo "PHASE 8 VERIFIED"
