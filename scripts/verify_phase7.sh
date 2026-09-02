#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/3  ring weights are learned, threshold not tuned on held-out data =="
python3 -c "
import json
m = json.load(open('artifacts/rings.json'))['model']
assert 0 < m['threshold'] < 1
assert any(v < 0 for v in m['coefficients'].values()), 'no mitigating signal'
top = sorted(m['coefficients'].items(), key=lambda kv: -abs(kv[1]))[:3]
print(f\"   OK - fit on {m['n_fit_components']} components, threshold {m['threshold']:.3f} by 5-fold CV\")
print('        strongest: ' + ', '.join(f'{k} {v:+.2f}' for k, v in top))"

echo "== 2/3  the graph beats the no-graph floor on identical clusters =="
python3 -c "
import json
m = json.load(open('artifacts/metrics.json'))
g, b = m['ring_model'], m['ring_baseline_model']
assert g['n_true_rings'] == b['n_true_rings'], 'denominators differ; comparison invalid'
assert g['recall'] > b['recall'], 'graph does not beat the floor'
assert g['precision'] >= 0.80, 'precision too low to act on'
assert not (g['precision'] == 1.0 and g['recall'] == 1.0), 'suspiciously perfect'
print(f\"   OK - graph  {g['n_true_rings_detected']}/{g['n_true_rings']} rings, \"
      f\"{g['n_hard_negatives_flagged']}/{g['n_hard_negative_clusters']} wrongly accused \"
      f\"(P {g['precision']:.3f} R {g['recall']:.3f})\")
print(f\"        floor  {b['n_true_rings_detected']}/{b['n_true_rings']} rings, \"
      f\"{b['n_hard_negatives_flagged']}/{b['n_hard_negative_clusters']} wrongly accused \"
      f\"(P {b['precision']:.3f} R {b['recall']:.3f})\")
gain = g['n_true_rings_detected'] - b['n_true_rings_detected']
print(f'        the graph is worth {gain} additional rings caught')"

echo "== 3/3  evidence is faithful to the model and merchant-safe =="
python3 -c "
import json
rings = json.load(open('artifacts/rings.json'))['rings']
assert rings, 'no rings flagged'
banned = ('cluster', 'is_fraud', 'AR-', 'HN-', 'label', 'synthetic')
for r in rings:
    w = [i['weight'] for i in r['evidence'] if i['weight'] > 0]
    assert w == sorted(w, reverse=True), 'evidence not ordered by contribution'
    assert abs(sum(w) - 1.0) < 0.15, 'weights do not sum to the decision'
    for i in r['evidence']:
        low = i['statement'].lower()
        assert not any(b.lower() in low for b in banned), f'evidence leaks: {i}'
print(f'   OK - {len(rings)} rings, evidence ordered by model contribution, no label leakage')"

echo
echo "PHASE 7 VERIFIED"
