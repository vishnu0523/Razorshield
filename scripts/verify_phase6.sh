#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/3  graph builds and does not collapse =="
python3 -c "
from ml.graph import build_graph, components, load_events, CUSTOMER_PREFIX, HUB_THRESHOLDS
ev = load_events()
b = build_graph(ev['transactions'])
n_cust = ev['transactions'].customer_id.nunique()
largest = max(sum(1 for n in c if n.startswith(CUSTOMER_PREFIX + ':'))
              for c in components(b, min_accounts=1))
assert largest < 0.05 * n_cust, f'giant component: {largest} of {n_cust}'
print(f'   OK - {b.graph.number_of_nodes():,} nodes, {b.graph.number_of_edges():,} edges, '
      f'largest component {largest} accounts of {n_cust:,}')
print('        hubs suppressed: ' + ', '.join(f'{k} {v}' for k, v in b.suppressed.items()))"

echo "== 2/3  clusters are recovered into single components =="
python3 -c "
from ml.inspect_graph import evaluate, MIN_RING_RECOVERY
r = evaluate()
assert r['passes'], f\"ring recovery {r['ring_recovery_rate']:.1%} below {MIN_RING_RECOVERY:.0%}\"
print(f\"   OK - {r['ring_recovery_rate']:.1%} of rings recovered, \"
      f\"{r['lookalike_recovery_rate']:.1%} of lookalikes (they must survive too)\")
print(f\"        {r['n_components']:,} components with 3+ accounts\")"

echo "== 3/3  the graph contributes signal the row model cannot express =="
python3 -c "
from ml.inspect_graph import evaluate
s = evaluate()['signal'].set_index('feature')
p = s.loc['bridging_entity_ratio', 'power']
assert p > 0.30, f'bridging_entity_ratio power {p} too weak'
top = s.index[0]
assert s.iloc[0]['power'] <= 0.90, f'{top} is a giveaway'
print(f'   OK - bridging_entity_ratio power {p:.3f} (component-only feature)')
print(f'        strongest is {top} at {s.iloc[0][\"power\"]:.3f}, below the giveaway bar')"

echo
echo "PHASE 6 VERIFIED"
