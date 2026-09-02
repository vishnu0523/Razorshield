#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/3  features build clean =="
python3 -c "
import numpy as np
from ml.generate import Generator
from ml.features import build_features, feature_matrix, FEATURE_COLUMNS
f = Generator(seed=42).run()
feat = build_features({k: f[k] for k in ('customers','transactions','returns','refunds')})
X = feature_matrix(feat).to_numpy(float)
assert np.isfinite(X).all(), 'non-finite values in the feature matrix'
print(f'   OK - {len(feat):,} rows x {len(FEATURE_COLUMNS)} features, all finite')"

echo "== 2/3  point-in-time correctness (recompute against truncated history) =="
python3 -c "
import numpy as np
from ml.generate import Generator
from ml.features import build_features, canonical_transactions, FEATURE_COLUMNS
f = Generator(seed=42).run()
ev = {k: f[k] for k in ('customers','transactions','returns','refunds')}
full = build_features(ev)
tx = canonical_transactions(ev['transactions'])
rng = np.random.default_rng(0)
probes = sorted(int(p) for p in rng.choice(np.arange(200, len(tx)), size=12, replace=False))
for i in probes:
    t = tx.loc[i, 'timestamp']
    trunc = {'customers': ev['customers'], 'transactions': tx.iloc[:i+1],
             'returns': ev['returns'][ev['returns'].timestamp <= t],
             'refunds': ev['refunds'][ev['refunds'].timestamp <= t]}
    r = build_features(trunc).iloc[-1]; a = full.iloc[i]
    for c in FEATURE_COLUMNS:
        assert np.isclose(float(a[c]), float(r[c]), rtol=1e-9, atol=1e-9), f'{c} at row {i} saw the future'
print(f'   OK - {len(probes)} probed rows match a truncated-history recomputation exactly')"

echo "== 3/3  no feature is a label proxy =="
python3 -c "
import numpy as np
from sklearn.metrics import roc_auc_score
from ml.generate import Generator
from ml.features import build_features, feature_matrix, FEATURE_COLUMNS
from ml import config as C
f = Generator(seed=42).run()
feat = build_features({k: f[k] for k in ('customers','transactions','returns','refunds')})
assert not (set(FEATURE_COLUMNS) & C.FORBIDDEN_FEATURE_COLUMNS)
y = f['transaction_labels'].set_index('transaction_id').loc[feat.transaction_id,'is_fraud'].to_numpy()
X = feature_matrix(feat)
worst, name = 0.0, ''
for c in FEATURE_COLUMNS:
    v = X[c].to_numpy(float)
    if np.ptp(v) == 0: continue
    p = abs(roc_auc_score(y, v) - 0.5) * 2
    if p > worst: worst, name = p, c
assert worst < 0.95, f'{name} is a label proxy (power {worst:.3f})'
print(f'   OK - strongest standalone feature is {name} at power {worst:.3f}')"

echo
echo "PHASE 3 VERIFIED"
