#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/3  generator is reproducible from seed =="
python3 -c "
from ml.generate import Generator
a = Generator(seed=42).run()['transactions']
b = Generator(seed=42).run()['transactions']
assert a.equals(b), 'seed 42 is not reproducible'
print('   OK -', len(a), 'transactions, byte-identical across runs')"

echo "== 2/3  no label column is reachable from the event log =="
python3 -c "
from ml.generate import Generator
from ml import config as C
f = Generator(seed=42).run()
for name in ('customers','transactions','returns','refunds'):
    bad = set(f[name].columns) & C.FORBIDDEN_FEATURE_COLUMNS
    assert not bad, f'{name} leaks {bad}'
print('   OK - events carry no labels; labels live in', C.LABELS_DIR.name + '/')"

echo "== 3/3  hard negatives are genuinely hard =="
python3 -c "
from ml.generate import Generator
from ml.inspect_overlap import evaluate
r = evaluate(Generator(seed=42).run())
assert r['passes'], 'dataset is too easy'
print(f\"   OK - most separating feature '{r['most_separating_feature']}' power {r['max_separating_power']:.3f}\")
print(f\"        {r['n_ambiguous_features']}/10 features carry no standalone signal\")"

echo
echo "PHASE 2 VERIFIED"
