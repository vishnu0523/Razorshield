#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/3  ring detail conforms to the frozen contract =="
python3 -c "
import json
from jsonschema import Draft202012Validator, FormatChecker
c = json.load(open('contracts/api.schema.json'))
raw = json.load(open('artifacts/rings.json'))['rings']
schema = {'\$defs': c['\$defs'], **c['endpoints']['GET /api/rings/{ring_id}']}
v = Draft202012Validator(schema, format_checker=FormatChecker())
for r in raw:
    payload = {k: r[k] for k in ('summary','evidence','graph','policy','explanation','component_features')}
    errs = list(v.iter_errors(payload))
    assert not errs, (r['ring_id'], errs[0].message)
print(f'   OK - {len(raw)} ring details validate against the contract')"

echo "== 2/3  graphs are drawable and never hide an account =="
python3 -c "
import json
rings = json.load(open('artifacts/rings.json'))['rings']
for r in rings:
    g = r['graph']
    ids = {n['id'] for n in g['nodes']}
    accounts = {n for n in ids if n.startswith('c:')}
    assert len(accounts) == r['n_accounts'], f\"{r['ring_id']} hid an account\"
    assert len(g['nodes']) <= 40, f\"{r['ring_id']} too many nodes to read\"
    for e in g['edges']:
        assert e['source'] in ids and e['target'] in ids, 'dangling edge'
trunc = sum(1 for r in rings if r['graph']['truncated'])
print(f'   OK - every account rendered in all {len(rings)} graphs; {trunc} truncated entities, flagged in the UI')"

echo "== 3/3  every field the case view reads exists in the live API =="
(uvicorn backend.app.main:app --port 8000 >/tmp/rs-api9.log 2>&1 &)
trap 'pkill -f "uvicorn backend.app.main" 2>/dev/null || true' EXIT
sleep 5
node frontend/scripts/check-api-shape.mjs http://localhost:8000

echo
echo "PHASE 9 VERIFIED"
