#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/3  audit trail is populated and the chain verifies =="
python3 -c "
from backend.app.core import audit
from backend.app.core.seed import seed
audit.init_db(); seed()
r = audit.verify_chain()
assert r['valid'], f\"chain broken at {r['broken_at']}: {r['detail']}\"
assert r['checked'] > 0, 'audit log is empty'
print(f\"   OK - {r['checked']} entries, hash chain intact\")"

echo "== 2/3  tampering is detected and located =="
python3 -c "
import tempfile, os, importlib
from pathlib import Path
db = Path(tempfile.mkdtemp()) / 'a.db'
os.environ['DATABASE_URL'] = f'sqlite:///{db}'
from backend.app.core import audit as m
audit = importlib.reload(m); audit.init_db()
for i in range(4):
    audit.append(event=f'e{i}', actor='model', subject_id='AR-1',
                 input_summary='a', decision=f'risk {i}', reason='c')
assert audit.verify_chain()['valid']
from sqlalchemy import update
with audit.SessionLocal() as s:
    s.execute(update(audit.AuditRow).where(audit.AuditRow.seq==2).values(decision='risk 0')); s.commit()
r = audit.verify_chain()
assert not r['valid'] and r['broken_at'] == 'AUD-00000002'
print(f\"   OK - an edited row is caught and named ({r['broken_at']})\")"

echo "== 3/3  merchant decisions append, endpoints conform =="
(uvicorn backend.app.main:app --port 8000 >/tmp/rs-api10.log 2>&1 &)
trap 'pkill -f "uvicorn backend.app.main" 2>/dev/null || true' EXIT
sleep 5
python3 -c "
import json, urllib.request
from jsonschema import Draft202012Validator, FormatChecker
c = json.load(open('contracts/api.schema.json'))
def check(ep, payload):
    v = Draft202012Validator({'\$defs': c['\$defs'], **c['endpoints'][ep]}, format_checker=FormatChecker())
    errs = list(v.iter_errors(payload)); assert not errs, (ep, errs[0].message)

ring = json.load(urllib.request.urlopen('http://localhost:8000/api/rings?limit=1'))['rings'][0]['ring_id']
before = json.load(urllib.request.urlopen(f'http://localhost:8000/api/audit?subject_id={ring}'))['total']
req = urllib.request.Request(f'http://localhost:8000/api/rings/{ring}/decision',
    data=json.dumps({'decision':'APPROVED','note':'verified'}).encode(),
    headers={'Content-Type':'application/json'}, method='POST')
d = json.load(urllib.request.urlopen(req))
check('POST /api/rings/{ring_id}/decision', d)
after = json.load(urllib.request.urlopen(f'http://localhost:8000/api/audit?subject_id={ring}'))
check('GET /api/audit', after)
assert after['total'] == before + 1, 'decision did not append'
chain = json.load(urllib.request.urlopen('http://localhost:8000/api/audit/verify'))
check('GET /api/audit/verify', chain)
assert chain['valid']
print(f\"   OK - {ring} decision recorded as {d['audit_entry_id']}, trail grew {before} -> {after['total']}, chain still intact\")"

echo
echo "PHASE 10 VERIFIED"
