#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/3  the system works with no language model configured =="
python3 -c "
import json, os
from backend.app.adapters import llm
os.environ['RAZORSHIELD_DISABLE_LLM'] = '1'
ring = json.load(open('artifacts/rings.json'))['rings'][0]
e, log = llm.explain(ring, ring['explanation']['text'])
assert e.source == 'deterministic' and e.degraded is False, 'absent credentials must not read as a failure'
assert len(e.text) > 60, 'the deterministic text must stand on its own'
print('   OK - deterministic explanation is the default path, not a fallback')"

echo "== 2/3  an outage degrades without changing any decision =="
(uvicorn backend.app.main:app --port 8000 >/tmp/rs-api14.log 2>&1 &)
trap 'pkill -f "uvicorn backend.app.main" 2>/dev/null || true' EXIT
sleep 5
python3 -c "
import json, urllib.request
rid = json.load(urllib.request.urlopen('http://localhost:8000/api/rings?limit=1'))['rings'][0]['ring_id']
before = json.load(urllib.request.urlopen(f'http://localhost:8000/api/rings/{rid}'))
e = json.load(urllib.request.urlopen(f'http://localhost:8000/api/rings/{rid}/explanation?simulate_failure=true'))
after = json.load(urllib.request.urlopen(f'http://localhost:8000/api/rings/{rid}'))
assert e['degraded'] is True and e['source'] == 'deterministic'
assert after['summary']['risk_score'] == before['summary']['risk_score'], 'the score moved'
assert after['policy'] == before['policy'], 'the recommended action moved'
print(f\"   OK - model killed; {rid} keeps risk {after['summary']['risk_score']:.0f} and action {after['policy']['recommended_action']}\")"

echo "== 3/3  the model cannot state a verdict or invent a figure =="
python3 -c "
import json
from backend.app.adapters import llm
ring = json.load(open('artifacts/rings.json'))['rings'][0]
class Echo:
    name='t'
    def __init__(s,t): s.t=t
    def available(s): return True
    def complete(s,a,b): return s.t
for bad in ['You should block these accounts, they are fraudsters and this proves it.',
            'This group shows unusual behaviour worth about 987654 rupees in linked orders here.']:
    e,_ = llm.explain(ring, ring['explanation']['text'], provider=Echo(bad))
    assert e.source == 'deterministic', f'accepted unsafe output: {bad[:40]}'
prompt = llm.build_prompt(ring)
for banned in ('CUST-','TXN-','MANUAL_REVIEW','VERIFY'):
    assert banned not in prompt, f'prompt leaked {banned}'
print('   OK - verdicts and invented figures rejected; prompt carries no identifiers or actions')"

echo
echo "PHASE 14 VERIFIED"
