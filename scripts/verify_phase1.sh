#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/4  contract parses as valid JSON Schema =="
python3 - <<'PY'
import json
from jsonschema import Draft202012Validator
c = json.load(open("contracts/api.schema.json"))
Draft202012Validator.check_schema({"$defs": c["$defs"], "type": "object"})
for name, sub in c["endpoints"].items():
    Draft202012Validator.check_schema({"$defs": c["$defs"], **sub})
print(f"   OK - {len(c['endpoints'])} endpoints, {len(c['$defs'])} shared definitions")
PY

echo "== 2/4  app imports and every route is registered =="
python3 -c "
from backend.app.main import app
routes = sorted(r.path for r in app.routes if r.path.startswith('/api/'))
print('   OK -', len(routes), 'routes')
[print('     ', r) for r in routes]"

echo "== 3/4  contract conformance suite =="
pytest tests/ -q

echo "== 4/4  secrets hygiene =="
if [ -f .env ]; then echo "   NOTE - .env exists; confirm it is gitignored"; else echo "   OK - no .env present"; fi
LIVE_PREFIX="rzp_""live_"
if grep -rIl --exclude-dir=node_modules --exclude-dir=.git --exclude-dir=scripts -F "$LIVE_PREFIX" . >/dev/null 2>&1; then
  echo "   FAIL - live Razorpay key found in:"
  grep -rIl --exclude-dir=node_modules --exclude-dir=.git --exclude-dir=scripts -F "$LIVE_PREFIX" .
  exit 1
else
  echo "   OK - no live Razorpay keys"
fi

echo
echo "PHASE 1 VERIFIED"
