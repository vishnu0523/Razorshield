#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/3  dashboard type-checks and builds =="
# Check for a real installed package, not just the directory: a partial or
# failed install leaves node_modules present but unusable.
if [ ! -f frontend/node_modules/vite/package.json ]; then
  echo "   SKIP - dependencies not installed; run 'make install-ui' first"
else
  (cd frontend && npm run build >/tmp/rs-fe-build.log 2>&1) \
    && echo "   OK - tsc + vite build clean" \
    || { echo "   FAIL - see /tmp/rs-fe-build.log"; tail -20 /tmp/rs-fe-build.log; exit 1; }
fi

echo "== 2/3  API serves real metrics, not placeholders =="
(uvicorn backend.app.main:app --port 8000 >/tmp/rs-api.log 2>&1 &)
trap 'pkill -f "uvicorn backend.app.main" 2>/dev/null || true' EXIT
sleep 5
python3 -c "
import json, urllib.request
m = json.load(urllib.request.urlopen('http://localhost:8000/api/metrics'))
assert m['is_placeholder'] is False, 'API is still serving placeholders'
f = m['financial']
print(f\"   OK - net protected INR {f['net_protected_value']:,.0f}, \"
      f\"PR-AUC {m['transaction_model']['pr_auc']}\")"

echo "== 3/3  every field the dashboard reads exists in the live API =="
node frontend/scripts/check-api-shape.mjs http://localhost:8000

echo
echo "PHASE 5 VERIFIED"
