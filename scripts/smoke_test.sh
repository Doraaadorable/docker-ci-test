#!/usr/bin/env bash
# Smoke-test a running container: health, a real prediction, and input validation.
# Usage: scripts/smoke_test.sh [base_url]
set -euo pipefail
URL="${1:-http://127.0.0.1:5050}"

for _ in $(seq 1 30); do
  curl -sf "$URL/health" >/dev/null && break
  sleep 1
done
curl -sf "$URL/health" | tee /dev/stderr | python3 -c 'import json,sys; assert json.load(sys.stdin)["status"] == "ok"'

sample=$(curl -sf "$URL/sample/0")
payload=$(python3 -c 'import json,sys; print(json.dumps({"pixels": json.loads(sys.argv[1])["pixels"]}))' "$sample")
curl -sf -X POST -H 'Content-Type: application/json' -d "$payload" "$URL/predict" \
  | python3 -c 'import json,sys; b=json.load(sys.stdin); assert 0 <= b["prediction"] <= 9, b'

code=$(curl -s -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d '{"pixels":[1,2]}' "$URL/predict")
[ "$code" = "400" ] || { echo "expected 400 for bad input, got $code"; exit 1; }

echo "Smoke test passed for $URL"
