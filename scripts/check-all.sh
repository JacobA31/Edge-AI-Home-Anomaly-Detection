#!/usr/bin/env bash
# Run every check CI runs. Usage: ./scripts/check-all.sh
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"

echo "== backend";  (cd "$root/backend"  && ruff check . && pytest -q)
echo "== edge";     (cd "$root/edge"     && ruff check . && pytest -q)
echo "== ml";       (cd "$root"          && ruff check ml)
echo "== frontend"; (cd "$root/frontend" && npm run lint && npm run build)
echo "All checks passed."
