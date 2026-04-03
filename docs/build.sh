#!/usr/bin/env bash
# Build the westlean wheel and the Docusaurus documentation site.
#
# Usage: ./docs/build.sh
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Building westlean wheel..."
uv build
cp dist/westlean-0.1.0-py3-none-any.whl docs/static/

echo "==> Building documentation site..."
cd docs
npm run build

echo "==> Done. Serve with: cd docs && npm run serve"
