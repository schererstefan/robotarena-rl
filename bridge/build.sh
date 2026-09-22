#!/usr/bin/env bash
# Build the Node bridge: template -> esbuild bundle -> dist/bridge.mjs
set -euo pipefail

ROBOTARENA_PATH="${ROBOTARENA_PATH:-/Users/stefan/dev/robotarena}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ESBUILD="$ROBOTARENA_PATH/node_modules/.bin/esbuild"

if [[ ! -x "$ESBUILD" ]]; then
  echo "esbuild not found at $ESBUILD (need ROBOTARENA_PATH with node_modules)" >&2
  exit 1
fi

mkdir -p "$SCRIPT_DIR/dist"
sed "s#__ROBOTARENA_PATH__#${ROBOTARENA_PATH}#g" \
  "$SCRIPT_DIR/bridge.template.ts" > "$SCRIPT_DIR/dist/bridge.gen.ts"

"$ESBUILD" "$SCRIPT_DIR/dist/bridge.gen.ts" \
  --bundle --platform=node --format=esm \
  --outfile="$SCRIPT_DIR/dist/bridge.mjs" \
  --log-level=warning

echo "built $SCRIPT_DIR/dist/bridge.mjs"
