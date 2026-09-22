#!/usr/bin/env bash
# Train: ./scripts/train.sh [--timesteps N] [--resume runs/<run>]
set -euo pipefail
cd "$(dirname "$0")/.."
./bridge/build.sh
exec .venv/bin/python -m rl.train "$@"
