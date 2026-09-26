#!/bin/bash
# Pluto-upgrade pipeline: smoke run -> full campaign (runs/campaign_pluto).
#
#  1. Short smoke training (1.5M steps, configs/ppo_pluto_smoke.yaml) to
#     validate the frame-stacked + strategy-conditioned architecture learns.
#  2. On success (best_roster.zip exists), launch the full campaign seeded
#     from the smoke run's best roster:
#       CAMP_DIR=runs/campaign_pluto CONFIG=configs/ppo_pluto_campaign.yaml
#
# Run in tmux on the mini: ./scripts/pluto_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")/.."

CAMP="runs/campaign_pluto"
mkdir -p "$CAMP"
LOG="$CAMP/pipeline.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "pipeline start: smoke training (configs/ppo_pluto_smoke.yaml)"
./scripts/train.sh --config configs/ppo_pluto_smoke.yaml 2>&1 | tee -a "$LOG"
log "smoke training finished"

SMOKE_DIR="$(ls -dt runs/ppo_pluto_smoke_* | head -1)"
log "smoke run dir: $SMOKE_DIR"
if [[ ! -f "$SMOKE_DIR/best_roster.zip" ]]; then
  log "SMOKE FAILED: no best_roster.zip in $SMOKE_DIR — campaign NOT launched"
  exit 1
fi

log "smoke OK — launching campaign_pluto (CONFIG=configs/ppo_pluto_campaign.yaml)"
CAMP_DIR="$CAMP" CONFIG="configs/ppo_pluto_campaign.yaml" \
  ./scripts/campaign.sh "$SMOKE_DIR" 50 2>&1 | tee -a "$LOG"
log "pipeline done"
