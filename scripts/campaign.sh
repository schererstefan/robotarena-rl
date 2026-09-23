#!/bin/bash
# Campaign: sequential continued-training runs until the champion bar is met.
#
# Champion bar (Stefan, Sep 2026; broadened Sep 22, 2026): >80% deterministic
# win rate vs EACH of wanderer, rusher, hunter, ghost, hunter-hc2, brawler,
# >=20 episodes per opponent, actual wins not rewards.
#
# Each iteration:
#   1. seed a fresh run dir with the current global-best checkpoint
#      (model.zip + vecnormalize.pkl)
#   2. train 8M steps with configs/ppo_continue.yaml (override via CONFIG env);
#      from iteration 6 onward (Stefan, Sep 22, 2026) the campaign switches to
#      configs/ppo_continue_hard_selfplay.yaml: 70% scripted / 30% VALIDATED
#      league self-play (snapshots admitted only after a deterministic 6-bot
#      roster eval shows a new run-best aggregate >= min_roster_rate).
#   3. STRICT deterministic eval: scripts/eval_roster.py, 20 eps/bot, TWICE,
#      on the run's best_roster.zip. This is the ONLY metric used for
#      cross-run checkpoint selection -- in-training evals overstated rusher
#      performance before, so they are not trusted here.
#   4. Elo tournament: python -m rl.evaluate, 20 eps/bot x17 bots (recorded).
#   5. score = min strict win rate over the 6 gate bots, averaged over the
#      two evals; new global best iff score strictly improves.
#
# Stop when: champion bar met, or MAX_RUNS iterations, or PATIENCE straight
# iterations with no improvement. Progress: runs/campaign/STATUS.md,
# runs/campaign/campaign.log, per-iteration rows in runs/campaign/results.jsonl.
# Global best checkpoint: runs/campaign/best/{model.zip,vecnormalize.pkl}.
#
# Usage: ./scripts/campaign.sh <seed_dir_with_best_roster> [max_runs]
#   seed_dir must contain best_roster.zip + best_roster_vecnormalize.pkl
set -euo pipefail
cd "$(dirname "$0")/.."

SEED_DIR="${1:?usage: scripts/campaign.sh <seed_dir> [max_runs]}"
MAX_RUNS="${2:-50}"
PATIENCE=10
TIMESTEPS=8000000
CONFIG="${CONFIG:-configs/ppo_continue.yaml}"  # override: CONFIG=configs/ppo_continue_hard.yaml ./scripts/campaign.sh ...
# Full 17-bot game roster (base 8 + 9 hardened champions), for tournament eval.
FULL_ROSTER="wanderer rusher hunter orbiter sniper brawler ghost turret wanderer-hc1 rusher-hc1 hunter-hc1 hunter-hc2 orbiter-hc1 turret-hc1 sniper-hc1 brawler-hc1 ghost-hc1"
# Gate bots for the strict champion-bar eval (broadened Sep 22, 2026 per Stefan:
# ghost, hunter-hc2, brawler join the original wanderer/rusher/hunter bar).
GATE_BOTS="wanderer rusher hunter ghost hunter-hc2 brawler"

CAMP="runs/campaign"
BEST="$CAMP/best"
mkdir -p "$BEST"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$CAMP/campaign.log"; }

strict_eval() {
  # $1=model.zip $2=vecnormalize.pkl $3=out_prefix
  # prints 6 win rates (w r h g hhc br) averaged over 2 strict evals; empty on failure
  local m="$1" v="$2" o="$3"
  # NOTE: eval_roster.py has no shebang; invoke via the venv python explicitly.
  .venv/bin/python scripts/eval_roster.py "$m" "$v" --bots $GATE_BOTS \
    --episodes 20 --arena open --out "${o}_1.json" > /dev/null 2>&1 || return 1
  .venv/bin/python scripts/eval_roster.py "$m" "$v" --bots $GATE_BOTS \
    --episodes 20 --arena open --out "${o}_2.json" > /dev/null 2>&1 || return 1
  .venv/bin/python - "$o" <<'PYEOF' 2>/dev/null || return 1
import json, sys
o = sys.argv[1]
acc = {}
for i in ("1", "2"):
    d = json.load(open(f"{o}_{i}.json"))
    for b, r in d["results"].items():
        acc[b] = acc.get(b, 0) + r["win_rate"] / 2
bots = ("wanderer", "rusher", "hunter", "ghost", "hunter-hc2", "brawler")
print(" ".join(f"{acc[b]:.4f}" for b in bots))
PYEOF
}

tourney_eval() {
  # $1=model.zip $2=vecnormalize.pkl $3=out.json -> prints agent elo, empty on failure
  local m="$1" v="$2" o="$3"
  .venv/bin/python -m rl.evaluate --model "$m" --vecnormalize "$v" \
    --bots $FULL_ROSTER --episodes 20 --out "$o" > /dev/null 2>&1 || return 1
  .venv/bin/python -c "import json; print(json.load(open('$o'))['agent_elo'])" 2>/dev/null || return 1
}

write_status() {
  # $1=iter $2=score $3..$8 = w r h g hhc br $9=elo ${10}=note
  cat > "$CAMP/STATUS.md" <<EOF
# Campaign status
- updated: $(date '+%F %T %Z')
- iteration: $1 / $MAX_RUNS
- best score (min strict win rate vs gate bots): $2
- best strict win rates: wanderer=$3 rusher=$4 hunter=$5 ghost=$6 hunter-hc2=$7 brawler=$8
- best agent Elo: $9
- note: ${10}
EOF
}

# ---- seed global best ----
[[ -f "$SEED_DIR/best_roster.zip" ]] || { echo "missing $SEED_DIR/best_roster.zip"; exit 1; }
[[ -f "$SEED_DIR/best_roster_vecnormalize.pkl" ]] || { echo "missing vecnormalize"; exit 1; }
cp "$SEED_DIR/best_roster.zip" "$BEST/model.zip"
cp "$SEED_DIR/best_roster_vecnormalize.pkl" "$BEST/vecnormalize.pkl"
log "campaign start: seeded global best from $SEED_DIR (max_runs=$MAX_RUNS)"

SEED_RATES=$(strict_eval "$BEST/model.zip" "$BEST/vecnormalize.pkl" "$CAMP/seed_eval") || SEED_RATES=""
SEED_ELO=$(tourney_eval "$BEST/model.zip" "$BEST/vecnormalize.pkl" "$CAMP/seed_elo.json") || SEED_ELO="?"
if [[ -n "$SEED_RATES" ]]; then
  read -r BEST_W BEST_R BEST_H BEST_G BEST_HHC BEST_BR <<< "$SEED_RATES"
  BEST_ELO="$SEED_ELO"
  BEST_SCORE=$(python3 -c "print(min($BEST_W,$BEST_R,$BEST_H,$BEST_G,$BEST_HHC,$BEST_BR))")
else
  BEST_W="?"; BEST_R="?"; BEST_H="?"; BEST_G="?"; BEST_HHC="?"; BEST_BR="?"; BEST_ELO="$SEED_ELO"; BEST_SCORE=0
fi
write_status 0 "$BEST_SCORE" "$BEST_W" "$BEST_R" "$BEST_H" "$BEST_G" "$BEST_HHC" "$BEST_BR" "$BEST_ELO" "seeded; training not started"
log "seed: score=$BEST_SCORE (w=$BEST_W r=$BEST_R h=$BEST_H g=$BEST_G hhc=$BEST_HHC br=$BEST_BR) elo=$BEST_ELO"

no_improve=0
stop_reason=""
for (( i=1; i<=MAX_RUNS; i++ )); do
  stamp=$(date +%Y%m%d-%H%M%S)
  rundir="runs/ppo_campaign_${i}_${stamp}"
  mkdir -p "$rundir/league"
  cp "$BEST/model.zip" "$rundir/model.zip"
  cp "$BEST/vecnormalize.pkl" "$rundir/vecnormalize.pkl"
  log "=== iteration $i/$MAX_RUNS: $rundir (seeded from best score=$BEST_SCORE) ==="

  # Validated self-play blend from iteration 6 (Stefan, Sep 22, 2026):
  # fights scripted-bot overfitting on the binding gate bots (brawler/ghost).
  CONFIG_ROUND="$CONFIG"
  if (( i >= 6 )) && [[ "$CONFIG" == "configs/ppo_continue_hard.yaml" ]]; then
    CONFIG_ROUND="configs/ppo_continue_hard_selfplay.yaml"
    log "iteration $i: switching to validated self-play blend ($CONFIG_ROUND)"
  fi

  rc=0
  ./scripts/train.sh --config "$CONFIG_ROUND" --timesteps "$TIMESTEPS" --resume "$rundir" \
    >> "$CAMP/campaign.log" 2>&1 || rc=$?
  if (( rc != 0 )) || [[ ! -f "$rundir/best_roster.zip" ]]; then
    log "WARN: iteration $i training failed (rc=$rc); keeping previous best"
    write_status "$i" "$BEST_SCORE" "$BEST_W" "$BEST_R" "$BEST_H" "$BEST_G" "$BEST_HHC" "$BEST_BR" "$BEST_ELO" \
      "iteration $i training failed; previous best kept"
    no_improve=$((no_improve+1))
    continue
  fi

  RATES=$(strict_eval "$rundir/best_roster.zip" "$rundir/best_roster_vecnormalize.pkl" \
    "$rundir/eval_strict") || RATES=""
  ELO=$(tourney_eval "$rundir/best_roster.zip" "$rundir/best_roster_vecnormalize.pkl" \
    "$rundir/elo.json") || ELO="?"
  if [[ -z "$RATES" ]]; then
    log "WARN: iteration $i eval failed; keeping previous best"
    no_improve=$((no_improve+1))
    continue
  fi
  read -r w r h g hhc br <<< "$RATES"
  score=$(python3 -c "print(min($w,$r,$h,$g,$hhc,$br))")
  improved=$(python3 -c "print(1 if $score > $BEST_SCORE + 1e-9 else 0)")
  if [[ "$improved" == "1" ]]; then
    cp "$rundir/best_roster.zip" "$BEST/model.zip"
    cp "$rundir/best_roster_vecnormalize.pkl" "$BEST/vecnormalize.pkl"
    cp "$rundir/eval_strict_1.json" "$BEST/eval_strict_1.json"
    cp "$rundir/eval_strict_2.json" "$BEST/eval_strict_2.json"
    cp "$rundir/elo.json" "$BEST/elo.json"
    BEST_W=$w; BEST_R=$r; BEST_H=$h; BEST_G=$g; BEST_HHC=$hhc; BEST_BR=$br; BEST_ELO=$ELO; BEST_SCORE=$score
    no_improve=0
    log "iter $i: NEW GLOBAL BEST score=$score (w=$w r=$r h=$h g=$g hhc=$hhc br=$br) elo=$ELO"
  else
    no_improve=$((no_improve+1))
    log "iter $i: score=$score (w=$w r=$r h=$h g=$g hhc=$hhc br=$br) elo=$ELO -- no improvement ($no_improve/$PATIENCE)"
  fi
  echo "{\"iter\": $i, \"rundir\": \"$rundir\", \"w\": $w, \"r\": $r, \"h\": $h, \"g\": $g, \"hhc\": $hhc, \"br\": $br, \"score\": $score, \"elo\": \"$ELO\", \"improved\": $improved}" \
    >> "$CAMP/results.jsonl"
  write_status "$i" "$BEST_SCORE" "$BEST_W" "$BEST_R" "$BEST_H" "$BEST_G" "$BEST_HHC" "$BEST_BR" "$BEST_ELO" \
    "last iter score=$score (w=$w r=$r h=$h g=$g hhc=$hhc br=$br) elo=$ELO"

  # CHAMPION GATE (strict): EVERY gate bot in BOTH independent 20-episode
  # strict evals must exceed 0.80. Averages are NOT used here -- a weak
  # reproduction must not hide behind a strong one. 0.80 exactly does not pass.
  champ=$(.venv/bin/python3 - "$rundir" <<'PYEOF'
import json, sys
ok = True
for i in ("1", "2"):
    d = json.load(open(f"{sys.argv[1]}/eval_strict_{i}.json"))
    for b in ("wanderer", "rusher", "hunter", "ghost", "hunter-hc2", "brawler"):
        if d["results"][b]["win_rate"] <= 0.8:
            ok = False
print(1 if ok else 0)
PYEOF
)
  if [[ "$champ" == "1" ]]; then
    stop_reason="CHAMPION: all of wanderer/rusher/hunter/ghost/hunter-hc2/brawler > 0.80 in BOTH strict runs at iteration $i"
    break
  fi
  if (( no_improve >= PATIENCE )); then
    stop_reason="no improvement for $PATIENCE straight iterations (best score=$BEST_SCORE)"
    break
  fi
done

[[ -z "$stop_reason" ]] && stop_reason="reached MAX_RUNS=$MAX_RUNS (best score=$BEST_SCORE)"
echo "$stop_reason" > "$CAMP/DONE"
write_status "$i" "$BEST_SCORE" "$BEST_W" "$BEST_R" "$BEST_H" "$BEST_G" "$BEST_HHC" "$BEST_BR" "$BEST_ELO" "FINISHED: $stop_reason"
log "CAMPAIGN DONE: $stop_reason"
