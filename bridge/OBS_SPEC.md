# Observation spec — bridge output vector

Fixed layout, 124 float32 dims, built Node-side in `bridge.template.ts`
(`buildObs`). All values are clamped; absent entities are zero-padded with
`present = 0`. Source of truth is the code; this doc mirrors it.

| Dims    | Contents |
| ------- | -------- |
| 0–16    | **self (17):** x/960, y/640, sin/cos(heading), sin/cos(tower), speed/(2.5·150), health/maxHealth, cooldown/24, charge 0–1, charged flag, dashCd/480, empCd/720, slowed flag, lastDamage amount/100, sin/cos(lastDamage bearing − heading) |
| 17–32   | **foes (2 nearest × 8):** present, dx/960, dy/640, dist/540, sin/cos(bearing − heading), speed/(2.5·150), health/maxHealth |
| 33–37   | **stale track (nearest × 5):** seenNow flag, dx/960, dy/640, age/600 ticks, dist/960 |
| 38–73   | **bullets (6 nearest × 6):** present, dx/960, dy/640, dist/540, closing/430, damage/25 |
| 74–77   | **walls (4):** left/960, right/960, top/640, bottom/640 |
| 78–81   | **zone (4):** distToSafety/960, inside flag, suddenDeathIn/11400, shrinking flag |
| 82–89   | **turrets (2 × 4):** active flag, owned-by-me flag, owned-by-foe flag, progress −1..1 |
| 90–113  | **pickup pads (4 × 6):** active flag, dx/960, dy/640, kind one-hot (amp/repair/overdrive) |
| 114–117 | **match (4):** killsYou/5, killsTeam/5, aliveFoes/3, tick/11400 |
| 118–123 | **last-tick events (6):** wasHit flag, **leadFeat sin** (sin of signed lead-bearing−tower err, nearest foe), gotKill flag, **leadFeat cos** (cos of signed lead-bearing−tower err, nearest foe), pickup flag, turret-captured flag |

Notes:

- `bearing − heading` is wrapped to [−π, π] before sin/cos (rotation-invariant).
- Event flags lag one physics tick: `SenseState.events` describes the step
  completed before the current tick's `update()` ran. Damage/kill reward
  components are computed from exact snapshot diffs instead, so this lag only
  affects the two reserved/pickup/turret indicator dims.
- `OBS_DIM` is asserted at runtime — the bridge throws if the builder drifts.
- Dims 119/121 (leadFeat) carry the bridge-computed first-order target lead: the foe's velocity (heading+speed from SenseState) is extrapolated over the bullet time-of-flight, and the signed error between the lead bearing and the tower angle is exposed as sin/cos (0/1 when no foe is visible). The policy cannot infer foe velocity from a single frame (obs has foe speed but not direction), so the bridge computes the lead directly. Saved weights stay valid: these dims were constant 0 before 2026-09-22.
- The exported policy (`rl/export_ts.py`) and the in-bridge policy opponent
  both consume exactly this layout; changing it invalidates saved weights.

## Action space

`MultiDiscrete([5, 5, 5, 2, 2, 2, 2])` → Intent:

- dims 0–2: index 0..4 → −1, −0.5, 0, 0.5, 1 for throttle / turn / towerTurn
- dims 3–6: index 1 → true for fire / charge / dash / emp
