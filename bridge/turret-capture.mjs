// Turret-capture detection for the bridge reward components.
//
// Map turrets are static (fixed per-arena positions) and SenseTurret
// carries no id — only x, y, state, owner, progress. sense.turrets is
// cone-gated and nearest-first, so as the learner moves a different
// turret becomes index 0: list index is sense order, not identity.
// Ownership is therefore keyed by position (`x,y`), which is stable
// because the engine copies static layout coordinates verbatim.
//
// Semantics:
// - Owners persist across steps, including while a turret is outside
//   the cone (absent from sense.turrets): last seen owner stands.
// - A capture is "owner is now my team and the recorded owner was
//   not". Turrets spawn neutral (-1), so an unrecorded turret counts
//   as neutral: capture is by presence and can complete outside the
//   tower cone, so first seeing a turret as mine credits that capture
//   once. The reset path seeds match-start owners from the primed
//   sense (seedTurretOwners) so anything already owned at match start
//   is baseline and never pays as a fresh capture.
// - Regression coverage: bridge/test-turret-capture.mjs.

/** Stable identity of a static map turret. */
export function turretKey(t) {
    return `${t.x},${t.y}`;
}

/** Record current owners without judging captures (match-start baseline). */
export function seedTurretOwners(owners, turrets) {
    for (const t of turrets ?? []) owners.set(turretKey(t), t.owner);
}

/**
 * Diff one sense against recorded owners. Returns true iff some turret
 * became myTeam's since it was last recorded; updates records in place.
 */
export function detectTurretCapture(owners, turrets, myTeam) {
    let captured = false;
    for (const t of turrets ?? []) {
        const key = turretKey(t);
        const prev = owners.get(key) ?? -1;
        if (t.owner === myTeam && prev !== myTeam) captured = true;
        owners.set(key, t.owner);
    }
    return captured;
}
