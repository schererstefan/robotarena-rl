#!/usr/bin/env node
// Regression test for turret-capture detection (bridge/turret-capture.mjs).
//
// sense.turrets is cone-gated and nearest-first, so list order changes as
// the learner moves. Capture detection must key ownership by turret
// position (turrets are static; SenseTurret has no id), never by list
// index. The old index-keyed check paid a spurious +1.0 `turret` reward
// when an already-owned turret moved to an index whose previous occupant
// was foe/neutral, and missed real captures hidden by the same shuffle.
//
// Run: node bridge/test-turret-capture.mjs   (exit 0 = pass)
// Not run by CI: the pure-tests job is Python-only and building the full
// bridge needs the sibling robotarena checkout. The wiring is guarded in
// CI by tests/test_turret_capture_contract.py instead.
import assert from 'node:assert/strict';

import {
    detectTurretCapture,
    seedTurretOwners,
    turretKey,
} from './turret-capture.mjs';

const ME = 0;
const FOE = 1;
const NEUTRAL = -1;
// Two static turret positions (engine: exactly 2 per arena; `open` pair).
const A = { x: 480, y: 162 };
const B = { x: 480, y: 378 };
const at = (pos, owner) => ({
    ...pos,
    state: owner === NEUTRAL ? 'disabled' : 'active',
    owner,
    progress: 0,
});

let cases = 0;
function check(name, actual, expected) {
    cases += 1;
    assert.equal(actual, expected, `${name}: got ${actual}, want ${expected}`);
    console.log(`ok ${cases} - ${name}`);
}

// Position is identity.
check('key is position', turretKey(at(A, ME)), '480,162');
check('key ignores owner', turretKey(at(A, ME)) === turretKey(at(A, FOE)), true);
check('key distinguishes turrets', turretKey(at(A, ME)) !== turretKey(at(B, ME)), true);

// Required regression 1: order swaps, ownership unchanged -> no capture.
// Baseline: A mine (nearest), B foe. Next sense B is nearest. The old
// index-keyed check compared A against B's slot (foe) and fired.
{
    const owners = new Map();
    seedTurretOwners(owners, [at(A, ME), at(B, FOE)]);
    check('swap with unchanged ownership pays nothing',
        detectTurretCapture(owners, [at(B, FOE), at(A, ME)], ME), false);
}

// Required regression 2: real flip on the same position -> capture, even
// when the flip coincides with a reorder.
{
    const owners = new Map();
    seedTurretOwners(owners, [at(A, FOE), at(B, NEUTRAL)]);
    check('flip foe->mine pays',
        detectTurretCapture(owners, [at(A, ME), at(B, NEUTRAL)], ME), true);
}
{
    const owners = new Map();
    seedTurretOwners(owners, [at(A, FOE), at(B, NEUTRAL)]);
    check('flip + reorder still pays',
        detectTurretCapture(owners, [at(B, NEUTRAL), at(A, ME)], ME), true);
}

// Match-start baseline (reset seeding): a turret already mine in the
// primed sense is never a fresh capture on the first step.
{
    const owners = new Map();
    seedTurretOwners(owners, [at(A, ME), at(B, NEUTRAL)]);
    check('owned at match start pays nothing',
        detectTurretCapture(owners, [at(A, ME), at(B, NEUTRAL)], ME), false);
}

// First-sighting semantics: turrets spawn neutral and capture is by
// presence (it can complete outside the tower cone), so a never-seen
// turret first seen as mine is credited once, then never again.
{
    const owners = new Map();
    check('first sight of own turret credits once',
        detectTurretCapture(owners, [at(A, ME)], ME), true);
    check('steady own turret does not re-pay',
        detectTurretCapture(owners, [at(A, ME)], ME), false);
}

// Foe capture pays nothing; recapture pays. Out-of-cone gaps keep memory.
{
    const owners = new Map();
    seedTurretOwners(owners, [at(A, NEUTRAL)]);
    check('foe capture pays nothing',
        detectTurretCapture(owners, [at(A, FOE)], ME), false);
    check('empty sense (out of cone) pays nothing',
        detectTurretCapture(owners, [], ME), false);
    check('flip while unseen pays on next sighting',
        detectTurretCapture(owners, [at(A, ME)], ME), true);
}

// Absent turrets field is tolerated (sense.turrets is optional).
{
    const owners = new Map();
    seedTurretOwners(owners, undefined);
    check('undefined turrets pays nothing',
        detectTurretCapture(owners, undefined, ME), false);
}

console.log(`TURRET CAPTURE OK (${cases} cases)`);
