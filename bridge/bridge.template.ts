// RobotArena RL bridge — stdio JSONL server.
// Bundled with esbuild (see build.sh); __ROBOTARENA_PATH__ is replaced at build time.
//
// Protocol (one JSON object per line):
//   -> {"cmd":"reset","seed":1234,"opponent":{"bot":"hunter"},"arena":"open","loadout":{...}}
//   <- {"type":"reset","obs":[...],"info":{...}}
//   -> {"cmd":"step","intent":{"throttle":0.5,"turn":0,"towerTurn":0,"fire":true,...}}
//   <- {"type":"step","obs":[...],"reward":0.12,"components":{...},"terminated":false,"truncated":false,"info":{...}}
//
// Opponent spec is either {"bot":"<registry id>"} or
// {"policy":{"weights":"/path/to/weights.json","deterministic":false}}.
// The policy controller runs the exported MLP forward pass in plain TS
// (shared logic with export/robot_policy_template.ts).

import { Match } from '__ROBOTARENA_PATH__/src/sim/engine';
import { getRobot } from '__ROBOTARENA_PATH__/src/robots/registry';
import {
    ARENA_WIDTH,
    ARENA_HEIGHT,
    SENSOR_RANGE,
    MAX_TICKS_TOTAL,
    MAX_SPEED,
    GUN_COOLDOWN_TICKS,
    BULLET_SPEED,
    DASH_COOLDOWN_TICKS,
    EMP_COOLDOWN_TICKS,
} from '__ROBOTARENA_PATH__/src/sim/constants';
import type {
    Intent,
    RobotController,
    SenseState,
} from '__ROBOTARENA_PATH__/src/sim/types';
import * as fs from 'node:fs';
import * as readline from 'node:readline';

// ---------------------------------------------------------------------------
// Observation spec — 124 dims. Full layout documented in OBS_SPEC.md.
// ---------------------------------------------------------------------------
export const OBS_DIM = 124;
export const ACTION_DIMS = [5, 5, 5, 2, 2, 2, 2];

const clamp = (v: number, lo: number, hi: number): number =>
    Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : 0;
const normAngle = (a: number): number => {
    let x = a % (Math.PI * 2);
    if (x > Math.PI) x -= Math.PI * 2;
    if (x < -Math.PI) x += Math.PI * 2;
    return x;
};

interface ObsCtx {
    maxHealth: number;
    myTeam: 0 | 1;
}


/**
 * First-order lead solution for the nearest visible foe.
 * Returns sin/cos of the SIGNED lead error (lead bearing minus tower angle)
 * plus the unsigned error in radians. The agent cannot infer foe velocity
 * from a single frame (obs carries speed but not velocity direction), so the
 * bridge computes the lead directly; the policy just drives these features
 * to sin=0, cos=1. err is -1 when no foe is visible.
 */
function leadError(sense: SenseState): { sin: number; cos: number; err: number } {
    const s = sense.self;
    const towerAngle = s.tower as number;
    let best: { sin: number; cos: number; err: number } | null = null;
    for (const f of sense.foes ?? []) {
        if (typeof f.bearing !== "number" || !isFinite(f.bearing)) continue;
        const vx = (f.speed ?? 0) * Math.cos(f.heading ?? 0);
        const vy = (f.speed ?? 0) * Math.sin(f.heading ?? 0);
        // Two fixed-point iterations on time-of-flight; plenty at 60Hz.
        let tof = (f.distance ?? 0) / BULLET_SPEED;
        for (let i = 0; i < 2; i += 1) {
            const px = f.x + vx * tof;
            const py = f.y + vy * tof;
            tof = Math.hypot(px - s.x, py - s.y) / BULLET_SPEED;
        }
        const px = f.x + vx * tof;
        const py = f.y + vy * tof;
        const leadBearing = Math.atan2(py - s.y, px - s.x);
        const signed = normAngle(leadBearing - towerAngle);
        const err = Math.abs(signed);
        if (best === null || err < best.err) {
            best = { sin: Math.sin(signed), cos: Math.cos(signed), err };
        }
    }
    return best ?? { sin: 0, cos: 1, err: -1 };
}

function buildObs(sense: SenseState, ctx: ObsCtx): number[] {
    const o: number[] = [];
    const s = sense.self;
    const hx = s.x;
    const hy = s.y;

    // -- self (17) --
    o.push(
        clamp(hx / ARENA_WIDTH, 0, 1),
        clamp(hy / ARENA_HEIGHT, 0, 1),
        Math.sin(s.heading), Math.cos(s.heading),
        Math.sin(s.tower), Math.cos(s.tower),
        clamp(s.speed / (MAX_SPEED * 2.5), -1, 1),
        clamp(s.health / ctx.maxHealth, 0, 1),
        clamp(s.cooldown / GUN_COOLDOWN_TICKS, 0, 2),
        clamp(s.charge ?? 0, 0, 1),
        s.charged ? 1 : 0,
        clamp((s.dashCd ?? 0) / DASH_COOLDOWN_TICKS, 0, 1),
        clamp((s.empCd ?? 0) / EMP_COOLDOWN_TICKS, 0, 1),
        s.slowed ? 1 : 0,
        s.lastDamage ? clamp(s.lastDamage.amount / 100, 0, 2) : 0,
        s.lastDamage ? Math.sin(normAngle(s.lastDamage.bearing - s.heading)) : 0,
        s.lastDamage ? Math.cos(normAngle(s.lastDamage.bearing - s.heading)) : 0,
    );

    // -- foes: 2 nearest x8 (16) --
    const foes = (sense.foes ?? []).slice(0, 2);
    for (let i = 0; i < 2; i += 1) {
        const f = foes[i];
        if (!f) { o.push(0, 0, 0, 0, 0, 0, 0, 0); continue; }
        const rel = normAngle(f.bearing - s.heading);
        o.push(
            1,
            clamp((f.x - hx) / ARENA_WIDTH, -1, 1),
            clamp((f.y - hy) / ARENA_HEIGHT, -1, 1),
            clamp(f.distance / SENSOR_RANGE, 0, 2),
            Math.sin(rel), Math.cos(rel),
            clamp((f.speed ?? 0) / (MAX_SPEED * 2.5), -1, 1),
            clamp((f.health ?? 0) / ctx.maxHealth, 0, 1),
        );
    }

    // -- stale track: nearest x5 (5) --
    const tracks = (sense.tracks ?? []).slice().sort((a, b) =>
        (Math.hypot(a.x - hx, a.y - hy) - Math.hypot(b.x - hx, b.y - hy)));
    const tr = tracks[0];
    if (tr) {
        const b = Math.atan2(tr.y - hy, tr.x - hx);
        const rel = normAngle(b - s.heading);
        void rel;
        o.push(
            tr.seenNow ? 1 : 0,
            clamp((tr.x - hx) / ARENA_WIDTH, -1, 1),
            clamp((tr.y - hy) / ARENA_HEIGHT, -1, 1),
            clamp((sense.tick - (tr.lastSeenTick ?? sense.tick)) / 600, 0, 2),
            clamp(Math.hypot(tr.x - hx, tr.y - hy) / ARENA_WIDTH, 0, 2),
        );
    } else {
        o.push(0, 0, 0, 0, 0);
    }

    // -- bullets: 6 nearest x6 (36) --
    const bullets = (sense.bullets ?? []).slice(0, 6);
    for (let i = 0; i < 6; i += 1) {
        const b = bullets[i];
        if (!b) { o.push(0, 0, 0, 0, 0, 0); continue; }
        o.push(
            1,
            clamp((b.x - hx) / ARENA_WIDTH, -1, 1),
            clamp((b.y - hy) / ARENA_HEIGHT, -1, 1),
            clamp(b.distance / SENSOR_RANGE, 0, 2),
            clamp((b.closing ?? 0) / 430, -2, 2),
            clamp((b.damage ?? 0) / 25, 0, 2),
        );
    }

    // -- walls (4) --
    const w = sense.walls;
    o.push(
        clamp((w?.left ?? 0) / ARENA_WIDTH, 0, 1),
        clamp((w?.right ?? 0) / ARENA_WIDTH, 0, 1),
        clamp((w?.top ?? 0) / ARENA_HEIGHT, 0, 1),
        clamp((w?.bottom ?? 0) / ARENA_HEIGHT, 0, 1),
    );

    // -- zone (4) --
    const z = sense.zone;
    o.push(
        clamp((z?.distToSafety ?? 0) / ARENA_WIDTH, 0, 2),
        z?.inside ? 1 : 0,
        clamp((z?.suddenDeathIn ?? 0) / MAX_TICKS_TOTAL, 0, 1),
        z?.phase === 'shrinking' ? 1 : 0,
    );

    // -- turrets x4 (8) --
    const turrets = sense.turrets ?? [];
    for (let i = 0; i < 2; i += 1) {
        const t = turrets[i];
        if (!t) { o.push(0, 0, 0, 0); continue; }
        o.push(
            t.state === 'active' ? 1 : 0,
            t.owner === ctx.myTeam ? 1 : 0,
            t.owner !== -1 && t.owner !== ctx.myTeam ? 1 : 0,
            clamp(t.progress ?? 0, -1, 1),
        );
    }

    // -- pads x6 (24) --
    const pads = sense.pickups ?? [];
    for (let i = 0; i < 4; i += 1) {
        const p = pads[i];
        if (!p) { o.push(0, 0, 0, 0, 0, 0); continue; }
        o.push(
            p.active ? 1 : 0,
            clamp((p.x - hx) / ARENA_WIDTH, -1, 1),
            clamp((p.y - hy) / ARENA_HEIGHT, -1, 1),
            p.kind === 'amp' ? 1 : 0,
            p.kind === 'repair' ? 1 : 0,
            p.kind === 'overdrive' ? 1 : 0,
        );
    }

    // -- match (4) --
    const m = sense.match;
    o.push(
        clamp((m?.killsYou ?? 0) / 5, 0, 2),
        clamp((m?.killsTeam ?? 0) / 5, 0, 2),
        clamp((m?.aliveFoes ?? 0) / 3, 0, 1),
        clamp(sense.tick / MAX_TICKS_TOTAL, 0, 1),
    );

    // -- last-tick events (6) --
    // NOTE: sense.events describes the step just completed *before* the current
    // tick's update ran, i.e. these lag the latest physics step by one tick.
    // Damage/kill reward components come from exact snapshot diffs instead.
    const evts = sense.events ?? [];
    const kinds = new Set(evts.map((e) => e.kind));
    const lead = leadError(sense);
    o.push(
        kinds.has('hit-by') || kinds.has('blast') ? 1 : 0,
        lead.sin, // leadFeat sin(signed lead err), nearest foe (was reserved)
        kinds.has('kill') ? 1 : 0,
        lead.cos, // leadFeat cos(signed lead err), nearest foe (was reserved)
        kinds.has('pickup') ? 1 : 0,
        kinds.has('turret-captured') ? 1 : 0,
    );

    if (o.length !== OBS_DIM) {
        throw new Error(`obs dim mismatch: got ${o.length}, want ${OBS_DIM}`);
    }
    return o;
}

// ---------------------------------------------------------------------------
// Policy controller — runs an exported MLP (see rl/export_ts.py) as a robot.
// ---------------------------------------------------------------------------
interface PolicyWeights {
    obs_dim: number;
    action_dims: number[];
    activation: 'tanh' | 'relu';
    layers: Array<{ W: number[][]; b: number[] }>; // W is [out][in]
    // VecNormalize statistics (see rl/export_ts.py). Absent = identity.
    norm?: { mean: number[]; var: number[]; epsilon: number; clip_obs: number };
}

// Normalize a raw obs with the exported VecNormalize stats (matches
// stable_baselines3 VecNormalize: (x - mean)/sqrt(var + eps), clipped).
function normalizeObs(w: PolicyWeights, obs: number[]): number[] {
    const n = w.norm;
    if (!n) return obs;
    const out = new Array<number>(obs.length);
    for (let i = 0; i < obs.length; i += 1) {
        const v = (obs[i] - n.mean[i]) / Math.sqrt(n.var[i] + n.epsilon);
        out[i] = Math.min(n.clip_obs, Math.max(-n.clip_obs, v));
    }
    return out;
}

function mlpForward(w: PolicyWeights, obs: number[]): number[] {
    let x = obs;
    for (let li = 0; li < w.layers.length; li += 1) {
        const { W, b } = w.layers[li];
        const y = new Array<number>(b.length);
        for (let j = 0; j < b.length; j += 1) {
            let acc = b[j];
            const row = W[j];
            for (let k = 0; k < x.length; k += 1) acc += row[k] * x[k];
            y[j] = acc;
        }
        const last = li === w.layers.length - 1;
        x = last ? y : y.map((v) => (w.activation === 'tanh' ? Math.tanh(v) : Math.max(0, v)));
    }
    return x;
}

function sampleCategorical(logits: number[]): number {
    const m = Math.max(...logits);
    const exps = logits.map((l) => Math.exp(l - m));
    const sum = exps.reduce((a, b) => a + b, 0);
    let r = Math.random() * sum;
    for (let i = 0; i < exps.length; i += 1) {
        r -= exps[i];
        if (r <= 0) return i;
    }
    return exps.length - 1;
}

function decodeAction(logits: number[], actionDims: number[], deterministic: boolean): number[] {
    const out: number[] = [];
    let off = 0;
    for (const d of actionDims) {
        const seg = logits.slice(off, off + d);
        off += d;
        if (deterministic) {
            let bi = 0;
            for (let i = 1; i < seg.length; i += 1) if (seg[i] > seg[bi]) bi = i;
            out.push(bi);
        } else {
            out.push(sampleCategorical(seg));
        }
    }
    return out;
}

function actionToIntent(a: number[]): Intent {
    const bin5 = (i: number): number => (i / 2) - 1; // 0..4 -> -1..1
    return {
        throttle: bin5(a[0] ?? 2),
        turn: bin5(a[1] ?? 2),
        towerTurn: bin5(a[2] ?? 2),
        fire: (a[3] ?? 0) === 1,
        charge: (a[4] ?? 0) === 1,
        dash: (a[5] ?? 0) === 1,
        emp: (a[6] ?? 0) === 1,
    };
}

function makePolicyController(weightsPath: string, deterministic: boolean, maxHealth: number): RobotController {
    const raw = fs.readFileSync(weightsPath, 'utf8');
    const w = JSON.parse(raw) as PolicyWeights;
    if (w.obs_dim !== OBS_DIM) {
        throw new Error(`weights obs_dim ${w.obs_dim} != bridge OBS_DIM ${OBS_DIM}`);
    }
    return {
        meta: {
            id: 'rl-policy',
            name: 'RL Policy',
            author: 'robotarena-rl',
            version: '1.0.0',
            description: 'Exported PPO policy snapshot (self-play opponent).',
        },
        loadout: {},
        update(sense: SenseState): Intent {
            const obs = buildObs(sense, { maxHealth, myTeam: sense.self.team as 0 | 1 });
            const logits = mlpForward(w, normalizeObs(w, obs));
            return actionToIntent(decodeAction(logits, w.action_dims, deterministic));
        },
    };
}

// ---------------------------------------------------------------------------
// RL controller — returns the pending intent queued by Python.
// ---------------------------------------------------------------------------
const IDLE_INTENT: Required<Intent> = {
    throttle: 0, turn: 0, towerTurn: 0,
    fire: false, charge: false, dash: false, emp: false,
    strafe: 0, moveX: 0, moveY: 0, moveMode: 0,
    aimMode: 0, aimTarget: -1, aimLead: false, fireMode: 0,
    radio: null,
};

interface RlHandle {
    controller: RobotController;
    setIntent(partial: Partial<Intent>): void;
    lastSense(): SenseState | null;
}

function makeRlController(): RlHandle {
    let pending: Required<Intent> = { ...IDLE_INTENT };
    let sense: SenseState | null = null;
    return {
        controller: {
            meta: {
                id: 'rl-learner',
                name: 'RL Learner',
                author: 'robotarena-rl',
                version: '1.0.0',
                description: 'External PPO learner (bridge-controlled).',
            },
            loadout: {},
            update(s: SenseState): Intent {
                sense = s;
                return { ...pending };
            },
        },
        setIntent(partial: Partial<Intent>): void {
            pending = { ...IDLE_INTENT };
            if (typeof partial.throttle === 'number') pending.throttle = clamp(partial.throttle, -1, 1);
            if (typeof partial.turn === 'number') pending.turn = clamp(partial.turn, -1, 1);
            if (typeof partial.towerTurn === 'number') pending.towerTurn = clamp(partial.towerTurn, -1, 1);
            if (partial.fire === true) pending.fire = true;
            if (partial.charge === true) pending.charge = true;
            if (partial.dash === true) pending.dash = true;
            if (partial.emp === true) pending.emp = true;
        },
        lastSense(): SenseState | null {
            return sense;
        },
    };
}

// ---------------------------------------------------------------------------
// Match session
// ---------------------------------------------------------------------------
interface Session {
    match: Match;
    rl: RlHandle;
    ctx: ObsCtx;
    prevHealth: [number, number];
    prevAlive: [boolean, boolean];
    prevTurretOwner: number[];
}

function resolveOpponent(spec: unknown, maxHealth: number): RobotController {
    const s = spec as { bot?: string; policy?: { weights: string; deterministic?: boolean } };
    if (s && typeof s === 'object' && s.policy && typeof s.policy.weights === 'string') {
        return makePolicyController(s.policy.weights, s.policy.deterministic === true, maxHealth);
    }
    const botId = (s && typeof s.bot === 'string' && s.bot) || 'wanderer';
    const entry = getRobot(botId);
    if (!entry) throw new Error(`unknown opponent bot: ${botId}`);
    return entry.create();
}

function newSession(seed: number, opponentSpec: unknown, arena: string, loadout: Record<string, number>): Session {
    const rl = makeRlController();
    const spec = opponentSpec as { bot?: string; policy?: { weights: string } } | null;
    const botId = spec?.bot;
    const isPolicy = !!(spec && typeof spec === 'object' && spec.policy);
    const oppEntry = botId ? getRobot(botId) : undefined;
    // Policy snapshots fight with the learner's loadout (same maxHealth on
    // both sides); scripted bots keep their registry loadout.
    const oppLoadout = (isPolicy ? { ...loadout }
        : { ...(oppEntry?.loadout ?? {}) }) as Record<string, number>;
    // maxHealth = START_HEALTH (100) + 15 per plating point — mirrors skills.ts.
    const oppMaxHealth = 100 + 15 * (oppLoadout.plating ?? 0);
    const opp = resolveOpponent(opponentSpec, oppMaxHealth);
    const match = new Match(
        [
            { team: 0, controller: rl.controller, loadout: { ...loadout } },
            { team: 1, controller: opp, loadout: { ...oppLoadout } },
        ],
        seed >>> 0,
        { arena: arena as 'open' | 'blocks' | 'ruins' | 'foundry' | 'crossfire' },
    );
    const snaps = match.robotSnapshots;
    const ctx: ObsCtx = { maxHealth: snaps[0]?.maxHealth ?? 100, myTeam: 0 };
    return {
        match, rl, ctx,
        prevHealth: [snaps[0]?.health ?? 0, snaps[1]?.health ?? 0],
        prevAlive: [true, true],
        prevTurretOwner: [-1, -1],
    };
}

interface Components {
    dealt: number;
    taken: number;
    killed: boolean;
    died: boolean;
    pickup: boolean;
    turret: boolean;
    aimErr: number;
    leadErr: number;
}

function stepSession(s: Session, intent: Partial<Intent>): {
    obs: number[]; components: Components; terminated: boolean; truncated: boolean; result: { winner: number; tick: number; over: boolean };
} {
    s.rl.setIntent(intent);
    s.match.step();
    const sense = s.rl.lastSense();
    if (!sense) throw new Error('no sense captured — engine did not call update()');
    const obs = buildObs(sense, s.ctx);

    // Exact per-step components from snapshots.
    const snaps = s.match.robotSnapshots;
    const h0 = snaps[0]?.health ?? 0;
    const h1 = snaps[1]?.health ?? 0;
    const a0 = (snaps[0]?.alive ?? false) as boolean;
    const a1 = (snaps[1]?.alive ?? false) as boolean;
    const dealt = Math.max(0, s.prevHealth[1] - h1);
    const taken = Math.max(0, s.prevHealth[0] - h0);
    const killed = s.prevAlive[1] && !a1;
    const died = s.prevAlive[0] && !a0;
    s.prevHealth = [h0, h1];
    s.prevAlive = [a0, a1];

    // Sparse event bonuses lag one tick (sense.events describes the previous step).
    const kinds = new Set((sense.events ?? []).map((e) => e.kind));
    const pickup = kinds.has('pickup');
    let turret = false;
    const turrets = sense.turrets ?? [];
    turrets.forEach((t, i) => {
        if (t.owner === s.ctx.myTeam && s.prevTurretOwner[i] !== s.ctx.myTeam) turret = true;
        s.prevTurretOwner[i] = t.owner;
    });

    // Aim error: min |bearing - tower| over visible foes (-1 if none visible).
    // Used for aim-assist shaping (see rl/reward.py).
    // NOTE: `s` is the Session here; the tower angle lives on sense.self.
    const towerAngle = sense.self.tower as number;
    let aimErr = -1;
    for (const f of sense.foes ?? []) {
        if (typeof f.bearing !== "number" || !isFinite(f.bearing)) continue;
        const e = Math.abs(normAngle(f.bearing - towerAngle));
        if (aimErr < 0 || e < aimErr) aimErr = e;
    }
    const leadErr = leadError(sense).err; // min |lead bearing - tower|

    const r = s.match.result;
    const terminated = r.over && r.winner !== -1;
    const truncated = r.over && r.winner === -1;
    return {
        obs,
        components: { dealt, taken, killed, died, pickup, turret, aimErr, leadErr },
        terminated, truncated,
        result: { winner: r.winner, tick: r.tick, over: r.over },
    };
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------
const out = (obj: unknown): void => {
    process.stdout.write(`${JSON.stringify(obj)}\n`);
};

let session: Session | null = null;

async function main(): Promise<void> {
    const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
    for await (const line of rl) {
        const text = line.trim();
        if (!text) continue;
        let msg: { cmd?: string } & Record<string, unknown>;
        try {
            msg = JSON.parse(text) as typeof msg;
        } catch {
            out({ type: 'error', message: 'invalid JSON' });
            continue;
        }
        try {
            if (msg.cmd === 'reset') {
                const seed = typeof msg.seed === 'number' ? msg.seed : (Math.random() * 2 ** 31) | 0;
                session = newSession(
                    seed,
                    msg.opponent ?? { bot: 'wanderer' },
                    typeof msg.arena === 'string' ? msg.arena : 'open',
                    (msg.loadout as Record<string, number>) ?? {},
                );
                // Prime: one idle step so the learner's first obs is post-tick-0.
                const primed = stepSession(session, {});
                const sense = session.rl.lastSense();
                out({
                    type: 'reset',
                    obs: primed.obs,
                    info: {
                        seed,
                        tick: primed.result.tick,
                        maxHealth: session.ctx.maxHealth,
                        senseTick: sense?.tick ?? 0,
                    },
                });
            } else if (msg.cmd === 'step') {
                if (!session) throw new Error('no active session — send reset first');
                if (session.match.result.over) throw new Error('match over — send reset first');
                const r = stepSession(session, (msg.intent as Partial<Intent>) ?? {});
                out({
                    type: 'step',
                    obs: r.obs,
                    components: r.components,
                    terminated: r.terminated,
                    truncated: r.truncated,
                    info: {
                        winner: r.result.winner,
                        tick: r.result.tick,
                        health: session.prevHealth,
                    },
                });
            } else if (msg.cmd === 'ping') {
                out({ type: 'pong', obsDim: OBS_DIM, actionDims: ACTION_DIMS });
            } else if (msg.cmd === 'shutdown') {
                out({ type: 'ok' });
                process.exit(0);
            } else {
                out({ type: 'error', message: `unknown cmd: ${msg.cmd}` });
            }
        } catch (err) {
            out({ type: 'error', message: err instanceof Error ? err.message : String(err) });
        }
    }
}

main().catch((err) => {
    process.stderr.write(`bridge fatal: ${err instanceof Error ? err.stack : String(err)}\n`);
    process.exit(1);
});
