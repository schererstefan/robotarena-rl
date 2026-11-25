"""Contract test for turret-capture reward detection wiring.

The ``turret`` reward component (+1.0, rl/reward.py) is computed in the
Node bridge by diffing turret owners between steps. sense.turrets is
cone-gated and nearest-first (robotarena src/sim/types.ts), and
SenseTurret carries no id, so ownership must be keyed by turret
position. Keying by list index (the old ``prevTurretOwner[i]``) paid
spurious captures whenever nearest-first order swapped and hid real
flips behind the shuffle.

Behavior is covered by bridge/test-turret-capture.mjs (Node, zero-dep:
order swap with unchanged ownership must pay nothing; a real flip on
the same position must pay). CI's pure-tests job is Python-only and
cannot build or run the bridge (no sibling robotarena checkout), so
this module guards the wiring at source level, like
tests/test_obs_spec_contract.py: position-keyed Map in the session,
detection delegated to bridge/turret-capture.mjs, and reset seeding
from the primed sense so match-start owners are baseline, never a
first-step capture.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE_TEMPLATE_PATH = REPO_ROOT / "bridge" / "bridge.template.ts"
TURRET_MODULE_PATH = REPO_ROOT / "bridge" / "turret-capture.mjs"
TURRET_TEST_PATH = REPO_ROOT / "bridge" / "test-turret-capture.mjs"
BUILD_SH_PATH = REPO_ROOT / "bridge" / "build.sh"


def _template():
    return BRIDGE_TEMPLATE_PATH.read_text(encoding="utf-8")


def test_session_keys_turret_owners_by_position_not_index():
    text = _template()
    assert "prevTurretOwner: Map<string, number>" in text
    assert "prevTurretOwner: number[]" not in text
    assert "prevTurretOwner: [-1, -1]" not in text
    assert "prevTurretOwner: new Map()" in text
    # No list-index access anywhere: index is sense order, not identity.
    assert "prevTurretOwner[" not in text


def test_step_detection_delegates_to_position_keyed_module():
    text = _template()
    assert "from './turret-capture.mjs'" in text
    assert "detectTurretCapture(s.prevTurretOwner" in text
    module = TURRET_MODULE_PATH.read_text(encoding="utf-8")
    # Turrets are static and SenseTurret has no id: position is identity.
    assert "${t.x},${t.y}" in module
    assert "owners.get(key)" in module
    assert "owners.set(key, t.owner)" in module
    assert TURRET_TEST_PATH.exists()


def test_reset_seeds_owners_from_primed_sense():
    text = _template()
    reset_at = text.index("msg.cmd === 'reset'")
    block = text[reset_at:]
    primed_at = block.index("stepSession(session, {})")
    seed_at = block.index("seedTurretOwners(session.prevTurretOwner")
    reply_at = block.index("type: 'reset'")
    assert primed_at < seed_at < reply_at, (
        "reset must seed turret owners from the primed sense before replying"
    )
    assert "sense?.turrets ?? []" in block


def test_build_copies_turret_module_next_to_generated_template():
    build = BUILD_SH_PATH.read_text(encoding="utf-8")
    assert 'cp "$SCRIPT_DIR/turret-capture.mjs" "$SCRIPT_DIR/dist/turret-capture.mjs"' in build
