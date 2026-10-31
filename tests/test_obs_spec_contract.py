"""Contract test for the 128-dim observation layout.

The per-tick obs layout is the riskiest cross-language contract in the
repo: the Node bridge builds it (bridge/bridge.template.ts ``buildObs``),
bridge/OBS_SPEC.md documents it, and rl/env.py indexes into it with named
constants (SELF_SPEED, FOE_PRESENT, BULLET_BASE, TICK). It has already
drifted twice (124 -> 128 dims; leadFeat documented at 119/121 while the
bridge actually emits it at 123/125 inside the last-tick events block),
yet no test previously touched the layout beyond FRAME_DIM/OBS_DIM
totals in tests/test_pure.py.

This module parses the OBS_SPEC.md table and cross-checks it against
rl/env.py so the next drift fails CI instead of silently invalidating
weights. Pure Python only (stdlib + rl.env constants): no torch, no
Node, no sibling checkout, no bridge subprocess -- it runs inside the
existing pure-tests job unchanged.

Note on the 119/121 regression: the stale prose note at OBS_SPEC.md:36
("Dims 119/121 (leadFeat)") is owned by draft PR #11
(fix/obsspec-leadfeat-dims) and deliberately not fixed here. The
regression checks below therefore bind the *table* (the layout the code
and weights actually follow): leadFeat must live at 123/125 inside the
events row 122-127, never in the match block 118-121.
"""

import re
from pathlib import Path

from rl.env import BULLET_BASE, FOE_PRESENT, FRAME_DIM, OBS_DIM, SELF_SPEED, TICK

REPO_ROOT = Path(__file__).resolve().parents[1]
OBS_SPEC_PATH = REPO_ROOT / "bridge" / "OBS_SPEC.md"
BRIDGE_TEMPLATE_PATH = REPO_ROOT / "bridge" / "bridge.template.ts"

_RANGE_RE = re.compile(r"(\d+)\s*[–—-]\s*(\d+)")


def _parse_obs_spec_table():
    """Return [(lo, hi, contents), ...] from the OBS_SPEC.md dims table."""
    rows = []
    for line in OBS_SPEC_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) < 2:
            continue
        m = _RANGE_RE.fullmatch(cells[0])
        if m is None:
            continue  # header / separator rows
        lo, hi = int(m.group(1)), int(m.group(2))
        rows.append((lo, hi, cells[1]))
    return rows


def _row_containing(rows, needle):
    for lo, hi, contents in rows:
        if needle in contents:
            return lo, hi, contents
    raise AssertionError(f"no OBS_SPEC table row containing {needle!r}: {rows}")


def _events_items(contents):
    """Ordered event names in the last-tick events row, parens stripped.

    The row reads e.g. "**last-tick events (6):** wasHit flag,
    **leadFeat sin** (sin of ..., nearest foe), gotKill flag, ...".
    Parentheticals contain commas, so strip them before splitting.
    """
    without_parens = re.sub(r"\([^)]*\)", "", contents)
    _, _, tail = without_parens.partition(":")
    return [item.strip() for item in tail.split(",") if item.strip()]


def _leadfeat_dims(rows):
    lo, hi, contents = _row_containing(rows, "last-tick events")
    items = _events_items(contents)
    sin_idx = next(i for i, it in enumerate(items) if "leadFeat sin" in it)
    cos_idx = next(i for i, it in enumerate(items) if "leadFeat cos" in it)
    return lo + sin_idx, lo + cos_idx, (lo, hi, contents), items


def test_obs_spec_table_contiguous_0_to_127():
    rows = _parse_obs_spec_table()
    assert rows, "OBS_SPEC.md dims table parsed empty"
    assert rows[0][0] == 0, f"table must start at dim 0, got {rows[0]}"
    assert rows[-1][1] == 127, f"table must end at dim 127, got {rows[-1]}"
    for (plo, phi, _), (lo, hi, _) in zip(rows, rows[1:]):
        assert lo == phi + 1, f"gap/overlap: ...{plo}-{phi} then {lo}-{hi}"
        assert lo <= hi, f"inverted range {lo}-{hi}"
    total = sum(hi - lo + 1 for lo, hi, _ in rows)
    assert total == 128, f"table ranges total {total} dims, want 128"


def test_leadfeat_sin_cos_at_123_125_inside_events():
    rows = _parse_obs_spec_table()
    sin_dim, cos_dim, (lo, hi, _), items = _leadfeat_dims(rows)
    assert (lo, hi) == (122, 127), f"events row is {lo}-{hi}, want 122-127"
    assert len(items) == 6, f"events row lists {len(items)} items, want 6: {items}"
    assert sin_dim == 123, f"leadFeat sin at dim {sin_dim}, want 123"
    assert cos_dim == 125, f"leadFeat cos at dim {cos_dim}, want 125"
    assert lo <= sin_dim <= hi and lo <= cos_dim <= hi


def test_env_named_indices_match_obs_spec_table():
    rows = _parse_obs_spec_table()
    total = sum(hi - lo + 1 for lo, hi, _ in rows)
    assert FRAME_DIM == 128 == total
    assert OBS_DIM == 516

    self_lo, self_hi, _ = _row_containing(rows, "self (17)")
    assert (self_lo, self_hi) == (0, 16)
    assert SELF_SPEED == 6
    assert self_lo <= SELF_SPEED <= self_hi

    foes_lo, _, _ = _row_containing(rows, "foes (2 nearest")
    assert FOE_PRESENT == 17 == foes_lo

    bullets_lo, _, _ = _row_containing(rows, "bullets (6 nearest")
    assert BULLET_BASE == 38 == bullets_lo

    match_lo, match_hi, _ = _row_containing(rows, "match (4)")
    assert (match_lo, match_hi) == (118, 121)
    assert TICK == 121
    assert match_lo <= TICK <= match_hi


def test_no_leadfeat_at_119_121_regression():
    rows = _parse_obs_spec_table()
    sin_dim, cos_dim, _, _ = _leadfeat_dims(rows)
    assert (sin_dim, cos_dim) == (123, 125)
    assert (sin_dim, cos_dim) != (119, 121)
    # The match block 118-121 (which contains the stale 119/121 claim)
    # must not carry leadFeat; only the events row may mention it.
    _, _, match_contents = _row_containing(rows, "match (4)")
    assert "leadFeat" not in match_contents
    for lo, hi, contents in rows:
        if "leadFeat" in contents:
            assert (lo, hi) == (122, 127), (
                f"leadFeat claimed in row {lo}-{hi}, want only 122-127"
            )


def test_bridge_template_events_order_matches_spec():
    text = BRIDGE_TEMPLATE_PATH.read_text(encoding="utf-8")
    match_at = text.index("// -- match")
    events_at = text.index("// -- last-tick events")
    assert match_at < events_at, "bridge must push match before events"
    block = text[events_at:]
    block = block[: block.index("if (o.length")]
    order = ["hit-by", "lead.sin", "'kill'", "lead.cos", "'pickup'", "'turret-captured'"]
    positions = [block.index(tok) for tok in order]
    assert positions == sorted(positions), (
        f"bridge events push order drifted: {list(zip(order, positions))}"
    )
    # Events start at 122 (after match 118-121): hit=122, sin=123,
    # kill=124, cos=125, pickup=126, turret-captured=127.
    rows = _parse_obs_spec_table()
    sin_dim, cos_dim, _, _ = _leadfeat_dims(rows)
    assert 122 + order.index("lead.sin") == sin_dim == 123
    assert 122 + order.index("lead.cos") == cos_dim == 125
