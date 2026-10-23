"""Pure RollbackCallback regression tests (stub eval callback only).

Scope: RollbackCallback is polled on EVERY training step, but the eval
callback's ``last_min6`` only changes when a periodic roster eval
completes. The bug: without a distinct-eval guard, one stale bad eval
was re-counted on every step, so ``patience: 2`` stopped training two
steps (not two evals) after the first drop, contradicting the
callback's "patience consecutive evals" contract.

These tests drive RollbackCallback with a stub eval callback — no
RobotArenaEnv, no Node bridge, no training run. The stub mirrors
WinRateEvalCallback's surface: ``last_min6`` plus a monotonic
``eval_count`` that increments once per completed eval. Between evals,
repeated ``_on_step()`` calls see the same (count, value) pair.
"""

from rl.train import RollbackCallback


class StubEvalCallback:
    """Minimal stand-in for WinRateEvalCallback."""

    def __init__(self):
        self.last_min6 = None
        self.eval_count = 0

    def set_eval(self, min6):
        """Simulate one completed roster eval publishing a new min6."""
        self.last_min6 = min6
        self.eval_count += 1


def _rollback(stub, patience=2):
    return RollbackCallback(stub, max_drop=0.05, patience=patience)


def test_no_eval_yet_never_stops():
    stub = StubEvalCallback()
    rb = _rollback(stub)
    for _ in range(10):
        assert rb._on_step() is True
    assert rb.reference is None
    assert rb.bad == 0


def test_repeated_same_eval_never_stops():
    stub = StubEvalCallback()
    rb = _rollback(stub)
    stub.set_eval(0.60)
    assert rb._on_step() is True
    assert rb.reference == 0.60
    # One distinct bad eval (drop 0.20 > max_drop 0.05)...
    stub.set_eval(0.40)
    # ...polled for many training steps before the next eval runs.
    for _ in range(10):
        assert rb._on_step() is True
    assert rb.bad == 1


def test_two_successive_distinct_drops_stop_at_patience2():
    stub = StubEvalCallback()
    rb = _rollback(stub)
    stub.set_eval(0.60)
    assert rb._on_step() is True
    stub.set_eval(0.40)
    for _ in range(5):
        assert rb._on_step() is True
    assert rb.bad == 1
    # Second successive distinct bad eval reaches patience=2.
    stub.set_eval(0.38)
    assert rb._on_step() is False
    assert rb.bad == 2


def test_distinct_eval_with_same_value_counts():
    stub = StubEvalCallback()
    rb = _rollback(stub)
    stub.set_eval(0.60)
    assert rb._on_step() is True
    stub.set_eval(0.40)
    assert rb._on_step() is True
    assert rb.bad == 1
    # A NEW eval that happens to produce the same min6 is still a
    # distinct eval: identity is eval_count, not the value.
    stub.set_eval(0.40)
    assert rb._on_step() is False
    assert rb.bad == 2


def test_good_distinct_eval_resets_bad():
    stub = StubEvalCallback()
    rb = _rollback(stub)
    stub.set_eval(0.60)
    assert rb._on_step() is True
    stub.set_eval(0.40)
    assert rb._on_step() is True
    assert rb.bad == 1
    # Distinct good eval (within max_drop of reference) resets the count.
    stub.set_eval(0.58)
    assert rb._on_step() is True
    assert rb.bad == 0
    # Re-polling that same good eval changes nothing...
    for _ in range(5):
        assert rb._on_step() is True
    assert rb.bad == 0
    # ...and the next single distinct bad eval does not stop training.
    stub.set_eval(0.40)
    assert rb._on_step() is True
    assert rb.bad == 1


def test_value_only_stub_repeated_same_eval_never_stops():
    # Eval callbacks without an eval_count fall back to value identity:
    # consecutive repeats of one value are a single eval.
    class ValueOnlyStub:
        last_min6 = None

    stub = ValueOnlyStub()
    rb = _rollback(stub)
    stub.last_min6 = 0.60
    assert rb._on_step() is True
    stub.last_min6 = 0.40
    for _ in range(10):
        assert rb._on_step() is True
    assert rb.bad == 1
    stub.last_min6 = 0.38
    assert rb._on_step() is False
    assert rb.bad == 2
