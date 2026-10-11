"""Pytest root conftest.

This file's presence at the repo root makes pytest add the repo root to
sys.path (prepend import mode imports this root conftest), so `import rl`
works when tests are run via the `pytest` console script in CI as well as
via `python -m pytest` locally. No fixtures, no other logic.
"""
