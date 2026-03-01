import sys
import os

# make sure workspace root is on path so that imports like "validation" succeed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from pathlib import Path

import numpy as np

from validation.permutation_test import run_permutation_test, _block_permute
from agents.statistical_validator import StatisticalValidatorAgent


def _make_history(tmp_path: Path, returns: list) -> str:
    path = tmp_path / "hist.json"
    data = [{"pnl_pct": r} for r in returns]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return str(path)


def test_permutation_basic(tmp_path):
    # Tiny dataset where net profit is zero; p-value should be roughly 0.5
    hist = _make_history(tmp_path, [1, -1, 2, -2, 3, -3])
    # run with default behaviour (bootstrap=True) - should execute without error
    res = run_permutation_test(
        trade_history_path=hist,
        n_permutations=100,
        metric="net_profit",
        use_block_permutation=False,
    )
    assert "p_value" in res
    assert res["original"] == 0
    assert 0 <= res["p_value"] <= 1

    # explicitly test non-bootstrap (pure permutation) yields valid output too
    res2 = run_permutation_test(
        trade_history_path=hist,
        n_permutations=100,
        metric="net_profit",
        use_block_permutation=False,
        bootstrap=False,
    )
    assert "p_value" in res2
    assert 0 <= res2["p_value"] <= 1


def test_block_permute_preserves(tmp_path):
    arr = np.arange(10)
    perm = _block_permute(arr, 3)
    assert len(perm) == len(arr)
    assert set(perm) == set(arr)


def test_bootstrap_resamples(tmp_path):
    # ensure bootstrapping can repeat elements
    arr = np.array([1, 2, 3])
    # mimic internal use by invoking choice directly
    samples = np.random.choice(arr, size=len(arr), replace=True)
    assert len(samples) == 3
    # at least one resample should have duplicates over many tries
    seen = False
    for _ in range(50):
        if len(np.unique(np.random.choice(arr, size=len(arr), replace=True))) < 3:
            seen = True
            break
    assert seen, "bootstrap sampling should sometimes produce duplicates"


def test_statistical_agent():
    agent = StatisticalValidatorAgent(p_value_threshold=0.1)
    assert agent.evaluate({"p_value": 0.05})
    assert not agent.evaluate({"p_value": 0.2})
    assert agent.evaluate([{"p_value": 0.01}, {"p_value": 0.09}])
    assert not agent.evaluate([{"p_value": 0.01}, {"p_value": 0.15}])
