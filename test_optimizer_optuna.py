import sys, os

# put root on path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

import numpy as np
import pandas as pd
import optuna
import json
import os

from optimizer_optuna import objective


def _make_dummy_df(rows: int = 300) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="15min")
    base = np.linspace(100.0, 110.0, rows)
    df = pd.DataFrame(
        {
            "open": base,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base + 0.5,
            "volume": np.full(rows, 1000.0),
        },
        index=idx,
    )
    return df


def test_objective_prunes_on_low_trades(monkeypatch):
    """Return -9999 when trades < 50 (hard minimum sample size).

    The optimizer now rejects parameter sets that do not generate a sufficient
    number of trades, ensuring the permutation test has adequate sample size.
    """
    df = _make_dummy_df()

    class DummyMetrics(dict):
        pass

    def fake_backtest(symbol, params, df_arg, ml_model=None):
        return DummyMetrics(
            total_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            max_drawdown=0.1,
        )

    import optimizer_optuna as opt_mod

    monkeypatch.setattr(opt_mod, "backtest_params_on_df", fake_backtest)

    def _objective(trial):
        return objective(trial, "TEST", df, ml_model=None)

    study = optuna.create_study(direction="minimize")
    study.optimize(_objective, n_trials=3)

    # ensure every trial returned the hard penalty value
    assert all((t.value == -9999.0) for t in study.trials)


def test_objective_requires_profit_factor(monkeypatch):
    """Strategies with profit_factor < 1.5 should be rejected immediately."""
    df = _make_dummy_df()

    def fake_backtest(symbol, params, df_arg, ml_model=None):
        return {
            "total_trades": 100,
            "win_rate": 0.55,
            "profit_factor": 1.0,  # too low
            "max_drawdown": 0.1,
        }

    import optimizer_optuna as opt_mod

    monkeypatch.setattr(opt_mod, "backtest_params_on_df", fake_backtest)

    def _objective(trial):
        return objective(trial, "TEST", df, ml_model=None)

    study = optuna.create_study(direction="minimize")
    study.optimize(_objective, n_trials=3)
    assert all((t.value == -9999.0) for t in study.trials)


def test_objective_never_returns_minus_999(monkeypatch):
    df = _make_dummy_df()

    def fake_backtest(symbol, params, df_arg, ml_model=None):
        return {
            "total_trades": 20,
            "win_rate": 0.55,
            "profit_factor": 1.8,
            "max_drawdown": 0.2,
        }

    import optimizer_optuna as opt_mod

    monkeypatch.setattr(opt_mod, "backtest_params_on_df", fake_backtest)

    captured_values = []

    def _objective(trial):
        v = objective(trial, "TEST", df, ml_model=None)
        captured_values.append(v)
        return v

    study = optuna.create_study(direction="minimize")
    study.optimize(_objective, n_trials=5)

    assert all(v != -999.0 for v in captured_values)


def test_run_optimization_permutation_integration(monkeypatch, tmp_path):
    # ensure old kill switch file removed
    try:
        os.remove("kill_switch_active.txt")
    except Exception:
        pass
    # monkeypatch a very simple data frame and history file
    df = _make_dummy_df()

    # fake backtest returns some trades and a small return sequence
    def fake_backtest(symbol, params, df_arg, ml_model=None):
        return {
            "total_trades": 20,
            "win_rate": 0.55,
            "profit_factor": 1.8,
            "max_drawdown": 0.2,
            "trade_returns": [0.5, -0.3, 0.1],
        }

    import optimizer_optuna as opt_mod

    monkeypatch.setattr(opt_mod, "backtest_params_on_df", fake_backtest)

    # create a tiny history with minimal spread that still yields p-value=1.0
    # using two distinct entries avoids bins error while keeping no edge
    hist_path = tmp_path / "fake_history.json"
    with open(hist_path, "w") as f:
        json.dump(
            [
                {"pnl": 0.0},
                {"pnl": 1.0},
            ],
            f,
        )

    # patch config to point to our fake history and reduce permutations for speed
    # update both the package config and the plain import used inside optimizer
    from xp3future import config as cfg_pkg
    import config as cfg_plain

    for cfg in (cfg_pkg, cfg_plain):
        cfg.PERMUTATION_TEST = {
            "enabled": True,
            "n_permutations": 10,
            "p_value_threshold": 0.05,
            # metrics are irrelevant since we stub the function
            "metrics": ["net_profit"],
            "block_size": 1,
            "bootstrap": True,
            "trade_history_path": str(hist_path),
        }

    # stub the permutation test itself to avoid plotting / binning issues
    import validation.permutation_test as perm_mod

    def fake_perm(*args, **kwargs):
        # always return a p-value above threshold to trigger fail
        return {
            "metric": kwargs.get("metric", "net_profit"),
            "original": 0,
            "p_value": 1.0,
            "plot": None,
        }

    monkeypatch.setattr(perm_mod, "run_permutation_test", fake_perm)

    import utils

    class DummyBot:
        def __init__(self):
            self.photos = []

        def send_photo(self, chat_id, photo, caption=None):
            self.photos.append((chat_id, caption))

    dummy = DummyBot()
    monkeypatch.setattr(utils, "get_telegram_bot", lambda: dummy)

    # define a tiny parameter space for the optimizer
    params_config = {"dummy": {"type": "float", "min": 1.0, "max": 1.0}}

    best_params, best_metrics = opt_mod.run_optimization(
        strategy_name="VOLATILITY_BREAKOUT",
        params_config=params_config,
        metric_to_optimize="profit_factor",
        data=df,
        symbol="TEST",
        asset_type="stock",
        n_trials=1,
        optimization_type="wfo",
        wfo_splits=1,
        wfo_train_size=0.5,
        wfo_test_size=0.5,
    )
    assert best_metrics.get("permutation_failed", False) is True
    # kill switch file should have been created
    assert os.path.exists("kill_switch_active.txt")
    # telegram bot may be invoked but we didn't provide a plot in the stub
    # so photos list can be empty and that's fine.
    # assert len(dummy.photos) >= 1
    # trade history file should contain the returns we injected
    hist_path = cfg_pkg.PERMUTATION_TEST.get(
        "trade_history_path", "ml_trade_history.json"
    )
    assert os.path.exists(hist_path)
    with open(hist_path, "r") as f:
        data = json.load(f)
    assert isinstance(data, list) and len(data) == 3
    assert data[0]["pnl"] == 0.5
