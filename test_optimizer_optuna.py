import numpy as np
import pandas as pd
import optuna

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

    assert all(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)


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

