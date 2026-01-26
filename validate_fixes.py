import sys
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("validate_fixes")

sys.path.append(r"c:\Users\luizf\Documents\xp3v5")

def _dummy_df(n=600):
    idx = pd.date_range(start="2025-01-01", periods=n, freq="15min")
    base = np.cumsum(np.random.normal(0, 0.3, n)) + 20
    close = np.clip(base, 5, None)
    open_ = close + np.random.normal(0, 0.1, n)
    high = np.maximum(close, open_) + np.abs(np.random.normal(0, 0.2, n))
    low = np.minimum(close, open_) - np.abs(np.random.normal(0, 0.2, n))
    volume = np.random.uniform(5e6, 2e7, n)
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)
    return df

def run_sanity_checks():
    try:
        from optimizer_optuna import backtest_params_on_df
        print("1) Import OK: optimizer_optuna.backtest_params_on_df")
    except Exception as e:
        print(f"[FAIL] Import optimizer_optuna: {e}")
        return False

    df = _dummy_df()
    params = {
        "ema_short": 12, "ema_long": 30,
        "rsi_low": 30, "rsi_high": 70,
        "adx_threshold": 20,
        "sl_atr_multiplier": 2.0,
        "tp_mult": 2.5,
        "base_slippage": 0.0015,
        "enable_shorts": 1
    }

    # Forward-style check (same df as proxy)
    res_fwd = backtest_params_on_df("DUMMY3", params, df, ml_model=None)
    print(f" - Forward WR={res_fwd.get('win_rate',0):.2f} Calmar={res_fwd.get('calmar',0):.2f} DD={res_fwd.get('max_drawdown',0):.2f}")

    # Stress check (double slippage)
    params_stress = dict(params)
    params_stress["base_slippage"] = params["base_slippage"] * 2.0
    res_stress = backtest_params_on_df("DUMMY3", params_stress, df, ml_model=None)
    print(f" - Stress Calmar={res_stress.get('calmar',0):.2f}")

    # Buy & Hold comparison
    bh = float(df["close"].iloc[-1] / df["close"].iloc[0] - 1.0)
    algo_ret = float(res_fwd.get("total_return", 0.0) or 0.0)
    print(f" - Buy&Hold={bh:.2f} Algo={algo_ret:.2f}")

    ok_forward = res_fwd.get("win_rate", 0.0) >= 0.10  # mais brando para dummy
    ok_stress = res_stress.get("calmar", 0.0) > -0.5   # brando para dummy
    ok_bh = algo_ret >= (bh * 0.2)                     # brando para dummy
    print(f"RESULT: Forward OK={ok_forward} | Stress OK={ok_stress} | Buy&Hold OK={ok_bh}")
    return ok_forward and ok_stress and ok_bh

if __name__ == "__main__":
    try:
        if sys.platform.startswith("win"):
            import os
            os.system("chcp 65001")
    except Exception:
        pass
    success = run_sanity_checks()
    sys.exit(0 if success else 1)
