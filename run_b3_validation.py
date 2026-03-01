
import sys
import os
import pandas as pd
from datetime import datetime, timedelta
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("validation")

try:
    import futures_optimizer
    import config_futures
    import futures_core
    import utils
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def _generate_synthetic_df(days: int):
    end = datetime.now()
    start = end - timedelta(days=days)
    dates = pd.date_range(start=start, end=end, freq="15min")
    n = len(dates)
    print(f"Generating {n} synthetic bars for {days} days...")
    import numpy as np
    np.random.seed(42)
    price = 130000.0
    close, high, low, volume = [], [], [], []
    for _ in range(n):
        change = np.random.normal(0, 100)
        price += change
        c = price
        h = c + abs(np.random.normal(0, 50))
        l = c - abs(np.random.normal(0, 50))
        v = np.random.randint(5000, 50000)
        close.append(c); high.append(h); low.append(l); volume.append(v)
    return pd.DataFrame({'close': close, 'high': high, 'low': low, 'volume': volume}, index=dates)

def run_test(symbol: str, days: int):
    print(f"Starting Validation for {symbol} ({days} days)")
    generic = symbol
    cfg_key = generic if generic in config_futures.FUTURES_CONFIGS else f"{generic[:3]}$N"
    cfg = config_futures.FUTURES_CONFIGS.get(cfg_key)
    if cfg:
        print(f"Config Loaded: Tick={cfg['tick_size']}, MarginStress={cfg['margin_stress']}")
    mt5_ok = futures_optimizer.ensure_mt5_futures()
    target = utils.resolve_current_symbol(generic) if mt5_ok else None
    if target:
        print(f"Resolved {generic} → {target}")
    bars = int(days * 28)
    df = None
    if mt5_ok:
        df = futures_optimizer.load_futures_data(generic, bars=bars, timeframe="M15")
        if df is not None and not df.empty:
            print(f"Loaded {len(df)} real bars from MT5 for {generic}")
        else:
            print("MT5 data not available; falling back to synthetic")
            df = _generate_synthetic_df(days)
    else:
        df = _generate_synthetic_df(days)
    print("⚙️ Running Backtest Engine...")
    params = {'ema_fast': 9, 'ema_slow': 21, 'sl_atr_mult': 2.0, 'tp_atr_mult': 3.0}
    try:
        symbol_for_bt = target or generic
        results = futures_optimizer.backtest_futures(symbol_for_bt, params, df)
        print("\n✅ Backtest Completed Successfully!")
        print(f"   Total Trades: {results.get('total_trades')}")
        print(f"   Win Rate: {results.get('win_rate'):.2%}")
        print(f"   Final Equity: R$ {results.get('final_equity'):.2f}")
        try:
            with open("validation_summary.txt", "w", encoding="utf-8") as f:
                f.write(f"Symbol: {symbol_for_bt}\n")
                f.write(f"Days: {days}\n")
                f.write(f"Total Trades: {int(results.get('total_trades', 0))}\n")
                f.write(f"Win Rate: {float(results.get('win_rate', 0.0)):.2%}\n")
                f.write(f"Final Equity: R$ {float(results.get('final_equity', 0.0)):.2f}\n")
        except Exception:
            pass

    # ✅ Permutation test post-backtest (uses config for parameters)
    try:
        import config
        if getattr(config, "PERMUTATION_TEST", {}).get("enabled", False):
            from validation.permutation_test import run_permutation_test
            cfg = config.PERMUTATION_TEST
            print("📊 Running permutation test (end of validation)")
            perm_res = run_permutation_test(
                trade_history_path=cfg.get("trade_history_path", "ml_trade_history.json"),
                n_permutations=cfg.get("n_permutations", 5000),
                metric=cfg.get("metrics", ["profit_factor"])[0],
                use_block_permutation=cfg.get("block_size", 3) > 1,
                block_size=cfg.get("block_size", 3),
                bootstrap=cfg.get("bootstrap", True),
            )
            print(f"Permutation p-value ({perm_res['metric']}): {perm_res['p_value']:.4f}")
    except Exception as e:
        print(f"Erro na permutation test: {e}")
    except Exception as e:
        print(f"❌ Backtest Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="WIN$N")
    parser.add_argument("--days", type=int, default=90)
    args = parser.parse_args()
    run_test(args.symbol, args.days)
