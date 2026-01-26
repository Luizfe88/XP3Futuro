# otimizador_final.py
# Walk-Forward Optimizer (WFO) - Final version with Backfill integration
# - Tries base.load_historical_bars -> utils.safe_copy_rates -> backfill.ensure_history (CSV)
# - Performs WFO with configurable windows, train/test sizes
# - Computes robustness metrics and saves JSON summaries per symbol
# Usage:
#   python otimizador_final.py --bars 4000 --maxevals 300 --workers 4 --wfo_windows 6 --train 500 --test 200

import os
import json
import time
import argparse
import logging
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("otimizador_final")

# Inicializa o MT5 se disponível
if mt5:
    if not mt5.initialize():
        logger.error("Falha ao inicializar o MT5. Verifique se o terminal está aberto e configurado corretamente.")
        mt5 = None
    else:
        logger.info("MT5 inicializado com sucesso.")

# try to import user's optimizer module(s)
try:
    import optimizer_updated as base
except Exception:
    try:
        import optimizer as base
    except Exception:
        base = None

try:
    import config
except Exception:
    config = None

try:
    import utils
except Exception:
    utils = None

# backfill helper (local file)
try:
    from backfill import ensure_history
except Exception:
    ensure_history = None

# defaults (can be overridden by CLI or config)
WFO_WINDOWS = int(getattr(config, "WFO_WINDOWS", 6))
TRAIN_PERIOD = int(getattr(config, "WFO_TRAIN_PERIOD", 500))
TEST_PERIOD = int(getattr(config, "WFO_TEST_PERIOD", 200))
OPT_OUTPUT_DIR = getattr(base, "OPT_OUTPUT_DIR", getattr(config, "OPTIMIZER_OUTPUT", "optimizer_output"))
os.makedirs(OPT_OUTPUT_DIR, exist_ok=True)

def load_all_symbols() -> List[str]:
    secmap = getattr(config, "SECTOR_MAP", {}) or {}
    syms = [k.upper().strip() for k in secmap.keys() if isinstance(k, str) and k.strip()]
    if not syms:
        syms = list(getattr(config, "PROXY_SYMBOLS", []) or [])
    return sorted(list(set(syms)))

def safe_save_json(fp: str, data: dict):
    tmp = fp + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, fp)

def compute_basic_metrics(equity_curve: List[float]) -> Dict[str, Any]:
    import math, statistics
    out = {"n": len(equity_curve)}
    if not equity_curve or len(equity_curve) < 2:
        out.update({"total_return": 0.0, "max_drawdown": 0.0, "sharpe": 0.0, "pf": 0.0})
        return out
    returns = []
    for i in range(1, len(equity_curve)):
        prev = equity_curve[i-1]
        cur = equity_curve[i]
        if prev == 0:
            returns.append(0.0)
        else:
            returns.append((cur - prev) / abs(prev))
    total_return = (equity_curve[-1] - equity_curve[0]) / (equity_curve[0] if equity_curve[0] != 0 else 1)
    # max drawdown
    peak = equity_curve[0]
    max_dd = 0.0
    for v in equity_curve:
        if v > peak:
            peak = v
        dd = (peak - v) / (peak if peak != 0 else 1)
        if dd > max_dd:
            max_dd = dd
    mean_r = statistics.mean(returns) if returns else 0.0
    std_r = statistics.pstdev(returns) if returns else 0.0
    sharpe = (mean_r / std_r) * (252 ** 0.5) if std_r and std_r > 0 else 0.0
    pos = sum([r for r in returns if r>0])
    neg = -sum([r for r in returns if r<0])
    pf = (pos / neg) if neg > 0 else (pos if pos>0 else 0.0)
    out.update({"total_return": total_return, "max_drawdown": max_dd, "sharpe": sharpe, "pf": pf})
    return out

def backtest_params_on_df(sym: str, params: dict, df):
    # prefer base.test_params_on_data if available
    try:
        if base and hasattr(base, "test_params_on_data"):
            return base.test_params_on_data(sym, params, df)
    except Exception:
        pass
    # fallback simple emulator: EMA crossover equity simulation
    try:
        close = df["close"].astype(float).copy()
        short = int(params.get("ema_short", params.get("ema_fast", 9)))
        long = int(params.get("ema_long", params.get("ema_slow", 21)))
        if len(close) < long + 2:
            return {"metrics": {}, "equity_curve": [1.0]}
        ema_s = close.ewm(span=short, adjust=False).mean()
        ema_l = close.ewm(span=long, adjust=False).mean()
        position = 0
        cash = 1.0
        equity_curve = []
        for i in range(len(close)):
            if ema_s.iat[i] > ema_l.iat[i] and position == 0:
                position = 1
            elif ema_s.iat[i] < ema_l.iat[i] and position == 1:
                position = 0
            ret = (close.iat[i] / close.iat[i-1] - 1) if i>0 else 0.0
            if position:
                cash = cash * (1 + ret)
            equity_curve.append(cash)
        metrics = compute_basic_metrics(equity_curve)
        return {"metrics": metrics, "equity_curve": equity_curve}
    except Exception as e:
        return {"metrics": {}, "equity_curve": [1.0]}

def optimize_window(sym: str, df_train, maxevals: int):
    try:
        if base and hasattr(base, "optimize_on_data"):
            best = base.optimize_on_data(sym, df_train, max_evals=maxevals, base_dir=OPT_OUTPUT_DIR)
            return best
    except Exception:
        logger.exception("optimize_on_data failed, falling back")
    try:
        if base and hasattr(base, "optimize_symbol_robust"):
            res = base.optimize_symbol_robust(sym, base_dir=OPT_OUTPUT_DIR, max_evals=maxevals)
            if isinstance(res, dict) and res.get("best_params"):
                return res.get("best_params")
    except Exception:
        logger.exception("optimize_symbol_robust fallback failed")
    # fallback: return defaults
    return {"ema_short":9, "ema_long":21, "rsi_low":30, "rsi_high":70, "mom_min":0.0}

def load_series_with_backfill(sym: str, bars: int, timeframe=None):
    df = None
    if timeframe is None:
        timeframe = mt5.TIMEFRAME_M15 if mt5 else None

    # Try base.load_historical_bars
    try:
        if base and hasattr(base, "load_historical_bars"):
            df = base.load_historical_bars(sym, bars=bars)
            logger.info(f"{sym} carregado via base.load_historical_bars")
    except Exception as e:
        logger.warning(f"{sym} falha em base.load_historical_bars: {e}")

    # Try utils.safe_copy_rates
    try:
        if (df is None or (hasattr(df, "empty") and df.empty)) and utils and hasattr(utils, "safe_copy_rates"):
            df = utils.safe_copy_rates(sym, timeframe, count=bars)
            logger.info(f"{sym} carregado via utils.safe_copy_rates")
    except Exception as e:
        logger.warning(f"{sym} falha em utils.safe_copy_rates: {e}")

    # Try direct MT5 copy_rates_from_pos
    try:
        if (df is None or (hasattr(df, "empty") and df.empty)) and mt5 and pd:
            if mt5.symbol_select(sym, True):
                rates = mt5.copy_rates_from_pos(sym, timeframe, 0, bars)
                if rates is not None and len(rates) > 0:
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    df.set_index('time', inplace=True)
                    df = df[['open', 'high', 'low', 'close', 'tick_volume']].rename(columns={'tick_volume': 'volume'})
                    logger.info(f"{sym} carregado diretamente via mt5.copy_rates_from_pos")
                else:
                    logger.warning(f"{sym} sem dados em mt5.copy_rates_from_pos")
            else:
                logger.warning(f"{sym} não selecionado no MT5")
    except Exception as e:
        logger.warning(f"{sym} falha em mt5 direto: {e}")

    # Try backfill.ensure_history -> CSV
    try:
        if (df is None or (hasattr(df, "empty") and df.empty)) and ensure_history:
            df = ensure_history(sym, period_days=60, interval='15m')
            if df is not None and not getattr(df, "empty", False):
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
                    df = df.set_index('time').sort_index()
            logger.info(f"{sym} carregado via backfill.ensure_history")
    except Exception as e:
        logger.warning(f"{sym} falha em backfill.ensure_history: {e}")

    # Final check
    if df is None or (hasattr(df, "empty") and df.empty):
        logger.error(f"{sym} sem dados de nenhuma fonte")
        return None

    # ensure index is datetime and sorted
    try:
        if pd is not None and "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
            df = df.set_index("time").sort_index()
    except Exception:
        pass
    return df

def worker_wfo(sym: str, bars: int, maxevals: int, wfo_windows: int, train_period: int, test_period: int) -> Dict[str, Any]:
    out = {"symbol": sym, "status": "ok", "wfo_windows": []}
    try:
        df_full = load_series_with_backfill(sym, bars)
        if df_full is None:
            return {"symbol": sym, "error": "no_data"}
        df_full = df_full.sort_index()
        n = len(df_full)
        step = test_period
        wins = []
        max_windows = int(wfo_windows)
        for i in range(max_windows):
            train_start_idx = i * step
            train_end_idx = train_start_idx + train_period
            test_end_idx = train_end_idx + test_period
            if test_end_idx > n:
                break
            df_train = df_full.iloc[train_start_idx:train_end_idx].copy()
            df_test = df_full.iloc[train_end_idx:test_end_idx].copy()
            if df_train is None or getattr(df_train, "empty", True) or df_test is None or getattr(df_test, "empty", True):
                continue
            best_params = optimize_window(sym, df_train, maxevals)
            test_res = backtest_params_on_df(sym, best_params, df_test)
            wins.append({"train_range": (str(df_train.index[0]), str(df_train.index[-1])),
                         "test_range": (str(df_test.index[0]), str(df_test.index[-1])),
                         "best_params": best_params,
                         "test_metrics": test_res.get("metrics", {}),
                         "equity_curve": test_res.get("equity_curve", [])})
        if not wins:
            return {"symbol": sym, "error": "wfo_no_windows"}
        def score_win(w):
            m = w.get("test_metrics", {}) or {}
            return (m.get("total_return", 0.0) * 1.0) + (m.get("sharpe", 0.0) * 0.1)
        wins_sorted = sorted(wins, key=score_win, reverse=True)
        best_overall = wins_sorted[0]
        out["wfo_windows"] = wins
        out["selected_params"] = best_overall.get("best_params", {})
        # save per-symbol WFO summary
        fp = os.path.join(OPT_OUTPUT_DIR, f"WFO_{sym}.json")
        safe_save_json(fp, out)
        return out
    except Exception as e:
        logger.exception(f"WFO worker failed for {sym}: {e}")
        return {"symbol": sym, "error": str(e)}

def run_parallel_wfo(symbols: List[str], bars: int, maxevals: int, workers: int, wfo_windows: int, train_period: int, test_period: int):
    results = {}
    if not symbols:
        logger.info("No symbols provided")
        return results
    if workers <= 0:
        workers = max(1, min((os.cpu_count() or 1) - 1, 4))
    logger.info(f"Running WFO on {len(symbols)} symbols with {workers} workers")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(worker_wfo, s, bars, maxevals, wfo_windows, train_period, test_period): s for s in symbols}
        for fut in as_completed(futures):
            s = futures[fut]
            try:
                r = fut.result()
                results[s] = r
                if r.get("error"):
                    logger.warning(f"WFO {s} error: {r.get('error')}")
                else:
                    logger.info(f"WFO {s} done")
            except Exception as e:
                results[s] = {"symbol": s, "error": str(e)}
                logger.exception(f"WFO {s} failed")
    return results

def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Optimizer Final")
    parser.add_argument('--bars', type=int, default=4000)
    parser.add_argument('--maxevals', type=int, default=300)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--wfo_windows', type=int, default=WFO_WINDOWS)
    parser.add_argument('--train', type=int, default=TRAIN_PERIOD)
    parser.add_argument('--test', type=int, default=TEST_PERIOD)
    args = parser.parse_args()
    symbols = load_all_symbols()
    logger.info(f"Weekly WFO loading {len(symbols)} symbols from SECTOR_MAP")
    filtered = []
    for s in symbols:
        # attempt to load series (this will call backfill if necessary)
        df = None
        try:
            df = load_series_with_backfill(s, bars=args.bars)
        except Exception as e:
            logger.exception(f"{s} exception during load: {e}")
            df = None
        if df is None or (hasattr(df, 'empty') and df.empty):
            logger.warning(f"{s} - no data, skipped")
            continue
        filtered.append(s)
    logger.info(f"Symbols to optimize (with data): {len(filtered)}")
    results = run_parallel_wfo(filtered, bars=args.bars, maxevals=args.maxevals, workers=args.workers, wfo_windows=args.wfo_windows, train_period=args.train, test_period=args.test)
    fp = os.path.join(OPT_OUTPUT_DIR, "wfo_summary.json")
    safe_save_json(fp, {"generated_at": time.time(), "results": results})
    logger.info("WFO finished. Summary saved.")
if __name__ == '__main__':
    main()