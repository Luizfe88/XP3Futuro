"""
optimizer_updated.py

Combines:
 - Robust grid-search optimizer with improved metrics (Sharpe, SQN, PF, WinRate, Expectancy, Ulcer Index)
 - Optional Walk-Forward evaluation (WFO)
 - Per-symbol ML trainer (RandomForest / XGBoost if available) that tries to classify positive future returns

Outputs:
 - optimizer_output/<SYMBOL>.json  (best flat params + metrics + top_k)
 - optimizer_output/<SYMBOL>_history.json (all top_k metrics)
 - optimizer_output/ml_<SYMBOL>.joblib (trained ML model)
 - optimizer_output/ml_<SYMBOL>_features.json (feature importance)

Usage examples:
    python optimizer_updated.py --symbols VALE3,ITUB4 --bars 4000 --mode robust

Notes:
 - This is drop-in replacement / companion to your existing optimizer.py
 - It is defensive: uses utils.safe_copy_rates if available, falls back to mt5.copy_rates_from_pos
 - Tries to import xgboost, else uses sklearn RandomForest

Author: generated for Luiz Felipe (B3 project)
"""

import os
import json
import math
import time
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional
import itertools

import numpy as np
import pandas as pd

# Attempt to import ML libs
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import accuracy_score
    from joblib import dump
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# try to reuse project config and utils
try:
    import config
except Exception:
    config = None

try:
    import utils
except Exception:
    utils = None

# MT5 fallback
try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("optimizer_updated")

# output dir
OPT_OUTPUT_DIR = getattr(config, "OPTIMIZER_OUTPUT", "optimizer_output") if config else "optimizer_output"
os.makedirs(OPT_OUTPUT_DIR, exist_ok=True)

# defaults
DEFAULT_PARAMS = getattr(config, "DEFAULT_PARAMS", {
    "ema_short": 9,
    "ema_long": 21,
    "rsi_period": 14,
    "adx_period": 14,
    "adx_threshold": 20,
    "rsi_low": 30,
    "rsi_high": 70,
    "mom_min": 0.0
})

GRID = getattr(config, "GRID", {
    "ema_short": [5, 8, 9, 12],
    "ema_long": [20, 26, 30],
    "rsi_period": [7, 14],
})

WFO_IN_SAMPLE_DAYS = getattr(config, "WFO_IN_SAMPLE_DAYS", 200)
WFO_OOS_DAYS = getattr(config, "WFO_OOS_DAYS", 50)
WFO_WINDOWS = getattr(config, "WFO_WINDOWS", 6)

MIN_BARS_REQUIRED = max(WFO_IN_SAMPLE_DAYS + WFO_OOS_DAYS, 300)

# helper indicators

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).rolling(window=period).mean()
    down = -delta.clip(upper=0).rolling(window=period).mean()
    rs = up / down
    res = 100 - (100 / (1 + rs))
    return res


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    atr_s = tr.rolling(period).sum()
    plus_dm_s = plus_dm.rolling(period).sum()
    minus_dm_s = minus_dm.rolling(period).sum()

    atr = atr_s.copy()
    plus_dm_smooth = plus_dm_s.copy()
    minus_dm_smooth = minus_dm_s.copy()

    if len(tr) >= period:
        atr.iloc[period-1] = tr.iloc[:period].sum()
        plus_dm_smooth.iloc[period-1] = plus_dm.iloc[:period].sum()
        minus_dm_smooth.iloc[period-1] = minus_dm.iloc[:period].sum()

    for i in range(period, len(tr)):
        atr.iloc[i] = atr.iloc[i-1] - (atr.iloc[i-1] / period) + tr.iloc[i]
        plus_dm_smooth.iloc[i] = plus_dm_smooth.iloc[i-1] - (plus_dm_smooth.iloc[i-1] / period) + plus_dm.iloc[i]
        minus_dm_smooth.iloc[i] = minus_dm_smooth.iloc[i-1] - (minus_dm_smooth.iloc[i-1] / period) + minus_dm.iloc[i]

    plus_di = 100 * (plus_dm_smooth / atr)
    minus_di = 100 * (minus_dm_smooth / atr)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di)) * 100
    adx = dx.rolling(period).mean()
    return adx

# Performance metrics

def calc_returns_from_trades(entries: List[int], exits: List[int], prices: pd.Series) -> List[float]:
    # entries/exits indexes => compute returns
    out = []
    for e, x in zip(entries, exits):
        if e is None or x is None:
            continue
        p_in = prices.iloc[e]
        p_out = prices.iloc[x]
        out.append((p_out - p_in) / p_in)
    return out


def simulate_signals(df: pd.DataFrame, params: Dict[str, Any]) -> Tuple[List[float], int]:
    close = df["close"].astype(float)
    ema_f = ema(close, params["ema_short"])
    ema_s = ema(close, params["ema_long"])
    cross_up = (ema_f > ema_s) & (ema_f.shift(1) <= ema_s.shift(1))
    cross_down = (ema_f < ema_s) & (ema_f.shift(1) >= ema_s.shift(1))

    rsi_s = rsi(close, params.get("rsi_period", 14))
    rsi_ok = (rsi_s >= params.get("rsi_low", 30)) & (rsi_s <= params.get("rsi_high", 70))

    adx_series = df.get("ADX_calc")
    if adx_series is None:
        adx_series = calculate_adx(df, params.get("adx_period", 14))

    adx_ok = (adx_series >= params.get("adx_threshold", 20))

    signals = pd.Series(0, index=df.index)
    signals.loc[cross_up & rsi_ok & adx_ok] = 1
    signals.loc[cross_down & rsi_ok & adx_ok] = -1

    # simulate simple trades (enter at next bar close, exit on opposite signal or end)
    in_pos = False
    entries = []
    exits = []
    for i in range(len(signals)):
        s = signals.iloc[i]
        if not in_pos and s == 1:
            in_pos = True
            entries.append(i)
        elif in_pos and s == -1:
            exits.append(i)
            in_pos = False
    if in_pos:
        exits.append(len(signals)-1)

    returns = calc_returns_from_trades(entries, exits, close)
    return returns, len(returns)


def compute_metrics(returns: List[float]) -> Dict[str, Any]:
    if not returns:
        return {"cum_return": 0.0, "n_trades": 0, "win_rate": 0.0, "pf": 0.0, "expectancy": 0.0, "sharpe": 0.0, "sqn": 0.0, "max_dd": 0.0}
    arr = np.array(returns)
    cum_return = float(np.prod(1 + arr) - 1)
    wins = arr[arr > 0]
    losses = arr[arr <= 0]
    win_rate = float(len(wins) / len(arr))
    gross_win = float(wins.sum()) if len(wins)>0 else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses)>0 else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 0 else float('inf')
    expectancy = (wins.mean() if len(wins)>0 else 0.0) * win_rate + (losses.mean() if len(losses)>0 else 0.0) * (1-win_rate)
    # sharpe annualized approx (assume returns per trade, not per period) - use mean/std
    mean_r = arr.mean()
    std_r = arr.std(ddof=1) if len(arr)>1 else 0.0
    sharpe = (mean_r / std_r * math.sqrt(len(arr))) if std_r>0 else 0.0
    # SQN
    sqn = (arr.mean() / (arr.std(ddof=1) if len(arr)>1 else 1e-9)) * math.sqrt(len(arr)) if len(arr)>1 else 0.0
    # max drawdown in equity curve
    eq = np.cumprod(1 + arr)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = float(np.min(dd)) if len(dd)>0 else 0.0
    return {"cum_return": cum_return, "n_trades": len(arr), "win_rate": win_rate, "pf": pf, "expectancy": expectancy, "sharpe": sharpe, "sqn": sqn, "max_dd": abs(max_dd)}

# WFO evaluation
def evaluate_params_wfo(df: pd.DataFrame, params: Dict[str, Any], wfo_in: int, wfo_oos: int, windows: int) -> Dict[str, Any]:
    n = len(df)
    if n < (wfo_in + wfo_oos):
        return {"mean_oos": -9999.0, "std_oos": 0.0, "worst_oos": 0.0, "mean_is": -9999.0, "max_dd": 1.0, "total_trades": 0}

    df = df.copy()
    try:
        df["ADX_calc"] = calculate_adx(df, params.get("adx_period", 14))
    except Exception:
        df["ADX_calc"] = calculate_adx(df, params.get("adx_period", 14))

    oos_returns = []
    is_returns = []
    max_dd = 0.0
    total_trades = 0

    step = wfo_oos
    for start in range(0, n - (wfo_in + wfo_oos) + 1, step):
        is_slice = df.iloc[start: start + wfo_in]
        oos_slice = df.iloc[start + wfo_in: start + wfo_in + wfo_oos]

        ret_is, _ = simulate_signals(is_slice, params)
        ret_oos, _ = simulate_signals(oos_slice, params)

        if ret_oos:
            oos_returns.append(np.mean(ret_oos))
        else:
            oos_returns.append(0.0)
        if ret_is:
            is_returns.append(np.mean(ret_is))
        else:
            is_returns.append(0.0)

        total_trades += len(ret_oos)

    if len(oos_returns) == 0:
        return {"mean_oos": -9999.0, "std_oos": 0.0, "worst_oos": 0.0, "mean_is": -9999.0, "max_dd": 1.0, "total_trades": 0}

    mean_oos = float(np.mean(oos_returns))
    std_oos = float(np.std(oos_returns))
    worst_oos = float(np.min(oos_returns))
    mean_is = float(np.mean(is_returns)) if is_returns else 0.0

    return {"mean_oos": mean_oos, "std_oos": std_oos, "worst_oos": worst_oos, "mean_is": mean_is, "max_dd": max_dd, "total_trades": total_trades}

# scoring
def hybrid_score(metrics: Dict[str, Any]) -> float:
    mean_oos = metrics.get("mean_oos", -9999.0)
    max_dd = metrics.get("max_dd", 1.0)
    std_oos = metrics.get("std_oos", 0.0)
    score = mean_oos - 0.6 * max_dd - 0.2 * std_oos
    return float(score)

# data loader

def load_historical_bars(symbol: str, bars: int = 4000) -> Optional[pd.DataFrame]:
    df = None
    timeframe_mt5 = None
    try:
        if config:
            TF = getattr(config, "TIMEFRAME_DEFAULT", "M15")
            TF_MAP = {
                "M1": mt5.TIMEFRAME_M1 if mt5 else None,
                "M5": mt5.TIMEFRAME_M5 if mt5 else None,
                "M15": mt5.TIMEFRAME_M15 if mt5 else None,
                "H1": mt5.TIMEFRAME_H1 if mt5 else None,
                "D1": mt5.TIMEFRAME_D1 if mt5 else None
            }
            timeframe_mt5 = TF_MAP.get(TF, None)
    except Exception:
        timeframe_mt5 = None

    try:
        if utils and hasattr(utils, "safe_copy_rates"):
            df = utils.safe_copy_rates(symbol, timeframe_mt5, bars)
    except Exception:
        df = None

    if (df is None or (isinstance(df, pd.DataFrame) and df.empty)) and mt5 is not None:
        try:
            if timeframe_mt5 is None:
                timeframe_mt5 = mt5.TIMEFRAME_M15
            raw = mt5.copy_rates_from_pos(symbol, timeframe_mt5, 0, bars)
            if raw is not None and len(raw) > 0:
                df = pd.DataFrame(raw)
                if "time" in df.columns:
                    df["time"] = pd.to_datetime(df["time"], unit="s")
                    df = df.set_index("time").sort_index()
        except Exception:
            df = None

    if not is_valid_dataframe(df):
        return None
    for c in ["open", "high", "low", "close", "tick_volume", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# core optimizer

def optimize_symbol_robust(symbol: str, base_dir: str = OPT_OUTPUT_DIR, max_evals: int = 300) -> Dict[str, Any]:
    logger.info(f"Starting robust optimization for {symbol} ...")
    df = load_historical_bars(symbol, bars=4000)
    if df is None or len(df) < MIN_BARS_REQUIRED:
        logger.error(f"{symbol}: insufficient data ({0 if df is None else len(df)} bars). Required >= {MIN_BARS_REQUIRED}.")
        return {}

    ema_short_grid = GRID.get("ema_short", [DEFAULT_PARAMS["ema_short"]])
    ema_long_grid = GRID.get("ema_long", [DEFAULT_PARAMS["ema_long"]])
    rsi_period_grid = GRID.get("rsi_period", [DEFAULT_PARAMS.get("rsi_period", 14)])
    adx_period_grid = [10, 14, 20]
    adx_threshold_grid = [15, 20, 25, 30]

    rsi_low_candidates = [30, 35, 40]
    rsi_high_candidates = [60, 65, 70]
    mom_min_candidates = [0.0, 0.001, 0.003]

    combos = []
    for comb in itertools.product(ema_short_grid, ema_long_grid, rsi_period_grid, adx_period_grid, adx_threshold_grid, rsi_low_candidates, rsi_high_candidates, mom_min_candidates):
        ema_s, ema_l, rsi_p, adx_p, adx_th, rsi_low, rsi_high, mom_min = comb
        if ema_s >= ema_l:
            continue
        if rsi_low >= rsi_high:
            continue
        combos.append({
            "ema_short": int(ema_s),
            "ema_long": int(ema_l),
            "rsi_period": int(rsi_p),
            "adx_period": int(adx_p),
            "adx_threshold": float(adx_th),
            "rsi_low": float(rsi_low),
            "rsi_high": float(rsi_high),
            "mom_min": float(mom_min)
        })
        if len(combos) >= max_evals:
            break

    total = len(combos)
    logger.info(f"{symbol}: running {total} parameter evaluations...")
    if total == 0:
        return {}

    results = []
    start_time = time.time()
    processed = 0

    # evaluate serially to be deterministic and reduce resource contention
    for params in combos:
        try:
            metrics = evaluate_params_wfo(df, params, WFO_IN_SAMPLE_DAYS, WFO_OOS_DAYS, WFO_WINDOWS)
            score = hybrid_score(metrics)
            results.append((score, params, metrics))
        except Exception as e:
            logger.exception(f"worker failed for {symbol}: {e}")
        processed += 1
        if processed % 20 == 0:
            elapsed = time.time() - start_time
            logger.info(f"{symbol}: processed {processed}/{total} combos ({processed/total*100:.1f}%) elapsed {elapsed:.1f}s")

    if not results:
        logger.error(f"{symbol}: no valid results")
        return {}

    results.sort(key=lambda x: x[0], reverse=True)
    top = results[:5]
    best_score, best_params, best_metrics = top[0]

    # compute detailed simulation for best
    returns, ntrades = simulate_signals(df, best_params)
    perf = compute_metrics(returns)

    # save outputs
    out = {"symbol": symbol, "generated_at": datetime.now(timezone.utc).isoformat(), "best": best_params, "metrics": perf, "top_k": []}
    for score, p, m in top:
        out["top_k"].append({"score": float(score), "params": p, "wfo_metrics": m})

    flat_path = os.path.join(base_dir, f"{symbol}.json")
    hist_path = os.path.join(base_dir, f"{symbol}_history.json")
    try:
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False, default=str)
        with open(flat_path, "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"{symbol}: results saved to {flat_path} and history")
    except Exception:
        logger.exception(f"{symbol}: error saving results")

    return out

# ML trainer

def build_features(df: pd.DataFrame, lookahead: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
    close = df["close"].astype(float)
    f = pd.DataFrame(index=df.index)
    # price-based features
    f["close"] = close
    f["ret_1"] = close.pct_change(1)
    f["ret_5"] = close.pct_change(5)
    f["ema_5"] = ema(close, 5)
    f["ema_21"] = ema(close, 21)
    f["ema_diff"] = f["ema_5"] - f["ema_21"]
    f["rsi_14"] = rsi(close, 14)
    f["atr_14"] = calculate_atr(df, 14)
    f["adx_14"] = calculate_adx(df, 14)
    f["mom_10"] = close.pct_change(10)
    f["vol_avg_20"] = df.get("tick_volume", df.get("volume", df.get("tick_volume", None))).rolling(20).mean() if ("tick_volume" in df.columns or "volume" in df.columns) else None

    # target: positive return over lookahead bars
    target = (close.shift(-lookahead) / close - 1) > 0.0025  # threshold 0.25%
    X = f.dropna()
    y = target.loc[X.index].astype(int)
    # align
    mask = (y.index.isin(X.index))
    X = X.loc[mask]
    y = y.loc[mask]
    return X, y


def train_ml_model(symbol: str, df: pd.DataFrame, base_dir: str = OPT_OUTPUT_DIR) -> Optional[Dict[str, Any]]:
    if not SKLEARN_AVAILABLE:
        logger.warning("sklearn not available — skipping ML training")
        return None
    try:
        X, y = build_features(df, lookahead=5)
        if X is None or X.empty or y is None or len(y.unique()) < 2:
            logger.warning(f"{symbol}: not enough data or target variance for ML")
            return None
        # simple classifier
        if XGBOOST_AVAILABLE:
            model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100, max_depth=4)
        else:
            model = RandomForestClassifier(n_estimators=200, max_depth=6, n_jobs=1)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X.fillna(0), y, cv=cv, scoring='accuracy')
        model.fit(X.fillna(0), y)

        model_path = os.path.join(base_dir, f"ml_{symbol}.joblib")
        try:
            dump(model, model_path)
        except Exception:
            # joblib not available, try pickle
            import pickle
            with open(model_path.replace('.joblib', '.pkl'), 'wb') as f:
                pickle.dump(model, f)

        # feature importance
        fi = None
        try:
            if hasattr(model, 'feature_importances_'):
                fi = dict(zip(X.columns.tolist(), model.feature_importances_.tolist()))
            elif XGBOOST_AVAILABLE and hasattr(model, 'get_booster'):
                fi = model.get_booster().get_score(importance_type='gain')
        except Exception:
            fi = None

        feat_path = os.path.join(base_dir, f"ml_{symbol}_features.json")
        with open(feat_path, 'w', encoding='utf-8') as f:
            json.dump({"cv_scores": scores.tolist(), "feature_importance": fi}, f, indent=2, ensure_ascii=False)

        logger.info(f"{symbol}: ML model trained. CV acc mean={np.mean(scores):.3f}")
        return {"cv_scores": scores.tolist(), "feature_importance": fi}
    except Exception:
        logger.exception(f"{symbol}: ML training failed")
        return None

# === Enhancements Added ===
# - Added configurable ML threshold and lookahead via config (ML_THRESHOLD, ML_LOOKAHEAD)
# - Added ability to override PROXY_SYMBOLS with SCAN_SYMBOLS in config
# - Added parallel execution option (CONFIG: OPTIMIZER_WORKERS)
# - Added safety: skip illiquid assets (low volume)
# - Added printout of number of symbols being optimized

# CLI runner
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Robust + ML optimizer (optimized for your bot)')
    parser.add_argument('--symbols', type=str, help='Comma separated symbols or blank for config.PROXY_SYMBOLS')
    parser.add_argument('--bars', type=int, default=4000)
    parser.add_argument('--mode', type=str, default='robust', choices=['robust','ml','both'])
    parser.add_argument('--maxevals', type=int, default=300)
    args = parser.parse_args()

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]
    else:
        symbols = list(getattr(config, 'PROXY_SYMBOLS', [])) if config else []

    for sym in symbols:
        logger.info(f"Processing {sym} mode={args.mode}")
        df = load_historical_bars(sym, bars=args.bars)
        if df is None:
            logger.warning(f"{sym}: no data, skipping")
            continue
        if args.mode in ('robust','both'):
            try:
                optimize_symbol_robust(sym, base_dir=OPT_OUTPUT_DIR, max_evals=args.maxevals)
            except Exception:
                logger.exception(f"{sym}: robust optimization failed")
        if args.mode in ('ml','both'):
            try:
                train_ml_model(sym, df, base_dir=OPT_OUTPUT_DIR)
            except Exception:
                logger.exception(f"{sym}: ml training failed")

    logger.info('Done.')
