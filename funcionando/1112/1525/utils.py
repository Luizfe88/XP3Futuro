# utils.py — utilitários do bot (versão atualizada)
import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

import pandas as pd
import numpy as np

logger = logging.getLogger("utils")

# -------------------------
# safe_copy_rates (robusto)
# -------------------------
def safe_copy_rates(symbol: str, timeframe, count: int = 500) -> Optional[pd.DataFrame]:
    """
    Robust wrapper for mt5.copy_rates... Returns DataFrame indexed by datetime or None.
    Expects timeframe as MT5 constant (e.g. mt5.TIMEFRAME_M15).
    """
    try:
        import MetaTrader5 as mt5
    except Exception:
        logger.warning("safe_copy_rates: MetaTrader5 não disponível.")
        return None

    try:
        if not mt5.initialize():
            try:
                mt5.initialize()
            except Exception:
                logger.warning("safe_copy_rates: MT5 init failed")
                return None

        attempts = 3
        for i in range(attempts):
            try:
                raw = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            except Exception as e:
                logger.debug(f"safe_copy_rates: copy_rates error {e}")
                raw = None

            if raw is None or len(raw) == 0:
                if i < attempts - 1:
                    import time; time.sleep(0.4)
                continue

            df = pd.DataFrame(raw)
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"], unit="s")
            else:
                df.reset_index(inplace=True); df["time"] = pd.to_datetime(df["time"], unit="s", errors="coerce")
            df = df.set_index("time").sort_index()
            for c in ["open", "high", "low", "close", "tick_volume", "volume"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            return df

        logger.debug(f"safe_copy_rates: sem dados para {symbol} após {attempts} tentativas")
        return None
    except Exception as e:
        logger.exception(f"safe_copy_rates ERROR {symbol}: {e}")
        return None

# -------------------------
# ATR
# -------------------------
def get_atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    try:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        return float(atr.iloc[-1])
    except Exception as e:
        logger.exception(f"get_atr ERROR: {e}")
        return None

# -------------------------
# posição / risco
# -------------------------
def calculate_position_size(symbol: str, sl_price: float, risk_pct: float = 0.01) -> Optional[float]:
    try:
        import MetaTrader5 as mt5
        acc = mt5.account_info()
        if not acc:
            return None
        equity = float(acc.equity or acc.balance or 0.0)
        risk_money = equity * risk_pct
        info = mt5.symbol_info(symbol)
        if not info:
            return None
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return None
        entry = float(tick.ask) if getattr(tick, "ask", None) and tick.ask > 0 else float(tick.bid)
        stop_distance = abs(entry - sl_price)
        if stop_distance <= 0:
            return None
        contract_size = getattr(info, "trade_contract_size", 1) or 1
        volume = risk_money / (stop_distance * contract_size)
        vol_step = getattr(info, "volume_step", 0.01) or 0.01
        min_vol = getattr(info, "volume_min", vol_step) or vol_step
        steps = max(1, int(round(volume / vol_step)))
        volume = max(min_vol, steps * vol_step)
        return float(volume)
    except Exception as e:
        logger.exception(f"calculate_position_size ERROR: {e}")
        return None

# -------------------------
# send order with SL/TP
# -------------------------
def send_order_with_sl_tp(symbol: str, side: str, volume: float, sl: float, tp: float) -> Dict[str, Any]:
    try:
        import MetaTrader5 as mt5
        info = mt5.symbol_info(symbol)
        if not info:
            return {"success": False, "reason": "symbol_info None"}
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return {"success": False, "reason": "no tick"}
        price = float(tick.ask) if side.upper() == "BUY" else float(tick.bid)
        tick_size = getattr(info, "trade_tick_size", 0.01) or 0.01
        sl = round(sl / tick_size) * tick_size
        tp = round(tp / tick_size) * tick_size
        if side.upper() == "BUY":
            if sl >= price:
                sl = price - (2 * tick_size)
            if tp <= price:
                tp = price + (2 * tick_size)
        else:
            if sl <= price:
                sl = price + (2 * tick_size)
            if tp >= price:
                tp = price - (2 * tick_size)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": mt5.ORDER_TYPE_BUY if side.upper() == "BUY" else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl,
            "tp": tp,
            "magic": 99,
            "comment": "bot_fast",
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        res = mt5.order_send(request)
        retcode = getattr(res, "retcode", None)
        if retcode is None:
            try:
                retcode = res.get("retcode")
            except Exception:
                retcode = None
        if retcode in (mt5.TRADE_RETCODE_DONE, 10009):
            return {"success": True, "order": getattr(res, "order", None)}
        else:
            reason = getattr(res, "comment", None) or str(res)
            return {"success": False, "reason": f"retcode={retcode} reason={reason}"}
    except Exception as e:
        logger.exception(f"send_order_with_sl_tp ERROR: {e}")
        return {"success": False, "reason": str(e)}

# -------------------------
# QUICK INDICATORS & SCAN
# -------------------------
def quick_indicators(symbol: str, timeframe, lookback: int = 300) -> Dict[str, Any]:
    out = {"symbol": symbol}
    try:
        df = safe_copy_rates(symbol, timeframe, lookback)
        if df is None or df.empty:
            out["error"] = "no_data"
            return out
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        # default EMAs for snapshot
        ema_short_n = 9
        ema_long_n = 21
        ema_fast = float(close.ewm(span=ema_short_n, adjust=False).mean().iloc[-1])
        ema_slow = float(close.ewm(span=ema_long_n, adjust=False).mean().iloc[-1])
        # RSI
        try:
            delta = close.diff()
            up = delta.clip(lower=0).rolling(14).mean()
            down = -delta.clip(upper=0).rolling(14).mean()
            rs = up / down
            rsi_val = float((100 - (100 / (1 + rs))).iloc[-1])
        except Exception:
            rsi_val = None
        # MOM
        try:
            mom = float(close.iloc[-1] / close.shift(10).iloc[-1] - 1) if len(close) > 10 else 0.0
        except Exception:
            mom = None
        # ATR
        try:
            atr_val = get_atr(df, 14)
        except Exception:
            atr_val = None
        # ADX using pandas_ta if possible, else fallback
        adx_val = None
        try:
            import pandas_ta as ta
            adx_series = ta.adx(high, low, close, length=14)
            if isinstance(adx_series, pd.DataFrame):
                for c in adx_series.columns:
                    if "ADX" in c.upper():
                        adx_val = float(adx_series[c].iloc[-1])
                        break
        except Exception:
            try:
                # naive ADX approx
                up_move = high.diff()
                down_move = -low.diff()
                plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
                minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
                tr1 = high - low
                tr2 = (high - close.shift(1)).abs()
                tr3 = (low - close.shift(1)).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr_s = tr.rolling(14).sum()
                if len(tr) >= 14:
                    atr_s.iloc[13] = tr.iloc[:14].sum()
                    for i in range(14, len(tr)):
                        atr_s.iloc[i] = atr_s.iloc[i-1] - (atr_s.iloc[i-1] / 14) + tr.iloc[i]
                    plus_dm_s = plus_dm.rolling(14).sum()
                    minus_dm_s = minus_dm.rolling(14).sum()
                    plus_di = 100 * (plus_dm_s / atr_s)
                    minus_di = 100 * (minus_dm_s / atr_s)
                    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di)) * 100
                    adx_val = float(dx.rolling(14).mean().iloc[-1])
            except Exception:
                adx_val = None
        # volume
        if "tick_volume" in df.columns:
            last_vol = int(df["tick_volume"].iloc[-1])
            vol_mean = int(df["tick_volume"].tail(20).mean())
        elif "volume" in df.columns:
            last_vol = int(df["volume"].iloc[-1])
            vol_mean = int(df["volume"].tail(20).mean())
        else:
            last_vol = 0
            vol_mean = 0
        out.update({
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "rsi": rsi_val,
            "mom": mom,
            "atr": atr_val,
            "adx": adx_val,
            "last_vol": last_vol,
            "vol_mean": vol_mean,
            "error": None
        })
        return out
    except Exception as e:
        logger.exception(f"quick_indicators ERROR {symbol}: {e}")
        out["error"] = "exception"
        return out

def scan_universe(symbols: List[str], timeframe, lookback: int = 300, workers: int = 8) -> Dict[str, Dict[str, Any]]:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    results = {}
    if not symbols:
        return results
    workers = min(workers, max(1, (os.cpu_count() or 4)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(quick_indicators, s, timeframe, lookback): s for s in symbols}
        for fut in as_completed(futures):
            s = futures[fut]
            try:
                res = fut.result()
                results[s] = res
            except Exception:
                logger.exception(f"scan_universe worker failed for {s}")
                results[s] = {"symbol": s, "error": "exception"}
    return results

# -------------------------
# loaders/savers (persistence)
# -------------------------
def save_bot_state(data: Dict[str, Any], file_path: str = None):
    try:
        import config
        fp = file_path or getattr(config, "BOT_STATE_FILE", "bot_state.json")
        tmp = fp + ".tmp"
        os.makedirs(os.path.dirname(fp) or ".", exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        os.replace(tmp, fp)
        logger.debug(f"save_bot_state: saved to {fp}")
        return True
    except Exception as e:
        logger.exception(f"save_bot_state ERROR: {e}")
        return False

def load_bot_state(file_path: str = None) -> Dict[str, Any]:
    try:
        import config
        fp = file_path or getattr(config, "BOT_STATE_FILE", "bot_state.json")
        if not os.path.exists(fp):
            return {}
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data or {}
    except Exception as e:
        logger.exception(f"load_bot_state ERROR: {e}")
        return {}

def load_optimized_summaries(symbols: List[str], base_dir: str = None) -> Dict[str, Dict[str, Any]]:
    """
    Lê optimizer_output/<SYMBOL>.json e retorna mapping.
    Se arquivo ausente, não erra — apenas não inclui o símbolo.
    """
    out = {}
    try:
        import config
    except Exception:
        config = None
    base_dir = base_dir or (getattr(config, "OPTIMIZER_OUTPUT", "optimizer_output") if config else "optimizer_output")
    if not os.path.isabs(base_dir):
        base_dir = os.path.join(os.getcwd(), base_dir)
    for s in symbols:
        fp = os.path.join(base_dir, f"{s}.json")
        if not os.path.exists(fp):
            continue
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            # normalize and return minimal flat dict
            mapped = {
                "ema_short": int(data.get("ema_short", getattr(config, "DEFAULT_PARAMS", {}).get("ema_short", 9))),
                "ema_long": int(data.get("ema_long", getattr(config, "DEFAULT_PARAMS", {}).get("ema_long", 21))),
                "rsi_period": int(data.get("rsi_period", getattr(config, "DEFAULT_PARAMS", {}).get("rsi_period", 14))),
                "adx_period": int(data.get("adx_period", getattr(config, "DEFAULT_PARAMS", {}).get("adx_period", 14))),
                "adx_threshold": float(data.get("adx_threshold", getattr(config, "DEFAULT_PARAMS", {}).get("adx_threshold", 20))),
                "rsi_low": float(data.get("rsi_low", getattr(config, "DEFAULT_PARAMS", {}).get("rsi_low", 30))),
                "rsi_high": float(data.get("rsi_high", getattr(config, "DEFAULT_PARAMS", {}).get("rsi_high", 70))),
                "mom_min": float(data.get("mom_min", getattr(config, "DEFAULT_PARAMS", {}).get("mom_min", 0.0)))
            }
            out[s] = mapped
        except Exception as e:
            logger.exception(f"load_optimized_summaries: erro lendo {fp}: {e}")
    return out
