# utils.py — XP3 Utils B3 (CONSOLIDADO FINAL)

import time
import logging
from datetime import datetime
from typing import Optional, Dict, Any

import pandas as pd
import config

logger = logging.getLogger("utils")

# =========================
# MT5 SAFE COPY
# =========================
def safe_copy_rates(symbol: str, timeframe, count: int = 500) -> Optional[pd.DataFrame]:
    try:
        import MetaTrader5 as mt5
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None or len(rates) == 0:
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        return df.sort_index()
    except Exception:
        logger.exception(f"safe_copy_rates erro {symbol}")
        return None

# =========================
# FAST RATES (ANTI-SPAM)
# =========================
_last_bar_time = {}

def get_fast_rates(symbol, timeframe):
    df = safe_copy_rates(symbol, timeframe, 3)
    if df is None or df.empty:
        return None

    last = df.index[-1]
    if _last_bar_time.get(symbol) == last:
        return None

    _last_bar_time[symbol] = last
    return df

# =========================
# INDICADORES
# =========================
def get_atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    return float(atr.iloc[-1])

def get_intraday_vwap(df: pd.DataFrame) -> Optional[float]:
    today = datetime.now().date()
    df = df[df.index.date == today]
    if df.empty:
        return None

    pv = (df["close"] * df["tick_volume"]).sum()
    vol = df["tick_volume"].sum()
    return float(pv / vol) if vol > 0 else None

def quick_indicators(
    symbol: str,
    timeframe,
    lookback: int = 300,
    df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:

    if df is None:
        df = safe_copy_rates(symbol, timeframe, lookback)

    if df is None or len(df) < 50:
        return {"error": "no_data"}

    close = df["close"]
    ema_fast = close.ewm(span=9).mean().iloc[-1]
    ema_slow = close.ewm(span=21).mean().iloc[-1]

    delta = close.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = -delta.clip(upper=0).rolling(14).mean()
    rs = up / down
    rsi = (100 - (100 / (1 + rs))).iloc[-1]

    atr = get_atr(df)

    return {
        "ema_fast": float(ema_fast),
        "ema_slow": float(ema_slow),
        "rsi": float(rsi),
        "atr": atr,
        "error": None
    }

# =========================
# RISCO DINÂMICO
# =========================
def get_current_risk_pct():
    now = datetime.now()
    if now.weekday() == 4 and now.strftime("%H:%M") >= config.FRIDAY_REDUCED_RISK_AFTER:
        return config.REDUCED_RISK_PCT
    return config.RISK_PER_TRADE_PCT

# =========================
# ORDENS
# =========================
def calculate_position_size(symbol, sl_price):
    import MetaTrader5 as mt5
    from bot import current_indicators

    acc = mt5.account_info()
    info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if not acc or not info or not tick:
        return None

    atr = current_indicators.get(symbol, {}).get("atr")
    risk_pct = get_current_risk_pct()

    risk_money = acc.equity * risk_pct
    entry = tick.ask
    stop_dist = abs(entry - sl_price)
    if stop_dist <= 0:
        return None

    volume = risk_money / stop_dist
    step = info.volume_step
    volume = max(info.volume_min, round(volume / step) * step)
    return float(volume)

def send_order_with_sl_tp(symbol, side, volume, sl, tp):
    import MetaTrader5 as mt5
    tick = mt5.symbol_info_tick(symbol)
    price = tick.ask if side == "BUY" else tick.bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": sl,
        "tp": tp,
        "magic": 2026,
        "comment": "XP3",
        "type_filling": mt5.ORDER_FILLING_IOC
    }

    return mt5.order_send(request)
