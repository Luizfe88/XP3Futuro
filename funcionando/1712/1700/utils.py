# utils.py — XP3 Utils B3 (CONSOLIDADO FINAL)

import time
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
import MetaTrader5 as mt5
import pandas as pd
import config
import numpy as np
import MetaTrader5 as mt5

TIMEFRAME_BASE = mt5.TIMEFRAME_M15  # fallback padrão
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

def quick_indicators_custom(
    symbol: str,
    timeframe,
    df: Optional[pd.DataFrame] = None,
    params: Dict[str, Any] = None
) -> Dict[str, Any]:
    if params is None:
        params = {}

    if df is None:
        df = safe_copy_rates(symbol, timeframe, 300)

    if df is None or len(df) < 50:
        return {"error": "no_data"}

    close = df["close"]

    ema_short = params.get("ema_short", 9)
    ema_long = params.get("ema_long", 21)
    rsi_low = params.get("rsi_low", 35)
    rsi_high = params.get("rsi_high", 70)

    ema_fast = close.ewm(span=ema_short, adjust=False).mean().iloc[-1]
    ema_slow = close.ewm(span=ema_long, adjust=False).mean().iloc[-1]

    delta = close.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = -delta.clip(upper=0).rolling(14).mean()
    rs = up / down
    rsi = (100 - (100 / (1 + rs))).iloc[-1]

    atr = get_atr(df)
    adx = get_adx(df)  # <<< NOVO: ADX calculado

    return {
        "ema_fast": float(ema_fast),
        "ema_slow": float(ema_slow),
        "rsi": float(rsi),
        "atr": atr,
        "adx": adx if adx is not None else 0.0,  # <<< NOVO
        "error": None
    }

def calculate_position_size_custom(symbol, sl_price, risk_pct):
    import MetaTrader5 as mt5
    from bot import current_indicators

    acc = mt5.account_info()
    info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if not acc or not info or not tick:
        return None

    atr = current_indicators.get(symbol, {}).get("atr")
    risk_money = acc.equity * risk_pct
    entry = tick.ask
    stop_dist = abs(entry - sl_price)
    if stop_dist <= 0:
        return None

    volume = risk_money / stop_dist
    step = info.volume_step
    volume = max(info.volume_min, round(volume / step) * step)
    return float(volume)

# utils.py — adicione no final do arquivo

def calculate_correlation_matrix(symbols: List[str], timeframe=mt5.TIMEFRAME_M15, lookback_days: int = 60) -> Dict[str, Dict[str, float]]:
    """
    Calcula matriz de correlação entre ativos usando retornos diários.
    Retorna dict: {sym1: {sym2: corr, ...}}
    """
    if len(symbols) < 2:
        return {}

    import MetaTrader5 as mt5
    from datetime import timedelta
    import pandas as pd

    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days * 2)  # margem extra

    rates_dict = {}
    for sym in symbols:
        rates = mt5.copy_rates_range(sym, timeframe, start_date, end_date)
        if rates is None or len(rates) < 100:
            continue
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        rates_dict[sym] = df["close"]

    if len(rates_dict) < 2:
        return {}

    # Align e calcular retornos diários
    closes = pd.DataFrame(rates_dict)
    daily = closes.resample("D").last().dropna()
    returns = daily.pct_change().dropna()

    if len(returns) < 20:
        return {}

    corr_matrix = returns.corr().to_dict()
    return corr_matrix

def get_avg_volume(df: pd.DataFrame, period: int = 20) -> float:
    return float(df["tick_volume"].rolling(period).mean().iloc[-1])

def get_open_gap(symbol: str, timeframe) -> Optional[float]:
    df = safe_copy_rates(symbol, timeframe, 50)
    if df is None or len(df) < 2:
        return None
    prev_close = df["close"].iloc[-2]
    today_open = df["open"].iloc[-1]
    gap = (today_open - prev_close) / prev_close
    return abs(gap)

def calculate_sl_price(entry_price: float, side: str, atr: float):
    mult = config.SL_ATR_MULTIPLIER
    if side == "BUY":
        return entry_price - atr * mult
    else:
        return entry_price + atr * mult

def calculate_position_size_atr(equity: float, risk_pct: float, atr: float):
    if atr <= 0:
        return None
    risk_money = equity * risk_pct
    stop_dist = atr * config.SL_ATR_MULTIPLIER
    volume = risk_money / stop_dist
    # Ajuste mínimo/máximo/step (mesmo que antes)
    info = mt5.symbol_info(symbol)
    if info:
        step = info.volume_step
        volume = max(info.volume_min, round(volume / step) * step)
    return float(volume)

import telebot
from datetime import datetime
import config

_bot_instance = None

def get_telegram_bot():
    global _bot_instance
    if _bot_instance is None and config.ENABLE_TELEGRAM_NOTIF:
        _bot_instance = telebot.TeleBot(config.TELEGRAM_BOT_TOKEN)
    return _bot_instance

def send_telegram_trade(symbol: str, side: str, volume: float, price: float, sl: float = None, tp: float = None):
    bot = get_telegram_bot()
    if not bot:
        return

    msg = f"🚨 <b>XP3 EXECUTOU ORDEM!</b>\n\n" \
          f"📊 <b>Ativo:</b> {symbol}\n" \
          f"📈 <b>Direção:</b> {side.upper()}\n" \
          f"📦 <b>Volume:</b> {volume:.0f} ações\n" \
          f"💰 <b>Preço:</b> R${price:.2f}\n"
    if sl:
        msg += f"🛑 <b>Stop Loss:</b> R${sl:.2f}\n"
    if tp:
        msg += f"🎯 <b>Take Profit:</b> R${tp:.2f}\n"
    msg += f"⏰ <b>Hora:</b> {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"

    try:
        bot.send_message(
            chat_id=config.TELEGRAM_CHAT_ID,
            text=msg,
            parse_mode="HTML"  # Para negrito e emojis ficarem bonitos
        )
        print("✅ Notificação Telegram enviada!")
    except Exception as e:
        print(f"❌ Erro Telegram: {e}")

def send_telegram_exit(symbol: str, side: str, volume: float, entry_price: float, exit_price: float, profit_loss: float):
    bot = get_telegram_bot()
    if not bot:
        return

    pl_color = "🟢" if profit_loss > 0 else "🔴" if profit_loss < 0 else "⚪"
    pl_str = f"{profit_loss:+.2f} ({(profit_loss / (entry_price * volume)) * 100:+.2f}%)"

    msg = f"🚨 <b>XP3 FECHOU POSIÇÃO!</b>\n\n" \
          f"📊 <b>Ativo:</b> {symbol}\n" \
          f"📈 <b>Direção:</b> {side.upper()}\n" \
          f"📦 <b>Volume:</b> {volume:.0f} ações\n" \
          f"💰 <b>Entrada:</b> R${entry_price:.2f} → <b>Saída:</b> R${exit_price:.2f}\n" \
          f"{pl_color} <b>P&L:</b> R${pl_str}\n" \
          f"⏰ <b>Hora:</b> {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"

    try:
        bot.send_message(
            chat_id=config.TELEGRAM_CHAT_ID,
            text=msg,
            parse_mode="HTML"
        )
        print("✅ Notificação de saída Telegram enviada!")
    except Exception as e:
        print(f"❌ Erro Telegram saída: {e}")

def get_adx(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """
    Calcula o ADX (Average Directional Index) com período padrão 14.
    Retorna o valor atual do ADX.
    """
    if len(df) < period * 2:
        return None

    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = low.diff() * -1

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/period, adjust=False).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/period, adjust=False).mean()

    return float(adx.iloc[-1])


def check_and_close_orphans(elite_symbols: dict):
    """
    Verifica se há posições abertas no MT5 que não constam no dicionário ELITE_SYMBOLS.
    Se encontrar, fecha a posição imediatamente.
    """
    import MetaTrader5 as mt5
    
    # Obtém todas as posições abertas
    positions = mt5.positions_get()
    
    if positions is None or len(positions) == 0:
        print("ℹ️ Nenhuma posição aberta para verificar.")
        return

    print(f"🔍 Verificando {len(positions)} posições abertas contra a nova Elite...")

    for pos in positions:
        symbol = pos.symbol
        
        # Se o ativo não estiver no dicionário da Elite
        if symbol not in elite_symbols:
            print(f"⚠️ ATENÇÃO: {symbol} não pertence mais à ELITE. Encerrando...")
            
            # Prepara a ordem de fecho (Market Order inversa)
            tick = mt5.symbol_info_tick(symbol)
            type_close = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
            price_close = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": pos.volume,
                "type": type_close,
                "position": pos.ticket,
                "price": price_close,
                "deviation": 10,
                "magic": pos.magic,
                "comment": "Fechamento por saída da Elite",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"✅ {symbol} encerrado com sucesso.")
            else:
                print(f"❌ Erro ao fechar {symbol}: {result.comment}")

def calculate_signal_score(symbol: str, ind: dict, params: dict, current_symbols: List[str], timeframe) -> float:
    score = 0.0

    # Força da tendência EMA
    ema_diff = abs(ind["ema_fast"] - ind["ema_slow"]) / ind["ema_slow"]
    score += min(ema_diff * 10000, 30)

    # RSI na zona ideal
    rsi = ind["rsi"]
    if ind["ema_fast"] > ind["ema_slow"]:
        distance = 70 - rsi
        score += min(distance * 1.5, 20)
    else:
        distance = rsi - 30
        score += min(distance * 1.5, 20)

    # ADX forte
    adx = ind.get("adx", 0)
    if adx >= params.get("adx_threshold", 25):
        score += min((adx - 25) * 1.5, 25)

    # VWAP intraday
    df_vwap = safe_copy_rates(symbol, timeframe, 100)  # ← CORRIGIDO AQUI
    if df_vwap is not None and not df_vwap.empty:
        vwap = get_intraday_vwap(df_vwap)
        price = ind.get("close", 0)
        if vwap and price > 0:
            if (ind["ema_fast"] > ind["ema_slow"] and price > vwap) or \
               (ind["ema_fast"] < ind["ema_slow"] and price < vwap):
                score += 15

    # Macro trend alinhado (pontos fixos ou implementar macro_trend_ok aqui se preferir)
    score += 20

    # Volatilidade ideal
    price = ind.get("close", 0) or 0.01
    atr_pct = ind.get("atr", 0) / price
    if 0.005 <= atr_pct <= 0.03:
        score += 15
    elif atr_pct > 0.05:
        score -= 20

    # Penalidade por correlação
    if len(current_symbols) > 0:
        avg_corr = get_average_correlation_with_portfolio(symbol, current_symbols)
        if avg_corr > 0.8:
            score -= 40
        elif avg_corr > 0.6:
            score -= 20
        elif avg_corr > 0.4:
            score -= 10

    return max(score, 0)