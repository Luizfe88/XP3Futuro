# utils.py — XP3 Utils B3 (CORRIGIDO ANTI-DEADLOCK — FINAL COMPLETO)
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from collections import defaultdict
import json
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import telebot
import config
from threading import Lock
import threading
import queue
import os
import redis
import pickle
import hashlib

# =========================================================
# CONFIG GERAL
# =========================================================

TIMEFRAME_BASE = mt5.TIMEFRAME_M15
TIMEFRAME_MACRO = getattr(mt5, f"TIMEFRAME_{config.MACRO_TIMEFRAME}", mt5.TIMEFRAME_H1)
logger = logging.getLogger("utils")

mt5_lock = Lock()  # Lock global APENAS para operações críticas (ordens / account)
sector_weights: Dict[str, Dict[str, float]] = {}
symbol_weights: Dict[str, Dict[str, float]] = {}
# Conexão Redis (ajuste se necessário)
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)
    redis_client.ping()  # Testa conexão
    REDIS_AVAILABLE = True
    logger.info("✅ Redis conectado - cache ativado")
except Exception as e:
    redis_client = None
    REDIS_AVAILABLE = False
    logger.warning(f"⚠️ Redis não disponível: {e} - cache desativado")
# =========================================================
# MT5 SAFE COPY (ANTI-DEADLOCK)
# =========================================================

def safe_copy_rates(symbol: str, timeframe, count: int = 500, timeout: int = 12) -> Optional[pd.DataFrame]:
    if not mt5.symbol_select(symbol, True):
        logger.warning(f"⚠️ {symbol} não pôde ser selecionado no Market Watch.")
        return None

    try:
        bars = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        bars_available = 0 if bars is None else len(bars)
    except Exception:
        bars_available = 0

    if bars_available < count:
        mt5.copy_rates_from_pos(symbol, timeframe, 0, 1)
        time.sleep(0.2)

    q = queue.Queue()

    def worker():
        try:
            q.put(mt5.copy_rates_from_pos(symbol, timeframe, 0, count))
        except Exception as e:
            q.put(e)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout)

    if t.is_alive():
        logger.error(f"🚨 TIMEOUT MT5 em {symbol}")
        return None

    try:
        rates = q.get_nowait()
        if isinstance(rates, Exception) or rates is None or len(rates) == 0:
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        return df.sort_index()
    except queue.Empty:
        return None

# =========================================================
# SLIPPAGE
# =========================================================

def get_real_slippage(symbol: str) -> float:
    tick = mt5.symbol_info_tick(symbol)
    if not tick or tick.bid <= 0:
        return config.SLIPPAGE_MAP.get("DEFAULT", 0.005)

    spread_pct = (tick.ask - tick.bid) / tick.bid

    # Multiplicador por perfil de liquidez
    if symbol in config.LOW_LIQUIDITY_SYMBOLS:
        mult = 2.0
    elif is_power_hour():
        mult = 1.2
    else:
        mult = 1.5

    mapped = config.SLIPPAGE_MAP.get(symbol, config.SLIPPAGE_MAP.get("DEFAULT"))
    return max(spread_pct * mult, mapped)


# =========================================================
# REGIME DE MERCADO
# =========================================================

def detect_market_regime() -> str:
    ibov = safe_copy_rates("IBOV", mt5.TIMEFRAME_D1, 50)
    if ibov is None or len(ibov) < 30:
        return "RISK_ON"

    close = ibov["close"]
    ma20 = close.rolling(20).mean().iloc[-1]
    ma50 = close.rolling(50).mean().iloc[-1]
    cur = close.iloc[-1]

    return "RISK_ON" if cur > ma20 > ma50 else "RISK_OFF"

# =========================================================
# EXPOSIÇÃO SETORIAL
# =========================================================

def calculate_sector_exposure_pct(equity: float) -> Dict[str, float]:
    with mt5_lock:
        positions = mt5.positions_get() or []

    sector_risk = defaultdict(float)
    for p in positions:
        sector = config.SECTOR_MAP.get(p.symbol, "UNKNOWN")
        sector_risk[sector] += p.volume * p.price_open

    return {s: v / equity for s, v in sector_risk.items()} if equity > 0 else {}

# =========================================================
# FAST RATES
# =========================================================

_last_bar_time = {}

def get_fast_rates(symbol, timeframe):
    df = safe_copy_rates(symbol, timeframe, 3)
    if df is None or df.empty:
        return None

    last = df.index[-1]
    key = (symbol, timeframe)
    if _last_bar_time.get(key) == last:
        return None

    _last_bar_time[key] = last
    return df

# =========================================================
# INDICADORES BÁSICOS
# =========================================================

def get_atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    return float(atr.iloc[-1])

def get_adx(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    if len(df) < period * 2:
        return None

    high, low, close = df["high"], df["low"], df["close"]

    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    return float(adx.iloc[-1])

def get_intraday_vwap(df: pd.DataFrame) -> Optional[float]:
    """
    VWAP desde a abertura do pregão (10h00) até agora.
    Acumula volume * preço médio de cada candle.
    """
    
    now = datetime.now()
    today = now.date()
    market_open = datetime.combine(today, datetime.strptime("10:00", "%H:%M").time())
    
    # Filtra apenas candles desde a abertura de hoje
    df_today = df[df.index >= market_open]
    
    if df_today.empty or len(df_today) < 3:
        # Menos de 3 candles = VWAP não confiável
        return None
    
    # Preço médio de cada candle (HLCC/4 - mais preciso que só close)
    typical_price = (df_today['high'] + df_today['low'] + 2 * df_today['close']) / 4
    
    # Volume (tick_volume ou real_volume)
    volume = df_today.get('real_volume', df_today['tick_volume'])
    
    # VWAP = Soma(preço * volume) / Soma(volume)
    pv = (typical_price * volume).sum()
    total_vol = volume.sum()
    
    return float(pv / total_vol) if total_vol > 0 else None

# =========================================================
# MACRO TREND
# =========================================================

def macro_trend_ok(symbol: str, side: str) -> bool:
    df = safe_copy_rates(symbol, TIMEFRAME_MACRO, 300)
    if df is None or len(df) < config.MACRO_EMA_LONG:
        return False

    close = df["close"]
    ema = close.ewm(span=config.MACRO_EMA_LONG, adjust=False).mean().iloc[-1]
    tick = mt5.symbol_info_tick(symbol)
    if not tick or (tick.last <= 0 and tick.bid <= 0):
        return False

    price = tick.last if tick.last > 0 else tick.bid

    adx = get_adx(df)
    if adx is not None and adx < 20:
        return False

    return price > ema if side == "BUY" else price < ema

# =========================================================
# INDICADORES CONSOLIDADOS (SEM SCORE)
# =========================================================

def quick_indicators_custom(symbol, timeframe, df=None, params=None):
    params = params or {}
    df = df if df is not None else safe_copy_rates(symbol, timeframe, 300)

    if df is None or len(df) < 50:
        return {"error": "no_data"}

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # --- MÉDIAS E RSI ---
    ema_fast = close.ewm(span=params.get("ema_short", 9), adjust=False).mean().iloc[-1]
    ema_slow = close.ewm(span=params.get("ema_long", 21), adjust=False).mean().iloc[-1]

    delta = close.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = -delta.clip(upper=0).rolling(14).mean()
    rsi = (100 - (100 / (1 + up / down))).iloc[-1]

    # --- ATR E ADX ---
    atr = get_atr(df)
    adx = get_adx(df) or 0.0
    price = float(close.iloc[-1])

    # --- CÁLCULO ATR% REAL (TIME-FRAME ATUAL M15) ---
    # Normalização se for pontos (índice) ou R$ (ações)
    if atr > price * 2:
        atr_price = atr * mt5.symbol_info(symbol).point
    else:
        atr_price = atr
    
    # ATR% real sem projeções arbitrárias
    atr_pct_real = (atr_price / price) * 100 if price > 0 else 0

    # --- Z-SCORE DE VOLATILIDADE (A nova métrica de consistência) ---
    # Calculamos o desvio padrão dos retornos percentuais para ver se o ATR atual é um outlier
    vol_series = df['close'].pct_change().rolling(20).std() * 100
    atr_mean = vol_series.mean()
    atr_std = vol_series.std()
    
    # Z-Score: quão longe estamos da média de volatilidade (em desvios padrão)
    z_score = (atr_pct_real - atr_mean) / atr_std if (atr_std and atr_std > 0) else 0
    
    # Cap de segurança apenas para o Score, mas mantemos o valor real para o log
    atr_pct_capped = min(round(atr_pct_real, 3), 10.0) 

    # --- VOLUME ---
    avg_vol = get_avg_volume(df)
    cur_vol = df["real_volume"].iloc[-1] if "real_volume" in df.columns else df["tick_volume"].iloc[-1]
    volume_ratio = round(cur_vol / avg_vol, 2) if avg_vol > 0 else 1.0

    atr_series_data = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1).ewm(alpha=1 / 14, adjust=False).mean()

    atr_mean_val = atr_series_data.rolling(20).mean().iloc[-1]
    side = "BUY" if ema_fast > ema_slow else "SELL"

    return {
        "ema_fast": float(ema_fast),
        "ema_slow": float(ema_slow),
        "rsi": float(rsi),
        "adx": float(adx),
        "atr": float(atr),
        "atr_pct": atr_pct_capped, # Para o cálculo de score
        "atr_real": round(atr_pct_real, 3), # Para o painel
        "atr_zscore": round(z_score, 2), # Para detecção de anomalias
        "volume_ratio": volume_ratio,
        "vol_breakout": is_volatility_breakout(df, atr, atr_mean_val, volume_ratio, side),
        "vwap": get_intraday_vwap(df),
        "close": price,
        "macro_trend_ok": macro_trend_ok(symbol, side),
        "tick_size": mt5.symbol_info(symbol).point,
        "error": None
    }
# =========================================================
# SCORE FINAL
# =========================================================
def calculate_signal_score(ind: dict) -> float:
    """
    Calcula o score operacional final do ativo.
    Retorna APENAS float.
    """

    score = 0.0
    score_log = {}
    ind["block_reason"] = " "

    # =========================
    # 📦 INPUTS
    # =========================
    rsi = ind.get("rsi", 50)
    adx = ind.get("adx", 0)
    atr_pct = ind.get("atr_pct", 0)
    volume_ratio = ind.get("volume_ratio", 1.0)
    ema_fast = ind.get("ema_fast", 0)
    ema_slow = ind.get("ema_slow", 0)
    corr = ind.get("corr", 0)
    macro_ok = ind.get("macro_trend_ok", True)
    vol_break = ind.get("vol_breakout", False)

    # =========================
    # ⏰ TIME-AWARE
    # =========================
    _, time_cfg = get_time_bucket()
    adx_min = time_cfg["adx_min"]
    min_score = time_cfg["min_score"]
    atr_max = time_cfg["atr_max"]

    # =========================
    # 📈 EMA
    # =========================
    if ema_fast > ema_slow:
        score += 15
        score_log["EMA"] = 15
    else:
        score -= 10
        score_log["EMA"] = -10

    # =========================
    # 📊 RSI + ADX
    # =========================
    if 40 <= rsi <= 60:
        rsi_score = 20
    elif 30 <= rsi < 40 or 60 < rsi <= 70:
        rsi_score = 10
    else:
        rsi_score = -10

    adx_factor = min(max((adx - adx_min) / 20, 0), 1)
    rsi_adx_score = rsi_score * adx_factor

    score += rsi_adx_score
    score_log["RSI_ADX"] = round(rsi_adx_score, 1)

    # =========================
    # 🌊 ATR
    # =========================
    if atr_pct > atr_max * 1.5:
        ind["score_log"] = score_log
        return 0.0

    if atr_pct <= atr_max:
        score += 15
        score_log["ATR"] = 15
    else:
        score -= 10
        score_log["ATR"] = -10

    # =========================
    # 🌍 MACRO
    # =========================
    if macro_ok:
        score += 5
        score_log["MACRO"] = 5
    else:
        score -= 15
        score_log["MACRO"] = -15

    # =========================
    # ⚡ POWER-HOUR
    # =========================
    if is_power_hour():
        if atr_pct < config.POWER_HOUR["min_atr_pct"]:
            ind["block_reason"] = "POWER_NO_VOL"
            return 0.0

        if volume_ratio < config.POWER_HOUR["min_volume_ratio"]:
            ind["block_reason"] = "POWER_NO_VOLUME"
            return 0.0

        score += config.POWER_HOUR["score_boost"]
        score_log["POWER"] = config.POWER_HOUR["score_boost"]

    # =========================
    # 🚀 BREAKOUT
    # =========================
    if vol_break:
        score += config.VOL_BREAKOUT["score_boost"]
        score_log["VOL_BREAK"] = config.VOL_BREAKOUT["score_boost"]

    # =========================
    # 🔗 CORRELAÇÃO
    # =========================
    if corr > 0.85:
        score -= 20
        score_log["CORR"] = -20
    elif corr > 0.65:
        score -= 12
        score_log["CORR"] = -12
    elif corr > 0.45:
        score -= 5
        score_log["CORR"] = -5

    # =========================
    # ✅ FINAL
    # =========================
    final_score = round(max(score, 0), 1)

    if final_score < min_score and not ind["block_reason"]:
        ind["block_reason"] = "TIME_FILTER"

    ind["score_log"] = score_log
    return final_score


def check_and_close_orphans(active_signals: dict):
    with mt5_lock:
        positions = mt5.positions_get() or []
    for pos in positions:
        if pos.symbol not in active_signals:
            logger.warning(f"Posição órfã detectada: {pos.symbol}")
            send_telegram_exit(
                symbol=pos.symbol,
                reason="Posição órfã (sem sinal ativo)"
            )

def get_avg_volume(df, window: int = 20):
    if df is None or df.empty:
        return 0

    if "real_volume" in df.columns:
        vol_col = "real_volume"
    elif "tick_volume" in df.columns:
        vol_col = "tick_volume"
    else:
        return 0

    return df[vol_col].tail(window).mean()

def resolve_signal_weights(symbol, sector, base_weights,
                           sector_weights=None, symbol_weights=None):
    w = base_weights.copy()

    if sector_weights and sector in sector_weights:
        for k, v in sector_weights[sector].items():
            w[k] *= v

    if symbol_weights and symbol in symbol_weights:
        for k, v in symbol_weights[symbol].items():
            w[k] *= v

    return w

def update_symbol_weights(symbol, sector, score_log, trade_result):
    global symbol_weights

    alpha = 0.03

    if symbol not in symbol_weights:
        symbol_weights[symbol] = {}

    for k, contribution in score_log.items():
        current = symbol_weights[symbol].get(k, 1.0)
        delta = 1 + alpha * np.tanh(trade_result)
        symbol_weights[symbol][k] = max(0.5, min(1.8, current * delta))

_bot_instance = None

def get_telegram_bot():
    global _bot_instance
    if _bot_instance is None and config.ENABLE_TELEGRAM_NOTIF:
        _bot_instance = telebot.TeleBot(config.TELEGRAM_BOT_TOKEN)
    return _bot_instance

def send_telegram_exit(
    symbol: str, 
    side: str = "",
    volume: float = 0,
    entry_price: float = 0,
    exit_price: float = 0,
    profit_loss: float = 0,
    reason: str = ""
):
    bot = get_telegram_bot()
    if not bot:
        return

    pl_emoji = "🟢" if profit_loss > 0 else "🔴"
    pl_pct = (profit_loss / (entry_price * volume)) * 100 if entry_price > 0 and volume > 0 else 0

    msg = (
        f"{pl_emoji} <b>XP3 — POSIÇÃO ENCERRADA</b>\n\n"
        f"📊 <b>Ativo:</b> {symbol}\n"
        f"📍 <b>Direção:</b> {side}\n"
        f"📦 <b>Volume:</b> {volume:.2f}\n"
        f"💰 <b>Entrada:</b> R${entry_price:.2f}\n"
        f"🚪 <b>Saída:</b> R${exit_price:.2f}\n"
        f"💵 <b>P&L:</b> R${profit_loss:+.2f} ({pl_pct:+.2f}%)\n"
        f"📝 <b>Motivo:</b> {reason}\n"
        f"⏱ {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
    )

    try:
        bot.send_message(
            chat_id=config.TELEGRAM_CHAT_ID,
            text=msg,
            parse_mode="HTML"
        )
    except Exception as e:
        logger.warning(f"Erro Telegram saída: {e}")

def close_position(symbol: str, volume: float, position_type: int):
    with mt5_lock:  
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return

    order_type = (
        mt5.ORDER_TYPE_SELL if position_type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
    )

    price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "deviation": 10,
        "magic": 2026,
        "comment": "XP3 close orphan",
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    with mt5_lock:
        mt5.order_send(request)

def save_adaptive_weights():
    data = {
        "symbol": symbol_weights,
        "sector": sector_weights
    }
    with open("adaptive_weights.json", "w") as f:
        json.dump(data, f, indent=2)

def load_adaptive_weights():
    global symbol_weights, sector_weights
    path = "adaptive_weights.json"

    if not os.path.exists(path):
        logger.info("ℹ️ Pesos adaptativos não encontrados. Usando padrão.")
        return

    try:
        with open(path, "r") as f:
            data = json.load(f)
            symbol_weights = data.get("symbol", {})
            sector_weights = data.get("sector", {})
            logger.info("🧠 Pesos adaptativos carregados com sucesso.")
    except Exception as e:
        logger.error(f"Erro ao carregar pesos adaptativos: {e}")


def get_open_gap(symbol, timeframe):
    df = safe_copy_rates(symbol, timeframe, 2)
    if df is None or len(df) < 2:
        return None
    prev_close = df["close"].iloc[-2]
    open_price = df["open"].iloc[-1]
    return abs((open_price - prev_close) / prev_close) * 100

def calculate_position_size_atr(equity, risk_pct, atr, tick_value=1.0, min_vol=0.01):
    risk_money = equity * risk_pct
    if atr <= 0:
        return None
    volume = risk_money / (atr * tick_value)

    return max(min_vol, round(volume / min_vol) * min_vol)

import signal

def signal_handler(sig, frame):
    with mt5_lock:
        logger.info("Encerrando bot - salvando pesos adaptativos...")
        save_adaptive_weights()
        mt5.shutdown()
    exit(0)

signal.signal(signal.SIGINT, signal_handler)

def send_telegram_trade(symbol: str, side: str, volume: float, price: float, sl: float, tp: float):
    bot = get_telegram_bot()
    if not bot:
        return

    if side == "BUY":
        direction = "🟢 COMPRA"
        arrow = "⬆️"
    else:
        direction = "🔴 VENDA"
        arrow = "⬇️"

    dist_sl = abs(price - sl)
    dist_tp = abs(tp - price)
    risk_pct = round((dist_sl / price) * 100, 2)
    reward_pct = round((dist_tp / price) * 100, 2)
    rr_ratio = round(dist_tp / dist_sl, 2) if dist_sl > 0 else 0

    msg = (
        f"<b>🚀 XP3 – NOVA ENTRADA</b>\n\n"
        f"<b>Ativo:</b> {symbol}\n"
        f"<b>Direção:</b> {direction} {arrow}\n"
        f"<b>Volume:</b> {volume:.2f} contratos/ações\n"
        f"<b>Preço de entrada:</b> R${price:.2f}\n\n"
        f"<b>🛑 Stop Loss:</b> R${sl:.2f}  <i>(-{dist_sl:.2f} | -{risk_pct}%)</i>\n"
        f"<b>🎯 Take Profit:</b> R${tp:.2f}  <i>(+{dist_tp:.2f} | +{reward_pct}%)</i>\n"
        f"<b>Risk:Reward:</b> 1:{rr_ratio}\n\n"
        f"<i>⏱ {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</i>"
    )

    try:
        bot.send_message(
            chat_id=config.TELEGRAM_CHAT_ID,
            text=msg,
            parse_mode="HTML"
        )
    except Exception as e:
        logger.warning(f"Erro ao enviar Telegram entrada: {e}")

def send_order_with_sl_tp(symbol, side, volume, sl, tp):
    with mt5_lock:
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return None
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
        with mt5_lock:
            result = mt5.order_send(request)
        return result

def send_order_with_retry(symbol, side, volume, sl, tp, max_retries=3):
    """Tenta enviar ordem com retry automático"""
    for attempt in range(max_retries):
        result = send_order_with_sl_tp(symbol, side, volume, sl, tp)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            return result
        
        # Erros recuperáveis
        if result and result.retcode in [
            mt5.TRADE_RETCODE_REQUOTE,
            mt5.TRADE_RETCODE_PRICE_OFF,
            mt5.TRADE_RETCODE_TIMEOUT
        ]:
            logger.warning(f"Tentativa {attempt+1}/{max_retries} falhou: {result.comment}")
            time.sleep(0.5)  # Aguarda meio segundo
            continue
        
        # Erro irrecuperável
        break
    
    return result

def validate_order_params(symbol: str, volume: float, price: float, sl: float, tp: float) -> bool:
    """Valida parâmetros antes de enviar ordem"""
    with mt5_lock:
        info = mt5.symbol_info(symbol)
    
    if not info:
        logger.error(f"Símbolo {symbol} não disponível")
        return False
    
    # Volume mínimo/máximo
    if volume < info.volume_min or volume > info.volume_max:
        logger.error(f"Volume {volume} fora dos limites [{info.volume_min}, {info.volume_max}]")
        return False
    
    # Stop Loss não pode ser muito próximo
    min_sl_distance = info.trade_stops_level * info.point
    if abs(price - sl) < min_sl_distance:
        logger.error(f"SL muito próximo. Mínimo: {min_sl_distance}")
        return False
    
    # Price dentro de limites
    if price < info.bid * 0.95 or price > info.ask * 1.05:
        logger.error(f"Preço {price} muito distante de Bid/Ask")
        return False
    
    return True

def get_time_bucket():
    now = datetime.now().time()
    for bucket, cfg in config.TIME_SCORE_RULES.items():
        start = datetime.strptime(cfg["start"], "%H:%M").time()
        end   = datetime.strptime(cfg["end"], "%H:%M").time()
        if start <= now <= end:
            return bucket, cfg
    return "MID", config.TIME_SCORE_RULES["MID"]

def is_power_hour():
    now = datetime.now().time()
    cfg = config.POWER_HOUR
    if not cfg["enabled"]:
        return False
    start = datetime.strptime(cfg["start"], "%H:%M").time()
    end   = datetime.strptime(cfg["end"], "%H:%M").time()
    return start <= now <= end

def is_volatility_breakout(df, atr_now, atr_mean, volume_ratio, side=None):
    if not config.VOL_BREAKOUT["enabled"]:
        return False

    if atr_now is None or atr_mean is None:
        return False

    if atr_now < atr_mean * config.VOL_BREAKOUT["atr_expansion"]:
        return False

    if volume_ratio < config.VOL_BREAKOUT["volume_ratio"]:
        return False

    lookback = config.VOL_BREAKOUT["lookback"]

    high_break = df["high"].iloc[-1] > df["high"].rolling(lookback).max().iloc[-2]
    low_break  = df["low"].iloc[-1]  < df["low"].rolling(lookback).min().iloc[-2]
    if len(df) < lookback + 2:
        return False

    if side == "BUY":
        return high_break
    if side == "SELL":
        return low_break

    return high_break or low_break

def get_current_risk_pct() -> float:
    """
    Retorna o risco percentual atual por trade,
    respeitando regras de gestão de risco do config.
    """

    # =========================
    # 🎯 RISCO BASE
    # =========================
    risk = config.RISK_PER_TRADE_PCT  # ex: 1%

    now = datetime.now()
    weekday = now.weekday()  # 0=segunda ... 4=sexta
    hour = now.hour

    # =========================
    # 📉 REDUÇÃO NA SEXTA À TARDE
    # =========================
    if weekday == 4 and hour >= 15:
        risk = min(risk, config.REDUCED_RISK_PCT)

    # =========================
    # 🌍 REGIME DE MERCADO
    # =========================
    regime = detect_market_regime()
    if regime == "RISK_OFF":
        risk *= 0.7  # defensivo em mercado ruim

    # =========================
    # ⚡ POWER-HOUR
    # =========================
    if is_power_hour():
        # Se quiser agressivar no fechamento
        risk *= config.POWER_HOUR.get("risk_multiplier", 1.0)

    # =========================
    # 🔒 CLAMP FINAL (SEGURANÇA)
    # =========================
    max_allowed = min(
        config.MAX_RISK_PER_SYMBOL_PCT,
        config.MAX_DAILY_DRAWDOWN_PCT
    )

    return max(0.001, min(risk, max_allowed))

def send_telegram_eod_report():
    bot = get_telegram_bot()
    if not bot:
        return

    equity = mt5.account_info().equity
    balance = mt5.account_info().balance
    positions = mt5.positions_total()

    msg = (
        "📅 <b>XP3 — FECHAMENTO DO DIA</b>\n\n"
        f"💰 Equity: R${equity:,.2f}\n"
        f"🏦 Balance: R${balance:,.2f}\n"
        f"📊 Posições abertas: {positions}\n"
        f"⏱ {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
    )

    bot.send_message(
        chat_id=config.TELEGRAM_CHAT_ID,
        text=msg,
        parse_mode="HTML"
    )

def get_dynamic_slippage(symbol, hour):
    base = config.SLIPPAGE_MAP.get(symbol, config.SLIPPAGE_MAP["DEFAULT"])

    
    # Reduz pela metade na power hour (maior liquidez)
    if 15 <= hour <= 17:
        base *= 0.6
    
    # Aumenta 50% na abertura (spread maior)
    elif 10 <= hour <= 11:
        base *= 1.5
    
    return base

def calculate_smart_sl(symbol, entry_price, side, atr, df):
    """
    Calcula stop loss considerando:
    1. ATR (risco estatístico)
    2. Suporte/Resistência mais próximo
    3. Mínimo de 1.5 ATR (nunca muito apertado)
    """
    
    # Stop base (ATR)
    base_distance = atr * 2.0
    
    # Encontra suporte/resistência relevante
    lookback = 50
    if side == "BUY":
        # Para compra: busca último fundo relevante
        recent_lows = df['low'].tail(lookback)
        support = recent_lows.min()
        
        # Stop 0.5 ATR abaixo do suporte
        structure_stop = support - (atr * 0.5)
        
        # Usa o MENOR entre estrutura e ATR (mais conservador)
        final_sl = max(structure_stop, entry_price - base_distance)
        
    else:  # SELL
        recent_highs = df['high'].tail(lookback)
        resistance = recent_highs.max()
        structure_stop = resistance + (atr * 0.5)
        final_sl = min(structure_stop, entry_price + base_distance)
    
    # Garante mínimo de 1.5 ATR
    min_distance = atr * 1.5
    if side == "BUY":
        final_sl = min(final_sl, entry_price - min_distance)
    else:
        final_sl = max(final_sl, entry_price + min_distance)
    
    return round(final_sl, 2)

def analyze_order_book_depth(symbol, side, volume_needed):
    """
    Verifica se há liquidez suficiente no order book
    para executar a ordem sem slippage excessivo.
    """
    
    # MT5 não expõe order book diretamente
    # Alternativa: usar DOM (Depth of Market) via API
    # Ou estimar por volume recente
    
    df = safe_copy_rates(symbol, TIMEFRAME_BASE, 20)
    if df is None:
        return True  # Assume OK se não conseguir dados
    
    # Volume médio dos últimos 20 candles
    avg_volume_per_candle = df['tick_volume'].mean()
    
    # Nossa ordem não deve ser > 10% do volume médio
    volume_ratio = volume_needed / avg_volume_per_candle
    
    if volume_ratio > 0.10:  # 10% do candle médio
        logger.warning(
            f"⚠️ {symbol}: Volume da ordem ({volume_needed}) "
            f"representa {volume_ratio*100:.1f}% do candle médio. "
            f"Possível alto impacto no preço."
        )
        return False
    
    return True

def apply_trailing_stop(symbol: str, side: str, current_price: float, atr: float):
    """
    Move o Stop Loss para proteger o lucro conforme o preço avança.
    
    Regras:
    - Trailing baseado em ATR
    - Nunca afrouxa stop
    - Só ajusta se houver ganho mínimo real
    """

    if atr is None or atr <= 0:
        return

    with mt5_lock:
        positions = mt5.positions_get(symbol=symbol)

    if not positions:
        return

    # Distância do trailing (ex: 1.5 ATR)
    trail_dist = atr * 1.5

    for pos in positions:
        # =========================
        # 🟢 COMPRA
        # =========================
        if side == "BUY" and pos.type == mt5.POSITION_TYPE_BUY:
            new_sl = round(current_price - trail_dist, 2)

            # Só move se:
            # 1) Novo SL > SL atual
            # 2) Novo SL ainda abaixo do preço atual
            if pos.sl is not None and new_sl <= pos.sl:
                continue

            if new_sl >= current_price:
                continue

            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": pos.ticket,
                "sl": new_sl,
                "tp": pos.tp
            }

            with mt5_lock:
                result = mt5.order_send(request)

            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(
                    f"🔒 Trailing BUY ajustado | {symbol} | SL: {pos.sl:.2f} → {new_sl:.2f}"
                )
            else:
                logger.warning(
                    f"⚠️ Falha ao mover trailing BUY {symbol}: "
                    f"{getattr(result, 'comment', 'sem retorno')}"
                )

        # =========================
        # 🔴 VENDA
        # =========================
        elif side == "SELL" and pos.type == mt5.POSITION_TYPE_SELL:
            new_sl = round(current_price + trail_dist, 2)

            # Só move se:
            # 1) Novo SL < SL atual
            # 2) Novo SL ainda acima do preço atual
            if pos.sl is not None and new_sl >= pos.sl:
                continue

            if new_sl <= current_price:
                continue

            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": pos.ticket,
                "sl": new_sl,
                "tp": pos.tp
            }

            with mt5_lock:
                result = mt5.order_send(request)

            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(
                    f"🔒 Trailing SELL ajustado | {symbol} | SL: {pos.sl:.2f} → {new_sl:.2f}"
                )
            else:
                logger.warning(
                    f"⚠️ Falha ao mover trailing SELL {symbol}: "
                    f"{getattr(result, 'comment', 'sem retorno')}"
                )


def can_enter_symbol(symbol: str, equity: float) -> bool:
    """
    Verifica se pode entrar em novo trade no símbolo.
    Considera:
    - Limite de risco por símbolo
    - Bloqueio temporário após perdas
    """
    
    # 1. Risco atual no símbolo
    with mt5_lock:
        positions = [p for p in mt5.positions_get() or [] if p.symbol == symbol]
    
    if not positions:
        return True  # Sem posição = pode entrar
    
    # Soma exposição atual
    total_risk = sum(p.volume * p.price_open for p in positions)
    risk_pct = total_risk / equity if equity > 0 else 0
    
    if risk_pct >= config.MAX_RISK_PER_SYMBOL_PCT:
        logger.warning(f"{symbol}: Limite de risco por símbolo atingido ({risk_pct*100:.1f}%)")
        return False
    
    # 2. Bloqueio temporário (opcional - baseado em perdas recentes)
    # Implementar lógica de bloqueio se necessário
    
    return True

def calculate_correlation_matrix(symbols: List[str], lookback: int = 60) -> Dict[str, Dict[str, float]]:
    """
    Calcula matriz de correlação entre símbolos.
    Retorna: {symbol1: {symbol2: corr_value}}
    """
    
    if len(symbols) < 2:
        return {}
    
    # Coleta dados de fechamento
    closes = {}
    for sym in symbols:
        df = safe_copy_rates(sym, mt5.TIMEFRAME_D1, lookback)
        if df is not None and len(df) >= 30:
            closes[sym] = df['close']
    
    if len(closes) < 2:
        return {}
    
    # Alinha datas
    df_all = pd.DataFrame(closes)
    df_all = df_all.dropna()
    
    if len(df_all) < 30:
        return {}
    
    # Calcula correlação
    corr_matrix = df_all.corr()
    
    # Converte para dict
    result = {}
    for sym1 in symbols:
        if sym1 not in corr_matrix.columns:
            continue
        result[sym1] = {}
        for sym2 in symbols:
            if sym2 not in corr_matrix.columns:
                continue
            result[sym1][sym2] = float(corr_matrix.loc[sym1, sym2])
    
    return result

# =========================================================
# RASTREAMENTO DE PERFORMANCE POR ATIVO (LOSS STREAK)
# =========================================================

LOSS_STREAK_FILE = "symbol_loss_streak.json"

_symbol_loss_streak = defaultdict(int)
_symbol_last_loss_time = {}
_symbol_block_until = {}

def load_loss_streak_data():
    global _symbol_loss_streak, _symbol_last_loss_time, _symbol_block_until
    if os.path.exists(LOSS_STREAK_FILE):
        try:
            with open(LOSS_STREAK_FILE, "r") as f:
                data = json.load(f)
                _symbol_loss_streak = defaultdict(int, data.get("streak", {}))
                _symbol_last_loss_time = {k: datetime.fromisoformat(v) for k, v in data.get("last_loss", {}).items()}
                _symbol_block_until = {k: datetime.fromisoformat(v) for k, v in data.get("block_until", {}).items()}
            logger.info("📉 Dados de loss streak carregados.")
        except Exception as e:
            logger.error(f"Erro ao carregar loss streak: {e}")

def save_loss_streak_data():
    data = {
        "streak": dict(_symbol_loss_streak),
        "last_loss": {k: v.isoformat() for k, v in _symbol_last_loss_time.items()},
        "block_until": {k: v.isoformat() for k, v in _symbol_block_until.items()}
    }
    try:
        with open(LOSS_STREAK_FILE, "w") as f:
            json.dump(data, f)
        logger.info("💾 Loss streak salvo.")
    except Exception as e:
        logger.error(f"Erro ao salvar loss streak: {e}")

def record_trade_outcome(symbol: str, profit_loss: float):
    """
    Chama após fechar uma posição.
    profit_loss = valor em R$ (positivo = lucro, negativo = perda)
    """
    global _symbol_loss_streak, _symbol_last_loss_time, _symbol_block_until

    now = datetime.now()

    if profit_loss >= 0:
        # Reset streak em caso de lucro
        if _symbol_loss_streak[symbol] > 0:
            logger.info(f"✅ {symbol}: Streak de perdas resetado (lucro detectado)")
        _symbol_loss_streak[symbol] = 0
    else:
        # Perda
        _symbol_loss_streak[symbol] += 1
        _symbol_last_loss_time[symbol] = now
        logger.warning(f"🔴 {symbol}: Perda consecutiva #{_symbol_loss_streak[symbol]}")

        if _symbol_loss_streak[symbol] >= config.SYMBOL_MAX_CONSECUTIVE_LOSSES:
            block_until = now + timedelta(hours=config.SYMBOL_COOLDOWN_HOURS)
            _symbol_block_until[symbol] = block_until
            logger.critical(f"🚫 {symbol}: BLOQUEADO por {config.SYMBOL_COOLDOWN_HOURS}h após {_symbol_loss_streak[symbol]} perdas seguidas")

    save_loss_streak_data()

def is_symbol_blocked(symbol: str) -> tuple[bool, str]:
    """
    Retorna (blocked: bool, reason: str)
    """
    now = datetime.now()

    # Limpa bloqueios expirados
    if symbol in _symbol_block_until:
        if now >= _symbol_block_until[symbol]:
            del _symbol_block_until[symbol]
            _symbol_loss_streak[symbol] = 0
            save_loss_streak_data()
            logger.info(f"✅ {symbol}: Bloqueio expirado e removido")

    if symbol in _symbol_block_until:
        remaining = int((_symbol_block_until[symbol] - now).total_seconds() / 3600)
        return True, f"Bloqueado ({remaining}h restantes) - {config.SYMBOL_MAX_CONSECUTIVE_LOSSES} perdas seguidas"

    return False, ""

def get_cached_indicators(symbol: str, timeframe, count: int = 300, ttl: int = 45):
    """
    Retorna indicadores com cache Redis (TTL 45s)
    Fallback: calcula normalmente se Redis off ou erro
    """
    if not REDIS_AVAILABLE:
        df = safe_copy_rates(symbol, timeframe, count)
        return quick_indicators_custom(symbol, timeframe, df=df) if df is not None else {"error": "no_data"}

    key = f"ind:v2:{symbol}:{timeframe}:{count}"
    
    try:
        cached = redis_client.get(key)
        if cached:
            ind = pickle.loads(cached)
            logger.debug(f"Cache HIT: {symbol}")
            return ind
    except Exception as e:
        logger.warning(f"Redis get error: {e}")

    # Cache miss → calcula
    df = safe_copy_rates(symbol, timeframe, count)
    if df is None or len(df) < 50:
        ind = {"error": "no_data"}
    else:
        ind = quick_indicators_custom(symbol, timeframe, df=df)

    # Salva no cache
    try:
        redis_client.setex(key, ttl, pickle.dumps(ind))
    except Exception as e:
        logger.warning(f"Redis set error: {e}")

    return ind