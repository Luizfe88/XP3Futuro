import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from collections import defaultdict
import json
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import config
from threading import RLock  # Changed from Lock to RLock as per refactor
import threading
import queue
import os
import redis
import pickle
import hashlib
import signal
from news_calendar import apply_blackout
from ml_optimizer import ml_optimizer
import re
from typing import Optional, Dict, Any, List, Tuple

def is_valid_dataframe(df, min_rows: int = 1) -> bool:
    """
    Valida DataFrame de forma segura.
    
    Args:
        df: Objeto a validar (pode ser DataFrame, lista, None, etc)
        min_rows: Número mínimo de linhas (padrão: 1)
    
    Returns:
        True se válido, False caso contrário
    """
    if df is None:
        return False
    
    if isinstance(df, pd.DataFrame):
        return not df.empty and len(df) >= min_rows
    
    if isinstance(df, (list, tuple)):
        return len(df) >= min_rows
    
    return False


mt5_lock = RLock()
try:
    import telebot
except ImportError:
    telebot = None
    logger.warning("telebot não instalado - comandos Telegram desativados")
# =========================================================
# CONFIG GERAL
# =========================================================

TIMEFRAME_BASE = mt5.TIMEFRAME_M15
TIMEFRAME_MACRO = getattr(mt5, f"TIMEFRAME_{config.MACRO_TIMEFRAME}", mt5.TIMEFRAME_H1)
logger = logging.getLogger("utils")

mt5_lock = RLock()  # Lock global APENAS para operações críticas (ordens / account)
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

def get_dynamic_rr_min() -> float:
    """
    Retorna R:R mínimo dinâmico baseado no regime de mercado.
    - RISK_ON (bull): 1.25
    - RISK_OFF (incertezas): 1.5
    """
    regime = detect_market_regime()
    thresholds = config.ADAPTIVE_THRESHOLDS[regime]
    
    # Usa thresholds do regime
    min_score = thresholds["min_signal_score"]
    adx_min = thresholds["min_adx"]
    min_volume_ratio = thresholds["min_volume_ratio"]
    if regime == "RISK_ON":
        logger.info(f"🟢 Regime: {regime} | R:R mínimo: 1.25 (bull market)")
        return 1.25
    else:
        logger.info(f"🔴 Regime: {regime} | R:R mínimo: 1.5 (incertezas)")
        return 1.5

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
    if not is_valid_dataframe(df):
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

def get_momentum(df: pd.DataFrame, period: int = 10) -> Optional[float]:
    """
    Calcula momentum (Rate of Change)
    
    Momentum = (preço_atual - preço_passado) / preço_passado
    
    Args:
        df: DataFrame com coluna 'close'
        period: Quantos candles olhar para trás (padrão: 10)
    
    Returns:
        Momentum como float (ex: 0.05 = 5% de alta)
    """
    if df is None or len(df) < period + 1:
        return None
    
    close = df['close']
    
    # Momentum = mudança percentual em N períodos
    momentum = (close.iloc[-1] - close.iloc[-period - 1]) / close.iloc[-period - 1]
    
    return float(momentum)


def quick_indicators_custom(symbol, timeframe, df=None, params=None):
    """
    ✅ VERSÃO COMPLETA: Inclui Momentum
    """
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

    # --- ✅ MOMENTUM (NOVO!) ---
    momentum = get_momentum(df, period=10)  # ROC de 10 períodos
    
    # Tratamento de valores extremos
    if momentum is not None:
        # Cap em ±50% (protege contra outliers)
        momentum = max(-0.5, min(momentum, 0.5))
    else:
        momentum = 0.0

    # --- CÁLCULO ATR% REAL ---
    if atr > price * 2:
        atr_price = atr * mt5.symbol_info(symbol).point
    else:
        atr_price = atr
    
    atr_pct_real = (atr_price / price) * 100 if price > 0 else 0

    # --- Z-SCORE DE VOLATILIDADE ---
    vol_series = df['close'].pct_change().rolling(20).std() * 100
    atr_mean = vol_series.mean()
    atr_std = vol_series.std()
    z_score = (atr_pct_real - atr_mean) / atr_std if (atr_std and atr_std > 0) else 0
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
        "symbol": symbol,
        "ema_fast": float(ema_fast),
        "ema_slow": float(ema_slow),
        "rsi": float(rsi),
        "adx": float(adx),
        "atr": float(atr),
        "atr_pct": atr_pct_capped,
        "atr_real": round(atr_pct_real, 3),
        "atr_zscore": round(z_score, 2),
        "momentum": round(momentum, 6),  # ✅ NOVO!
        "volume_ratio": volume_ratio,
        "vol_breakout": is_volatility_breakout(df, atr, atr_mean_val, volume_ratio, side),
        "vwap": get_intraday_vwap(df),
        "close": price,
        "macro_trend_ok": macro_trend_ok(symbol, side),
        "tick_size": mt5.symbol_info(symbol).point,
        "params": params,
        "error": None
    }


# =========================================================
# SCORE FINAL
# =========================================================
def calculate_signal_score(ind: dict) -> float:
    """
    ✅ VERSÃO MELHORADA: Mais restritiva com novos filtros de volume e VWAP.
    Integra min_volume_ratio e require_vwap_proximity do TIME_SCORE_RULES.
    Aumenta thresholds para reduzir falsos positivos.
    """
    
    if not isinstance(ind, dict):
        logger.warning(f"calculate_signal_score recebeu tipo inválido: {type(ind)}")
        return 0.0
    
    if ind.get("error"):
        return 0.0

    score = 0.0
    score_log = {}
    ind.setdefault("block_reason", " ")

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
    momentum = ind.get("momentum", 0.0)
    current_price = ind.get("close", 0)  # 🆕 Para VWAP check
    vwap = ind.get("vwap", None)  # 🆕 Intraday VWAP (assuma calculado em get_intraday_vwap)
    
    # ✅ Busca parâmetros específicos do ativo
    params = ind.get("params", {})
    
    # RSI limites otimizados
    rsi_low = params.get("rsi_low", 30)
    rsi_high = params.get("rsi_high", 70)
    rsi_mid = (rsi_low + rsi_high) / 2
    
    # ADX mínimo otimizado
    adx_threshold = params.get("adx_threshold", None)
    
    if adx_threshold is None:
        _, time_cfg = get_time_bucket()
        adx_min = time_cfg["adx_min"]
    else:
        _, time_cfg = get_time_bucket()
        time_factor = time_cfg["adx_min"] / 8
        adx_min = adx_threshold * max(0.75, min(time_factor, 1.25))
    
    # ✅ MOMENTUM MÍNIMO
    mom_min = params.get("mom_min", time_cfg.get("min_momentum", 0.0015))  # 🆕 Dinâmico do time_cfg
    
    # ATR máximo
    min_score = time_cfg["min_score"]
    atr_max = time_cfg["atr_max"]
    
    # 🆕 Novos do TIME_SCORE_RULES
    min_volume_ratio = time_cfg.get("min_volume_ratio", 1.3)
    require_vwap_proximity = time_cfg.get("require_vwap_proximity", True)
    vwap_tolerance = 0.01  # ±1%

    # =========================
    # 🚫 FILTROS OBRIGATÓRIOS
    # =========================
    
    # 🆕 FILTRO DE VOLUME RATIO (NOVO! Antes de ADX/Momentum)
    if volume_ratio < min_volume_ratio:
        ind["block_reason"] = f"VOLUME_BAIXO ({volume_ratio:.2f} < {min_volume_ratio:.2f})"
        ind["score_log"] = {"VOLUME": 0}
        return 0.0
    
    # 🆕 FILTRO DE VWAP PROXIMITY (NOVO!)
    if require_vwap_proximity and vwap is not None and current_price > 0:
        vwap_dist = abs(current_price - vwap) / vwap
        if vwap_dist > vwap_tolerance:
            ind["block_reason"] = f"VWAP_LONGE ({vwap_dist:.2%} > {vwap_tolerance:.2%})"
            ind["score_log"] = {"VWAP": 0}
            return 0.0
    
    # FILTRO DE ADX MÍNIMO
    if adx < adx_min:
        ind["block_reason"] = f"ADX_LOW ({adx:.0f} < {adx_min:.0f})"
        ind["score_log"] = {"ADX": 0}
        return 0.0

    # FILTRO DE MOMENTUM
    if abs(momentum) < mom_min:
        ind["block_reason"] = f"MOMENTUM_FRACO ({abs(momentum):.6f} < {mom_min:.6f})"
        ind["score_log"] = {"MOMENTUM": 0}
        return 0.0
    
    # ATR extremo
    if atr_pct > atr_max * 1.5:  # 🆕 Aumentei o multiplicador para 1.5 → mais permissivo, mas ainda protetor
        ind["score_log"] = score_log
        ind["block_reason"] = "ATR_EXTREME"
        return 0.0
    
    # Spread
    if not ind.get("spread_ok", True):
        return 0.0

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
    
    # RSI Score com limites do config (🆕 Tornei mais restritivo: penalidade maior para extremos)
    if rsi_low <= rsi <= rsi_mid:
        rsi_score = 20
    elif rsi_mid < rsi <= rsi_high:
        rsi_score = 15
    elif (rsi_low - 10) <= rsi < rsi_low:
        rsi_score = 10
    elif rsi_high < rsi <= (rsi_high + 10):
        rsi_score = 10
    else:
        rsi_score = -20  # 🆕 Aumentei penalidade de -10 para -20 em oversold/overbought extremos

    # ADX Factor (🆕 Mais granular)
    if adx >= adx_min * 1.5:  # Novo tier para ADX muito alto
        adx_factor = 1.2  # Bônus extra
    elif adx >= adx_min * 1.3:
        adx_factor = 1.0
    elif adx >= adx_min:
        adx_factor = 0.8
    else:
        adx_factor = 0.3
    
    rsi_adx_score = rsi_score * adx_factor
    score += rsi_adx_score
    score_log["RSI_ADX"] = round(rsi_adx_score, 1)

    # =========================
    # 🚀 MOMENTUM SCORE
    # =========================
    mom_abs = abs(momentum)
    
    if mom_abs >= mom_min * 5:
        mom_score = 15
        score_log["MOMENTUM"] = 15
    elif mom_abs >= mom_min * 3:
        mom_score = 10
        score_log["MOMENTUM"] = 10
    elif mom_abs >= mom_min:
        mom_score = 5
        score_log["MOMENTUM"] = 5
    else:
        mom_score = 0
        score_log["MOMENTUM"] = 0
    
    score += mom_score

    # =========================
    # 🌊 ATR
    # =========================
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
        score += 10
        score_log["MACRO"] = 10
    else:
        score -= 5
        score_log["MACRO"] = -5

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
    # 🆕 VOLUME SCORE (NOVO!)
    # =========================
    # Bônus para volume alto (além do filtro mínimo)
    if volume_ratio >= min_volume_ratio * 2:
        vol_score = 10  # Volume muito alto
        score_log["VOLUME"] = 10
    elif volume_ratio >= min_volume_ratio * 1.5:
        vol_score = 5  # Volume bom
        score_log["VOLUME"] = 5
    else:
        vol_score = 0
        score_log["VOLUME"] = 0
    
    score += vol_score
    # =========================
    # 🆕 BÔNUS ML (Integração com EnsembleOptimizer)
    # =========================
    try:
        from ml_optimizer import ml_optimizer
        symbol = ind.get('symbol', 'UNKNOWN')  # 🆕 Extraia de ind (adicione 'symbol' no quick_indicators_custom se necessário)
        ml_features = ml_optimizer.extract_features(ind, symbol)  # Correto
        ml_pred = ml_optimizer.predict_signal_score(ml_features)
    
        ml_bonus = ml_pred * 100
        ml_bonus = np.clip(ml_bonus, -10, 15)
    
        score += ml_bonus
        score_log["ML_BONUS"] = round(ml_bonus, 1)
    
        if ml_pred < -0.02:
            ind["block_reason"] = "ML_LOW_PRED"
            return 0.0
    except Exception as e:
        logger.warning(f"ML predição falhou: {e} - Ignorando bônus")
    # =========================
    # ✅ FINAL
    # =========================
    final_score = round(max(score, 0), 1)

    if final_score < min_score and not ind["block_reason"]:
        ind["block_reason"] = "TIME_FILTER"

    ind["score_log"] = score_log
    
    # Debug info
    ind["params_used"] = {
        "rsi_low": rsi_low,
        "rsi_high": rsi_high,
        "adx_min": round(adx_min, 1),
        "mom_min": mom_min,
        "min_volume_ratio": min_volume_ratio  # 🆕 Para debug
    }
    
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
    if not is_valid_dataframe(df):
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
    if _bot_instance is None and getattr(config, "ENABLE_TELEGRAM_NOTIF", False):
        if telebot is None:
            logger.error("telebot não está instalado. Instale com: pip install pyTelegramBotAPI")
            return None
        try:
            _bot_instance = telebot.TeleBot(config.TELEGRAM_BOT_TOKEN)
            logger.info("Bot do Telegram inicializado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao criar bot do Telegram: {e}")
            _bot_instance = None
    return _bot_instance

def send_telegram_exit(symbol: str, side: str = "", volume: float = 0, entry_price: float = 0, exit_price: float = 0, profit_loss: float = 0, reason: str = ""):
    bot = get_telegram_bot()
    if not bot:
        logger.warning("⚠️ Telegram: Bot não disponível para saída")
        return

    # 1. PEGA O LUCRO ACUMULADO DO DIA NO ARQUIVO TXT
    # Usando a função que criamos antes
    lucro_realizado_total, _ = calcular_lucro_realizado_txt()

    # Cálculo do Valor Total da Operação
    total_value = volume * exit_price 
    pl_emoji = "🟢" if profit_loss > 0 else "🔴"
    pl_pct = (profit_loss / (entry_price * volume)) * 100 if entry_price > 0 and volume > 0 else 0

    msg = (
        f"{pl_emoji} <b>XP3 — POSIÇÃO ENCERRADA</b>\n\n"
        f"<b>Ativo:</b> {symbol}\n"
        f"<b>Direção:</b> {side}\n"
        f"<b>Volume:</b> {volume:.0f} ações\n"
        f"<b>Entrada:</b> R${entry_price:.2f} | <b>Saída:</b> R${exit_price:.2f}\n"
        f"<b>Resultado:</b> R${profit_loss:+.2f} ({pl_pct:+.2f}%)\n"
        f"<b>Motivo:</b> {reason}\n"
        f"---------------------------\n"
        f"💰 <b>LUCRO NO BOLSO HOJE: R$ {lucro_realizado_total:,.2f}</b>\n" # AQUI A NOVIDADE!
        f"---------------------------\n"
        f"<i>⏱ {datetime.now().strftime('%H:%M:%S')}</i>"
    )

    try:
        bot.send_message(
            chat_id=config.TELEGRAM_CHAT_ID,
            text=msg,
            parse_mode="HTML"
        )
        logger.info(f"✅ Telegram: Notificação de SAÍDA enviada com Lucro Acumulado")
    except Exception as e:
        logger.error(f"Erro ao enviar Telegram: {e}")

def close_position(symbol: str, ticket: int, volume: float, price: float, reason: str = "Saída Estratégica"):
    """
    Fecha uma posição específica no MT5 e envia notificação.
    """
    # Identifica o tipo da posição pelo ticket para saber o lado oposto
    pos = mt5.positions_get(ticket=ticket)
    if not pos:
        logger.error(f"❌ Erro ao fechar: Posição {ticket} não encontrada.")
        return False

    pos = pos[0]
    # Se a posição é de COMPRA (0), fechamos com VENDA (1) e vice-versa
    order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(volume),
        "type": order_type,
        "position": ticket, # OBRIGATÓRIO para fechar a posição correta
        "price": price,
        "deviation": 10,
        "magic": 2026,
        "comment": f"XP3:{reason}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    with mt5_lock:
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"❌ Falha ao fechar {symbol}: {result.comment}")
            return False
        
        # Se fechou com sucesso, envia o log de saída e notifica Telegram
        logger.info(f"✅ SAÍDA EXECUTADA: {symbol} | Motivo: {reason} | P&L: R${pos.profit:.2f}")
        
        # Chama sua função de Telegram já existente no utils.py
        send_telegram_exit(
            symbol=symbol,
            side="BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL",
            volume=volume,
            entry_price=pos.price_open,
            exit_price=price,
            profit_loss=pos.profit,
            reason=reason
        )
        return True

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

def calculate_position_size_atr(symbol: str, price: float, atr_dist: float, risk_money: float = None) -> float:
    """
    Calcula tamanho da posição com LIMITE INTELIGENTE.
    """
    try:
        # ✅ VALIDAÇÃO ADICIONAL
        if not all(isinstance(x, (int, float)) for x in [price, atr_dist]):
            logger.error(f"calculate_position_size_atr: Tipos inválidos")
            return 0.0
        
        if price <= 0 or atr_dist <= 0:  # ✅ Já estava OK
            logger.warning(f"{symbol}: Preço ou ATR inválidos")
            return 0.0
        
        # Risco padrão
        if risk_money is None:
            acc = mt5.account_info()
            if not acc:
                return 0.0
            risk_money = acc.balance * 0.01  # 1%
        
        # Cálculo base
        volume = risk_money / atr_dist
        
        # Ajuste para lote B3 (múltiplos de 100)
        step_vol = 100.0
        final_vol = round(volume / step_vol) * step_vol
        
        # Validações
        info = mt5.symbol_info(symbol)
        if info:
            final_vol = max(info.volume_min, min(final_vol, info.volume_max))
            
            # === 🔴 LIMITE INTELIGENTE POR PREÇO ===
            # Ações até R$ 5,00: máx 50.000
            # Ações R$ 5-20: máx 20.000
            # Ações R$ 20-50: máx 10.000
            # Ações acima R$ 50: máx 5.000
            
            if price <= 5.0:
                max_vol = 50000.0
            elif price <= 20.0:
                max_vol = 20000.0
            elif price <= 50.0:
                max_vol = 10000.0
            else:
                max_vol = 5000.0
            
            if final_vol > max_vol:
                logger.info(
                    f"📊 {symbol}: Volume ajustado por limite de preço | "
                    f"Calculado: {final_vol:.0f} → Máx: {max_vol:.0f} "
                    f"(preço: R${price:.2f})"
                )
                final_vol = max_vol
        
        return max(0.0, final_vol)
    
    except Exception as e:
        logger.error(f"Erro em calculate_position_size_atr: {e}", exc_info=True)
        return 0.0

def signal_handler(sig, frame):
    with mt5_lock:
        logger.info("Encerrando bot - salvando pesos adaptativos...")
        save_adaptive_weights()
        mt5.shutdown()
    exit(0)
    if threading.current_thread() is threading.main_thread():
        try:
            signal.signal(signal.SIGINT, signal_handler)
            logger.info("✅ Handler de sinal (Ctrl+C) registrado com sucesso")
        except ValueError:
            logger.debug("Não foi possível registrar handler de sinal (ambiente restrito, ex: Streamlit)")

def send_telegram_trade(symbol: str, side: str, volume: float, price: float, sl: float, tp: float, comment: str = ""):
    bot = get_telegram_bot()
    if not bot:
        logger.warning("⚠️ Telegram: Bot não inicializado (token ausente ou inválido)")
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
        f"<b>Volume:</b> {volume:.0f} ações\n"
        f"<b>Entrada:</b> R${price:.2f}\n\n"
        f"<b>🛑 SL:</b> R${sl:.2f} <i>(-{risk_pct}%)</i>\n"
        f"<b>🎯 TP:</b> R${tp:.2f} <i>(+{reward_pct}%)</i>\n"
        f"<b>R:R:</b> 1:{rr_ratio}\n"
        f"<b>Comentário:</b> {comment}\n\n"
        f"<i>⏱ {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</i>"
    )

    try:
        bot.send_message(
            chat_id=config.TELEGRAM_CHAT_ID,
            text=msg,
            parse_mode="HTML"
        )
        logger.info(f"✅ Telegram: Notificação de ENTRADA enviada para {symbol}")
    except Exception as e:
        logger.error(f"❌ ERRO ao enviar Telegram (entrada {symbol}): {e}")

def validate_mt5_connection():
    """
    Verifica se MT5 está conectado e operável.
    """
    try:
        terminal_info = mt5.terminal_info()
        if terminal_info is None:
            logger.critical("❌ MT5 não está inicializado")
            return False
        
        if not terminal_info.connected:
            logger.critical("❌ MT5 não está conectado ao servidor")
            return False
        
        if not terminal_info.trade_allowed:
            logger.error("⚠️ Trading não permitido no MT5")
            return False
        
        return True
    
    except Exception as e:
        logger.error(f"Erro ao validar conexão MT5: {e}")
        return False

def send_order_with_sl_tp(symbol, side, volume, sl, tp, comment="XP3_BOT"):
    """
    Versão BLINDADA com validações completas.
    """
    # 1. Valida conexão
    if not validate_mt5_connection():
        logger.error(f"❌ Ordem abortada ({symbol}): MT5 desconectado")
        return False

    if config.ENABLE_NEWS_FILTER:
        is_blackout, reason = apply_blackout(symbol)
        if is_blackout:
            logger.warning(f"⚠️ Ordem bloqueada por notícia: {symbol} - {reason}")
            send_telegram_message(f"⚠️ Ordem bloqueada por notícia: {symbol} - {reason}")
            return False
    
    # 2. Valida parâmetros
    try:
        volume = float(volume)
        sl = float(sl)
        tp = float(tp)
    except (ValueError, TypeError) as e:
        logger.error(f"❌ Parâmetros inválidos para {symbol}: {e}")
        return False
    
    if volume <= 0:
        logger.error(f"❌ Volume inválido para {symbol}: {volume}")
        return False
    
    # 3. Valida símbolo
    info = mt5.symbol_info(symbol)
    if info is None:
        logger.error(f"❌ Símbolo inválido: {symbol}")
        return False
    
    if not info.visible:
        mt5.symbol_select(symbol, True)
        time.sleep(0.3)
    
    # 4. Prepara ordem
    order_type = mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL
    tick = mt5.symbol_info_tick(symbol)
    
    if not tick:
        logger.error(f"❌ Não foi possível obter cotação de {symbol}")
        return False
    
    price = tick.ask if side == "BUY" else tick.bid
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(volume),
        "type": order_type,
        "price": price,
        "sl": float(sl),
        "tp": float(tp),
        "magic": 123456,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    # 5. Envia ordem com lock
    with mt5_lock:
        result = mt5.order_send(request)
    
    # 6. Valida resultado
    if result is None:
        logger.error(f"❌ MT5 retornou None para {symbol}")
        return False
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(
            f"❌ Ordem {side} {symbol} REJEITADA | "
            f"Retcode: {result.retcode} | "
            f"Comentário: {result.comment}"
        )
        return False
    
    logger.info(f"✅ Ordem {side} executada: {symbol} | Ticket: {result.deal}")
    return True

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

# ===== SUBSTITUIR A FUNÇÃO get_current_risk_pct() NO utils.py =====

def get_current_risk_pct() -> float:
    """
    Retorna o risco percentual atual por trade
    """
    risk = config.RISK_PER_TRADE_PCT
    now = datetime.now()
    weekday = now.weekday()
    hour = now.hour

    # Redução na sexta à tarde
    if weekday == 4 and hour >= 15:
        risk = min(risk, config.REDUCED_RISK_PCT)

    # Regime de mercado
    regime = detect_market_regime()
    if regime == "RISK_OFF":
        risk *= 0.7

    # Power-Hour
    if is_power_hour():
        risk *= config.POWER_HOUR.get("risk_multiplier", 1.0)
    
    # ✅ PROFIT LOCK SEM IMPORT CIRCULAR
    if config.PROFIT_LOCK["enabled"] and config.PROFIT_LOCK["reduce_risk"]:
        with mt5_lock:
            acc = mt5.account_info()
        
        if acc:
            try:
                # Tenta ler do arquivo compartilhado
                if os.path.exists("daily_equity.txt"):
                    with open("daily_equity.txt", "r") as f:
                        equity_inicio = float(f.read().strip())
                    
                    daily_pnl_pct = (acc.equity - equity_inicio) / equity_inicio
                    
                    if daily_pnl_pct >= config.PROFIT_LOCK["daily_target_pct"]:
                        risk *= 0.5
                        logger.debug(f"🔒 Risco reduzido (meta diária atingida)")
            except Exception as e:
                logger.debug(f"Erro ao ler daily_equity: {e}")

    max_allowed = min(
        config.MAX_RISK_PER_SYMBOL_PCT,
        config.MAX_DAILY_DRAWDOWN_PCT
    )

    return max(0.001, min(risk, max_allowed))

def get_dynamic_slippage(symbol, hour):
    base = config.SLIPPAGE_MAP.get(symbol, config.SLIPPAGE_MAP["DEFAULT"])

    
    # Reduz pela metade na power hour (maior liquidez)
    if 15 <= hour <= 17:
        base *= 0.6
    
    # Aumenta 50% na abertura (spread maior)
    elif 10 <= hour <= 11:
        base *= 1.5
    
    return base

def update_adaptive_weights():
    """
    Inicializa pesos adaptativos caso estejam vazios
    """
    global symbol_weights, sector_weights
    
    if not symbol_weights:
        # Inicializa com pesos neutros para todos os ativos do config
        for sym in config.SECTOR_MAP.keys():
            symbol_weights[sym] = {
                "EMA": 1.0,
                "RSI_ADX": 1.0,
                "ATR": 1.0,
                "MACRO": 1.0,
                "CORR": 1.0
            }
        logger.info("✅ Pesos adaptativos inicializados com valores padrão")

def calculate_smart_sl(symbol, entry_price, side, atr, df):
    """
    Calcula stop loss considerando:
    1. ATR (risco estatístico)
    2. Suporte/Resistência mais próximo
    3. Mínimo de 1.5 ATR (nunca muito apertado)
    """
    # ✅ PROTEÇÃO ATR MÍNIMO
    if atr < 0.01:
        atr = 0.01
        logger.warning(f"{symbol}: ATR muito baixo - usando mínimo 0.01")
    
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
    Validação de liquidez ADAPTATIVA por horário.
    """
    try:
        now = datetime.now().time()
        is_after_hours = now >= datetime.strptime("16:30", "%H:%M").time()
        
        # --- CAMADA 1: BOOK REAL (DOM) ---
        book = mt5.market_book_get(symbol)
        
        # ✅ PROTEÇÃO ADICIONAL
        if book is None or len(book) == 0:
            logger.debug(f"⚠️ {symbol}: Book não disponível - permitindo entrada")
            return True

        if book is not None and len(book) > 0:
            target_type = mt5.BOOK_TYPE_SELL if side == "BUY" else mt5.BOOK_TYPE_BUY
            available_liquidity = sum(item.volume for item in book if item.type == target_type)
            
            # === 🔴 THRESHOLDS ADAPTATIVOS ===
            if is_after_hours:
                # After-hours: aceita 20% da liquidez disponível
                min_ratio = 0.20
            else:
                # Horário normal: 50% (mais conservador)
                min_ratio = 0.50
            
            if available_liquidity >= (volume_needed * min_ratio):
                logger.debug(
                    f"✅ Book OK: {symbol} "
                    f"({available_liquidity:.0f}/{volume_needed:.0f} = "
                    f"{(available_liquidity/volume_needed)*100:.0f}%)"
                )
                return True
            else:
                logger.warning(
                    f"⚠️ {symbol}: Book insuficiente "
                    f"({available_liquidity:.0f}/{volume_needed:.0f})"
                )
                return False
        
        # --- CAMADA 2: VOLUME HISTÓRICO (FALLBACK) ---
        df = safe_copy_rates(symbol, mt5.TIMEFRAME_M5, 20)
        
        if df is not None and not df.empty:
            if 'real_volume' in df.columns and df['real_volume'].sum() > 0:
                median_vol = df['real_volume'].median()
            else:
                median_vol = df['tick_volume'].median() * 100
            
            if median_vol <= 0:
                return True
            
            # === 🔴 TOLERÂNCIA ADAPTATIVA ===
            if is_after_hours:
                max_impact = 0.35  # 35% do volume (era 20%)
            else:
                max_impact = 0.20  # 20% normal
            
            impact_ratio = volume_needed / median_vol
            
            if impact_ratio > max_impact:
                logger.warning(
                    f"⚠️ {symbol}: Alto impacto no volume | "
                    f"Ordem: {volume_needed:.0f} | Mediana: {median_vol:.0f} | "
                    f"Impacto: {impact_ratio*100:.1f}% (máx {max_impact*100:.0f}%)"
                )
                return False
            
            logger.debug(f"✅ Volume OK: {symbol} (impacto {impact_ratio*100:.1f}%)")
                
        return True

    except Exception as e:
        logger.error(f"Erro ao analisar profundidade de {symbol}: {e}", exc_info=True)
        return True  # Fail-open

def apply_trailing_stop(symbol: str, side: str, current_price: float, atr: float):
    
    if atr is None or atr <= 0:
        return
    with mt5_lock:
        positions = mt5.positions_get(symbol=symbol)

    if not is_valid_dataframe(positions):
        return

    # Distância do trailing (ex: 1.5 ATR)
    trail_dist = atr * 1.5
    for pos in positions:
        
        # =========================
        # 🟢 COMPRA
        # =========================
        # ✅ NOVO: Só move se houver lucro mínimo
        if side == "BUY" and pos.type == mt5.POSITION_TYPE_BUY:
            profit_dist = current_price - pos.price_open
            
            if profit_dist < atr * 1.0:  # ✅ Lucro mínimo de 1 ATR
                continue
            
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
    
    if not is_valid_dataframe(positions):
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
    
    if not is_valid_dataframe(symbols, min_rows=2):
        return {}
    
    # Coleta dados de fechamento
    closes = {}
    for sym in symbols:
        df = safe_copy_rates(sym, mt5.TIMEFRAME_D1, lookback)
        if df is not None and len(df) >= 30:
            closes[sym] = df['close']
    
    if not is_valid_dataframe(closes, min_rows=2):
        return {}
    
    # Alinha datas
    df_all = pd.DataFrame(closes)
    df_all = df_all.dropna()
    
    if not is_valid_dataframe(df_all, min_rows=30):
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

def mt5_with_retry(max_retries: int = 4, base_delay: float = 1.0):
    """
    Decorator para operações MT5 com retry exponencial
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        logger.error(f"🚨 Falha definitiva em {func.__name__} após {max_retries} tentativas: {e}")
                        raise
                    logger.warning(f"⚠️ Tentativa {attempt}/{max_retries} falhou em {func.__name__}: {e}. Tentando novamente em {delay}s...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
            return None
        return wrapper
    return decorator

def calculate_advanced_metrics(trades_df: pd.DataFrame) -> Dict[str, float]:
    metrics = {}
    
    if not is_valid_dataframe(trades_df):
        return metrics
    
    # Profit Factor
    gross_profit = trades_df[trades_df['pnl_money'] > 0]['pnl_money'].sum()
    gross_loss = abs(trades_df[trades_df['pnl_money'] < 0]['pnl_money'].sum())
    metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # MAE/MFE (assuma que você salva esses dados no save_trade; se não, calcule de outro jeito)
    if 'mae' in trades_df.columns and 'mfe' in trades_df.columns:
        metrics['avg_mae'] = trades_df['mae'].mean()
        metrics['avg_mfe'] = trades_df['mfe'].mean()
    
    # Ulcer Index
    equity_curve = (100000 + trades_df['pnl_money'].cumsum()).values  # Equity inicial fictícia
    peak = np.maximum.accumulate(equity_curve)
    drawdown_pct = ((equity_curve - peak) / peak) ** 2
    metrics['ulcer_index'] = np.sqrt(np.mean(drawdown_pct))
    
    # Recovery Factor
    total_return = (equity_curve[-1] / equity_curve[0]) - 1
    max_dd = np.min((equity_curve - peak) / peak)
    metrics['recovery_factor'] = total_return / abs(max_dd) if max_dd != 0 else float('inf')
    
    return metrics

def is_spread_acceptable(symbol, max_spread_pct=None):
    """
    Valida spread com ajuste automático por horário.
    Power-hour (15:30-17:00) = mais permissivo
    """
    tick = mt5.symbol_info_tick(symbol)
    if not tick or tick.bid <= 0:
        return False

    spread_financeiro = tick.ask - tick.bid
    spread_atual_pct = (spread_financeiro / tick.bid) * 100

    # === 🔴 AJUSTE DINÂMICO POR HORÁRIO ===
    now = datetime.now().time()
    
    if max_spread_pct is None:
        # Horário normal (10:00-15:30)
        if now < datetime.strptime("15:30", "%H:%M").time():
            max_spread_pct = 0.15  # 0.15% (era 0.10%)
        
        # Power-hour e after (15:30-18:00)
        else:
            max_spread_pct = 0.30  # 0.30% (dobro de tolerância)
    
    if spread_atual_pct > max_spread_pct:
        logger.debug(
            f"⚠️ {symbol}: Spread {spread_atual_pct:.3f}% > {max_spread_pct}% "
            f"(horário: {now.strftime('%H:%M')})"
        )
        return False
    
    return True

def adjust_global_sl_after_pyr(symbol, side, current_price, atr):
    """
    Ajusta o SL para o ponto de entrada da primeira perna ou um pouco além,
    garantindo que se o preço voltar, você saia no lucro positivo.
    """
    positions = mt5.positions_get(symbol=symbol)
    if not is_valid_dataframe(positions):
        return

    # Calcula um novo SL que protege a operação (Ex: Preço atual - 1.5 ATR)
    if side == "BUY":
        new_sl = current_price - (atr * 1.0) 
    else: # SELL
        new_sl = current_price + (atr * 1.0)

    for p in positions:
        # Só atualiza se o novo SL for melhor (mais seguro) que o atual
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": p.ticket,
            "sl": new_sl,
            "tp": p.tp, 
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Erro ao ajustar SL de {symbol}: {result.comment}")

def calculate_dynamic_sl_tp(symbol, side, entry_price, ind):
    atr = ind.get("atr", 0.10)
    adx = ind.get("adx", 20)
    
    # Detecta regime
    if adx >= 30:
        regime = "TRENDING"
        tp_mult = 4.5  # Deixa o lucro correr
    elif ind.get("vol_breakout"):
        regime = "BREAKOUT"
        tp_mult = 5.0  # Máxima agressividade
    else:
        regime = "RANGING"
        tp_mult = 2.5  # Conservador
    
    sl_mult = 2.0  # Mantém fixo
    
    if side == "BUY":
        sl = entry_price - (atr * sl_mult)
        tp = entry_price + (atr * tp_mult)
    else:
        sl = entry_price + (atr * sl_mult)
        tp = entry_price - (atr * tp_mult)
    
    # Normalização
    info = mt5.symbol_info(symbol)
    sl = round(sl / info.trade_tick_size) * info.trade_tick_size
    tp = round(tp / info.trade_tick_size) * info.trade_tick_size
    
    return sl, tp

def normalize_price(symbol, price):
    info = mt5.symbol_info(symbol)
    if not info: return price
    
    # Use trade_tick_size aqui também
    normalized = round(price / info.trade_tick_size) * info.trade_tick_size
    return round(normalized, info.digits)

def check_and_apply_breakeven(symbol, current_indicators, move_threshold_atr=1.0):
    """
    Se o preço andou 1x o ATR a favor, move o SL para o preço de entrada.
    """
    positions = mt5.positions_get(symbol=symbol)
    if not is_valid_dataframe(positions):
        return

    ind = current_indicators.get(symbol)
    if not ind: 
        return

    atr = ind.get("atr", 0.10)
    
    for p in positions:
        if p.type == mt5.POSITION_TYPE_BUY:
            if p.price_current >= (p.price_open + (atr * move_threshold_atr)):
                if p.sl < p.price_open:
                    logger.info(f"🛡️ {symbol}: Movendo para Breakeven (COMPRA)")
                    modify_sl_tp(p.ticket, p.price_open + (atr * 0.1), p.tp)
        
        elif p.type == mt5.POSITION_TYPE_SELL:
            if p.price_current <= (p.price_open - (atr * move_threshold_atr)):
                if p.sl > p.price_open or p.sl == 0:
                    logger.info(f"🛡️ {symbol}: Movendo para Breakeven (VENDA)")
                    modify_sl_tp(p.ticket, p.price_open - (atr * 0.1), p.tp)

def modify_sl_tp(ticket, new_sl, new_tp):
    """
    Envia a solicitação de modificação de SL/TP para um ticket específico.
    """
    # Normaliza os preços antes de enviar para evitar erro de tick_size
    pos = mt5.positions_get(ticket=ticket)
    if not pos: return False
    
    symbol = pos[0].symbol
    new_sl = normalize_price(symbol, new_sl)
    new_tp = normalize_price(symbol, new_tp)

    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": ticket,
        "sl": float(new_sl),
        "tp": float(new_tp),
    }

    with mt5_lock:
        result = mt5.order_send(request)
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"❌ Falha ao mover Stop: {result.comment}")
        return False
    
    return True

def update_correlations(top15_symbols):
    """
    Calcula matriz de correlação dos ativos.
    CORRIGIDO: Agora usa o nome correto do parâmetro.
    """
    # ✅ CORREÇÃO: Era 'symbols', agora é 'top15_symbols'
    if not isinstance(top15_symbols, (list, tuple)):
        logger.error(f"update_correlations recebeu tipo inválido: {type(top15_symbols)}")
        return
    
    if not top15_symbols:
        logger.warning("update_correlations: Lista de símbolos vazia")
        return
    
    logger.info(f"📊 Atualizando correlação para {len(top15_symbols)} ativos...")
    
    try:
        # Coleta dados de fechamento dos últimos 50 candles
        data = {}
        for sym in top15_symbols:
            rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M15, 0, 50)
            if rates is not None:
                data[sym] = [r['close'] for r in rates]
        
        if len(data) > 1:
            df = pd.DataFrame(data)
            corr_matrix = df.corr()
            
            # Salva na variável global
            global last_corr_matrix
            last_corr_matrix = corr_matrix
            logger.info("✅ Matriz de correlação atualizada")
            
    except Exception as e:
        logger.error(f"Erro ao calcular correlações: {e}", exc_info=True)

def send_daily_performance_report():
    """
    ✅ VERSÃO CORRIGIDA: Previne erro de DataFrame ambíguo
    """
    from database import get_trades_by_date
    from datetime import date, timedelta

    today = date.today()
    yesterday = today - timedelta(days=1)

    # Tenta trades de hoje, senão de ontem
    trades_today = get_trades_by_date(today)
    
    # ✅ CORREÇÃO: Valida corretamente se tem dados
    if not is_valid_dataframe(trades_today):
        trades_today = get_trades_by_date(yesterday)
        report_date = yesterday.strftime("%d/%m/%Y")
    else:
        report_date = today.strftime("%d/%m/%Y")

    acc = mt5.account_info()
    if not acc:
        return
    equity = acc.equity

    # Usa a variável global daily_max_equity do bot.py
    try:
        from bot import daily_max_equity as daily_max_global
        max_dd_pct = ((daily_max_global - equity) / daily_max_global * 100) if daily_max_global > equity else 0.0
    except:
        max_dd_pct = 0.0  # fallback se não conseguir acessar

    # ✅ CORREÇÃO: Valida novamente antes de processar
    if not is_valid_dataframe(trades_today):
        msg = (
            f"📊 <b>RELATÓRIO DIÁRIO XP3 - {report_date}</b>\n\n"
            f"ℹ️ <i>Nenhum trade realizado hoje.</i>\n\n"
            f"💰 Equity: R${equity:,.2f}\n"
            f"📉 Drawdown do Dia: {max_dd_pct:.2f}%\n"
            f"📊 Posições Abertas: {mt5.positions_total()}\n\n"
            f"✅ Sistema operando normalmente"
        )
        send_telegram_message(msg)
        return

    df = pd.DataFrame(trades_today)

    total_trades = len(df)
    wins = len(df[df['pnl_money'] > 0])
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0

    gross_profit = df[df['pnl_money'] > 0]['pnl_money'].sum()
    gross_loss = abs(df[df['pnl_money'] < 0]['pnl_money'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    rr_ratio = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    best_trade = df.loc[df['pnl_money'].idxmax()]
    worst_trade = df.loc[df['pnl_money'].idxmin()]

    daily_pnl = df['pnl_money'].sum()
    daily_pnl_pct = (daily_pnl / (equity - daily_pnl)) * 100 if (equity - daily_pnl) > 0 else 0

    meta_atingida = daily_pnl_pct >= config.PROFIT_LOCK.get("daily_target_pct", 0.02) * 100

    msg = (
        f"📊 <b>RELATÓRIO DIÁRIO XP3 - {report_date}</b>\n\n"
        f"💰 <b>Patrimônio Final:</b> R${equity:,.2f} ({daily_pnl_pct:+.2f}%)\n"
        f"📈 <b>PnL do Dia:</b> R${daily_pnl:+,.2f}\n"
        f"📉 <b>Max Drawdown:</b> {max_dd_pct:.2f}%\n"
        f"🎯 <b>Meta Diária:</b> {'✅ Atingida' if meta_atingida else '❌ Não atingida'}\n\n"
        f"📊 <b>PERFORMANCE</b>\n"
        f"Trades: {total_trades} | Win Rate: {win_rate:.1f}% ({wins}/{total_trades})\n"
        f"R:R Médio: 1:{rr_ratio:.2f} | Profit Factor: {profit_factor:.2f}\n"
        f"Melhor: +R${best_trade['pnl_money']:,.0f} ({best_trade['symbol']})\n"
        f"Pior: R${worst_trade['pnl_money']:,.0f} ({worst_trade['symbol']})\n\n"
        f"🏆 <b>DESTAQUES</b>\n"
        f"Top Ativo: {best_trade['symbol']} ({best_trade['pnl_pct']:+.2f}%)\n"
        f"🔒 Profit Lock: {'Ativado' if meta_atingida else 'Não ativado'}\n"
        f"📊 Posições EOD: {mt5.positions_total()}\n\n"
        f"✅ Sistema operando normalmente"
    )

    send_telegram_message(msg)

def send_telegram_message(text: str):
    bot = get_telegram_bot()
    if bot and getattr(config, "ENABLE_TELEGRAM_NOTIF", False):
        try:
            bot.send_message(
                chat_id=config.TELEGRAM_CHAT_ID,
                text=text,
                parse_mode="HTML",
                disable_web_page_preview=True
            )
        except Exception as e:
            logger.warning(f"Erro ao enviar relatório Telegram: {e}")

def calcular_lucro_realizado_txt():
    import os
    import re
    from datetime import datetime
    
    filename = f"trades_log_{datetime.now().strftime('%Y-%m-%d')}.txt"
    
    if not os.path.exists(filename):
        return 0.0, 0
    
    total_pnl = 0.0
    contagem_trades = 0
    
    with open(filename, "r", encoding="utf-8") as f:
        conteudo = f.read()
        
        # ✅ REGEX MAIS ROBUSTO
        # Busca linhas de fechamento (não de abertura)
        for linha in conteudo.split('\n'):
            if "Abertura de Posição" in linha or "---" in linha:
                continue
            
            # Match: P&L: +1550.00 ou P&L: -320.50
            match = re.search(r'P&L:\s*([+-]?\d+\.?\d*)', linha)
            if match:
                try:
                    pnl = float(match.group(1))
                    total_pnl += pnl
                    contagem_trades += 1
                except ValueError:
                    continue
                    
    return total_pnl, contagem_trades

def obter_resumo_financeiro_do_dia():
    lucro_realizado, total_ordens = calcular_lucro_realizado_txt()
    lucro_aberto_total = sum(p.profit for p in mt5.positions_get()) if mt5.positions_get() else 0.0
    return lucro_realizado, lucro_aberto_total, total_ordens

def responder_comando_lucro(message):
    bot = get_telegram_bot()
    if not bot: return

    # 1. Busca o Lucro Realizado no seu arquivo TXT (o que já está no bolso)
    realizado, qtd = calcular_lucro_realizado_txt()

    # 2. Busca o Lucro Flutuante (o que está aberto agora no MT5)
    posicoes_abertas = mt5.positions_get()
    aberto = sum(p.profit for p in posicoes_abertas) if posicoes_abertas else 0.0
    total_do_dia = realizado + aberto
    
    emoji = "🚀" if total_do_dia >= 0 else "⚠️"
    
    msg = (
        f"{emoji} <b>STATUS XP3 - AGORA</b>\n\n"
        f"💰 <b>Realizado:</b> R$ {realizado:,.2f}\n"
        f"📈 <b>Flutuante:</b> R$ {aberto:,.2f}\n"
        f"---------------------------\n"
        f"🏆 <b>TOTAL DO DIA: R$ {total_do_dia:,.2f}</b>\n\n"
        f"<i>Baseado em {qtd} ordens e {len(posicoes_abertas) if posicoes_abertas else 0} posições abertas.</i>"
    )

    bot.reply_to(message, msg, parse_mode="HTML")

# ============================================
# 🔥 PRIORIDADE 1 - ANTI-CHOP
# ============================================

import json
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional, Tuple

# Arquivos persistentes
ANTI_CHOP_FILE = "anti_chop_data.json"
DAILY_LIMITS_FILE = "daily_symbol_limits.json"

# Estado global
_symbol_sl_timestamps = {}  # {symbol: timestamp_último_sl}
_symbol_sl_prices = {}  # {symbol: preço_quando_bateu_sl}
_daily_symbol_trades = defaultdict(lambda: {"total": 0, "losses": 0})  # Contador diário

# ============================================
# 📁 PERSISTÊNCIA
# ============================================

def load_anti_chop_data():
    """Carrega dados de cooldown"""
    global _symbol_sl_timestamps, _symbol_sl_prices
    
    if os.path.exists(ANTI_CHOP_FILE):
        try:
            with open(ANTI_CHOP_FILE, "r") as f:
                data = json.load(f)
                _symbol_sl_timestamps = {
                    k: datetime.fromisoformat(v) 
                    for k, v in data.get("timestamps", {}).items()
                }
                _symbol_sl_prices = data.get("prices", {})
            logger.info("✅ Dados anti-chop carregados")
        except Exception as e:
            logger.error(f"Erro ao carregar anti-chop: {e}")

def save_anti_chop_data():
    """Salva dados de cooldown"""
    data = {
        "timestamps": {k: v.isoformat() for k, v in _symbol_sl_timestamps.items()},
        "prices": _symbol_sl_prices
    }
    try:
        with open(ANTI_CHOP_FILE, "w") as f:
            json.dump(data, f)
    except Exception as e:
        logger.error(f"Erro ao salvar anti-chop: {e}")

def load_daily_limits():
    """Carrega contadores diários"""
    global _daily_symbol_trades
    
    if os.path.exists(DAILY_LIMITS_FILE):
        try:
            with open(DAILY_LIMITS_FILE, "r") as f:
                data = json.load(f)
                
                # Valida se é do dia atual
                saved_date = data.get("date")
                today = datetime.now().date().isoformat()
                
                if saved_date == today:
                    _daily_symbol_trades = defaultdict(
                        lambda: {"total": 0, "losses": 0},
                        data.get("trades", {})
                    )
                    logger.info("✅ Limites diários carregados")
                else:
                    logger.info("🔄 Novo dia detectado - resetando limites")
        except Exception as e:
            logger.error(f"Erro ao carregar limites: {e}")

def save_daily_limits():
    """Salva contadores diários"""
    data = {
        "date": datetime.now().date().isoformat(),
        "trades": dict(_daily_symbol_trades)
    }
    try:
        with open(DAILY_LIMITS_FILE, "w") as f:
            json.dump(data, f)
    except Exception as e:
        logger.error(f"Erro ao salvar limites: {e}")

# ============================================
# 🚫 ANTI-CHOP: COOLDOWN APÓS SL
# ============================================

def register_sl_hit(symbol: str, sl_price: float):
    """
    Registra que o SL foi atingido
    Chame isso em close_position() quando reason contém "SL"
    """
    if not config.ANTI_CHOP["enabled"]:
        return
    
    _symbol_sl_timestamps[symbol] = datetime.now()
    _symbol_sl_prices[symbol] = sl_price
    save_anti_chop_data()
    
    logger.info(
        f"🛑 {symbol}: SL registrado @ R${sl_price:.2f} | "
        f"Cooldown: {config.ANTI_CHOP['cooldown_after_sl_minutes']} min"
    )

def check_anti_chop_filter(symbol: str, current_price: float, atr: float) -> Tuple[bool, str]:
    """
    ✅ VERSÃO PROGRESSIVA: Cooldown aumenta a cada perda
    """
    if not config.ANTI_CHOP["enabled"]:
        return True, ""
    
    # === 1️⃣ COOLDOWN TEMPORAL PROGRESSIVO ===
    last_sl_time = _symbol_sl_timestamps.get(symbol)
    
    if last_sl_time:
        # 🆕 Calcula quantas perdas consecutivas teve
        stats = _daily_symbol_trades.get(symbol, {"losses": 0})
        loss_count = stats.get("losses", 0)
        if loss_count >= 3:
            logger.error(f"🔒 {symbol}: BLOQUEADO - 3+ perdas consecutivas")
            return False, "BLOQUEADO_PERDAS_EXCESSIVAS"
        
        # 🆕 Aumentar cooldown base e multiplicadores (mais restritivo)
        cooldown_minutes = config.ANTI_CHOP["cooldown_after_sl_minutes"] * 1.5  # Novo: 180min base (120*1.5)
        if config.ANTI_CHOP.get("progressive_cooldown", False):
            multipliers = config.ANTI_CHOP.get("cooldown_multipliers", {})
            multiplier = multipliers.get(loss_count, 4.0)  # Default: 4x se >3 perdas
            cooldown_minutes *= multiplier
        
        elapsed = (datetime.now() - last_sl_time).total_seconds() / 60
        
        if elapsed < cooldown_minutes:
            remaining = int(cooldown_minutes - elapsed)
            logger.warning(f"🚫 Anti-Chop bloqueou {symbol}: Cooldown SL ({remaining} min | {loss_count} perdas)")  # Novo: Log aviso
            return False, f"Cooldown SL ({remaining} min restantes | {loss_count} perdas)"
        
    # === 2️⃣ MOVIMENTO MÍNIMO ===
    last_sl_price = _symbol_sl_prices.get(symbol)
    
    if last_sl_price:
        price_change_pct = abs((current_price - last_sl_price) / last_sl_price) * 100
        min_range = config.ANTI_CHOP["min_range_pct"]
        
        if price_change_pct < min_range:
            return False, f"Range insuficiente ({price_change_pct:.2f}% < {min_range}%)"
    
    # === 3️⃣ VOLATILIDADE ANORMAL ===
    df = safe_copy_rates(symbol, TIMEFRAME_BASE, 50)
    if df is not None:
        vol_series = df['close'].pct_change().rolling(20).std() * 100
        atr_mean = vol_series.mean()
        atr_std = vol_series.std()
        atr_pct_real = (atr / current_price) * 100 if current_price > 0 else 0
        z_score = (atr_pct_real - atr_mean) / atr_std if atr_std > 0 else 0
        
        if abs(z_score) > 2.5:  # ⬆️ Era 2.0 → Agora 2.5 (mais restritivo)
            return False, f"Volatilidade anormal (z_score: {z_score:.2f})"
    
    # 🆕 Novo: Bloqueio total após max perdas (integra com DAILY_SYMBOL_LIMITS)
    if stats.get("losses", 0) >= config.DAILY_SYMBOL_LIMITS["max_losing_trades_per_symbol"]:
        logger.error(f"🔒 {symbol} bloqueado pelo dia: {stats['losses']} perdas")  # Novo: Log erro
        return False, f"Bloqueado: Máx perdas diárias atingidas ({stats['losses']})"
    return True, ""

def clear_anti_chop_cooldown(symbol: str):
    """Limpa cooldown após entrada bem-sucedida"""
    if symbol in _symbol_sl_timestamps:
        del _symbol_sl_timestamps[symbol]
    if symbol in _symbol_sl_prices:
        del _symbol_sl_prices[symbol]
    save_anti_chop_data()

# ============================================
# 📊 LIMITE DIÁRIO POR ATIVO
# ============================================

def check_daily_symbol_limit(symbol: str, is_loss: bool = False) -> Tuple[bool, str]:
    """
    Verifica limites diários
    
    Args:
        symbol: Ativo
        is_loss: Se True, conta como perda (para validação futura)
    
    Returns:
        (pode_operar: bool, motivo: str)
    """
    if not config.DAILY_SYMBOL_LIMITS["enabled"]:
        return True, ""
    
    stats = _daily_symbol_trades[symbol]
    
    # Limite de perdas
    max_losses = config.DAILY_SYMBOL_LIMITS["max_losing_trades_per_symbol"]
    if stats["losses"] >= max_losses:
        return False, f"Limite de perdas diário ({max_losses})"
    
    # Limite total de trades
    max_total = config.DAILY_SYMBOL_LIMITS["max_total_trades_per_symbol"]
    if stats["total"] >= max_total:
        return False, f"Limite total diário ({max_total})"
    
    return True, ""

def register_trade_result(symbol: str, is_loss: bool):
    if not config.DAILY_SYMBOL_LIMITS["enabled"]:
        return
    
    _daily_symbol_trades[symbol]["total"] += 1
    
    if is_loss:
        _daily_symbol_trades[symbol]["losses"] += 1
        logger.warning(f"📉 {symbol}: Perda #{_daily_symbol_trades[symbol]['losses']}/{config.DAILY_SYMBOL_LIMITS['max_losing_trades_per_symbol']}")
        
        # 🆕 Novo: Bloqueio imediato se exceder (redundância)
        if _daily_symbol_trades[symbol]["losses"] > config.DAILY_SYMBOL_LIMITS["max_losing_trades_per_symbol"]:
            logger.critical(f"🚨 {symbol} excedeu perdas! Bloqueando permanentemente hoje.")
            # Adicione lógica para bloquear entradas (ex: flag global)
    
    save_daily_limits()

def reset_daily_limits():
    """Reseta contadores diários (chamar em handle_daily_cycle)"""
    global _daily_symbol_trades
    _daily_symbol_trades.clear()
    save_daily_limits()
    logger.info("🔄 Limites diários resetados")

# ============================================
# 🔺 PIRÂMIDE INTELIGENTE
# ============================================

def check_pyramid_eligibility(symbol: str, side: str, ind: dict) -> Tuple[bool, str]:
    """
    ✅ VERSÃO REFORÇADA: Valida pirâmide com requisitos críticos
    
    Returns:
        (pode_piramidar: bool, motivo: str)
    """
    with mt5_lock:
        positions = mt5.positions_get(symbol=symbol)
    
    if not positions or len(positions) == 0:
        return True, "Primeira entrada"
    
    pos = positions[0]
    
    # Valida direção
    existing_side = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
    if existing_side != side:
        return False, "Direção oposta à posição"
    
    # Conta pernas atuais
    pyr_count = pos.comment.count("PYR") if pos.comment else 0
    if pyr_count >= config.PYRAMID_MAX_LEGS:
        return False, f"Limite de pernas ({config.PYRAMID_MAX_LEGS})"
    
    # ============================================
    # ✅ NOVAS VALIDAÇÕES CRÍTICAS
    # ============================================
    
    atr = ind.get("atr", 0.01)
    current_price = pos.price_current
    entry_price = pos.price_open
    
    # 1️⃣ BREAKEVEN OBRIGATÓRIO
    if config.PYRAMID_REQUIREMENTS_ENHANCED["require_breakeven"]:
        sl = pos.sl
        
        if side == "BUY":
            at_breakeven = sl >= entry_price
        else:
            at_breakeven = sl <= entry_price
        
        if not at_breakeven:
            return False, "SL não está no breakeven"
    
    # 2️⃣ +1R FLUTUANTE (alternativa ao BE)
    if config.PYRAMID_REQUIREMENTS_ENHANCED["require_1r_floating"]:
        profit_dist = abs(current_price - entry_price)
        profit_in_r = profit_dist / atr if atr > 0 else 0
        
        if profit_in_r < 1.5:
            return False, f"Lucro flutuante < 1R ({profit_in_r:.2f}R)"
    
    # 3️⃣ TEMPO MÍNIMO ENTRE PERNAS
    min_time = config.PYRAMID_REQUIREMENTS_ENHANCED["min_time_between_legs_minutes"]
    
    # Busca timestamp da última perna
    try:
        from bot import position_open_times
        last_entry_time = position_open_times.get(pos.ticket, 0)
        
        if last_entry_time:
            elapsed_minutes = (time.time() - last_entry_time) / 60
            
            if elapsed_minutes < min_time:
                remaining = int(min_time - elapsed_minutes)
                return False, f"Aguardar {remaining} min entre pernas"
    except:
        pass  # Não bloqueia se não conseguir obter timestamp
    
    # 4️⃣ CORRELAÇÃO (não piramidar se carteira correlacionada)
    max_corr = config.PYRAMID_REQUIREMENTS_ENHANCED["max_correlation_for_pyramid"]
    
    try:
        with mt5_lock:
            all_positions = mt5.positions_get() or []
        
        symbols_in_portfolio = [p.symbol for p in all_positions if p.symbol != symbol]
        
        if symbols_in_portfolio:
            # Importa a função do próprio utils
            from utils import get_average_correlation_with_portfolio
            avg_corr = get_average_correlation_with_portfolio(symbol, symbols_in_portfolio)
            
            if avg_corr > max_corr:
                return False, f"Carteira correlacionada ({avg_corr:.2f} > {max_corr})"
    except Exception as e:
        logger.debug(f"Não foi possível validar correlação: {e}")
    
    # ============================================
    # ✅ VALIDAÇÕES ANTIGAS (mantidas)
    # ============================================
    
    adx = ind.get("adx", 0)
    if adx < config.PYRAMID_REQUIREMENTS["min_adx"]:
        return False, f"ADX baixo ({adx:.0f})"
    
    rsi = ind.get("rsi", 50)
    if side == "BUY" and rsi > config.PYRAMID_REQUIREMENTS["max_rsi_long"]:
        return False, "RSI sobrecomprado"
    
    if side == "SELL" and rsi < config.PYRAMID_REQUIREMENTS["min_rsi_short"]:
        return False, "RSI sobrevendido"
    
    volume_ratio = ind.get("volume_ratio", 1.0)
    if volume_ratio < config.PYRAMID_REQUIREMENTS["volume_ratio"]:
        return False, "Volume insuficiente"
    
    return True, "Elegível para pirâmide"

# ============================================
# 🛡️ RANGE MÍNIMO
# ============================================

def check_minimum_price_movement(symbol: str, df: pd.DataFrame, atr: float) -> Tuple[bool, str]:
    """
    Valida se houve movimento mínimo antes de entrar
    """
    if not config.MIN_PRICE_MOVEMENT["enabled"]:
        return True, ""
    
    lookback = config.MIN_PRICE_MOVEMENT["lookback_candles"]
    
    if df is None or len(df) < lookback:
        return True, ""  # Fail-open
    
    recent = df.tail(lookback)
    price_range = recent["high"].max() - recent["low"].min()
    
    min_movement = atr * config.MIN_PRICE_MOVEMENT["min_atr_multiplier"]
    
    if price_range < min_movement:
        return False, f"Range baixo ({price_range:.2f} < {min_movement:.2f})"
    
    return True, ""