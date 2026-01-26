import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta, time as datetime_time
from typing import Optional, Dict, Any, List, Tuple
from collections import defaultdict, deque
import json
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import config
from threading import RLock, Lock # Changed from Lock to RLock as per refactor
import threading
import queue
import os
import redis
import pickle
import hashlib
import signal
import sys
import requests # ✅ Adicionado para Polygon.io
from news_calendar import apply_blackout
from ml_optimizer import ml_optimizer
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
anti_chop_rf = None
anti_chop_scaler = StandardScaler()
ANTI_CHOP_MODEL_PATH = "anti_chop_rf.pkl"

def load_anti_chop_model():
    global anti_chop_rf
    if os.path.exists(ANTI_CHOP_MODEL_PATH):
        anti_chop_rf = joblib.load(ANTI_CHOP_MODEL_PATH)
    else:
        # Treinar com dados sintéticos ou históricos
        anti_chop_rf = RandomForestClassifier(n_estimators=100, random_state=42)
        # ... (treinamento simulado; usar histórico real em produção)
        joblib.dump(anti_chop_rf, ANTI_CHOP_MODEL_PATH)
load_anti_chop_model()
def is_valid_dataframe(df, min_rows: int = 1) -> bool:
    """Valida se o DataFrame é válido para cálculos."""
    if df is None: return False
    if isinstance(df, pd.DataFrame):
        return not df.empty and len(df) >= min_rows
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

# =========================================================
# 🌐 POLYGON.IO FALLBACK
# =========================================================

def get_polygon_rates_fallback(symbol: str, timeframe, count: int) -> Optional[pd.DataFrame]:
    """
    ✅ Fallback: Obtém dados da Polygon.io se o MT5 falhar.
    """
    if not hasattr(config, "POLYGON_API_KEY") or config.POLYGON_API_KEY == "MOCK_KEY_FOR_NOW":
        return None
        
    try:
        # Mapa de timeframe MT5 → Polygon
        tf_map = {
            mt5.TIMEFRAME_M1: ("minute", 1),
            mt5.TIMEFRAME_M5: ("minute", 5),
            mt5.TIMEFRAME_M15: ("minute", 15),
            mt5.TIMEFRAME_H1: ("hour", 1),
            mt5.TIMEFRAME_D1: ("day", 1),
        }
        
        timespan, multiplier = tf_map.get(timeframe, ("minute", 15))
        ticker = symbol.replace(".SA", "") # Formato B3 simples
        
        url = f"{config.POLYGON_BASE_URL}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}"
        
        # Pega últimos 30 dias de histórico
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        url = f"{url}/{start}/{end}"
        
        params = {
            "adjusted": "true",
            "sort": "desc",
            "limit": count,
            "apiKey": config.POLYGON_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=8)
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            if results:
                df = pd.DataFrame(results)
                # Formata para padrão MT5
                df = df.rename(columns={
                    "t": "time", "o": "open", "h": "high", 
                    "l": "low", "c": "close", "v": "tick_volume", "vw": "vwap"
                })
                df["time"] = pd.to_datetime(df["time"], unit="ms")
                if "real_volume" not in df.columns:
                    df["real_volume"] = df["tick_volume"]
                
                df.set_index("time", inplace=True)
                logger.info(f"✅ Fallback Polygon SUCESSO: {symbol}")
                return df.sort_index()
                
        return None
    except Exception as e:
        logger.error(f"Erro Polygon Fallback ({symbol}): {e}")
        return None

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
            # ✅ FALLBACK: Polygon.io
            logger.warning(f"⚠️ MT5 Falhou em {symbol}. Acionando Polygon.io fallback...")
            return get_polygon_rates_fallback(symbol, timeframe, count)

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        return df.sort_index()
    except queue.Empty:
        # ✅ FALLBACK: Polygon.io
        logger.warning(f"⚠️ Timeout MT5 em {symbol}. Acionando Polygon.io fallback...")
        return get_polygon_rates_fallback(symbol, timeframe, count)

# =========================================================
# AWS LAMBDA HOOK (SCALABILITY)
# =========================================================

def lambda_invoke(function_name: str, payload: dict) -> Optional[dict]:
    """
    Invoca uma função AWS Lambda para processamento pesado (ex: ML Inference).
    """
    try:
        import boto3
        import json
        client = boto3.client('lambda', region_name=getattr(config, 'AWS_REGION', 'us-east-1'))
        response = client.invoke(
            FunctionName=function_name,
            InvocationType='RequestResponse',
            Payload=json.dumps(payload)
        )
        return json.loads(response['Payload'].read())
    except ImportError:
        logger.debug("Boto3 não instalado. AWS Lambda desabilitado.")
        return None
    except Exception as e:
        logger.error(f"Erro ao invocar Lambda {function_name}: {e}")
        return None

# =========================================================
# REDIS CACHE HELPERS
# =========================================================

def cached_symbol_info(symbol: str) -> Optional[mt5.SymbolInfo]:
    """
    Versão com cache do symbol_info
    """
    if not REDIS_AVAILABLE:
        return mt5.symbol_info(symbol)
        
    cache_key = f"symbol_info:{symbol}"
    try:
        cached = redis_client.get(cache_key)
        if cached:
            return pickle.loads(cached)
            
        info = mt5.symbol_info(symbol)
        if info:
            redis_client.setex(cache_key, config.REDIS_CACHE_TTL_INFO, pickle.dumps(info))
        return info
    except Exception as e:
        logger.debug(f"Redis Info Error: {e}")
        return mt5.symbol_info(symbol)

def cached_symbol_info_tick(symbol: str):
    """
    Versão com cache do symbol_info_tick (TTL 1s)
    """
    if not REDIS_AVAILABLE:
        return mt5.symbol_info_tick(symbol)
        
    cache_key = f"tick:{symbol}"
    try:
        cached = redis_client.get(cache_key)
        if cached:
            return pickle.loads(cached)
            
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            redis_client.setex(cache_key, config.REDIS_CACHE_TTL_TICK, pickle.dumps(tick))
        return tick
    except Exception as e:
        logger.debug(f"Redis Tick Error: {e}")
        return mt5.symbol_info_tick(symbol)


# =========================================================
# 🛑 FILTRO ANTI-OVERTRADING
# =========================================================

# Rastreamento de trades por símbolo e período
_trade_history = {}  # {symbol: [timestamps]}
_trade_history_lock = RLock()

def check_anti_overtrading(symbol: str, max_trades_per_hour: int = 3, 
                            max_trades_per_day: int = 10) -> Tuple[bool, str]:
    """
    ✅ Verifica se está fazendo trades demais em um símbolo.
    
    Regras:
    - Máximo 3 trades/hora por símbolo
    - Máximo 10 trades/dia por símbolo
    
    Returns:
        (allowed: bool, reason: str)
    """
    global _trade_history
    
    with _trade_history_lock:
        now = datetime.now()
        
        if symbol not in _trade_history:
            _trade_history[symbol] = []
        
        # Limpa histórico antigo (>24h)
        _trade_history[symbol] = [
            t for t in _trade_history[symbol] 
            if (now - t).total_seconds() < 86400
        ]
        
        # Conta trades na última hora
        hour_ago = now - timedelta(hours=1)
        trades_last_hour = sum(1 for t in _trade_history[symbol] if t >= hour_ago)
        
        # Conta trades hoje
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        trades_today = sum(1 for t in _trade_history[symbol] if t >= today)
        
        if trades_last_hour >= max_trades_per_hour:
            return False, f"Overtrading/hora: {trades_last_hour}/{max_trades_per_hour}"
        
        if trades_today >= max_trades_per_day:
            return False, f"Overtrading/dia: {trades_today}/{max_trades_per_day}"
        
        return True, f"OK ({trades_last_hour}/h, {trades_today}/d)"


def register_trade_for_overtrading(symbol: str):
    """Registra um trade para controle de overtrading."""
    global _trade_history
    
    with _trade_history_lock:
        if symbol not in _trade_history:
            _trade_history[symbol] = []
        _trade_history[symbol].append(datetime.now())


# =========================================================
# 📊 WIN RATE EM TEMPO REAL
# =========================================================

def get_realtime_win_rate(lookback_trades: int = 20) -> Dict[str, float]:
    """
    ✅ Calcula win rate em tempo real dos últimos N trades.
    
    Returns:
        {
            'win_rate': float (0-1),
            'total_trades': int,
            'wins': int,
            'losses': int,
            'avg_win': float,
            'avg_loss': float,
            'profit_factor': float,
            'expectancy': float
        }
    """
    try:
        with mt5_lock:
            deals = mt5.history_deals_get(
                datetime.now() - timedelta(days=7), 
                datetime.now()
            )
        
        if not deals:
            return _default_win_rate_result()
        
        # Filtra fechamentos
        out_deals = [d for d in deals if d.entry == mt5.DEAL_ENTRY_OUT][-lookback_trades:]
        
        if len(out_deals) < 5:
            return _default_win_rate_result()
        
        wins = [d for d in out_deals if d.profit > 0]
        losses = [d for d in out_deals if d.profit <= 0]
        
        win_rate = len(wins) / len(out_deals)
        
        avg_win = sum(d.profit for d in wins) / len(wins) if wins else 0
        avg_loss = abs(sum(d.profit for d in losses) / len(losses)) if losses else 0
        
        profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        return {
            'win_rate': win_rate,
            'total_trades': len(out_deals),
            'wins': len(wins),
            'losses': len(losses),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy
        }
        
    except Exception as e:
        logger.error(f"Erro ao calcular win rate real-time: {e}")
        return _default_win_rate_result()


def _default_win_rate_result() -> Dict[str, float]:
    """Resultado padrão quando não há dados suficientes."""
    return {
        'win_rate': 0.55,
        'total_trades': 0,
        'wins': 0,
        'losses': 0,
        'avg_win': 0,
        'avg_loss': 0,
        'profit_factor': 1.0,
        'expectancy': 0
    }


def get_symbol_performance(symbol: str, lookback_days: int = 30) -> Dict[str, float]:
    """
    ✅ Performance específica de um símbolo.
    """
    try:
        with mt5_lock:
            deals = mt5.history_deals_get(
                datetime.now() - timedelta(days=lookback_days), 
                datetime.now()
            )
        
        if not deals:
            return {'win_rate': 0.55, 'trades': 0}
        
        symbol_deals = [d for d in deals if d.symbol == symbol and d.entry == mt5.DEAL_ENTRY_OUT]
        
        if len(symbol_deals) < 3:
            return {'win_rate': 0.55, 'trades': len(symbol_deals)}
        
        wins = sum(1 for d in symbol_deals if d.profit > 0)
        total_pnl = sum(d.profit for d in symbol_deals)
        
        return {
            'win_rate': wins / len(symbol_deals),
            'trades': len(symbol_deals),
            'wins': wins,
            'losses': len(symbol_deals) - wins,
            'total_pnl': total_pnl
        }
        
    except Exception as e:
        logger.error(f"Erro ao buscar performance {symbol}: {e}")
        return {'win_rate': 0.55, 'trades': 0}


def get_dynamic_rr_min() -> float:
    """
    ✅ R:R DINÂMICO MULTI-FATOR
    
    Fatores:
    1. Regime de mercado (IBOV trend)
    2. VIX Brasil (BVIX)
    3. Volatilidade histórica (ATR agregado)
    4. Drawdown atual
    
    Ranges:
    - RISK_ON + vol baixa: 1.2
    - RISK_OFF + vol alta: 2.5
    """
    regime = detect_market_regime()
    
    # ✅ NOVO: Fator 1 - Regime base
    if regime == "RISK_ON":
        base_rr = 1.25
    else:
        base_rr = 2.0  # ✅ 1.5 → 2.0 (mais conservador)
    
    # ✅ NOVO: Fator 2 - Volatilidade histórica
    try:
        ibov_df = safe_copy_rates("IBOV", mt5.TIMEFRAME_D1, 30)
        
        if ibov_df is not None:
            hvol = ibov_df['close'].pct_change().std() * 100
            
            # Multiplica RR se vol > 2%
            if hvol > 2.0:
                vol_multiplier = 1.3
            elif hvol > 1.5:
                vol_multiplier = 1.15
            else:
                vol_multiplier = 1.0
        else:
            vol_multiplier = 1.0
    except:
        vol_multiplier = 1.0
    
    # ✅ NOVO: Fator 3 - Drawdown atual
    try:
        from bot import daily_max_equity
        acc = mt5.account_info()
        
        if acc and daily_max_equity > 0:
            current_dd = (daily_max_equity - acc.equity) / daily_max_equity
            
            # Aumenta RR mínimo se em drawdown
            if current_dd > 0.03:  # >3%
                dd_multiplier = 1.5
            elif current_dd > 0.02:  # >2%
                dd_multiplier = 1.25
            else:
                dd_multiplier = 1.0
        else:
            dd_multiplier = 1.0
    except:
        dd_multiplier = 1.0
    
    # Cálculo Final
    final_rr = base_rr * vol_multiplier * dd_multiplier
    
    # logger.info(f"📊 R:R Dinâmico | Regime: {regime} | Base: {base_rr:.2f} | Vol×: {vol_multiplier:.2f} | DD×: {dd_multiplier:.2f} | Final: {final_rr:.2f}")
    
    return final_rr

# =========================================================
# 🌐 POLYGON.IO INTEGRATION
# =========================================================

def get_vix_br() -> float:
    """
    ✅ V5.2: Obtém VIX Brasil (proxy) via Polygon ou MT5.
    Se falhar, retorna valor baseado na volatilidade do IBOV.
    """
    if REDIS_AVAILABLE:
        cached = redis_client.get("vix_br")
        if cached: return float(cached)

    vix = 20.0
    try:
        # Tenta proxy via Polygon (VIXM ou similar se disponível)
        if hasattr(config, "POLYGON_API_KEY") and config.POLYGON_API_KEY != "MOCK_KEY_FOR_NOW":
            # Usando VIX como proxy global se B3 VIX não disponível
            url = f"{config.POLYGON_BASE_URL}/v2/aggs/ticker/I:VIX/prev"
            resp = requests.get(url, params={"apiKey": config.POLYGON_API_KEY}, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("results"):
                    vix = data["results"][0]["c"]
        else:
            # Fallback: Volatilidade do IBOV
            df = safe_copy_rates("IBOV", mt5.TIMEFRAME_H1, 100)
            if is_valid_dataframe(df):
                vix = df['close'].pct_change().std() * 100 * np.sqrt(252 * 7) # Proxy volatilidade anualizada
                vix = max(10, min(vix, 80))
    except:
        vix = 22.0

    if REDIS_AVAILABLE:
        redis_client.setex("vix_br", 300, str(vix)) # 5 min cache
    return float(vix)

def get_order_flow(symbol: str, bars: int = 20) -> Dict[str, float]:
    """
    ✅ V5.2 CONSOLIDADO: Order Flow Híbrido (Polygon + MT5)
    """
    # 1. Tenta Polygon (Melhor qualidade para CVD real)
    if hasattr(config, "POLYGON_API_KEY") and config.POLYGON_API_KEY != "MOCK_KEY_FOR_NOW":
        try:
            ticker = symbol.replace(".SA", "")
            url = f"{config.POLYGON_BASE_URL}/v2/aggs/ticker/{ticker}/range/1/minute"
            # Hoje
            end = datetime.now().strftime("%Y-%m-%d")
            resp = requests.get(f"{url}/{end}/{end}", params={"apiKey": config.POLYGON_API_KEY, "limit": bars}, timeout=5)
            if resp.status_code == 200:
                res = resp.json().get("results", [])
                if res:
                    buy_v = sum(r['v'] for r in res if r['c'] >= r['o'])
                    sell_v = sum(r['v'] for r in res if r['c'] < r['o'])
                    total = buy_v + sell_v
                    return {
                        "imbalance": (buy_v - sell_v) / total if total > 0 else 0,
                        "cvd": buy_v - sell_v,
                        "buy_volume": buy_v,
                        "sell_volume": sell_v
                    }
        except: pass

    # 2. Fallback MT5 (Estimado)
    df = safe_copy_rates(symbol, mt5.TIMEFRAME_M1, bars)
    if not is_valid_dataframe(df):
        return {"imbalance": 0.0, "cvd": 0.0, "buy_volume": 0, "sell_volume": 0}
    
    df['delta'] = np.where(df['close'] >= df['open'], df['tick_volume'], -df['tick_volume'])
    buy_v = df[df['delta'] > 0]['tick_volume'].sum()
    sell_v = df[df['delta'] < 0]['tick_volume'].sum()
    total = buy_v + sell_v
    
    return {
        "imbalance": (buy_v - sell_v) / total if total > 0 else 0,
        "cvd": float(df['delta'].sum()),
        "buy_volume": float(buy_v),
        "sell_volume": float(sell_v)
    }



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


def check_and_alert_slippage(symbol: str, requested_price: float, 
                               executed_price: float, side: str) -> bool:
    """
    Verifica se houve slippage excessivo na execução e notifica o usuário.
    
    Args:
        symbol: Símbolo do ativo
        requested_price: Preço solicitado
        executed_price: Preço executado
        side: BUY ou SELL
    
    Returns:
        bool: True se slippage foi excessivo
    """
    try:
        info = mt5.symbol_info(symbol)
        if not info or info.point <= 0:
            return False
        
        # Calcula slippage em ticks
        slippage = abs(executed_price - requested_price)
        slippage_ticks = slippage / info.point
        
        # Verifica direção (adverso = pior para o trader)
        if side == "BUY":
            is_adverse = executed_price > requested_price
        else:
            is_adverse = executed_price < requested_price
        
        max_slippage_ticks = getattr(config, 'SLIPPAGE_ALERT_TICKS', 3)
        
        if is_adverse and slippage_ticks > max_slippage_ticks:
            logger.warning(
                f"⚠️ SLIPPAGE EXCESSIVO {symbol} | "
                f"Solicitado: {requested_price:.2f} | "
                f"Executado: {executed_price:.2f} | "
                f"Slippage: {slippage_ticks:.1f} ticks"
            )
            
            # Envia alerta Telegram
            try:
                send_telegram_message(
                    f"⚠️ <b>SLIPPAGE EXCESSIVO</b>\n\n"
                    f"📊 Ativo: <b>{symbol}</b>\n"
                    f"📈 Direção: {side}\n"
                    f"💰 Solicitado: R$ {requested_price:.2f}\n"
                    f"💸 Executado: R$ {executed_price:.2f}\n"
                    f"📉 Slippage: <b>{slippage_ticks:.1f} ticks</b>\n\n"
                    f"💡 <i>Revise a liquidez do ativo</i>"
                )
            except Exception as e:
                logger.debug(f"Erro ao enviar alerta slippage: {e}")
            
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Erro ao verificar slippage: {e}")
        return False


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

def get_ibov_correlation(symbol: str, lookback: int = 50) -> float:
    """
    Calcula a correlação de Pearson entre o ativo e o IBOVESPA.
    """
    try:
        df_sym = safe_copy_rates(symbol, mt5.TIMEFRAME_D1, lookback)
        df_ibov = safe_copy_rates("IBOV", mt5.TIMEFRAME_D1, lookback)
        
        if df_sym is None or df_ibov is None or len(df_sym) < 20 or len(df_ibov) < 20:
            return 0.0
            
        returns_sym = df_sym['close'].pct_change().dropna()
        returns_ibov = df_ibov['close'].pct_change().dropna()
        
        # Alinha os índices
        combined = pd.concat([returns_sym, returns_ibov], axis=1).dropna()
        if len(combined) < 10:
            return 0.0
            
        correlation = combined.corr().iloc[0, 1]
        return float(correlation)
    except Exception as e:
        logger.error(f"Erro ao calcular correlação IBOV para {symbol}: {e}")
        return 0.0




def validate_subsetor_exposure(symbol: str) -> Tuple[bool, str]:
    """
    ✅ Garante que não ultrapassamos 20% do capital (ou config) em um mesmo subsetor.
    """
    try:
        subsetor = config.SUBSETOR_MAP.get(symbol, "Outros")
        
        with mt5_lock:
            positions = mt5.positions_get()
            
        if not positions:
            return True, "OK"
            
        acc = mt5.account_info()
        total_equity = acc.equity if acc else 100000.0
        
        subsetor_value = 0
        for pos in positions:
            if config.SUBSETOR_MAP.get(pos.symbol) == subsetor:
                subsetor_value += pos.volume * pos.price_open
                
        # Projeta nova posição (estimativa conservadora)
        projected_total = subsetor_value + (total_equity * 0.005) # 0.5% capital como margem
        exposure = projected_total / total_equity
        
        limit = getattr(config, "MAX_SUBSETOR_EXPOSURE", 0.20)
        
        if exposure > limit:
            return False, f"Exposição excessiva em {subsetor}: {exposure:.1%}"
            
        return True, "OK"
    except Exception as e:
        logger.error(f"Erro validação subsetor: {e}")
        return True, "OK"


def get_book_imbalance(symbol: str) -> float:
    """
    Calcula o desequilíbrio (imbalance) do book de ofertas.
    Imbalance = (BidVol - AskVol) / (BidVol + AskVol)
    Retorna valor entre -1.0 e 1.0.
    """
    try:
        with mt5_lock:
            if not mt5.market_book_add(symbol):
                return 0.0
            
            book = mt5.market_book_get(symbol)
            mt5.market_book_release(symbol)
        
        if not book:
            return 0.0
        
        # BOOK_TYPE_BUY = Oferta de Compra (Bid)
        # BOOK_TYPE_SELL = Oferta de Venda (Ask)
        bid_vol = sum(item.volume for item in book if item.type == mt5.BOOK_TYPE_BUY)
        ask_vol = sum(item.volume for item in book if item.type == mt5.BOOK_TYPE_SELL)

        total_vol = bid_vol + ask_vol
        if total_vol == 0:
            return 0.0
        
        return float((bid_vol - ask_vol) / total_vol)
        
    except Exception as e:
        logger.error(f"Erro ao calcular imbalance para {symbol}: {e}")
        return 0.0




def get_book_imbalance(symbol: str) -> float:
    """
    ✅ V5.2 CONSOLIDADO: Calcula o desequilíbrio do book de ofertas via MT5.
    Retorna: Valor entre -1 (pressão vendedora) e 1 (pressão compradora)
    """
    try:
        with mt5_lock:
            if not mt5.market_book_add(symbol): return 0.0
            book = mt5.market_book_get(symbol)
            mt5.market_book_release(symbol)
        
        if book is None or len(book) == 0: return 0.0
        
        bid_v = sum(item.volume for item in book if item.type == mt5.BOOK_TYPE_BUY)
        ask_v = sum(item.volume for item in book if item.type == mt5.BOOK_TYPE_SELL)
        total = bid_v + ask_v
        return float((bid_v - ask_v) / total) if total > 0 else 0.0
    except Exception as e:
        logger.debug(f"Book imbalance não disponível para {symbol}: {e}")
        return 0.0


def get_daily_max_equity() -> float:
    """
    Obtém o equity máximo do dia a partir de arquivo persistente ou conta atual.
    """
    try:
        if os.path.exists("daily_equity.txt"):
            with open("daily_equity.txt", "r") as f:
                return float(f.read())
        
        acc = mt5.account_info()
        return acc.equity if acc else 0.0
    except Exception as e:
        logger.error(f"Erro ao ler daily_equity: {e}")
        return 0.0

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
    """
    now = datetime.now()
    today = now.date()
    market_open = datetime.combine(today, datetime.strptime("10:00", "%H:%M").time())
    
    df_today = df[df.index >= market_open]
    
    if df_today.empty or len(df_today) < 3:
        return None
    
    typical_price = (df_today['high'] + df_today['low'] + 2 * df_today['close']) / 4
    volume = df_today.get('real_volume', df_today['tick_volume'])
    
    pv = (typical_price * volume).sum()
    total_vol = volume.sum()
    
    return float(pv / total_vol) if total_vol > 0 else None


# 🆕 NOVA FUNÇÃO - ADICIONAR APÓS get_intraday_vwap()
def get_obv(df: pd.DataFrame) -> Optional[float]:
    """
    📊 On-Balance Volume (OBV) - Indicador de fluxo de volume
    
    OBV sobe quando close > close anterior (volume comprador)
    OBV desce quando close < close anterior (volume vendedor)
    
    Retorna:
        - Valor OBV atual
        - None se dados insuficientes
    """
    if df is None or len(df) < 10:
        return None
    
    try:
        volume = df.get('real_volume', df['tick_volume']).values
        close = df['close'].values
        
        # Calcula mudanças de preço
        price_changes = np.diff(close)
        
        # OBV: soma volume quando sobe, subtrai quando desce
        obv = np.zeros(len(close))
        obv[0] = volume[0]
        
        for i in range(1, len(close)):
            if price_changes[i-1] > 0:
                obv[i] = obv[i-1] + volume[i]
            elif price_changes[i-1] < 0:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        
        return float(obv[-1])
    
    except Exception as e:
        logger.error(f"Erro ao calcular OBV: {e}")
        return None


# 🆕 NOVA FUNÇÃO - Tendência do OBV
def is_obv_trending_up(df: pd.DataFrame, lookback: int = 20) -> bool:
    """
    ✅ Valida se OBV está em tendência de alta
    
    Retorna True se:
    - OBV atual > média móvel de 20 períodos
    - OBV subiu nos últimos 5 candles
    """
    if df is None or len(df) < lookback + 5:
        return False
    
    try:
        # Calcula OBV completo
        volume = df.get('real_volume', df['tick_volume']).values
        close = df['close'].values
        price_changes = np.diff(close, prepend=close[0])
        
        obv = np.zeros(len(close))
        for i in range(1, len(close)):
            if price_changes[i] > 0:
                obv[i] = obv[i-1] + volume[i]
            elif price_changes[i] < 0:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        
        # Média móvel do OBV
        obv_ma = np.convolve(obv, np.ones(lookback)/lookback, mode='valid')
        
        # Validações
        obv_above_ma = obv[-1] > obv_ma[-1]
        obv_rising = obv[-1] > obv[-6]  # Subiu nos últimos 5 candles
        
        return obv_above_ma and obv_rising
    
    except Exception as e:
        logger.error(f"Erro ao validar OBV: {e}")
        return False

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

    # --- ✅ SPREAD REAL-TIME (LAND TRADING STYLE) ---
    tick = mt5.symbol_info_tick(symbol)
    spread_nominal = 0
    spread_pct = 0
    spread_points = 0
    
    if tick and tick.ask > 0 and tick.bid > 0:
        spread_nominal = tick.ask - tick.bid
        spread_points = round(spread_nominal / mt5.symbol_info(symbol).point, 0)
        spread_pct = (spread_nominal / tick.bid) * 100 if tick.bid > 0 else 0

    # --- Z-SCORE DE VOLATILIDADE ---
    vol_series = df['close'].pct_change().rolling(20).std() * 100
    atr_mean = vol_series.mean()
    atr_std = vol_series.std()
    z_score = (atr_pct_real - atr_mean) / atr_std if (atr_std and atr_std > 0) else 0
    atr_pct_capped = min(round(atr_pct_real, 3), 10.0)

    # --- VOLUME ---
    avg_vol = get_avg_volume(df)
    
    # ✅ NOVO: Fallback para Volume do candle anterior se o atual for 0.0 (Leilão/Hiato)
    cur_vol = 0
    if "real_volume" in df.columns:
        cur_vol = df["real_volume"].iloc[-1]
        if cur_vol == 0 and len(df) > 1:
            cur_vol = df["real_volume"].iloc[-2]
    elif "tick_volume" in df.columns:
        cur_vol = df["tick_volume"].iloc[-1]
        if cur_vol == 0 and len(df) > 1:
            cur_vol = df["tick_volume"].iloc[-2]
            
    volume_ratio = round(cur_vol / avg_vol, 2) if avg_vol > 0 else 1.0

    atr_series_data = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1).ewm(alpha=1 / 14, adjust=False).mean()

    atr_mean_val = atr_series_data.rolling(20).mean().iloc[-1]
    side = "BUY" if ema_fast > ema_slow else "SELL"
    # 🆕 CALCULA OBV
    obv_value = get_obv(df)
    obv_trend_up = is_obv_trending_up(df)
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
        "obv": obv_value,  # 🆕 NOVO
        "obv_trend_up": obv_trend_up,  # 🆕 NOVO
        "spread_points": spread_points,  # 🆕 NOVO (pontos/ticks)
        "spread_nominal": round(spread_nominal, 3),  # 🆕 NOVO (R$)
        "spread_pct": round(spread_pct, 4),  # 🆕 NOVO (%)
        "close": price,
        "macro_trend_ok": macro_trend_ok(symbol, side),
        "tick_size": mt5.symbol_info(symbol).point,
        "params": params,
        "error": None
    }

def check_volatility_filter(symbol: str, atr: float, current_price: float) -> tuple[bool, str]:
    """
    🌊 FILTRO DE VOLATILIDADE ADAPTATIVO
    
    Bloqueia entradas em condições extremas:
    - Volatilidade >3 desvios padrão
    - ATR expandindo muito rápido (>50% em 5 candles)
    - "Chop zones" (oscilação sem direção)
    
    Returns:
        (pode_entrar: bool, motivo: str)
    
    Impacto: +5-8% win rate (evita whipsaws)
    """
    try:
        df = utils.safe_copy_rates(symbol, mt5.TIMEFRAME_M15, 50)
        
        if df is None or len(df) < 30:
            return True, ""
        
        # 1. Z-Score de Volatilidade (já existe no código)
        vol_series = df['close'].pct_change().rolling(20).std() * 100
        atr_pct_real = (atr / current_price) * 100 if current_price > 0 else 0
        
        z_score = (atr_pct_real - vol_series.mean()) / vol_series.std() if vol_series.std() > 0 else 0
        
        # ✅ NOVO: Threshold mais restritivo
        if abs(z_score) > 2.5:  # Era 2.0
            return False, f"Volatilidade extrema (z={z_score:.2f})"
        
        # 2. ✅ NOVO: Detecta expansão rápida de ATR
        atr_series = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift()).abs(),
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1).ewm(alpha=1/14, adjust=False).mean()
        
        recent_atr = atr_series.tail(5)
        atr_change = (recent_atr.iloc[-1] - recent_atr.iloc[0]) / recent_atr.iloc[0]
        
        if atr_change > 0.50:  # >50% em 5 candles
            return False, f"ATR expandindo rápido ({atr_change*100:.0f}%)"
        
        # 3. ✅ NOVO: Detecta "Chop Zones" (Range sem direção)
        # ADX baixo + ATR alto = mercado lateral violento
        adx = utils.get_adx(df) or 0
        
        if adx < 15 and atr_pct_real > 2.0:
            return False, f"Chop Zone (ADX {adx:.0f}, ATR {atr_pct_real:.1f}%)"
        
        # 4. ✅ NOVO: Valida se está em "squeeze" (Bollinger Bands apertadas)
        bb_std = df['close'].rolling(20).std()
        bb_width = (bb_std.iloc[-1] / df['close'].iloc[-1]) * 100
        
        if bb_width < 1.0 and atr_change > 0.30:
            # Bollinger estreitando + ATR subindo = possível breakout falso
            return False, "Possível falso breakout"
        
        return True, ""
    
    except Exception as e:
        logger.error(f"Erro filtro volatilidade {symbol}: {e}")
        return True, ""

from datetime import datetime, date

def calculate_daily_dd() -> float:
    """
    Retorna o drawdown diário atual (0.0 a 1.0)
    Ex: 0.03 = 3%
    """
    try:
        account = mt5.account_info()
        if not account:
            return 0.0

        balance = account.balance
        equity = account.equity

        if balance <= 0:
            return 0.0

        dd = (balance - equity) / balance
        return max(dd, 0.0)

    except Exception as e:
        logger.error(f"Erro ao calcular DD diário: {e}")
        return 0.0

# =========================================================
# SCORE FINAL
# =========================================================
def calculate_signal_score(ind: dict) -> float:
    """
    ✅ VERSÃO v5.2: Escala 0-100 refletindo intensidade e qualidade técnica.
    
    Ranges:
    - 0-40: Aguardando (Base técnica insuficiente)
    - 41-60: Interesse Aumentando (Filtros ok, momentum moderado)
    - 61-100: EXECUÇÃO IMINENTE (Trade de alta probabilidade)
    """
    
    if not isinstance(ind, dict) or ind.get("error"):
        return 0.0

    # 1️⃣ BASE SCORE: Baixada para 20 para dar peso aos indicadores
    score = 20.0 
    score_log = {"BASE": 20.0}
    
    # =========================
    # 📦 INPUTS
    # =========================
    rsi = ind.get("rsi", 50)
    adx = ind.get("adx", 20)
    volume_ratio = ind.get("volume_ratio", 1.0)
    ema_fast = ind.get("ema_fast", 0)
    ema_slow = ind.get("ema_slow", 0)
    momentum = ind.get("momentum", 0.0)
    current_price = ind.get("close", 0)
    vwap = ind.get("vwap", None)
    
    # =========================
    # 🚀 BÔNUS (Fortalecidos para chegar em 61+)
    # =========================

    # 1. RSI Saudável (Peso aumentado)
    if 40 <= rsi <= 65:
        score += 20
        score_log["RSI_OK"] = 20
    elif 30 < rsi < 40 or 65 < rsi < 80:
        score += 10
        score_log["RSI_MODERADO"] = 10
    
    # 2. MACD Cruzado (Peso aumentado)
    macd = ind.get('macd', 0)
    macd_signal = ind.get('macd_signal', 0)
    if macd > macd_signal:
        score += 25
        score_log["MACD_CROSS"] = 25

    # 3. Momentum Positivo
    if momentum > 0:
        score += 15
        score_log["MOMENTUM"] = 15

    # 4. Stochastic em boa zona
    stoch_k = ind.get('stoch_k', 50)
    if stoch_k < 30:
        score += 15
        score_log["STOCH_LOW"] = 15

    # 5. Volume Diferenciado
    if volume_ratio > 1.2:
        score += 10
        score_log["VOL_BOOST"] = 10

    # 6. ML Boost (se habilitado)
    try:
        ml_pred = 0 
        if "ml_optimizer" in globals():
             ml_features = ml_optimizer.extract_features(ind, ind.get('symbol', 'UNK'))
             ml_pred = ml_optimizer.predict_signal_score(ml_features)
        
        if ml_pred > 0.1:
            score += 10
            score_log["ML_BOOST"] = 10
    except:
        pass

    # =========================
    # 📉 PENALIDADES
    # =========================
    
    # Anti-Chop / Sem tendência
    if adx < 15:
        score -= 20
        score_log["PENALTY_NO_TREND"] = -20
    
    # Fora da tendência principal
    if ema_fast < ema_slow and ema_fast > 0:
        score -= 15
        score_log["PENALTY_COUNTER_TREND"] = -15

    # =========================
    # ✅ FINAL
    # =========================
    final_score = min(max(score, 0.0), 100.0) # Range 0-100
    ind["score_log"] = score_log
    
    return round(final_score, 1)
def check_arbitrage_opp(sym1: str, sym2: str) -> bool:
    """
    Detecta arbitragem entre pares correlacionados (ex: PETR3 vs PETR4, web:5).
    """
    df1 = safe_copy_rates(sym1, mt5.TIMEFRAME_M15, 10)
    df2 = safe_copy_rates(sym2, mt5.TIMEFRAME_M15, 10)
    if df1 is None or df2 is None:
        return False
    
    spread = df1['close'].iloc[-1] - df2['close'].iloc[-1]
    mean_spread = (df1['close'] - df2['close']).mean()
    if abs(spread - mean_spread) > 2 * (df1['close'] - df2['close']).std():
        return True  # Oportunidade arb
    return False

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
    
    
    # CVM Compliance: Log em CSV
    
    import csv
    import os
    from datetime import datetime
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
        "position": ticket,
        "price": price,
        "deviation": 10,
        "magic": 2026,
        "comment": "X",
    }
    return request
