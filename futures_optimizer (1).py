"""
OTIMIZADOR DEDICADO PARA ÍNDICES FUTUROS B3
============================================
✅ WIN$ (Mini Índice), WDO$ (Mini Dólar), SMALL$ (Small Caps)
✅ Gestão de Margem e Alavancagem
✅ Custos Reais (Taxa + Slippage)
✅ Filtros de Volatilidade e Liquidez
✅ WFO + Optuna + Monte Carlo
"""

import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import functools
print = functools.partial(print, flush=True)

import json
import time
import logging
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import utils

# ===========================
# CONFIGURAÇÕES DE FUTUROS
# ===========================

@dataclass
class FuturesConfig:
    """Especificações técnicas dos contratos futuros"""
    
    # Contratos disponíveis
    SYMBOLS = ["WING26", "WDOG26", "WSPH26", "BGIG26"]
    
    # Especificações por contrato
    SPECS = {
        "WING26": {  # <--- Nome exato do contrato
        "name": "Mini Índice (Fev/26)",
        "point_value": 0.20,
        "tick_size": 5.0,
        "margin_required": 3500,
        "fee_per_contract": 0.25,
        "slippage_points": 10,
        "min_volume": 50000,
        "session_start": "09:00",
        "session_end": "17:55"
    },
    "WDOG26": {  # <--- Nome exato do contrato
        "name": "Mini Dólar (Fev/26)",
        "point_value": 10.0,
        "tick_size": 0.5,
        "margin_required": 2500,
        "fee_per_contract": 1.10,
        "slippage_points": 1.0,
        "min_volume": 30000,
        "session_start": "09:00",
        "session_end": "17:55"
    },
    "WSPH26": {  # <--- Nome exato do contrato (Março)
        "name": "Micro S&P 500 (Mar/26)",
        "point_value": 14.50, # USD 2.50 * R$ 5.80
        "tick_size": 0.25,
        "margin_required": 3000,
        "fee_per_contract": 0.80,
        "slippage_points": 0.50,
        "min_volume": 1000,
        "session_start": "09:00",
        "session_end": "17:55"
    },
    "BGIG26": {  # Símbolo adicional solicitado para verificação
        "name": "BGI (Fev/26)",
        "point_value": 1.0,
        "tick_size": 1.0,
        "margin_required": 3000,
        "fee_per_contract": 1.00,
        "slippage_points": 1.0,
        "min_volume": 1000,
        "session_start": "09:00",
        "session_end": "17:55"
    }
}
    
    # Parâmetros de risco
    MAX_LEVERAGE = 3.0          # Máximo 3x margem
    RISK_PER_TRADE = 0.02       # 2% do capital por trade
    MAX_DD_ALLOWED = 0.35       # 35% drawdown máximo
    MIN_SHARPE = 0.8            # Sharpe mínimo aceitável
    
    # Timeframes para otimização
    TIMEFRAMES = ["M5", "M15", "H1"]  # 5min, 15min, 1hora
    BARS_PER_DAY = {
        "M5": 96,   # ~8h sessão
        "M15": 32,
        "H1": 8
    }

config_fut = FuturesConfig()

# ===========================
# LOGGING
# ===========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("futures_optimizer")

OUTPUT_DIR = "futures_optimizer_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
ACTIVE_FUTURES: Dict[str, str] = {}
 
class TradeValidationManager:
    def __init__(self, required: int = 10):
        self.required = int(required)
        self.counts: Dict[str, int] = {}
        self.logs: Dict[str, List[Dict[str, Any]]] = {}
        self.unlocked: Dict[str, bool] = {}
    def record(self, symbol: str, trade: Dict[str, Any]):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = "WIN" if float(trade.get("pnl", 0.0)) > 0 else "LOSS"
        entry = {
            "timestamp": ts,
            "type": str(trade.get("type", "")),
            "exit_reason": str(trade.get("exit_reason", "")),
            "entry": float(trade.get("entry", 0.0)),
            "exit": float(trade.get("exit", 0.0)),
            "pnl": float(trade.get("pnl", 0.0)),
            "status": status
        }
        self.logs.setdefault(symbol, []).append(entry)
        self.counts[symbol] = self.counts.get(symbol, 0) + 1
        if self.counts[symbol] >= self.required and not self.unlocked.get(symbol, False):
            self.unlocked[symbol] = True
            logger.info(f"🔓 {symbol}: validação concluída ({self.counts[symbol]}/{self.required})")
    def get_status(self, symbol: str) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "count": int(self.counts.get(symbol, 0)),
            "required": self.required,
            "unlocked": bool(self.unlocked.get(symbol, False)),
            "logs": list(self.logs.get(symbol, []))
        }
    def is_unlocked(self, symbol: str) -> bool:
        return bool(self.unlocked.get(symbol, False))
    def summary(self) -> Dict[str, Any]:
        return {s: self.get_status(s) for s in sorted(self.counts.keys())}
 
VALIDATION_MANAGER = TradeValidationManager(required=10)

# ===========================
# CONEXÃO MT5 PARA FUTUROS
# ===========================

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except:
    MT5_AVAILABLE = False
    logger.warning("⚠️ MT5 não disponível - modo simulação")

def _detect_broker() -> str:
    if not MT5_AVAILABLE:
        return ""
    try:
        info = mt5.terminal_info()
        return str(getattr(info, "server", "") or "")
    except Exception:
        return ""

def _month_code_to_month(c: str) -> int:
    m = {
        "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
        "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12
    }
    return m.get(c.upper(), 0)

def _symbol_score(name: str) -> int:
    score = 0
    try:
        info = mt5.symbol_info(name)
        if info and getattr(info, "visible", False):
            score += 2
        if info and getattr(info, "selectable", False):
            score += 1
    except Exception:
        pass
    if "$" in name:
        score += 3
    return score

def _estimate_expiration_distance_days(name: str) -> int:
    import re
    try:
        m = re.search(r"([A-Z])(\d{2})$", name)
        if not m:
            return 9999
        mc = m.group(1)
        yy = int(m.group(2))
        mm = _month_code_to_month(mc)
        if mm == 0:
            return 9999
        from datetime import date
        year_full = 2000 + yy
        exp_date = date(year_full, mm, 28)
        today = date.today()
        d = (exp_date - today).days
        if d < 0:
            return 9999
        return d
    except Exception:
        return 9999

def _liquidity_score(name: str) -> float:
    if not MT5_AVAILABLE:
        return 0.0
    try:
        mt5.symbol_select(name, True)
        rates = mt5.copy_rates_from_pos(name, mt5.TIMEFRAME_M15, 0, 200)
        if rates is None or len(rates) == 0:
            return 0.0
        df = pd.DataFrame(rates)
        vol = float(df.get("tick_volume", pd.Series(dtype=float)).tail(100).sum() or 0.0)
        return vol
    except Exception:
        return 0.0

def get_current_futures_candidates(base_code: str) -> List[str]:
    if not MT5_AVAILABLE:
        return []
    candidates = []
    seen = set()
    masks = [f"{base_code}*", f"{base_code}$*", f"{base_code}@*"]
    broker = _detect_broker().lower()
    if "xp" in broker:
        masks += [f"{base_code}N*", f"{base_code}Z*"]
    if "btg" in broker or "clear" in broker:
        masks += [f"{base_code}?*"]
    for mask in masks:
        try:
            syms = mt5.symbols_get(mask) or []
            for s in syms:
                name = getattr(s, "name", "")
                if name and name not in seen:
                    seen.add(name)
                    candidates.append(name)
        except Exception:
            pass
    import re
    today = datetime.now()
    filtered = []
    for n in candidates:
        if not n.startswith(base_code):
            continue
        info = mt5.symbol_info(n)
        selectable = bool(info and getattr(info, "selectable", False))
        exp = getattr(info, "expiration_time", None)
        is_contract = bool(re.search(rf"^{base_code}[FGHJKMNQUVXZ]\d{{2}}$", n))
        if not is_contract:
            continue
        if not selectable:
            continue
        if not isinstance(exp, datetime) or exp <= today:
            continue
        filtered.append(n)
    filtered.sort(key=lambda n: (_estimate_expiration_distance_days(n), -_liquidity_score(n), -_symbol_score(n)))
    return filtered

def get_current_futures_symbol(base_code: str) -> Optional[str]:
    sym = utils.resolve_current_symbol(base_code)
    if sym:
        return sym
    alt_bases = {
        "WIN": ["WIN", "IND"],
        "WDO": ["WDO", "DOL"],
        "SMLL": ["SMLL", "SMAL", "SMALL"],
        "WSP": ["WSP", "SP"]
    }
    bases = alt_bases.get(base_code, [base_code])
    cands = []
    for b in bases:
        cands = get_current_futures_candidates(b)
        if cands:
            break
    if not cands:
        return None
    return cands[0]

def discover_active_futures() -> Dict[str, str]:
    mapping = {}
    return mapping

def ensure_mt5_futures():
    """Garante conexão MT5 e adiciona futuros ao Market Watch"""
    if not MT5_AVAILABLE:
        return False
    
    try:
        init_ok = False
        try:
            _res = mt5.initialize()
            init_ok = True
        except Exception:
            init_ok = False
        try:
            term = mt5.terminal_info()
            if term and getattr(term, "connected", False):
                init_ok = True
        except Exception:
            pass
        if not init_ok:
            logger.error("❌ Falha ao inicializar/conectar MT5")
            return False
        for symbol in config_fut.SYMBOLS:
            target = symbol
            if not mt5.symbol_select(target, True):
                logger.warning(f"⚠️ Não foi possível adicionar {symbol}")
            else:
                logger.info(f"✅ {target} adicionado ao Market Watch")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro MT5: {e}")
        return False

# ===========================
# CARREGAMENTO DE DADOS
# ===========================

def load_futures_data(symbol: str, bars: int = 5000, timeframe: str = "M15") -> Optional[pd.DataFrame]:
    """Carrega dados históricos de futuros"""
    
    if not MT5_AVAILABLE:
        logger.warning(f"📊 Gerando dados sintéticos para {symbol}")
        return generate_synthetic_futures_data(symbol, bars)
    
    try:
        # Mapeia timeframe para MT5
        tf_map = {
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "H1": mt5.TIMEFRAME_H1
        }
        tf = tf_map.get(timeframe, mt5.TIMEFRAME_M15)
        import re
        direct_contract = bool(re.search(r"^(WIN|WDO|WSP|SMLL|SMAL)[FGHJKMNQUVXZ]\d{2}$", symbol))
        mapped = ACTIVE_FUTURES.get(symbol, None)
        if direct_contract:
            target_list = [symbol]
        elif mapped:
            target_list = [mapped]
        else:
            base = symbol.split("$")[0]
            target_list = get_current_futures_candidates(base)
            if not target_list:
                logger.warning(f"⚠️ Símbolo {symbol} não encontrado - usando dados sintéticos")
                return generate_synthetic_futures_data(symbol, bars)
        for target in target_list:
            if not mt5.symbol_select(target, True):
                continue
            rates = mt5.copy_rates_from_pos(target, tf, 0, bars)
            if rates is None or len(rates) == 0:
                continue
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            df = df.rename(columns={'tick_volume': 'volume'})
            df = df[['open', 'high', 'low', 'close', 'volume']]
            logger.info(f"✅ {target}: {len(df)} barras carregadas ({timeframe})")
            return df
        logger.warning(f"⚠️ Falha ao carregar histórico MT5 para {symbol} - usando dados sintéticos")
        return generate_synthetic_futures_data(symbol, bars)
        
    except Exception as e:
        logger.error(f"❌ Erro ao carregar {symbol}: {e}")
        logger.warning(f"⚠️ Fallback sintético ativado para {symbol}")
        return generate_synthetic_futures_data(symbol, bars)

def generate_synthetic_futures_data(symbol: str, bars: int) -> pd.DataFrame:
    """Gera dados sintéticos para testes"""
    
    def _root_for(symbol: str) -> str:
        import re
        if re.match(r"^WIN", symbol):
            return "WIN$"
        if re.match(r"^WDO", symbol):
            return "WDO$"
        if re.match(r"^(SMLL|SMAL|SMALL)", symbol):
            return "SMALL$"
        if re.match(r"^WSP|^SP", symbol):
            return "WSP$"
        return symbol
    root = _root_for(symbol)
    spec = config_fut.SPECS.get(symbol) or config_fut.SPECS.get(root) or config_fut.SPECS.get("WIN$", {
        "point_value": 0.20,
        "tick_size": 5.0,
        "margin_required": 3500,
        "fee_per_contract": 0.80,
        "slippage_points": 10,
        "min_volume": 50000
    })
    
    # Parâmetros baseados no ativo
    if root == "WIN$":
        base_price = 125000
        volatility = 0.015
    elif root == "WDO$":
        base_price = 5500
        volatility = 0.012
    elif root == "WSP$":
        base_price = 5000
        volatility = 0.010
    else:  # SMALL$
        base_price = 2500
        volatility = 0.018
    
    # Gera série de preços
    returns = np.random.normal(0, volatility, bars)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Cria OHLCV
    df = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.003, bars))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.003, bars))),
        'close': prices * (1 + np.random.normal(0, 0.001, bars)),
        'volume': np.random.randint(
            spec['min_volume'] // 2, 
            spec['min_volume'] * 2, 
            bars
        )
    })
    
    # Arredonda para tick size
    tick = spec['tick_size']
    for col in ['open', 'high', 'low', 'close']:
        df[col] = (df[col] / tick).round() * tick
    
    # Adiciona timestamp
    df.index = pd.date_range(end=datetime.now(), periods=bars, freq='15min')
    
    return df

# ===========================
# DIAGNÓSTICOS E FORÇA DE EXECUÇÃO
# ===========================
def _is_generic_result(m: Dict[str, Any]) -> bool:
    try:
        tt = int(m.get("total_trades", 0))
        wr = float(m.get("win_rate", 0.0))
        dd = float(m.get("max_dd", 1.0))
        return tt <= 3 or (wr >= 0.99 and dd <= 0.0001)
    except Exception:
        return True

def deep_trade_diagnostics(symbol: str, params: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    res = backtest_futures(symbol, params, df)
    trades = res.get("trades", [])
    wins = [t for t in trades if t.get("pnl", 0.0) > 0]
    losses = [t for t in trades if t.get("pnl", 0.0) <= 0]
    type_counts = {}
    exit_counts = {}
    for t in trades:
        type_counts[t.get("type","")] = type_counts.get(t.get("type",""), 0) + 1
        exit_counts[t.get("exit_reason","")] = exit_counts.get(t.get("exit_reason",""), 0) + 1
    return {
        "symbol": symbol,
        "total_trades": len(trades),
        "win_rate": res.get("win_rate", 0.0),
        "avg_pnl_win": float(np.mean([t.get("pnl",0.0) for t in wins])) if wins else 0.0,
        "avg_pnl_loss": float(np.mean([t.get("pnl",0.0) for t in losses])) if losses else 0.0,
        "type_counts": type_counts,
        "exit_counts": exit_counts
    }

def force_execution_for_symbol(symbol: str) -> Optional[List[Dict[str, Any]]]:
    df = load_futures_data(symbol, bars=15000, timeframe="M15")
    if df is None or len(df) < 500:
        df = load_futures_data(symbol, bars=15000, timeframe="M5")
    if df is None or len(df) < 500:
        return None
    results = []
    res_wfo = walk_forward_futures(symbol, bars=12000, windows=8)
    if res_wfo:
        results = res_wfo
    return results
# ===========================
# INDICADORES TÉCNICOS
# ===========================

def calculate_indicators(df: pd.DataFrame, symbol: str) -> dict:
    """Calcula indicadores otimizados para futuros"""
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    
    # EMAs adaptativas
    ema_fast = pd.Series(close).ewm(span=8).mean().values
    ema_slow = pd.Series(close).ewm(span=21).mean().values
    
    # ATR para volatilidade
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum.reduce([tr1, tr2, tr3])
    atr = pd.Series(tr).ewm(span=14).mean().values
    
    # ADX para força de tendência
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14).mean().values / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14).mean().values
    
    # Volume Profile (simplificado)
    vol_ma = pd.Series(volume).rolling(20).mean().values
    vol_ratio = volume / (vol_ma + 1)
    
    # RSI
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).values

    # MOM (Momentum percentual 20 períodos)
    mom_20 = pd.Series(close).pct_change(20).values
    
    try:
        if not np.isfinite(ema_fast).all() or not np.isfinite(ema_slow).all():
            logger.warning(f"⚠️ EMA inválida para {symbol}")
        if not np.isfinite(adx).all():
            logger.warning(f"⚠️ ADX inválido para {symbol}")
        if not np.isfinite(vol_ratio).all():
            logger.warning(f"⚠️ VOL inválido para {symbol}")
        if not np.isfinite(mom_20).all():
            logger.warning(f"⚠️ MOM inválido para {symbol}")
    except Exception:
        pass
    
    return {
        'close': close,
        'high': high,
        'low': low,
        'volume': volume,
        'ema_fast': ema_fast,
        'ema_slow': ema_slow,
        'atr': atr,
        'adx': adx,
        'rsi': rsi,
        'vol_ma': vol_ma,
        'vol_ratio': vol_ratio,
        'mom_20': mom_20
    }

# ===========================
# BACKTEST ENGINE PARA FUTUROS
# ===========================

def backtest_futures(symbol: str, params: dict, df: pd.DataFrame) -> dict:
    """
    Backtest especializado para futuros com gestão de margem
    """
    
    if df is None or len(df) < 100:
        return {
            "total_return": -1.0,
            "sharpe": 0.0,
            "max_dd": 1.0,
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "ema_fast_last": 0.0,
            "ema_slow_last": 0.0,
            "adx_last": 0.0,
            "rsi_last": 0.0,
            "vol_ma_last": 0.0,
            "mom_last": 0.0,
            "vol_ratio_mean": 0.0,
            "mom_mean": 0.0
        }
    
    import re
    def _root_for(sym: str) -> str:
        if re.match(r"^WIN", sym):
            return "WIN$"
        if re.match(r"^WDO", sym):
            return "WDO$"
        if re.match(r"^(SMLL|SMAL|SMALL)", sym):
            return "SMALL$"
        if re.match(r"^(WSP|SP)", sym):
            return "WSP$"
        return sym
    root = _root_for(symbol)
    spec = config_fut.SPECS.get(symbol) or config_fut.SPECS.get(root) or config_fut.SPECS["WIN$"]
    ind = calculate_indicators(df, symbol)
    
    # Estado inicial
    capital = 100000.0
    equity = capital
    contracts = 0
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    
    trades = []
    equity_curve = [capital]
    
    # Parâmetros
    ema_period_fast = params.get('ema_fast', 8)
    ema_period_slow = params.get('ema_slow', 21)
    adx_threshold = params.get('adx_threshold', 20)
    rsi_low = params.get('rsi_low', 30)
    rsi_high = params.get('rsi_high', 70)
    sl_atr_mult = params.get('sl_atr_mult', 2.5)
    tp_atr_mult = params.get('tp_atr_mult', 4.0)
    
    n = len(df)
    # EMAs dinâmicas conforme parâmetros
    ema_fast_dyn = pd.Series(ind['close']).ewm(span=max(2, int(ema_period_fast))).mean().values
    ema_slow_dyn = pd.Series(ind['close']).ewm(span=max(3, int(ema_period_slow))).mean().values
    # Momentum e Volume
    mom_20 = ind.get('mom_20')
    vol_ma = ind.get('vol_ma')
    # Validações básicas
    def _safe_val(arr, idx, default=0.0):
        try:
            v = float(arr[idx])
            if not np.isfinite(v):
                return default
            return v
        except Exception:
            return default
    
    for i in range(50, n):  # Aguarda warmup dos indicadores
        
        price = ind['close'][i]
        atr_val = ind['atr'][i]
        
        # ==================
        # GESTÃO DE POSIÇÃO
        # ==================
        if contracts != 0:
            
            # Verifica stops
            if contracts > 0:  # Long
                hit_stop = ind['low'][i] <= stop_price
                hit_target = ind['high'][i] >= target_price
            else:  # Short
                hit_stop = ind['high'][i] >= stop_price
                hit_target = ind['low'][i] <= target_price
            
            # Executa saída
            if hit_stop or hit_target:
                exit_price = stop_price if hit_stop else target_price
                
                # Calcula P&L em pontos
                if contracts > 0:
                    points = exit_price - entry_price
                else:
                    points = entry_price - exit_price
                
                # Converte para R$
                pnl = points * spec['point_value'] * abs(contracts)
                
                # Subtrai custos
                costs = spec['fee_per_contract'] * abs(contracts) * 2  # Entrada + Saída
                net_pnl = pnl - costs
                
                equity += net_pnl
                
                trades.append({
                    'entry': entry_price,
                    'exit': exit_price,
                    'contracts': contracts,
                    'pnl': net_pnl,
                    'type': 'LONG' if contracts > 0 else 'SHORT',
                    'exit_reason': 'STOP' if hit_stop else 'TARGET'
                })
                
                contracts = 0
            
            else:
                # Trailing stop (usando PSAR simplificado)
                if contracts > 0 and price > entry_price + atr_val:
                    new_stop = price - (atr_val * sl_atr_mult)
                    if new_stop > stop_price:
                        stop_price = new_stop
        
        # ==================
        # SINAIS DE ENTRADA
        # ==================
        else:
            
            # Condições de entrada
            trend_up = ema_fast_dyn[i] > ema_slow_dyn[i]
            trend_dn = ema_fast_dyn[i] < ema_slow_dyn[i]
            
            strong_trend = ind['adx'][i] > adx_threshold
            oversold = ind['rsi'][i] < rsi_low
            overbought = ind['rsi'][i] > rsi_high
            
            high_volume = ind['vol_ratio'][i] > 1.2
            
            # Setup LONG
            signal_long = trend_up and oversold and strong_trend and high_volume
            
            # Setup SHORT
            signal_short = trend_dn and overbought and strong_trend and high_volume
            
            if signal_long or signal_short:
                
                # Define direção
                is_long = signal_long
                
                # Calcula posição baseada em risco
                risk_amount = equity * config_fut.RISK_PER_TRADE
                stop_distance_points = atr_val * sl_atr_mult
                
                # Quantos contratos cabem no risco?
                contracts_by_risk = risk_amount / (stop_distance_points * spec['point_value'])
                
                # Limita pela margem disponível
                max_contracts_by_margin = (equity * config_fut.MAX_LEVERAGE) / spec['margin_required']
                
                # Usa o menor
                num_contracts = int(min(contracts_by_risk, max_contracts_by_margin))
                
                if num_contracts >= 1:
                    
                    # Executa entrada
                    entry_price = price
                    
                    # Aplica slippage
                    slippage_cost = spec['slippage_points'] * spec['tick_size']
                    if is_long:
                        entry_price += slippage_cost
                        stop_price = entry_price - (stop_distance_points * spec['tick_size'])
                        target_price = entry_price + (atr_val * tp_atr_mult)
                        contracts = num_contracts
                    else:
                        entry_price -= slippage_cost
                        stop_price = entry_price + (stop_distance_points * spec['tick_size'])
                        target_price = entry_price - (atr_val * tp_atr_mult)
                        contracts = -num_contracts
                    
                    # Desconta taxa de entrada
                    equity -= spec['fee_per_contract'] * num_contracts
        
        equity_curve.append(equity)
    
    # ==================
    # MÉTRICAS
    # ==================
    
    if len(trades) == 0:
        return {
            "total_return": 0.0,
            "sharpe": 0.0,
            "max_dd": 0.0,
            "total_trades": 0,
            "win_rate": 0.0,
            "equity_curve": equity_curve,
            "ema_fast_last": _safe_val(ema_fast_dyn, n-1),
            "ema_slow_last": _safe_val(ema_slow_dyn, n-1),
            "adx_last": _safe_val(ind['adx'], n-1),
            "rsi_last": _safe_val(ind['rsi'], n-1),
            "vol_ma_last": _safe_val(vol_ma, n-1),
            "mom_last": _safe_val(mom_20, n-1),
            "profit_factor": 0.0,
            "vol_ratio_mean": float(np.nanmean(ind.get('vol_ratio', np.array([0.0])))),
            "mom_mean": float(np.nanmean(mom_20 if isinstance(mom_20, np.ndarray) else np.array([0.0])))
        }
    
    # Retorno total
    total_return = (equity - capital) / capital
    
    # Drawdown
    equity_arr = np.array(equity_curve)
    peak = np.maximum.accumulate(equity_arr)
    dd = (peak - equity_arr) / peak
    max_dd = float(np.max(dd))
    
    # Win Rate
    wins = sum(1 for t in trades if t['pnl'] > 0)
    win_rate = wins / len(trades)
    
    # Sharpe (simplificado)
    returns = np.diff(equity_arr) / equity_arr[:-1]
    sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
    
    # Profit Factor
    gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
    
    return {
        "total_return": float(total_return),
        "sharpe": float(sharpe),
        "max_dd": float(max_dd),
        "total_trades": len(trades),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "equity_curve": equity_curve,
        "trades": trades,
        "ema_fast_last": _safe_val(ema_fast_dyn, n-1),
        "ema_slow_last": _safe_val(ema_slow_dyn, n-1),
        "adx_last": _safe_val(ind['adx'], n-1),
        "rsi_last": _safe_val(ind['rsi'], n-1),
        "vol_ma_last": _safe_val(vol_ma, n-1),
        "mom_last": _safe_val(mom_20, n-1),
        "vol_ratio_mean": float(np.nanmean(ind.get('vol_ratio', np.array([0.0])))),
        "mom_mean": float(np.nanmean(mom_20 if isinstance(mom_20, np.ndarray) else np.array([0.0])))
    }

# ===========================
# OTIMIZAÇÃO COM OPTUNA
# ===========================

def optimize_futures(symbol: str, df: pd.DataFrame, n_trials: int = 200):
    """Otimiza parâmetros usando Optuna"""
    
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except:
        logger.error("❌ Optuna não disponível")
        return None
    
    def objective(trial):
        params = {
            'ema_fast': trial.suggest_int('ema_fast', 5, 15),
            'ema_slow': trial.suggest_int('ema_slow', 20, 50),
            'adx_threshold': trial.suggest_int('adx_threshold', 10, 30),
            'rsi_low': trial.suggest_int('rsi_low', 20, 40),
            'rsi_high': trial.suggest_int('rsi_high', 60, 80),
            'sl_atr_mult': trial.suggest_float('sl_atr_mult', 1.0, 3.5),
            'tp_atr_mult': trial.suggest_float('tp_atr_mult', 1.5, 4.0)
        }
        
        metrics = backtest_futures(symbol, params, df)

        if metrics['max_dd'] > config_fut.MAX_DD_ALLOWED:
            return -1.0
        
        trade_penalty = 0.0
        if metrics['total_trades'] < 20:
            trade_penalty = (20 - metrics['total_trades']) * 0.5
        
        score = metrics['sharpe'] + (metrics['win_rate'] * 1.5) - trade_penalty
        
        return score
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, timeout=600, show_progress_bar=True)
    try:
        best_params = study.best_params
        best_value = study.best_value
        train_metrics = backtest_futures(symbol, best_params, df)
        if train_metrics.get('total_trades', 0) < 10:
            logger.warning("⚠️ Resultado Estatisticamente Fraco (< 10 trades). Considere descartar.")
        return {'best_params': best_params, 'best_score': best_value}
    except Exception as e:
        logger.warning(f"⚠️ Optuna sem trials válidos para {symbol}: {e}")
        presets = [
            {'ema_fast': 8, 'ema_slow': 21, 'adx_threshold': 18, 'rsi_low': 35, 'rsi_high': 65, 'sl_atr_mult': 2.0, 'tp_atr_mult': 4.0},
            {'ema_fast': 5, 'ema_slow': 20, 'adx_threshold': 15, 'rsi_low': 25, 'rsi_high': 75, 'sl_atr_mult': 1.8, 'tp_atr_mult': 3.5},
            {'ema_fast': 12, 'ema_slow': 30, 'adx_threshold': 22, 'rsi_low': 30, 'rsi_high': 70, 'sl_atr_mult': 2.5, 'tp_atr_mult': 5.0}
        ]
        best = None
        best_score = -1e9
        for p in presets:
            m = backtest_futures(symbol, p, df)
            trade_penalty = 0.0
            if m['total_trades'] < 20:
                trade_penalty = (20 - m['total_trades']) * 0.5
            s = m['sharpe'] + (m['win_rate'] * 1.5) - trade_penalty
            if s > best_score:
                best = p
                best_score = s
        if best is None:
            return None
        return {'best_params': best, 'best_score': best_score}

# ===========================
# WALK-FORWARD OPTIMIZATION
# ===========================

def walk_forward_futures(symbol: str, bars: int = 10000, windows: int = 6):
    """WFO especializado para futuros"""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🎯 WALK-FORWARD: {symbol}")
    logger.info(f"{'='*60}\n")
    
    # Carrega dados
    df = load_futures_data(symbol, bars)
    if df is None:
        return None
    
    train_size = len(df) // (windows + 1)
    test_size = max(train_size // 2, 800)
    
    results = []
    
    for w in range(windows):
        logger.info(f"\n📊 Window {w+1}/{windows}")
        
        # Divide dados
        train_start = w * test_size
        train_end = train_start + train_size
        test_end = train_end + test_size
        
        if test_end > len(df):
            break
        
        df_train = df.iloc[train_start:train_end]
        df_test = df.iloc[train_end:test_end]
        
        # Otimiza no treino
        logger.info("🔧 Otimizando parâmetros...")
        opt_result = optimize_futures(symbol, df_train, n_trials=200)
        
        if opt_result is None:
            continue
        
        # Valida no teste
        logger.info("✅ Validando OOS...")
        test_metrics = backtest_futures(symbol, opt_result['best_params'], df_test)
        for t in test_metrics.get("trades", []):
            VALIDATION_MANAGER.record(symbol, t)
        if test_metrics.get('total_trades', 0) < 10:
            logger.warning("⚠️ Resultado Estatisticamente Fraco (< 10 trades). Considere descartar.")
        
        results.append({
            'window': w + 1,
            'params': opt_result['best_params'],
            'train_score': opt_result['best_score'],
            'test_metrics': test_metrics
        })
        
        logger.info(f"   Sharpe OOS: {test_metrics['sharpe']:.2f}")
        logger.info(f"   Win Rate: {test_metrics['win_rate']:.1%}")
        logger.info(f"   Trades: {test_metrics['total_trades']}")
    
    return results

# ===========================
# RELATÓRIO FINAL
# ===========================

def generate_report(all_results: dict):
    """Gera relatório consolidado"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file_md = os.path.join(OUTPUT_DIR, f"futures_report_{timestamp}.md")
    report_file_txt = os.path.join(OUTPUT_DIR, f"futures_report_{timestamp}.txt")
    
    with open(report_file_md, 'w', encoding='utf-8') as f:
        f.write("# 📊 RELATÓRIO DE OTIMIZAÇÃO - FUTUROS B3\n\n")
        f.write(f"**Gerado em:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        for symbol, results in all_results.items():
            spec = config_fut.SPECS.get(symbol, {"name": symbol})
            f.write(f"## 🎯 {symbol} - {spec.get('name', symbol)}\n\n")
            if results is None or len(results) == 0:
                st = VALIDATION_MANAGER.get_status(symbol)
                f.write("### ℹ️ Sem resultados disponíveis\n\n")
                f.write(f"- Validações: {st['count']}/{st['required']}\n")
                f.write(f"- Desbloqueado: {st['unlocked']}\n\n")
            else:
                best = max(results, key=lambda x: x['test_metrics']['sharpe'])
                f.write("### ✅ Melhores Parâmetros\n\n")
                f.write("```python\n")
                f.write(f"params_{symbol.replace('$', '')} = {{\n")
                for k, v in best['params'].items():
                    f.write(f"    '{k}': {v},\n")
                f.write("}\n```\n\n")
                f.write("### 📈 Métricas OOS\n\n")
                m = best['test_metrics']
                f.write(f"- **Sharpe Ratio:** {m['sharpe']:.2f}\n")
                f.write(f"- **Win Rate:** {m['win_rate']:.1%}\n")
                f.write(f"- **Total Trades:** {m['total_trades']}\n")
                f.write(f"- **Max Drawdown:** {m['max_dd']:.1%}\n")
                f.write(f"- **Profit Factor:** {m.get('profit_factor', 0.0):.2f}\n")
                f.write(f"- **EMA Fast (último):** {m.get('ema_fast_last', 0.0):.2f}\n")
                f.write(f"- **EMA Slow (último):** {m.get('ema_slow_last', 0.0):.2f}\n")
                f.write(f"- **ADX (último):** {m.get('adx_last', 0.0):.2f}\n")
                f.write(f"- **VOL Ratio (média):** {m.get('vol_ratio_mean', 0.0):.2f}\n")
                f.write(f"- **MOM (média):** {m.get('mom_mean', 0.0):.4f}\n\n")
            f.write("---\n\n")
    
    try:
        with open(report_file_txt, 'w', encoding='utf-8') as ftxt:
            ftxt.write("c:\\Users\\luizf\\Documents\\xp3v5\\config.py#L921-931\n")
            ftxt.write("RELATORIO DE OTIMIZACAO - FUTUROS B3\n")
            ftxt.write(f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
            ftxt.write("------------------------------------------------------------\n")
            for symbol, results in all_results.items():
                spec = config_fut.SPECS.get(symbol, {"name": symbol})
                ftxt.write(f"SIMBOLO: {symbol} - {spec.get('name', symbol)}\n")
                if results is None or len(results) == 0:
                    st = VALIDATION_MANAGER.get_status(symbol)
                    ftxt.write("SEM RESULTADOS DISPONIVEIS\n")
                    ftxt.write(f"  Validações: {st['count']}/{st['required']}\n")
                    ftxt.write(f"  Desbloqueado: {st['unlocked']}\n")
                else:
                    best = max(results, key=lambda x: x['test_metrics']['sharpe'])
                    ftxt.write("PARAMETROS:\n")
                    for k, v in best['params'].items():
                        ftxt.write(f"  {k}: {v}\n")
                    m = best['test_metrics']
                    ftxt.write("METRICAS OOS:\n")
                    ftxt.write(f"  Sharpe: {m['sharpe']:.2f}\n")
                    ftxt.write(f"  Win Rate: {m['win_rate']:.1%}\n")
                    ftxt.write(f"  Trades: {m['total_trades']}\n")
                    ftxt.write(f"  Max Drawdown: {m['max_dd']:.1%}\n")
                    ftxt.write(f"  Profit Factor: {m.get('profit_factor', 0.0):.2f}\n")
                    ftxt.write(f"  EMA Fast (último): {m.get('ema_fast_last', 0.0):.2f}\n")
                    ftxt.write(f"  EMA Slow (último): {m.get('ema_slow_last', 0.0):.2f}\n")
                    ftxt.write(f"  ADX (último): {m.get('adx_last', 0.0):.2f}\n")
                    ftxt.write(f"  VOL Ratio (média): {m.get('vol_ratio_mean', 0.0):.2f}\n")
                    ftxt.write(f"  MOM (média): {m.get('mom_mean', 0.0):.4f}\n")
                ftxt.write("------------------------------------------------------------\n")
    except Exception:
        pass
    
    logger.info(f"\n💾 Relatórios salvos: {report_file_md} | {report_file_txt}")
    
    # Salva JSON também
    json_file = os.path.join(OUTPUT_DIR, f"futures_results_{timestamp}.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Diagnósticos adicionais
    diag_file = os.path.join(OUTPUT_DIR, f"diagnostics_{timestamp}.txt")
    try:
        with open(diag_file, 'w', encoding='utf-8') as fd:
            fd.write("DIAGNOSTICOS DE TRADES\n")
            fd.write(f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
            fd.write("------------------------------------------------------------\n")
            for symbol, results in all_results.items():
                fd.write(f"SIMBOLO: {symbol}\n")
                if not results:
                    fd.write("SEM RESULTADOS\n")
                else:
                    best = max(results, key=lambda x: x['test_metrics']['sharpe'])
                    params = best.get("params", {})
                    df = load_futures_data(symbol, bars=5000, timeframe="M15")
                    if df is not None:
                        diag = deep_trade_diagnostics(symbol, params, df)
                        fd.write(f"  Trades: {diag['total_trades']} | WR: {diag['win_rate']:.2f}\n")
                        fd.write(f"  Med PNL WIN: {diag['avg_pnl_win']:.2f} | Med PNL LOSS: {diag['avg_pnl_loss']:.2f}\n")
                        fd.write(f"  Tipos: {json.dumps(diag['type_counts'], ensure_ascii=False)}\n")
                        fd.write(f"  Saidas: {json.dumps(diag['exit_counts'], ensure_ascii=False)}\n")
                fd.write("------------------------------------------------------------\n")
        logger.info(f"🧪 Diagnósticos salvos: {diag_file}")
    except Exception:
        pass

# ===========================
# MAIN
# ===========================

def main():
    """Execução principal"""
    
    print("\n" + "="*60)
    print("🚀 OTIMIZADOR DE FUTUROS B3 - XP3")
    print("="*60 + "\n")
    
    # Conecta MT5
    if MT5_AVAILABLE:
        logger.info("🔌 Conectando ao MetaTrader 5...")
        if ensure_mt5_futures():
            logger.info("✅ MT5 conectado com sucesso!\n")
        else:
            logger.warning("⚠️ MT5 não disponível - usando dados sintéticos\n")
    
    # Descoberta desativada: apenas símbolos concretos informados pelo usuário
    
    # Otimiza futuros definidos + os presentes no SECTOR_MAP
    all_results = {}
    symbols_to_run = list(config_fut.SYMBOLS)
    
    for symbol in symbols_to_run:
        try:
            results = walk_forward_futures(symbol, bars=10000, windows=6)
            all_results[symbol] = results
        except Exception as e:
            logger.error(f"❌ Erro em {symbol}: {e}")
            all_results[symbol] = None
    
    # Força execução para ativos com falha ou resultados genéricos
    for symbol in symbols_to_run:
        res = all_results.get(symbol)
        try:
            if not res or len(res) == 0:
                fr = force_execution_for_symbol(symbol)
                if fr:
                    all_results[symbol] = fr
            else:
                best = max(res, key=lambda x: x['test_metrics']['sharpe'])
                if _is_generic_result(best.get("test_metrics", {})):
                    fr = force_execution_for_symbol(symbol)
                    if fr:
                        all_results[symbol] = fr
        except Exception as e:
            logger.warning(f"⚠️ Força de execução falhou em {symbol}: {e}")
    
    # Gera relatório
    logger.info("\n📝 Gerando relatório final...")
    generate_report(all_results)
    
    print("\n" + "="*60)
    print("✅ OTIMIZAÇÃO CONCLUÍDA!")
    print("="*60 + "\n")
    
    try:
        from investment_allocation import run as allocation_run
        assets_for_allocation = []
        for symbol, results in all_results.items():
            if not results:
                continue
            if not VALIDATION_MANAGER.is_unlocked(symbol):
                st = VALIDATION_MANAGER.get_status(symbol)
                logger.warning(f"⛔ {symbol}: bloqueado. Validações {st['count']}/{st['required']}")
                continue
            best = max(results, key=lambda x: x['test_metrics']['sharpe'])
            m = best['test_metrics']
            assets_for_allocation.append({
                "symbol": symbol,
                "ema_short": float(m.get("ema_fast_last", 0.0)),
                "ema_long": float(m.get("ema_slow_last", 0.0)),
                "rsi": float(m.get("rsi_last", 50.0)),
                "adx": float(m.get("adx_last", 0.0)),
                "mom": float(m.get("mom_mean", m.get("mom_last", 0.0))),
                "sl_atr": float(best.get('params', {}).get('sl_atr_mult', 3.0))
            })
        alloc_res = allocation_run(assets_for_allocation, "investment_config.json", "allocation_output")
        logger.info(f"📊 Alocação gerada: total={alloc_res.get('total_percent',0):.2f}% | itens={len(alloc_res.get('allocations',[]))}")
        summary = VALIDATION_MANAGER.summary()
        logger.info(f"🧭 Status de validação: {json.dumps(summary, indent=2, ensure_ascii=False)}")
    except Exception as e:
        logger.error(f"❌ Falha na alocação: {e}")

if __name__ == "__main__":
    main()
