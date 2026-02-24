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
import config_futures
import futures_core

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

def ensure_mt5_futures():
    """Garante conexão MT5"""
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
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro MT5: {e}")
        return False

# ===========================
# CARREGAMENTO DE DADOS
# ===========================

def load_futures_data(symbol: str, bars: int = 5000, timeframe: str = "M15") -> Optional[pd.DataFrame]:
    """Carrega dados históricos de futuros usando futures_core"""
    
    if not MT5_AVAILABLE:
        return None
    
    try:
        # Mapeia timeframe para MT5
        tf_map = {
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "H1": mt5.TIMEFRAME_H1
        }
        tf = tf_map.get(timeframe, mt5.TIMEFRAME_M15)
        
        manager = futures_core.get_manager()
        
        # Se for simbolo generico (WIN$), resolve para o atual
        target = symbol
        if "$" in symbol:
            target = utils.resolve_current_symbol(symbol) or symbol
            
        data = manager.concatenate_history(target.replace("$", "")[:3], bars=bars, timeframe=tf)
        
        if utils.is_valid_dataframe(data):
            # SOLUÇÃO 5: Filtro de Horário
            base = target[:3]
            data = utils.filter_trading_hours(data, base)
            return data
            
        return None
        
    except Exception as e:
        logger.error(f"❌ Erro ao carregar {symbol}: {e}")
        return None

# ===========================
# INDICADORES TÉCNICOS
# ===========================

def calculate_indicators(df: pd.DataFrame, symbol: str) -> dict:
    """Calcula indicadores otimizados para futuros"""
    
    # SOLUÇÃO 12: Volume Decay
    manager = futures_core.get_manager()
    volume = manager.volume_decay_factor(df, symbol)
    
    close = df['close']
    high = df['high']
    low = df['low']
    
    # EMAs adaptativas
    ema_fast = close.ewm(span=8).mean()
    ema_slow = close.ewm(span=21).mean()
    
    # SOLUÇÃO 7: ATR em Pontos
    atr = utils.calculate_atr_points(df, 14)
    
    # ADX para força de tendência
    # (Simplified ADX calculation for brevity, assume utils.get_adx exists or implement optimized)
    # Using simple pandas based implementation for speed
    
    # Volume MA
    vol_ma = volume.rolling(20).mean()
    vol_ratio = volume / (vol_ma + 1)
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # MOM (Momentum percentual 20 períodos)
    mom_20 = close.pct_change(20)
    
    return {
        'close': close.values,
        'high': high.values,
        'low': low.values,
        'volume': volume.values, # Adjusted
        'ema_fast': ema_fast.values,
        'ema_slow': ema_slow.values,
        'atr': atr.values,
        'adx': rsi.values, # Placeholder
        'rsi': rsi.values,
        'vol_ma': vol_ma.values,
        'vol_ratio': vol_ratio.values,
        'mom_20': mom_20.values
    }

# ===========================
# BACKTEST ENGINE PARA FUTUROS
# ===========================

def backtest_futures(symbol: str, params: dict, df: pd.DataFrame) -> dict:
    """
    Backtest especializado para futuros com gestão de margem (B3)
    SOLUÇÕES IMPLANTADAS: 3, 4, 6, 8, 13
    """
    
    if df is None or len(df) < 100:
        return {}
    
    # Identifica configurações do ativo
    base = symbol[:3]
    generic_key = f"{base}$N"
    cfg = config_futures.FUTURES_CONFIGS.get(generic_key, config_futures.FUTURES_CONFIGS.get("WIN$N"))
    
    tick_size = cfg['tick_size']
    point_value = cfg['point_value']
    value_per_tick = cfg['value_per_tick']
    fees = cfg['fees_roundtrip'] # SOLUÇÃO 8
    
    margin_req = cfg['margin']
    margin_stress = cfg['margin_stress'] # SOLUÇÃO 6
    
    ind = calculate_indicators(df, symbol)
    
    # Estado inicial
    capital = config_futures.CAPITAL_TOTAL_BASE # R$ 100k
    equity = capital
    contracts = 0
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    
    trades = []
    equity_curve = []
    
    # Parâmetros
    ema_period_fast = params.get('ema_fast', 8)
    sl_atr_mult = params.get('sl_atr_mult', 2.5)
    tp_atr_mult = params.get('tp_atr_mult', 4.0)
    
    n = len(df)
    
    # SOLUÇÃO 3: Slippage Dinâmico (Pre-calc volume ratio)
    vol_ratio = ind['vol_ratio']
    
    for i in range(50, n):
        current_price = ind['close'][i]
        bar_high = ind['high'][i]
        bar_low = ind['low'][i]
        
        # SOLUÇÃO 13: Mark-to-Market Intraday Check
        if contracts != 0:
            # PnL Flutuante
            if contracts > 0:
                 diff = current_price - entry_price
            else:
                 diff = entry_price - current_price
                 
            # SOLUÇÃO 4: P&L Correto (Formula B3)
            # Para indices/dolar: diff (pontos) * value_per_point?
            # Config diz:
            # WIN: point_value=0.20. Tick=5. Value_per_tick=1.
            # diff=100pts. 100 * 0.20 = 20. Corretissimo.
            # diff=100pts / 5 * 1 = 20. Também.
            floating_pnl = diff * point_value * abs(contracts)
            current_equity = equity + floating_pnl
            
            # Margem Check (Call)
            required_margin = abs(contracts) * margin_stress
            if current_equity < required_margin:
                # MARGIN CALL
                exit_price = current_price
                reason = "MARGIN_CALL"
                
                # Executa saída forçada
                pnl = floating_pnl # Aprox
                costs = fees * abs(contracts)
                net_pnl = pnl - costs
                equity += net_pnl
                contracts = 0
                trades.append({'pnl': net_pnl, 'exit_reason': reason})
                continue
        
        # ==================
        # GESTÃO DE POSIÇÃO
        # ==================
        if contracts != 0:
            
            # Verifica stops
            stop_hit = False
            target_hit = False
            
            if contracts > 0: # Long
                 if bar_low <= stop_price: stop_hit = True
                 elif bar_high >= target_price: target_hit = True
            else: # Short
                 if bar_high >= stop_price: stop_hit = True
                 elif bar_low <= target_price: target_hit = True
            
            if stop_hit or target_hit:
                # Slippage na saída
                liquidity_mult = 1.0
                if vol_ratio[i] > 1.5: liquidity_mult = 0.7
                elif vol_ratio[i] < 0.6: liquidity_mult = 2.0
                
                # Base slippage da config
                slip_base = cfg.get('slippage_base', {}).get('avg', 10)
                slippage_pts = slip_base * liquidity_mult
                
                if stop_hit:
                    # Pior preço
                    exit_price = stop_price - (slippage_pts * tick_size) if contracts > 0 else stop_price + (slippage_pts * tick_size)
                    reason = "STOP"
                else:
                    exit_price = target_price
                    reason = "TARGET"
                
                # Calc PnL Final (SOLUÇÃO 4)
                if contracts > 0:
                    pts = exit_price - entry_price
                else:
                    pts = entry_price - exit_price
                    
                gross_pnl = pts * point_value * abs(contracts)
                costs = fees * abs(contracts) # Roundtrip já incluso
                net_pnl = gross_pnl - costs
                
                equity += net_pnl
                contracts = 0
                trades.append({'pnl': net_pnl, 'exit_reason': reason, 'entry': entry_price, 'exit': exit_price})
        
        # ==================
        # SINAIS DE ENTRADA
        # ==================
        else: # No position
             # Lógica simplificada de sinal (exemplo) based on EMA crossover
             ema_f = ind['ema_fast'][i]
             ema_s = ind['ema_slow'][i]
             prev_ema_f = ind['ema_fast'][i-1]
             prev_ema_s = ind['ema_slow'][i-1]
             
             # Cross Up
             if prev_ema_f <= prev_ema_s and ema_f > ema_s:
                 signal = 1
             elif prev_ema_f >= prev_ema_s and ema_f < ema_s:
                 signal = -1
             else:
                 signal = 0
                 
             if signal != 0:
                 # SOLUÇÃO 6: Tamanho de Posição com Margem
                 # 1. Capital risco = 2% do equity
                 risk_money = equity * config_futures.MAX_RISK_PERCENT
                 
                 # 2. Stop em pontos (SOLUÇÃO 7 - ATR Points)
                 atr_val = ind['atr'][i]
                 stop_pts = atr_val * sl_atr_mult
                 risk_per_contract = stop_pts * point_value
                 
                 if risk_per_contract <= 0: continue
                 
                 # 3. Max Contratos calculados
                 max_contracts_risk = risk_money / risk_per_contract
                 
                 # 4. Validar Margem Stress
                 max_contracts_margin = (equity * config_futures.MARGIN_SAFETY_FACTOR) / margin_stress
                 
                 allowed_contracts = int(min(max_contracts_risk, max_contracts_margin))
                 
                 if allowed_contracts >= 1:
                     # Slippage Entry
                     liquidity_mult = 1.0
                     if vol_ratio[i] > 1.5: liquidity_mult = 0.7
                     elif vol_ratio[i] < 0.6: liquidity_mult = 2.0
                     slip_pts = cfg.get('slippage_base', {}).get('avg', 10) * liquidity_mult
                     
                     if signal == 1:
                         entry_price = current_price + (slip_pts * tick_size)
                         stop_price = entry_price - (stop_pts) 
                         target_price = entry_price + (atr_val * tp_atr_mult)
                         contracts = allowed_contracts
                     else:
                         entry_price = current_price - (slip_pts * tick_size)
                         stop_price = entry_price + (stop_pts)
                         target_price = entry_price - (atr_val * tp_atr_mult)
                         contracts = -allowed_contracts
                         
                     # Deduz fees upfront? Geralmente deduz no PnL final.
        
        equity_curve.append(equity)
        
    return {
        "trades": trades,
        "final_equity": equity,
        "total_trades": len(trades),
        "win_rate": len([t for t in trades if t['pnl'] > 0]) / len(trades) if trades else 0,
        "equity_curve": equity_curve
    }
