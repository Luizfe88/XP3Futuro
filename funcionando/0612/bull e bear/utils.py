# utils.py – VERSÃO FINAL ELITE PRO (INSTITUCIONAL)
import logging
import csv
import os
import time
import winsound
import json
import random
import math
import numpy as np
import pandas as pd
import pandas_ta as ta
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import config
from colorama import init, Fore, Style

VERDE = Fore.GREEN + Style.BRIGHT
VERMELHO = Fore.RED + Style.BRIGHT
AMARELO = Fore.YELLOW + Style.BRIGHT
AZUL = Fore.CYAN + Style.BRIGHT
ROXO = Fore.MAGENTA + Style.BRIGHT
BRANCO = Fore.WHITE + Style.BRIGHT
RESET = Style.RESET_ALL

init(autoreset=True)

# Logger setup
def setup_logger():
    logger = logging.getLogger("BotElitePro")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        os.makedirs("logs", exist_ok=True)
        fh = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
        ch = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

logger = setup_logger()

# ==================== CIRCUIT BREAKERS & PROTECTION ====================

def check_circuit_breakers():
    """Verifica DD Diário e Equity Peak."""
    acc = mt5.account_info()
    if not acc: return False

    # 1. Equity Peak Protection
    equity = acc.equity
    hist_file = "data/equity_peak.json"
    peak = equity
    
    if os.path.exists(hist_file):
        with open(hist_file, 'r') as f:
            try: peak = json.load(f).get("peak", equity)
            except: pass
    
    if equity > peak:
        with open(hist_file, 'w') as f:
            json.dump({"peak": equity}, f)
    elif equity < (peak * config.PEAK_EQUITY_STOP):
        logger.critical(f"PROTEÇÃO MÁXIMA: Equity {equity:,.2f} < 92% do Pico {peak:,.2f}. FECHANDO TUDO.")
        close_all_positions(panic=True)
        return True # Stop Trading

    # 2. Daily Drawdown
    _, pct_loss = get_daily_profit_loss()
    if pct_loss <= config.MAX_DAILY_DRAWDOWN:
        logger.critical(f"DD DIÁRIO ATINGIDO: {pct_loss:.2f}%. Pausando 2h.")
        close_all_positions()
        time.sleep(7200) # Pausa 2h
        return True

    return False

def get_daily_profit_loss():
    now = datetime.now()
    start = now.replace(hour=0, minute=0, second=0)
    deals = mt5.history_deals_get(start, now)
    realized = sum(d.profit for d in deals) if deals else 0.0
    
    pos = mt5.positions_get()
    floating = sum(p.profit for p in pos) if pos else 0.0
    
    acc = mt5.account_info()
    balance = acc.balance if acc else 1.0
    total = realized + floating
    return total, (total/balance)*100

# ==================== POSITION SIZING (KELLY & VOL TARGET) ====================

def get_historical_stats():
    """Calcula WinRate e Payoff dos últimos 180 dias."""
    end = datetime.now()
    start = end - timedelta(days=180)
    deals = mt5.history_deals_get(start, end)
    
    if not deals or len(deals) < 10:
        return 0.50, 1.5 # Default conservador
        
    profits = [d.profit for d in deals if d.entry == mt5.DEAL_ENTRY_OUT]
    if not profits: return 0.50, 1.5
    
    wins = [p for p in profits if p > 0]
    losses = [abs(p) for p in profits if p <= 0]
    
    win_rate = len(wins) / len(profits)
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 1
    payoff = avg_win / avg_loss if avg_loss > 0 else 1.5
    
    return win_rate, payoff

def calculate_implied_vol_proxy(symbol):
    """Calcula volatilidade anualizada dos últimos 20 dias."""
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 22)
    if rates is None or len(rates) < 20: return 20.0 # Default normal
    
    closes = pd.Series([x['close'] for x in rates])
    returns = np.log(closes / closes.shift(1))
    daily_vol = returns.std()
    annual_vol = daily_vol * np.sqrt(252) * 100
    return annual_vol

def calcular_tamanho_posicao(symbol, sl_price, is_buy):
    """
    KELLY CRITERION FRACIONADO + VOL TARGETING
    """
    tick = mt5.symbol_info_tick(symbol)
    if not tick: return 0
    price = tick.ask if is_buy else tick.bid
    
    acc = mt5.account_info()
    equity = acc.equity
    
    # 1. Volatilidade Implícita (Panic Check)
    ibov_vol = calculate_implied_vol_proxy(config.IBOV_SYMBOL)
    risk_cap = config.MAX_RISK_PER_TRADE
    
    if ibov_vol > config.PANIC_IV_THRESHOLD:
        risk_cap = config.PANIC_RISK_REDUCTION
        if not is_buy: return 0 # Desabilita Short em Pânico
        logger.warning(f"REGIME DE PÂNICO DETECTADO (IV={ibov_vol:.1f}%). Risco reduzido.")

    # 2. Dados Históricos para Kelly
    win_rate, payoff = get_historical_stats()
    
    # Kelly Formula: K% = W - (1-W)/R
    kelly_pct = win_rate - ((1 - win_rate) / payoff)
    kelly_pct = max(0, kelly_pct) * config.KELLY_FRACTION # Kelly Fracionado
    
    # 3. Ajuste por Volatilidade do Ativo (Vol Targeting)
    asset_vol = calculate_implied_vol_proxy(symbol)
    vol_scalar = config.TARGET_VOL_ANNUAL / max(asset_vol, 10.0) # Normaliza para meta de 20%
    
    # 4. Risco Final %
    risk_pct = min(risk_cap/100, kelly_pct * 0.4) * vol_scalar # 0.4 é fator de suavização extra
    risk_cash = equity * risk_pct
    
    # 5. Converter para Lotes
    dist_sl = abs(price - sl_price)
    if dist_sl <= 0: return 0
    
    qtd = risk_cash / dist_sl
    
    # Ajuste Lote Padrão
    lote = (int(qtd) // 100) * 100
    return max(0, lote)

# ==================== DADOS TÉCNICOS & SCORE CONTÍNUO ====================

def get_market_regime():
    """Retorna o regime de mercado, preço atual do IBOV e MM200 para display."""
    ticker = config.IBOV_SYMBOL
    rates = mt5.copy_rates_from_pos(ticker, mt5.TIMEFRAME_D1, 0, config.IBOV_MM_PERIOD + 10)
    
    # Fallback com dados dummy
    if rates is None or len(rates) < config.IBOV_MM_PERIOD:
        last_price = rates[-1]['close'] if rates else 0.0
        return "BULL", last_price, last_price
    
    df = pd.DataFrame(rates)
    curr = df['close'].iloc[-1]
    ma200 = df['close'].rolling(config.IBOV_MM_PERIOD).mean().iloc[-1]
    
    # ADX do IBOV para detectar lateralidade
    df.ta.adx(length=14, append=True)
    adx = df['ADX_14'].iloc[-1] if 'ADX_14' in df else 0
    
    regime = "BULL"
    if adx < config.IBOV_ADX_SIDEWAYS:
        regime = "SIDEWAYS"
    elif curr < ma200:
        regime = "BEAR"
        
    return regime, curr, ma200 # Retorna Regime String, Preço Atual, MM200

def calculate_z_score(series, window=252):
    """Z-Score: (Valor - Média) / Desvio Padrão"""
    if len(series) < window: return 0
    roll = series.rolling(window=window)
    mean = roll.mean()
    std = roll.std()
    z = (series - mean) / std
    return z.iloc[-1]

def get_asset_score(symbol, params):
    """
    SCORE CONTÍNUO 0-100
    Baseado em Z-Score RSI, Distância EMA, ADX, Volume
    """
    bars_needed = 300 # Precisa de história para Z-Score
    
    # CORREÇÃO: Usa os períodos padrões de config.py para o cálculo dos indicadores.
    rsi_period = config.DEFAULT_PARAMS.get("rsi_period", 14)
    adx_period = config.DEFAULT_PARAMS.get("adx_period", 14)
    rsi_penalty_level = params.get("rsi_level", 70) 
    
    rates = mt5.copy_rates_from_pos(symbol, config.TIMEFRAME_MT5, 0, bars_needed)
    if rates is None or len(rates) < bars_needed: return 0, {}, "No Data"
    
    df = pd.DataFrame(rates)
    
    # Indicadores
    df["EMA_F"] = ta.ema(df["close"], length=params["ema_fast"])
    df["EMA_S"] = ta.ema(df["close"], length=params["ema_slow"])
    df["RSI"] = ta.rsi(df["close"], length=rsi_period)
    df.ta.adx(length=adx_period, append=True)
    col_adx = f"ADX_{adx_period}"
    df["VOL_MA"] = ta.sma(df["tick_volume"], length=20)
    
    last = df.iloc[-1]
    
    # 1. EMA Signal Strength (Normalized Distance)
    ema_dist_pct = (last["EMA_F"] - last["EMA_S"]) / last["close"] * 1000
    ema_score = min(100, abs(ema_dist_pct) * 10) 
    
    # 2. RSI Z-Score 
    rsi_z = calculate_z_score(df["RSI"], window=200)
    
    # Lógica de Score
    raw_score = 0
    is_buy = last["EMA_F"] > last["EMA_S"]
    
    # Composição do Score
    raw_score += 40 if is_buy else 0
    raw_score += min(30, ema_score)
    
    adx_val = last[col_adx] if col_adx in last else 0
    raw_score += min(20, adx_val) 
    
    if last["tick_volume"] > last["VOL_MA"]: raw_score += 10
    
    # Penalidades (RSI Extremo) - Usando o nível otimizado
    if is_buy and last["RSI"] > rsi_penalty_level: raw_score -= 30
    if not is_buy and last["RSI"] < (100 - rsi_penalty_level): raw_score -= 30
    
    # Z-Score Boost 
    if -1.5 < rsi_z < 1.5: raw_score += 10
    
    final_score = max(0, min(100, raw_score))
    
    data = {
        "price": last["close"],
        "tick": mt5.symbol_info_tick(symbol),
        "ema_fast": last["EMA_F"], "ema_slow": last["EMA_S"],
        "rsi": last["RSI"], "adx": adx_val
    }
    
    action = "COMPRA" if is_buy else "VENDA"
    return final_score, data, action


def calculate_asset_atr(symbol, timeframe=config.TIMEFRAME_MT5, period=14):
    """
    Calcula o ATR real (Average True Range) para uso em SL/TP.
    """
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, period + 5)
    if rates is None or len(rates) < period: 
        logger.warning(f"Dados insuficientes para ATR em {symbol}. Usando default 0.5.")
        return 0.5 
    
    df = pd.DataFrame(rates)
    atr_val = ta.atr(df['high'], df['low'], df['close'], length=period).iloc[-1]
    return atr_val


# ==================== TRAILING STOP AVANÇADO ====================

def trailing_stop_service():
    """Implementa Trailing Stop com regras de R-ratio."""
    logger.info("Service Trailing Stop: ON")
    while True:
        try:
            positions = mt5.positions_get()
            if positions:
                for pos in positions:
                    if pos.magic != 20251230: continue
                    
                    symbol = pos.symbol
                    # Precisa recalcular ATR
                    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 20)
                    df = pd.DataFrame(rates)
                    atr = ta.atr(df['high'], df['low'], df['close'], length=14).iloc[-1]
                    
                    initial_r = atr * 2.0 # Assumindo SL inicial 2 ATRs
                    
                    current_profit_pts = (pos.price_current - pos.price_open) if pos.type == 0 else (pos.price_open - pos.price_current)
                    
                    if current_profit_pts >= (1.8 * initial_r):
                        
                        if current_profit_pts >= (2.2 * initial_r):
                            # Mover para BE + 0.5 ATR (Proteção mais forte)
                            be_level = pos.price_open + (0.5 * atr) if pos.type == 0 else pos.price_open - (0.5 * atr)
                            
                            # Trailing step 0.6 ATR (Nível conservador)
                            trail_level = pos.price_current - (1.5 * atr) if pos.type == 0 else pos.price_current + (1.5 * atr)
                            
                            new_sl = max(be_level, trail_level) if pos.type == 0 else min(be_level, trail_level)
                            
                        else:
                            # Move SL para entrada
                            new_sl = pos.price_open
                        
                        update = False
                        if pos.type == 0 and new_sl > pos.sl: update = True
                        if pos.type == 1 and (pos.sl == 0 or new_sl < pos.sl): update = True
                        
                        if update:
                            mt5.order_send({
                                "action": mt5.TRADE_ACTION_SLTP, "position": pos.ticket, 
                                "sl": float(new_sl), "tp": pos.tp, "symbol": symbol
                            })
                            
        except Exception as e:
            logger.error(f"Trailing Error: {e}")
        
        time.sleep(1)

def close_all_positions(panic=False):
    positions = mt5.positions_get()
    for pos in positions:
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": 1 if pos.type == 0 else 0,
            "position": pos.ticket,
            "type_filling": mt5.ORDER_FILLING_IOC
        }
        mt5.order_send(req)
    
    if panic:
        winsound.Beep(2000, 1000)