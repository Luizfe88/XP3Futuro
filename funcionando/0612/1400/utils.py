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
from collections import deque
import config
from colorama import init, Fore, Style

# ... [Definições de Cores e setup do logger - Mantidas] ...
VERDE = Fore.GREEN + Style.BRIGHT
VERMELHO = Fore.RED + Style.BRIGHT
AMARELO = Fore.YELLOW + Style.BRIGHT
AZUL = Fore.CYAN + Style.BRIGHT
ROXO = Fore.MAGENTA + Style.BRIGHT
BRANCO = Fore.WHITE + Style.BRIGHT
RESET = Style.RESET_ALL

init(autoreset=True)

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
# -----------------------------------------------------------------------

# Variáveis Globais para Robustez e Vol Targeting
EQUITY_HISTORY = [] 
LAST_EQUITY_UPDATE_TIME = datetime.now()
LAST_TICK_TIME = datetime.now()
EQUITY_DROP_HISTORY = deque(maxlen=2)

# ==================== FUNÇÕES DE RISCO E VOLATILIDADE ====================

def calculate_equity_vol_20d(equity_value, window=20):
    """Calcula a Volatilidade Realizada Anualizada da Equity."""
    global EQUITY_HISTORY, LAST_EQUITY_UPDATE_TIME
    if (datetime.now() - LAST_EQUITY_UPDATE_TIME).total_seconds() > 3600 * 4: # A cada 4h
        EQUITY_HISTORY.append(equity_value)
        EQUITY_HISTORY = EQUITY_HISTORY[-22:] 
        LAST_EQUITY_UPDATE_TIME = datetime.now()
    if len(EQUITY_HISTORY) < window: return config.TARGET_VOL_ANNUAL 
    closes = pd.Series(EQUITY_HISTORY)
    returns = np.log(closes / closes.shift(1)).dropna()
    if len(returns) < 5: return config.TARGET_VOL_ANNUAL
    daily_vol = returns.std()
    annual_vol = daily_vol * np.sqrt(252)
    return annual_vol

def calculate_vix_br_proxy():
    """Calcula VIX_BR Proxy (Volatilidade Anualizada do WIN intradiário 20 dias)."""
    symbol = config.IBOV_SYMBOL
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 22)
    if rates is None or len(rates) < 20: return 25.0 
    closes = pd.Series([x['close'] for x in rates])
    returns = np.log(closes / closes.shift(1))
    daily_vol = returns.std()
    annual_vol = daily_vol * np.sqrt(252) * 100 # Em %
    return annual_vol

def calculate_asset_correlation(symbol_a, symbol_b, window=30):
    """Calcula Correlação Rolling 30 dias entre dois ativos."""
    try:
        rates_a = mt5.copy_rates_from_pos(symbol_a, mt5.TIMEFRAME_D1, 0, window + 5)
        rates_b = mt5.copy_rates_from_pos(symbol_b, mt5.TIMEFRAME_D1, 0, window + 5)
        df_a = pd.DataFrame(rates_a)
        df_b = pd.DataFrame(rates_b)
        returns_a = np.log(df_a['close'] / df_a['close'].shift(1)).dropna()
        returns_b = np.log(df_b['close'] / df_b['close'].shift(1)).dropna()
        df_returns = pd.concat([returns_a.rename('Ret_A'), returns_b.rename('Ret_B')], axis=1).dropna()
        if len(df_returns) < 15: return 0.0 
        corr = df_returns['Ret_A'].rolling(window=window).corr(df_returns['Ret_B']).iloc[-1]
        return abs(corr)
    except Exception as e:
        logger.error(f"Erro calculando correlação: {e}")
        return 0.0

def calculate_z_score(data_series, window):
    """Calcula o Z-Score do último ponto em relação ao rolling mean/std."""
    if len(data_series) < window: return 0.0
    rolling_mean = data_series.rolling(window=window).mean().iloc[-1]
    rolling_std = data_series.rolling(window=window).std().iloc[-1]
    last_value = data_series.iloc[-1]
    if rolling_std == 0: return 0.0
    return (last_value - rolling_mean) / rolling_std

def get_market_regime():
    """Retorna o regime de mercado, IBOV atual, MM200 e VIX_BR (PONTO 1)."""
    ticker = config.IBOV_SYMBOL
    rates = mt5.copy_rates_from_pos(ticker, mt5.TIMEFRAME_D1, 0, config.IBOV_MM_PERIOD + 252)
    
    vix_br = calculate_vix_br_proxy()
    
    # Fallback para IBOV e MM200
    if rates is None or len(rates) < 252: 
        curr = mt5.symbol_info_tick(ticker).last if mt5.symbol_info_tick(ticker) else 100000 
        return "BULL", curr, curr * 0.95, vix_br 

    df = pd.DataFrame(rates)
    df.ta.adx(length=14, append=True)

    curr = df['close'].iloc[-1]
    ma200 = df['close'].rolling(config.IBOV_MM_PERIOD).mean().iloc[-1]
    adx = df[f"ADX_14"].iloc[-1]
    
    df["Dist_MM200_Pct"] = (df["close"] - ma200) / ma200 * 100
    z_score_mm200 = calculate_z_score(df["Dist_MM200_Pct"], window=200)

    momentum_6m = (curr / df['close'].iloc[-120]) - 1
    momentum_12m = (curr / df['close'].iloc[-252]) - 1
    
    if vix_br > config.VIX_CRISIS_THRESHOLD:
        regime = "CRISIS"
    elif adx < config.IBOV_ADX_SIDEWAYS:
        regime = "SIDEWAYS"
    elif curr > ma200 and z_score_mm200 > 1.0 and momentum_6m > 0.10 and momentum_12m > 0.20:
        regime = "STRONG_BULL"
    elif curr > ma200 and momentum_6m > 0.0:
        regime = "BULL"
    else:
        regime = "BEAR"
        
    return regime, curr, ma200, vix_br # Retorna 4 valores para o bot.py

def calcular_tamanho_posicao(symbol, sl_price, is_buy):
    """
    KELLY CRITERION FRACIONADO + VOL TARGETING DINÂMICO + RISK OVERLAYS (PONTOS 2 & 6)
    ... [Lógica omitida por brevidade, assumida como funcional] ...
    """
    tick = mt5.symbol_info_tick(symbol)
    if not tick: return 0
    price = tick.ask if is_buy else tick.bid
    
    acc = mt5.account_info()
    equity = acc.equity
    
    # Risco base de Kelly
    win_rate, payoff = 0.55, 1.2 
    kelly_pct = win_rate - ((1 - win_rate) / payoff) if payoff > 0 else 0
    risk_base = max(0.001, kelly_pct * config.KELLY_FRACTION) 
    
    regime, _, _, vix_br = get_market_regime()
    max_risk_cap = config.MAX_RISK_PER_TRADE / 100 
    
    is_crisis = (regime == "CRISIS" or vix_br > config.VIX_CRISIS_THRESHOLD)
    
    if is_crisis:
        max_risk_cap = config.CRISIS_RISK_REDUCTION / 100 
        risk_base = min(risk_base, max_risk_cap) 
        if not is_buy: return 0 
        if symbol in config.HIGH_BETA_BLOCK_LIST: return 0
    
    risk_pct = min(risk_base, max_risk_cap)

    # 2. Vol Targeting Dinâmico
    leverage_factor = 1.0
    if config.USE_VOL_TARGETING:
        equity_vol = calculate_equity_vol_20d(equity) 
        leverage_factor = config.TARGET_VOL_ANNUAL / max(equity_vol, 0.01) 
        leverage_factor = min(config.MAX_LEVERAGE, leverage_factor)
        leverage_factor = max(config.MIN_LEVERAGE_FLOOR, leverage_factor)
        risk_pct *= leverage_factor
    
    risk_cash = equity * risk_pct
    
    # 3. Converter para Lotes
    dist_sl = abs(price - sl_price)
    if dist_sl <= 0: return 0
    
    qtd = risk_cash / dist_sl
    lote = (int(qtd) // 100) * 100 
    
    # 4. Limite de Exposição
    max_cash_per_asset = equity * (config.MAX_EXPOSURE_PER_ASSET_PCT / 100)
    qtd_max_cash = max_cash_per_asset / price
    
    lote_final = min(lote, (int(qtd_max_cash) // 100) * 100)
    
    return max(0, lote_final)


# ==================== UTILIDADES BÁSICAS (PNL & ATR) ====================

def calculate_asset_atr(symbol, timeframe=config.TIMEFRAME_MT5, window=30):
    """
    Calcula o ATR (Average True Range) de um ativo com fallback robusto.
    (Função extraída e ajustada da lógica anterior)
    """
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, window)
    if rates is None or len(rates) < 14: 
        logger.warning(f"Dados insuficientes para ATR de {symbol}.")
        return 0.0
        
    df = pd.DataFrame(rates)
    df.ta.atr(length=14, append=True)
    
    atr = 0.0
    if 'ATR_14' in df.columns and not pd.isna(df['ATR_14'].iloc[-1]): 
        atr = df['ATR_14'].iloc[-1]
        
    if atr <= 0.0:
        # Fallback para High/Low da barra anterior
        if len(df) >= 2:
            atr_fallback = df['high'].iloc[-2] - df['low'].iloc[-2]
            if atr_fallback > 0:
                atr = atr_fallback
            else:
                return 0.0
        else:
            return 0.0
            
    return atr

def get_daily_profit_loss():
    """
    Calcula o PnL flutuante total das posições abertas (R$ e %).
    """
    acc = mt5.account_info()
    positions = mt5.positions_get()
    
    total_profit_cash = sum(pos.profit for pos in positions)
    
    balance = acc.balance if acc else 1.0 
    total_profit_pct = (total_profit_cash / balance) * 100 
        
    return total_profit_cash, total_profit_pct

def calculate_z_score(data_series, window):
    """Calcula o Z-Score do último ponto em relação ao rolling mean/std."""
    if len(data_series) < window: return 0.0
    rolling_mean = data_series.rolling(window=window).mean().iloc[-1]
    rolling_std = data_series.rolling(window=window).std().iloc[-1]
    last_value = data_series.iloc[-1]
    
    if rolling_std == 0: return 0.0
    return (last_value - rolling_mean) / rolling_std

def get_market_regime():
    """Retorna um dos 5 regimes: STRONG_BULL, BULL, SIDEWAYS, BEAR, CRISIS (PONTO 1)."""
    ticker = config.IBOV_SYMBOL
    rates = mt5.copy_rates_from_pos(ticker, mt5.TIMEFRAME_D1, 0, config.IBOV_MM_PERIOD + 252)
    
    vix_br = calculate_vix_br_proxy()
    
    if rates is None or len(rates) < 252: 
        return "BULL", 0.0, 0.0, vix_br 

    df = pd.DataFrame(rates)
    df.ta.adx(length=14, append=True)

    curr = df['close'].iloc[-1]
    ma200 = df['close'].rolling(config.IBOV_MM_PERIOD).mean().iloc[-1]
    adx = df[f"ADX_14"].iloc[-1]
    
    df["Dist_MM200_Pct"] = (df["close"] - ma200) / ma200 * 100
    z_score_mm200 = calculate_z_score(df["Dist_MM200_Pct"], window=200)

    momentum_6m = (curr / df['close'].iloc[-120]) - 1
    momentum_12m = (curr / df['close'].iloc[-252]) - 1
    
    if vix_br > config.VIX_CRISIS_THRESHOLD:
        regime = "CRISIS"
    elif adx < config.IBOV_ADX_SIDEWAYS:
        regime = "SIDEWAYS"
    elif curr > ma200 and z_score_mm200 > 1.0 and momentum_6m > 0.10 and momentum_12m > 0.20:
        regime = "STRONG_BULL"
    elif curr > ma200 and momentum_6m > 0.0:
        regime = "BULL"
    else:
        regime = "BEAR"
        
    return regime, z_score_mm200, momentum_6m, vix_br

def get_targets(regime):
    # Lógica para encontrar targets baseada em sinais e CURRENT_PARAMS
    # Retorna: [{'symbol': 'PETR4', 'action': 'COMPRA', 'score': 120.0, 'data': {'price': 30.50}}]
    return []

def get_asset_score(symbol):
    return 100.0

def calcular_tamanho_posicao(symbol, sl_price, is_buy):
    """
    KELLY CRITERION FRACIONADO + VOL TARGETING DINÂMICO + RISK OVERLAYS (PONTOS 2 & 6)
    """
    tick = mt5.symbol_info_tick(symbol)
    if not tick: return 0
    price = tick.ask if is_buy else tick.bid
    
    acc = mt5.account_info()
    equity = acc.equity
    
    # Risco base de Kelly (Usando um placeholder para win_rate/payoff)
    win_rate, payoff = 0.55, 1.2 
    kelly_pct = win_rate - ((1 - win_rate) / payoff) if payoff > 0 else 0
    risk_base = max(0.001, kelly_pct * config.KELLY_FRACTION) 
    
    regime, _, _, vix_br = get_market_regime()
    max_risk_cap = config.MAX_RISK_PER_TRADE / 100 
    
    is_crisis = (regime == "CRISIS" or vix_br > config.VIX_CRISIS_THRESHOLD)
    
    if is_crisis:
        max_risk_cap = config.CRISIS_RISK_REDUCTION / 100 # Reduz risco para 0.15%
        risk_base = min(risk_base, max_risk_cap) 
        
        if not is_buy: 
            logger.warning(f"{symbol} BLOQUEADO: Zero Short em Regime CRISIS/VIX Alto.")
            return 0 
        
        if symbol in config.HIGH_BETA_BLOCK_LIST:
            logger.warning(f"{symbol} BLOQUEADO: Beta > 2.0 em Regime CRISIS.")
            return 0
    
    risk_pct = min(risk_base, max_risk_cap)

    # 2. Vol Targeting Dinâmico (PONTO 2)
    leverage_factor = 1.0
    if config.USE_VOL_TARGETING:
        equity_vol = calculate_equity_vol_20d(equity) 
        
        leverage_factor = config.TARGET_VOL_ANNUAL / max(equity_vol, 0.01) 
        leverage_factor = min(config.MAX_LEVERAGE, leverage_factor)
        leverage_factor = max(config.MIN_LEVERAGE_FLOOR, leverage_factor) # Floor 0.3
        
        risk_pct *= leverage_factor
    
    risk_cash = equity * risk_pct
    
    # 3. Converter para Lotes
    dist_sl = abs(price - sl_price)
    if dist_sl <= 0: return 0
    
    qtd = risk_cash / dist_sl
    lote = (int(qtd) // 100) * 100 
    
    # 4. Limite de Exposição (MAX_EXPOSURE_PER_ASSET_PCT)
    max_cash_per_asset = equity * (config.MAX_EXPOSURE_PER_ASSET_PCT / 100)
    qtd_max_cash = max_cash_per_asset / price
    
    lote_final = min(lote, (int(qtd_max_cash) // 100) * 100)
    
    return max(0, lote_final)


# ==================== TRAILING STOP AVANÇADO (SAÍDAS INTELIGENTES - PONTO 4) ====================

def get_position_r_ratio(pos):
    """Calcula o R-ratio (Lucro atual / Risco inicial)."""
    initial_r_dist = abs(pos.price_open - pos.sl) if pos.sl != 0 else 0.0
    
    # CORREÇÃO: Tratamento robusto para ATR_14 (KeyError)
    if initial_r_dist == 0:
        rates = mt5.copy_rates_from_pos(pos.symbol, config.TIMEFRAME_MT5, 0, 30) # Aumentado para 30 barras
        if rates is None or len(rates) < 14: 
            return 0.0
            
        df = pd.DataFrame(rates)
        df.ta.atr(length=14, append=True)
        
        atr = 0.0
        if 'ATR_14' in df.columns: # Verifica se a coluna foi criada
            atr = df['ATR_14'].iloc[-1]
            
        if atr > 0 and not pd.isna(atr):
            initial_r_dist = atr * config.DEFAULT_PARAMS.get("sl_atr_mult", 2.0)
        else:
            # Fallback para High/Low da barra anterior se ATR falhar (KeyError ou NaN/Zero)
            if len(df) >= 2:
                high_low_range = df['high'].iloc[-2] - df['low'].iloc[-2]
                initial_r_dist = high_low_range * config.DEFAULT_PARAMS.get("sl_atr_mult", 2.0)
                if initial_r_dist == 0:
                    return 0.0
            else:
                logger.warning(f"ATR falhou e dados insuficientes para fallback para {pos.symbol}.")
                return 0.0
        
    if initial_r_dist == 0: return 0.0
    
    price_dist = (pos.price_current - pos.price_open) if pos.type == 0 else (pos.price_open - pos.price_current)
    return price_dist / initial_r_dist

def update_position_comment(ticket, tag):
    """Adiciona uma tag ao comentário da posição para controle de parciais."""
    positions = mt5.positions_get(ticket=ticket)
    if not positions: return
    pos = positions[0]
    
    new_comment = pos.comment or ""
    if tag not in new_comment:
        new_comment += f"|{tag}"
    
    req = {
        "action": mt5.TRADE_ACTION_SLTP, 
        "position": pos.ticket, 
        "sl": pos.sl, 
        "tp": pos.tp, 
        "symbol": pos.symbol,
        "comment": new_comment
    }
    mt5.order_send(req)

def close_partial_position(pos, volume_to_close, reason):
    """Fecha um volume específico da posição."""
    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": pos.symbol,
        "volume": float(volume_to_close),
        "type": 1 if pos.type == 0 else 0, # Inverte a ordem
        "position": pos.ticket,
        "type_filling": mt5.ORDER_FILLING_IOC,
        "comment": reason
    }
    result = mt5.order_send(req)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"{AMARELO}FECHAMENTO PARCIAL: {pos.symbol} {volume_to_close:.0f} @ {reason}{RESET}")
    return result


def trailing_stop_service():
    """
    Saídas Inteligentes: Partial Profits + Time Stop (PONTO 4).
    """
    logger.info("Service Trailing Stop/Partial Profits: ON")
    while True:
        try:
            positions = mt5.positions_get()
            if positions:
                for pos in positions:
                    if pos.magic != 20251230: continue
                    
                    r_ratio = get_position_r_ratio(pos)
                    
                    # 1. Partial Profits
                    if r_ratio >= config.PARTIAL_PROFIT_1_R and (pos.comment is None or pos.comment.find("P1") == -1):
                        volume_venda = int(pos.volume * config.PARTIAL_PROFIT_1_PCT)
                        if volume_venda > 0 and pos.volume - volume_venda >= 100:
                            close_partial_position(pos, volume_venda, "Partial_1.0R")
                            update_position_comment(pos.ticket, "P1") 
                        elif volume_venda > 0: # Fecha o restante se sobrar menos de 100
                            close_partial_position(pos, pos.volume, "Partial_1.0R_Final")
                            update_position_comment(pos.ticket, "P1_Final")
                            
                    elif r_ratio >= config.PARTIAL_PROFIT_2_R and (pos.comment is None or pos.comment.find("P2") == -1):
                        volume_venda = int(pos.volume * config.PARTIAL_PROFIT_2_PCT)
                        if volume_venda > 0 and pos.volume - volume_venda >= 100:
                            close_partial_position(pos, volume_venda, "Partial_1.8R")
                            update_position_comment(pos.ticket, "P2") 
                        elif volume_venda > 0:
                            close_partial_position(pos, pos.volume, "Partial_1.8R_Final")
                            update_position_comment(pos.ticket, "P2_Final")

                    # 2. Time Stop
                    open_time = datetime.fromtimestamp(pos.time)
                    if (datetime.now() - open_time).total_seconds() > (config.TIME_STOP_HOURS * 3600):
                        if r_ratio >= config.TIME_STOP_PROFIT_MIN_R:
                            if pos.comment is None or pos.comment.find("Time") == -1:
                                close_partial_position(pos, pos.volume, "Time_Stop")
                                update_position_comment(pos.ticket, "Time")
                                continue
                            
                    # 3. Breakeven Trailing (Mantida a lógica anterior)
                    if r_ratio >= 0.5:
                        new_sl = pos.price_open
                        # Apenas move o SL se estiver mais vantajoso
                        if pos.type == 0 and new_sl > pos.sl:
                            mt5.order_send({"action": mt5.TRADE_ACTION_SLTP, "position": pos.ticket, "sl": float(new_sl), "tp": pos.tp, "symbol": pos.symbol})
                        elif pos.type == 1 and (pos.sl == 0 or new_sl < pos.sl):
                            mt5.order_send({"action": mt5.TRADE_ACTION_SLTP, "position": pos.ticket, "sl": float(new_sl), "tp": pos.tp, "symbol": pos.symbol})


        except Exception as e:
            logger.error(f"Trailing/Partial Error: {e}")
        
        time.sleep(1)


# ==================== CIRCUIT BREAKERS E UTILIDADES BÁSICAS (PONTOS 8 - CORREÇÃO) ====================
def check_mt5_connection():
    # ... [Implementação mantida] ...
    if not mt5.initialize():
        logger.critical(f"{VERMELHO}Falha ao inicializar MT5. Erro: {mt5.last_error()}{RESET}")
        return False
    
    account_info = mt5.account_info()
    if account_info is None:
        logger.critical(f"{VERMELHO}Não conectado à conta. Verifique as credenciais no MT5.{RESET}")
        mt5.shutdown()
        return False
        
    logger.info(f"{VERDE}MT5 conectado com sucesso. Conta: {account_info.login}{RESET}")
    return True

def check_circuit_breakers(account_info, tick_data):
    # ... [Implementação mantida] ...
    global LAST_TICK_TIME, EQUITY_DROP_HISTORY
    
    acc = account_info
    equity = acc.equity

    # 1. Heartbeat Check
    if tick_data is not None:
        LAST_TICK_TIME = datetime.now()
    
    if (datetime.now() - LAST_TICK_TIME).total_seconds() > config.HEARTBEAT_INTERVAL_SEC:
        logger.critical(f"{VERMELHO}HEARTBEAT FAILED: Sem ticks há {config.HEARTBEAT_INTERVAL_SEC}s. FECHAMENTO DE PÂNICO!{RESET}")
        close_all_positions(panic=True)
        return True
        
    # 2. Panic Close (Queda de 2% em 10 minutos)
    if len(EQUITY_DROP_HISTORY) == 2:
        start_equity = EQUITY_DROP_HISTORY[0]['equity']
        end_equity = EQUITY_DROP_HISTORY[1]['equity']
        time_diff = (EQUITY_DROP_HISTORY[1]['time'] - EQUITY_DROP_HISTORY[0]['time']).total_seconds() / 60
        
        if time_diff <= config.PANIC_DROP_WINDOW_MIN:
            drop_pct = ((end_equity / start_equity) - 1.0) * 100
            
            if drop_pct <= config.PANIC_DROP_PCT:
                logger.critical(f"{VERMELHO}PANIC CLOSE: Queda de {drop_pct:.2f}% em {time_diff:.1f} minutos. FECHAMENTO DE PÂNICO!{RESET}")
                close_all_positions(panic=True)
                return True
                
    return False

def close_all_positions(panic=False):
    # ... [Implementação mantida] ...
    positions = mt5.positions_get()
    for pos in positions:
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": 1 if pos.type == 0 else 0,
            "position": pos.ticket,
            "type_filling": mt5.ORDER_FILLING_IOC,
            "comment": "PANIC CLOSE" if panic else "CLOSE ALL"
        }
        mt5.order_send(req)
    logger.info("Todas as posições fechadas.")

def register_trade(symbol, action, price, volume, event, sl=0.0, tp=0.0):
    # ... [Implementação mantida] ...
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    trade_data = [timestamp, symbol, action, price, volume, event, sl, tp]
    
    file_exists = os.path.exists(config.TRADES_FILE)
    with open(config.TRADES_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists or os.path.getsize(config.TRADES_FILE) == 0:
            writer.writerow(['Timestamp', 'Symbol', 'Action', 'Price', 'Volume', 'Event', 'SL', 'TP'])
        writer.writerow(trade_data)

def get_trade_pnl_daily():
    # ... [Implementação mantida] ...
    total_profit = 0.0
    positions = mt5.positions_get()
    for pos in positions:
        total_profit += pos.profit
        
    return total_profit