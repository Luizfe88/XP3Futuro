# optimizer.py – OTIMIZADOR VECTORIZADO SEM VIÉS (REALISTIC BACKTEST)
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import pandas_ta as ta
import json
import time
import itertools
import os
from concurrent.futures import ProcessPoolExecutor # Melhor para cálculo pesado
from datetime import datetime, timedelta
import config
import utils
from utils import logger, VERDE, RESET, ROXO

# =================================================================
# Parâmetros de Busca (Aumentados para ~9520 combinações)
# =================================================================
EMA_FAST = range(2, 36, 2)     # Tamanho: 17 (2, 4, ..., 34)
EMA_SLOW = range(10, 50, 3)    # Tamanho: 14 (10, 13, ..., 49)
RSI_LEVEL = range(60, 85, 5)   # Tamanho: 5 (60, 65, 70, 75, 80)
MOMENTUM = [0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0] # Tamanho: 8
# Total de combinações = 17 * 14 * 5 * 8 = 9520
# =================================================================

def get_data(symbols, days=60):
    data_store = {}
    inicio = datetime.now() - timedelta(days=days)
    for s in symbols:
        rates = mt5.copy_rates_range(s, mt5.TIMEFRAME_M5, inicio, datetime.now())
        if rates is not None and len(rates) > 500:
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            data_store[s] = df
    return data_store

def simulate_trades(df, params, side='BUY'):
    """
    Simula trades verificando se o preço bateu no TP ou SL nas próximas N barras.
    NÃO usa shift negativo de preço fixo.
    """
    ema_f, ema_s, rsi_lim, mom_min = params
    
    # 1. Calcula Indicadores
    df['ema_f'] = ta.ema(df['close'], length=ema_f)
    df['ema_s'] = ta.ema(df['close'], length=ema_s)
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    # Momentum D1 (Aproximação via janela de 1 dia em barras M5 ~ 100 barras)
    df['mom'] = df['close'].pct_change(100) * 100 

    # 2. Gera Sinais
    if side == 'BUY':
        sinal = (df['ema_f'] > df['ema_s']) & (df['rsi'] < rsi_lim) & (df['mom'] > mom_min)
    else: # SELL
        # Para Venda: EMA Lenta < EMA Rápida, RSI acima do limite e Momentum negativo
        sinal = (df['ema_f'] < df['ema_s']) & (df['rsi'] > rsi_lim) & (df['mom'] < -mom_min)
        
    trades = df[sinal].copy()
    if trades.empty: return 0, 0, 0 # Score, Winrate, Trades

    # 3. Vectorized Outcome Check (Janela de 48 barras ~ 4 horas max)
    
    idx_sinais = trades.index.values
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    atrs = df['atr'].fillna(0).values
    
    limit_idx = len(closes) - 1
    
    wins = 0
    total = 0
    
    for idx in idx_sinais:
        if idx + 48 > limit_idx: continue
        
        entry_price = closes[idx]
        atr_val = atrs[idx]
        if atr_val == 0: continue
        
        # Define SL e TP baseados no ATR da entrada
        sl_dist = atr_val * config.ATR_MULTIPLIER_SL
        tp_dist = sl_dist * config.TAKE_PROFIT_MULTIPLIER
        
        if side == 'BUY':
            stop_price = entry_price - sl_dist
            target_price = entry_price + tp_dist
            
            # Olha o futuro janela de 48 barras (começa na barra idx+1)
            window_lows = lows[idx+1 : idx+49]
            window_highs = highs[idx+1 : idx+49]
            
            # Verifica qual bateu primeiro
            hit_tp = np.where(window_highs >= target_price)[0]
            hit_sl = np.where(window_lows <= stop_price)[0]
            
            first_tp = hit_tp[0] if len(hit_tp) > 0 else 999
            first_sl = hit_sl[0] if len(hit_sl) > 0 else 999
            
            if first_tp < first_sl: wins += 1
            total += 1
            
        else: # SELL
            stop_price = entry_price + sl_dist
            target_price = entry_price - tp_dist
            
            window_lows = lows[idx+1 : idx+49]
            window_highs = highs[idx+1 : idx+49]
            
            hit_tp = np.where(window_lows <= target_price)[0]
            hit_sl = np.where(window_highs >= stop_price)[0]
            
            first_tp = hit_tp[0] if len(hit_tp) > 0 else 999
            first_sl = hit_sl[0] if len(hit_sl) > 0 else 999
            
            if first_tp < first_sl: wins += 1
            total += 1

    if total < 10: return -100, 0, 0
    
    winrate = (wins / total) * 100
    # Score: Winrate alto é bom, mas precisa de volume de trades
    score = winrate * np.log(total) 
    
    return score, winrate, total

def worker_compra(args):
    params, data_dict = args
    total_score = 0
    total_wr = 0
    count = 0
    for sym, df in data_dict.items():
        s, w, t = simulate_trades(df, params, 'BUY')
        if t > 0:
            total_score += s
            total_wr += w
            count += 1
    
    avg_score = total_score / count if count > 0 else -999
    return avg_score, params

def worker_venda(args):
    params, data_dict = args
    total_score = 0
    total_wr = 0
    count = 0
    for sym, df in data_dict.items():
        s, w, t = simulate_trades(df, params, 'SELL')
        if t > 0:
            total_score += s
            total_wr += w
            count += 1
    
    avg_score = total_score / count if count > 0 else -999
    return avg_score, params


def otimizar():
    mt5.initialize()
    logger.info("Coletando dados...")
    # Limita a 10 ativos para a otimização ser mais rápida
    dados = get_data(config.CANDIDATOS_BASE[:10], days=45) 
    mt5.shutdown()
    
    if not dados:
        logger.error("Sem dados para otimizar.")
        return

    # Filtra as combinações onde EMA_FAST < EMA_SLOW
    combs = [p for p in itertools.product(EMA_FAST, EMA_SLOW, RSI_LEVEL, MOMENTUM) if p[0] < p[1]]
    
    # ------------------------------------------------------------------
    # Otimização de COMPRA
    # ------------------------------------------------------------------
    logger.info(f"Otimizando COMPRA ({len(combs)} combs)...")
    
    best_score_buy = -9999
    best_params_buy = None
    
    for i, p in enumerate(combs):
        if i % 10 == 0: print(f"Processando COMPRA {i}/{len(combs)}...", end='\r')
        score, _ = worker_compra((p, dados))
        if score > best_score_buy:
            best_score_buy = score
            best_params_buy = p
            
    print(f"\nMelhor COMPRA: Score {best_score_buy:.2f} | Params: {best_params_buy}")
    
    # Salvar resultados de Compra
    res_compra = {
        "score": best_score_buy,
        "ema_fast": best_params_buy[0], "ema_slow": best_params_buy[1],
        "rsi_max": best_params_buy[2], "momentum_min": best_params_buy[3],
        "data": datetime.now().strftime("%Y-%m-%d")
    }
    with open(config.PARAMETROS_FILE_COMPRA, 'w') as f:
        json.dump(res_compra, f)
        
    # ------------------------------------------------------------------
    # Otimização de VENDA
    # ------------------------------------------------------------------
    logger.info(f"Otimizando VENDA ({len(combs)} combs)...")
    
    best_score_sell = -9999
    best_params_sell = None
    
    for i, p in enumerate(combs):
        if i % 10 == 0: print(f"Processando VENDA {i}/{len(combs)}...", end='\r')
        score, _ = worker_venda((p, dados)) 
        if score > best_score_sell:
            best_score_sell = score
            best_params_sell = p
            
    print(f"\nMelhor VENDA: Score {best_score_sell:.2f} | Params: {best_params_sell}")
    
    # Salvar resultados de Venda
    res_venda = {
        "score": best_score_sell,
        "ema_fast": best_params_sell[0], "ema_slow": best_params_sell[1],
        # RSI_LEVEL para Venda é o limite MÍNIMO (rsi > rsi_lim)
        "rsi_min": best_params_sell[2], 
        # MOMENTUM para Venda é o limite MÍNIMO (mom < -mom_min)
        "momentum_max_neg": -best_params_sell[3], 
        "data": datetime.now().strftime("%Y-%m-%d")
    }
    with open(config.PARAMETROS_FILE_VENDA, 'w') as f: 
        json.dump(res_venda, f)

if __name__ == "__main__":
    otimizar()