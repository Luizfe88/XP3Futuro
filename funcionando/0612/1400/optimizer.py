# optimizer_pro_2026.py – OTIMIZADOR INSTITUCIONAL REALISTA (WALK-FORWARD)
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import pandas_ta as ta
import json
import time
import itertools
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
import config
from utils import logger, VERDE, RESET, ROXO, AZUL # Importando AZUL para as logs

# =================================================================
# PARÂMETROS DE BUSCA – EXPANDIDOS E REALISTAS
# =================================================================
EMA_FAST = [6, 8, 10, 12, 14, 16, 18, 21, 24]
EMA_SLOW = [18, 21, 26, 30, 35, 40, 45, 50, 60]
RSI_LEVEL = [60, 65, 68, 70, 72, 75]
MOMENTUM_MIN = [0.0, 0.1, 0.3, 0.5, 0.8] 
ADX_MIN = [18, 20, 22, 25, 28]
SL_MULT = [1.5, 2.0, 2.5, 3.0] # Adicionado aqui, mas não usado no worker para simplificar a busca principal
TP_MULT = [1.0, 1.5, 2.0, 2.5] # Adicionado aqui

def get_historical_data(symbol, start_date, end_date):
    """Obtém dados históricos para o backtest."""
    # Usando TIMEFRAME_D1 para simular otimização baseada em candles diários, mais rápido.
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_D1, start_date, end_date)
    df = pd.DataFrame(rates) if rates else pd.DataFrame()
    if not df.empty:
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
    return df

def calculate_sharpe(returns):
    """Calcula Sharpe Ratio (Anualizado)."""
    if returns.empty or returns.std() == 0:
        return 0.0
    daily_returns = returns.mean()
    daily_std = returns.std()
    annualized_sharpe = daily_returns / daily_std * np.sqrt(252)
    return annualized_sharpe

def backtest_strategy(df, params, side):
    """
    Backtest simulado. Retorna Sharpe Ratio do período de 30 dias (Proxy para Walk-Forward).
    """
    if df.empty: return -1.0, 0.0
    
    # Simulação da PnL (substituir pela sua lógica real)
    # Assumimos que o sinal é baseado em médias (EMA_FAST e EMA_SLOW)
    df.ta.ema(length=params[0], append=True, col_names=(f"EMA_{params[0]}",))
    df.ta.ema(length=params[1], append=True, col_names=(f"EMA_{params[1]}",))
    
    df['Signal'] = 0
    if side == "COMPRA":
        df.loc[df[f"EMA_{params[0]}"] > df[f"EMA_{params[1]}"], 'Signal'] = 1
    elif side == "VENDA":
        df.loc[df[f"EMA_{params[0]}"] < df[f"EMA_{params[1]}"], 'Signal'] = -1

    df['Returns'] = df['close'].pct_change() * df['Signal'].shift(1)
    
    # Score baseado no Sharpe dos ÚLTIMOS 30 DIAS (Walk-Forward Target)
    returns_30d = df['Returns'].iloc[-30:].dropna() 
    sharpe = calculate_sharpe(returns_30d)
    
    return sharpe, df['Returns'].sum() 

def worker_task(task):
    """Função worker para processamento paralelo."""
    symbol, start_date, end_date, params, side = task
    df = get_historical_data(symbol, start_date, end_date)
    
    score, _ = backtest_strategy(df, params, side)
    return score, params

def save_walk_forward_result(regime_name, best_params_set):
    """Salva os resultados do otimizador em um histórico (PONTO 7)."""
    
    history = []
    if os.path.exists(config.OPTIMIZER_HISTORY_FILE):
        with open(config.OPTIMIZER_HISTORY_FILE, 'r') as f:
            try: history = json.load(f)
            except json.JSONDecodeError: pass
            
    history.append(best_params_set)
    
    # Agrupa por regime e mantém os 8 mais recentes
    unique_regimes = {}
    for h in history:
        if h['regime'] not in unique_regimes:
            unique_regimes[h['regime']] = []
        unique_regimes[h['regime']].append(h)

    final_history = []
    for reg, items in unique_regimes.items():
        # Ordena por data (mais recente primeiro)
        items.sort(key=lambda x: datetime.strptime(x['data_otimizacao'], "%Y-%m-%d %H:%M"), reverse=True)
        final_history.extend(items[:8]) # Mantém os 8 mais recentes (PONTO 7)

    with open(config.OPTIMIZER_HISTORY_FILE, 'w') as f:
        json.dump(final_history, f, indent=2)
        logger.info(f"{AZUL}WF-Optimization History Salvo. Total de entradas: {len(final_history)}{RESET}")


def run_optimization(symbol="PETR4", regimes_to_optimize=["BULL"], side="COMPRA"):
    """
    Executa otimização Walk-Forward Rolling (PONTO 7).
    """
    logger.info(f"{ROXO}Iniciando Otimização WALK-FORWARD para {symbol}...{RESET}")
    
    END_DATE = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    WINDOW_DAYS = 120 # Rolling 120 dias (PONTO 7)
    START_DATE = END_DATE - timedelta(days=WINDOW_DAYS)

    combinacoes = list(itertools.product(EMA_FAST, EMA_SLOW, RSI_LEVEL, MOMENTUM_MIN, ADX_MIN))
    
    for regime_name in regimes_to_optimize:
        best_score = -100.0
        best_params = None
        tasks = []

        logger.info(f"Otimizando Regime: {regime_name.upper()} ({len(combinacoes)} combinações)")

        tasks = [(symbol, START_DATE, END_DATE, params, side) for params in combinacoes]

        with ProcessPoolExecutor(max_workers=os.cpu_count()-1) as executor:
            futures = {executor.submit(worker_task, task): task for task in tasks}

            for i, future in enumerate(as_completed(futures), 1):
                score, params = future.result()
                if score > best_score:
                    best_score = score
                    best_params = params
                    logger.info(f"{VERDE}NOVO MELHOR {regime_name.upper()}: Score {score:.3f} → {params}{RESET}")

                if i % 1000 == 0:
                    print(f"  → Processado {i}/{len(combinacoes)} | Melhor atual: {best_score:.3f}", end='\r')

        if best_params:
            result = {
                "regime": regime_name,
                "side": side,
                "score": round(best_score, 3), 
                "ema_fast": best_params[0],
                "ema_slow": best_params[1],
                "rsi_level": best_params[2],
                "momentum_min": best_params[3],
                "adx_min": best_params[4],
                # Assume que os multiplicadores SL/TP são fixos ou otimizados separadamente
                "sl_atr_mult": config.DEFAULT_PARAMS.get("sl_atr_mult", 2.0), 
                "tp_mult": config.DEFAULT_PARAMS.get("tp_mult", 1.5), 
                "data_otimizacao": datetime.now().strftime("%Y-%m-%d %H:%M")
            }
            
            save_walk_forward_result(regime_name, result)
            
            file_map = {
                "STRONG_BULL": config.PARAMS_STRONG_BULL, "BULL": config.PARAMS_BULL,
                "SIDEWAYS": config.PARAMS_SIDEWAYS, "BEAR": config.PARAMS_BEAR,
                "CRISIS": config.PARAMS_CRISIS
            }
            filename = file_map.get(regime_name) 
            if filename:
                 with open(filename, 'w') as f:
                    json.dump(result, f, indent=2)
                    logger.info(f"Parâmetros do regime {regime_name.upper()} atualizados: {filename}")


if __name__ == '__main__':
    if utils.check_mt5_connection():
        optimization_list = [
            ("PETR4", "STRONG_BULL", "COMPRA"),
            ("PETR4", "BULL", "COMPRA"),
            ("PETR4", "SIDEWAYS", "COMPRA"), 
            ("PETR4", "BEAR", "VENDA"),
            ("PETR4", "CRISIS", "NEUTRO")
        ]
        
        for symbol, regime, side in optimization_list:
            run_optimization(symbol=symbol, regimes_to_optimize=[regime], side=side)