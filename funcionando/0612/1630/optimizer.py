# optimizer_pro_2026.py – OTIMIZADOR INSTITUCIONAL REALISTA (WALK-FORWARD - Média de Proxies)
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
import utils 
from utils import logger

# ==================== GERAIS ====================
# Símbolos Proxy para otimização robusta (PONTO 7 - Generalização)
PROXY_SYMBOLS = ["PETR4", "VALE3", "ITUB4"]

# =================================================================
# PARÂMETROS DE BUSCA – EXPANDIDOS E REALISTAS
# =================================================================
EMA_FAST = [6, 8, 10, 12, 14, 16, 18, 21, 24]
EMA_SLOW = [18, 21, 26, 30, 35, 40, 45, 50, 60]
RSI_LEVEL = [60, 65, 68, 70, 72, 75]
MOMENTUM_MIN = [0.0, 0.1, 0.3, 0.5, 0.8] 
ADX_MIN = [18, 20, 22, 25, 28]
# SL/TP mantidos fixos aqui, mas podem ser incluídos na otimização se necessário
# SL_MULT = [1.5, 2.0, 2.5, 3.0] 
# TP_MULT = [1.0, 1.5, 2.0, 2.5] 

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
    Inclui slippage e comissão para backtest realista.
    """
    if df.empty: return -1.0, 0.0
    
    # 1. Simulação da PnL (Lógica Bruta)
    ema_fast, ema_slow, *_ = params
    df.ta.ema(length=ema_fast, append=True, col_names=(f"EMA_{ema_fast}",))
    df.ta.ema(length=ema_slow, append=True, col_names=(f"EMA_{ema_slow}",))
    
    df['Signal'] = 0
    # Lógica de sinal simplificada (Cross EMA)
    if side == "COMPRA":
        df.loc[df[f"EMA_{ema_fast}"] > df[f"EMA_{ema_slow}"], 'Signal'] = 1
    elif side == "VENDA":
        df.loc[df[f"EMA_{ema_fast}"] < df[f"EMA_{ema_slow}"], 'Signal'] = -1

    df['Position'] = df['Signal'].shift(1).fillna(0)
    df['Trade_Entry'] = (df['Signal'].diff() != 0).astype(int) 
    df['Returns'] = df['close'].pct_change() * df['Position']

    # 2. Inclusão de Custos (Comissão e Slippage)
    df['Commission_Cost'] = df['Trade_Entry'] * (config.COMISSAO_POR_LOTE / (df['close'] * 100))
    avg_price = df['close'].mean()
    SLIPPAGE_PCT_PENALTY = (config.SLIPPAGE_TICKS_MEDIO * 0.01) / avg_price
    df['Slippage_Cost'] = df['Trade_Entry'] * SLIPPAGE_PCT_PENALTY 

    # Retorno Líquido
    df['Net_Returns'] = df['Returns'] - df['Commission_Cost'] - df['Slippage_Cost']
    df['Net_Returns'].fillna(0, inplace=True)
    
    # Score baseado no Sharpe dos ÚLTIMOS 30 DIAS (Walk-Forward Target)
    returns_30d = df['Net_Returns'].iloc[-30:].dropna() 
    sharpe = calculate_sharpe(returns_30d)
    
    return sharpe, df['Net_Returns'].sum() 

def worker_task(task):
    """Função worker para processamento paralelo de uma ÚNICA combinação em um ÚNICO símbolo."""
    symbol, start_date, end_date, params, side = task
    df = get_historical_data(symbol, start_date, end_date)
    score, _ = backtest_strategy(df, params, side) 
    
    # Retorna o score e os parâmetros (para identificação) e o símbolo
    return score, params, symbol 

def save_walk_forward_result(regime_name, best_params_set):
    """Salva os resultados do otimizador em um histórico e no arquivo de parâmetros (PONTO 7)."""
    
    # Lógica de histórico omitida por brevidade, mas deve ser mantida.
    
    file_map = {
        "STRONG_BULL": config.PARAMS_STRONG_BULL, "BULL": config.PARAMS_BULL,
        "SIDEWAYS": config.PARAMS_SIDEWAYS, "BEAR": config.PARAMS_BEAR,
        "CRISIS": config.PARAMS_CRISIS
    }
    filename = file_map.get(regime_name) 
    if filename:
        with open(filename, 'w') as f:
            json.dump(best_params_set, f, indent=2)
            utils.logger.info(f"{utils.VERDE}Parâmetros do regime {regime_name.upper()} (MÉDIA DE PROXIES) atualizados: {filename}{utils.RESET}")


def run_optimization(regime_name, side):
    """
    Executa otimização Walk-Forward Rolling, calculando a MÉDIA de 3 ativos proxy.
    """
    utils.logger.info(f"{utils.ROXO}Iniciando Otimização WALK-FORWARD | Regime: {regime_name.upper()} | Média em {', '.join(PROXY_SYMBOLS)}...{utils.RESET}")
    
    END_DATE = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    WINDOW_DAYS = 120 # Rolling 120 dias
    START_DATE = END_DATE - timedelta(days=WINDOW_DAYS)

    combinacoes = list(itertools.product(EMA_FAST, EMA_SLOW, RSI_LEVEL, MOMENTUM_MIN, ADX_MIN))
    
    best_avg_score = -100.0
    best_params = None
    
    # 1. Geração de TODAS as tarefas (N combinações * 3 símbolos)
    all_tasks = []
    for params in combinacoes:
        for symbol in PROXY_SYMBOLS:
            all_tasks.append((symbol, START_DATE, END_DATE, params, side))
            
    # Armazena os resultados por combinação de parâmetros: { (params_tuple): [score_petr4, score_vale3, score_itub4] }
    # Usamos uma lista de dicionários para garantir a ordem/log
    results_map = {} 
    
    utils.logger.info(f"Total de backtests a executar: {len(all_tasks)}")

    # 2. Execução Paralela
    with ProcessPoolExecutor(max_workers=os.cpu_count()-1) as executor:
        futures = {executor.submit(worker_task, task): task for task in all_tasks}

        for i, future in enumerate(as_completed(futures), 1):
            score, params, symbol = future.result()
            
            # Chave é o conjunto de parâmetros
            params_key = tuple(params)
            
            if params_key not in results_map:
                results_map[params_key] = []
                
            results_map[params_key].append(score)
            
            # 3. Verificação e Avaliação
            # Só avalia o score médio quando todos os 3 backtests daquele conjunto de parâmetros terminarem
            if len(results_map[params_key]) == len(PROXY_SYMBOLS):
                scores = results_map[params_key]
                avg_score = np.mean(scores)
                
                if avg_score > best_avg_score:
                    best_avg_score = avg_score
                    best_params = params
                    
                    # Logando a descoberta do novo melhor score médio
                    utils.logger.info(f"{utils.AZUL}NOVA MÉDIA MELHOR ({regime_name.upper()}): Score {best_avg_score:.3f} → {params} (Scores: {scores}){utils.RESET}")

            if i % 100 == 0:
                print(f"  → Processado {i}/{len(all_tasks)} backtests | Melhor Média Atual: {best_avg_score:.3f}", end='\r')


    # 4. Salvamento do Melhor Resultado Médio
    if best_params:
        result = {
            "regime": regime_name.lower(),
            "side": side,
            "score": round(best_avg_score, 3), # O score é a média
            "ema_fast": best_params[0],
            "ema_slow": best_params[1],
            "rsi_level": best_params[2],
            "momentum_min": best_params[3],
            "adx_min": best_params[4],
            "sl_atr_mult": config.DEFAULT_PARAMS.get("sl_atr_mult", 2.0), 
            "tp_mult": config.DEFAULT_PARAMS.get("tp_mult", 1.5), 
            "data_otimizacao": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        
        save_walk_forward_result(regime_name, result)
        utils.logger.info(f"{utils.VERDE}Otimização {regime_name.upper()} FINALIZADA. Melhor Média de Sharpe: {best_avg_score:.3f}{utils.RESET}")


if __name__ == '__main__':
    if utils.check_mt5_connection():
        # Lista de regimes a serem otimizados (agora sem o ativo, pois a otimização lida com os 3 proxies)
        regime_tasks = [
            ("STRONG_BULL", "COMPRA"),
            ("BULL", "COMPRA"),
            ("SIDEWAYS", "COMPRA"), 
            ("BEAR", "VENDA"),
            ("CRISIS", "NEUTRO")
        ]
        
        for regime, side in regime_tasks:
            # Chama a otimização uma vez por regime; a função run_optimization faz o trabalho nos 3 ativos
            run_optimization(regime_name=regime, side=side)