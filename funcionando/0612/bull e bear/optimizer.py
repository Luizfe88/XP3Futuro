# optimizer_pro_2026.py – OTIMIZADOR INSTITUCIONAL REALISTA (ZERO LOOK-AHEAD)
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
from utils import logger, VERDE, RESET, ROXO

# =================================================================
# PARÂMETROS DE BUSCA – EXPANDIDOS E REALISTAS
# =================================================================
EMA_FAST = [6, 8, 10, 12, 14, 16, 18, 21, 24]
EMA_SLOW = [18, 21, 26, 30, 35, 40, 45, 50, 60]
RSI_LEVEL = [60, 65, 68, 70, 72, 75]
MOMENTUM_MIN = [0.1, 0.3, 0.5, 0.8, 1.0, 1.3, 1.6]
ADX_MIN = [18, 20, 22, 25, 28]

# Total: 9 × 9 × 6 × 7 × 5 = 17.010 combinações (perfeitamente viável com multiprocessing)

def get_historical_data(symbols, days=120):
    data_store = {}
    inicio = datetime.now() - timedelta(days=days)
    for s in symbols:
        rates = mt5.copy_rates_range(s, mt5.TIMEFRAME_M5, inicio, datetime.now())
        if rates is not None and len(rates) > 1000:
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            data_store[s] = df
    return data_store

def simulate_strategy_vectorized(df, params, side='BUY'):
    ema_f_len, ema_s_len, rsi_lim, mom_min, adx_min = params

    # --- Indicadores com LAG de 1 barra (zero look-ahead) ---
    df['ema_fast'] = ta.ema(df['close'], length=ema_f_len).shift(1)
    df['ema_slow'] = ta.ema(df['close'], length=ema_s_len).shift(1)
    df['rsi'] = ta.rsi(df['close'], length=14).shift(1)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14).shift(1)
    df['adx'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14'].shift(1)

    # Momentum D1 institucional (uma linha!)
    df['mom_d1'] = (df['close'].resample('1D').last()
                       .pct_change(fill_method=None)
                       .reindex(df.index, method='ffill')
                       .shift(1)
                       .fillna(0)) * 100

    # --- Sinal ---
    if side == 'BUY':
        signal = (df['ema_fast'] > df['ema_slow']) & \
                 (df['rsi'] < rsi_lim) & \
                 (df['mom_d1'] > mom_min) & \
                 (df['adx'] > adx_min)
    else:
        signal = (df['ema_fast'] < df['ema_slow']) & \
                 (df['rsi'] > rsi_lim) & \
                 (df['mom_d1'] < -mom_min) & \
                 (df['adx'] > adx_min)

    trades = df[signal].copy()
    if len(trades) < 15:
        return -999, 0, 0, 0, 0, 0

    # --- Simulação realista (48 barras = 4h) ---
    results = []
    for idx in trades.index:
        future = df.loc[idx:idx + timedelta(hours=4)]
        if len(future) < 10: continue

        entry = future['close'].iloc[0]
        atr_val = future['atr'].iloc[0]
        if pd.isna(atr_val) or atr_val <= 0: continue

        sl_dist = atr_val * 2.0
        tp_dist = sl_dist * 1.8  # RR 1:1.8

        if side == 'BUY':
            sl_price = entry - sl_dist
            tp_price = entry + tp_dist
            hit_sl = (future['low'] <= sl_price).any()
            hit_tp = (future['high'] >= tp_price).any()
            if hit_tp and (future['high'] >= tp_price).idxmin() < (future['low'] <= sl_price).idxmin() if hit_sl else True:
                results.append(tp_dist)
            else:
                results.append(-sl_dist)
        else:
            sl_price = entry + sl_dist
            tp_price = entry - tp_dist
            hit_sl = (future['high'] >= sl_price).any()
            hit_tp = (future['low'] <= tp_price).any()
            if hit_tp and (future['low'] <= tp_price).idxmin() < (future['high'] >= sl_price).idxmin() if hit_sl else True:
                results.append(tp_dist)
            else:
                results.append(-sl_dist)

    if len(results) < 15:
        return -999, 0, 0, 0, 0, 0

    wins = sum(1 for r in results if r > 0)
    total = len(results)
    winrate = wins / total
    payoff = np.mean([r for r in results if r > 0]) / abs(np.mean([r for r in results if r < 0])) if any(r < 0 for r in results) else 10
    expectancy = (winrate * payoff) - (1 - winrate)
    profit_factor = sum(r for r in results if r > 0) / abs(sum(r for r in results if r < 0)) if any(r < 0 for r in results) else 10

    equity = np.cumsum(results)
    peak = np.maximum.accumulate(equity)
    drawdown = peak - equity
    max_dd = drawdown.max()

    # Score institucional (quanto maior, melhor)
    score = expectancy * np.sqrt(total) * (profit_factor ** 1.5) / (1 + max_dd / abs(np.mean(results)))

    return score, winrate * 100, total, expectancy, profit_factor, max_dd

def worker_task(args):
    params, data_dict, side = args
    scores = []
    for sym, df in data_dict.items():
        score, wr, trades, exp, pf, dd = simulate_strategy_vectorized(df, params, side)
        if trades >= 15:
            scores.append(score)
        else:
            return -999, params  # penalidade forte

    if len(scores) < 5:
        return -999, params
    return np.mean(scores), params

def otimizar_por_regime():
    mt5.initialize()
    logger.info("Coletando 120 dias de dados M5 (alta qualidade)...")
    dados = get_historical_data(config.CANDIDATOS_BASE[:18], days=120)
    mt5.shutdown()

    if len(dados) < 10:
        logger.error("Dados insuficientes.")
        return

    combinacoes = list(itertools.product(EMA_FAST, EMA_SLOW, RSI_LEVEL, MOMENTUM_MIN, ADX_MIN))
    combinacoes = [c for c in combinacoes if c[0] < c[1]]  # EMA fast < slow

    regimes = [
        ("bull", "COMPRA"),
        ("bear", "VENDA"),
        ("sideways", "COMPRA")  # sideways usa mesmo que bull mas com ADX alto
    ]

    for regime_name, side in regimes:
        logger.info(f"OTIMIZANDO REGIME: {regime_name.upper()} ({side}) – {len(combinacoes)} combinações")

        best_score = -99999
        best_params = None

        tasks = [(params, dados, side) for params in combinacoes]

        with ProcessPoolExecutor(max_workers=os.cpu_count()-1) as executor:
            futures = {executor.submit(worker_task, task): task for task in tasks}

            for i, future in enumerate(as_completed(futures), 1):
                score, params = future.result()
                if score > best_score:
                    best_score = score
                    best_params = params
                    logger.info(f"{VERDE}NOVO MELHOR {regime_name.upper()}: Score {score:.3f} → {params}{RESET}")

                if i % 50 == 0:
                    print(f"  → Processado {i}/{len(combinacoes)} | Melhor atual: {best_score:.3f}")

        # Salvar
        filename = f"data/params_{regime_name}.json"
        result = {
            "regime": regime_name,
            "side": side,
            "score": round(best_score, 3),
            "ema_fast": best_params[0],
            "ema_slow": best_params[1],
            "rsi_level": best_params[2],
            "momentum_min": best_params[3],
            "adx_min": best_params[4],
            "data_otimizacao": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"{ROXO}REGIME {regime_name.upper()} OTIMIZADO → Score {best_score:.3f} | Params: {best_params}{RESET}")

    logger.info("OTIMIZAÇÃO INSTITUCIONAL CONCLUÍDA COM SUCESSO")

if __name__ == "__main__":
    otimizar_por_regime()