# yahoo_mt5_hybrid.py → OTIMIZAÇÃO TURBO USANDO DADOS DO MT5 (MELHOR DOS DOIS MUNDOS)
import pandas as pd
import numpy as np
import pandas_ta as ta
import MetaTrader5 as mt5
import json
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

VERDE = "\033[92m"
AZUL  = "\033[94m"
ROXO = "\033[95m"
RESET = "\033[0m"

# ==================== ATIVOS PROXY (OS MESMOS DO ORIGINAL) ====================
PROXY_SYMBOLS = ["VALE3", "PETR4", "ITUB4", "BBDC4", "WEGE3", "ABEV3"]

# ==================== PARÂMETROS DE OTIMIZAÇÃO ====================
EMA_FAST  = range(5, 30, 3)
EMA_SLOW  = range(25, 71, 5)
RSI_LEVEL = [60, 65, 70, 75]
MOMENTUM  = [0.4, 0.7, 1.0, 1.3, 1.6]
ADX_MIN   = [15, 20, 25, 30]

# Cache global
DATA_CACHE = {}

def get_data_mt5(symbol, days=500):
    """Pega dados do MT5 (D1) – muito mais rápido e confiável que yfinance"""
    if symbol in DATA_CACHE:
        return DATA_CACHE[symbol]

    # Pega até 1000 barras diárias (mais que suficiente)
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 1000)
    if rates is None or len(rates) < 200:
        logger.error(f"Sem dados suficientes para {symbol}")
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'tick_volume']]
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    df = df.sort_index()

    DATA_CACHE[symbol] = df
    logger.info(f"{symbol} → {len(df)} barras carregadas do MT5")
    return df

# ==================== BACKTEST (igual ao original) ====================
def backtest(df, params, side):
    if df.empty or len(df) < 100:
        return 0.0

    f, s, rsi_lv, mom, adx = params
    df = df.copy()

    df['ema_f'] = df['close'].ewm(span=f, adjust=False).mean()
    df['ema_s'] = df['close'].ewm(span=s, adjust=False).mean()
    df['rsi']   = ta.rsi(df['close'], length=14)
    adx_df      = ta.adx(df['high'], df['low'], df['close'], length=14)
    df = pd.concat([df, adx_df], axis=1)
    df['mom']   = df['close'].pct_change()

    df['signal'] = 0
    if side == "COMPRA":
        cond = (df['ema_f'] > df['ema_s']) & (df['rsi'] > rsi_lv) & (df['ADX_14'] > adx) & (df['mom'] > mom/100)
        df.loc[cond, 'signal'] = 1
    else:
        cond = (df['ema_f'] < df['ema_s']) & (df['rsi'] < (100 - rsi_lv)) & (df['ADX_14'] > adx) & (df['mom'] < -mom/100)
        df.loc[cond, 'signal'] = -1

    df['pos'] = df['signal'].shift(1).fillna(0)
    df['ret'] = df['close'].pct_change() * df['pos']
    trades = df['signal'].diff().abs() > 0
    df['custo'] = trades * (0.02 / df['close'] + 0.0005)
    df['net'] = df['ret'] - df['custo'].fillna(0)

    recent = df['net'].tail(60)
    if len(recent) < 10 or recent.std() == 0:
        return 0.0
    return (recent.mean() / recent.std()) * np.sqrt(252)

# ==================== OTIMIZAÇÃO (MESMO MOTOR TURBO) ====================
def otimizar(regime, lado):
    logger.info(f"{ROXO}=== {regime.upper()} | {lado} ==={RESET}")
    
    # Pré-carrega dados do MT5
    for s in PROXY_SYMBOLS:
        get_data_mt5(s)

    combos = [c for c in itertools.product(EMA_FAST, EMA_SLOW, RSI_LEVEL, MOMENTUM, ADX_MIN) if c[0] < c[1]]
    logger.info(f"Combinações válidas: {len(combos):,}")

    tasks = [(s, p, lado) for s in PROXY_SYMBOLS for p in combos]
    resultados = {}
    best_score = -999
    best_params = None

    with ProcessPoolExecutor(max_workers=os.cpu_count()-1) as pool:
        futures = {pool.submit(backtest, get_data_mt5(sym), params, lado): (sym, params) for sym, params, lado in tasks}
        
        for i, future in enumerate(as_completed(futures), 1):
            score = future.result()
            sym, params = futures[future]
            
            if params not in resultados:
                resultados[params] = []
            resultados[params].append(score)

            if len(resultados[params]) == len(PROXY_SYMBOLS):
                media = np.mean(resultados[params])
                if media > best_score:
                    best_score = media
                    best_params = params
                    logger.info(f"{AZUL}NOVO MELHOR → {regime.upper()}: {best_score:.3f} ← {params}{RESET}")
            
            if i % 500 == 0:
                print(f"   → {i}/{len(tasks)} processados | Melhor: {best_score:.3f}", end="\r")

    # Salva JSON
    res = {
        "regime": regime.lower(),
        "sharpe_medio": round(best_score, 3),
        "ema_fast": best_params[0],
        "ema_slow": best_params[1],
        "rsi_level": best_params[2],
        "momentum_min_pct": best_params[3],
        "adx_min": best_params[4],
        "data": datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    arq = f"params_{regime.lower()}.json"
    with open(arq, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    logger.info(f"{VERDE}FINALIZADO → {arq} (Sharpe: {best_score:.3f}){RESET}")

# ==================== EXECUÇÃO ====================
if __name__ == "__main__":
    if not mt5.initialize():
        logger.critical("ERRO: Não foi possível conectar ao MT5!")
        exit()

    regimes = [
        ("STRONG_BULL", "COMPRA"),
        ("BULL", "COMPRA"),
        ("SIDEWAYS", "COMPRA"),
        ("BEAR", "VENDA"),
        ("CRISIS", "VENDA")
    ]
    
    for r, l in regimes:
        otimizar(r, l)
        print("\n" + "="*80 + "\n")

    mt5.shutdown()
    print(f"{VERDE}OTIMIZAÇÃO HÍBRIDA CONCLUÍDA! Dados do MT5 + Motor Turbo = PERFEIÇÃO!{RESET}")