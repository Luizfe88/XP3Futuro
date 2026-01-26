# optimizer_yahoo_TURBO.py → VERSÃO FINAL RÁPIDA (com cache de dados)
import pandas as pd
import numpy as np
import pandas_ta as ta
import yfinance as yf
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
ROXO  = "\033[95m"
RESET = "\033[0m"

# ==================== CONFIGURAÇÃO ====================
PROXY_SYMBOLS = ["BBDC4.SA", "ITUB4.SA", "VALE3.SA", "PETR4.SA", "WEGE3.SA", "ABEV3.SA"]

EMA_FAST  = range(5, 30, 3)
EMA_SLOW  = range(25, 71, 5)
RSI_LEVEL = [60, 65, 70, 75]
MOMENTUM  = [0.4, 0.7, 1.0, 1.3, 1.6]
ADX_MIN   = [15, 20, 25, 30]

# Cache global de dados
DATA_CACHE = {}

def get_data_cached(symbol):
    if symbol in DATA_CACHE:
        return DATA_CACHE[symbol]
    
    for _ in range(3):
        try:
            df = yf.Ticker(symbol).history(period="500d", interval="1d", auto_adjust=True, actions=False)
            if not df.empty and len(df) > 200:
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                df.columns = ['open', 'high', 'low', 'close', 'volume']
                df = df.dropna()
                df.index = pd.to_datetime(df.index).tz_localize(None)
                DATA_CACHE[symbol] = df
                logger.info(f"{symbol} → {len(df)} dias (cache)")
                return df
        except:
            pass
    logger.error(f"Falha total: {symbol}")
    DATA_CACHE[symbol] = pd.DataFrame()
    return pd.DataFrame()

# ==================== BACKTEST ====================
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

# ==================== OTIMIZAÇÃO ====================
def otimizar(regime, lado):
    logger.info(f"{ROXO}=== {regime.upper()} | {lado} ==={RESET}")
    
    # Pré-carrega todos os dados (só 6 downloads!)
    for s in PROXY_SYMBOLS:
        get_data_cached(s)
    
    combos = [c for c in itertools.product(EMA_FAST, EMA_SLOW, RSI_LEVEL, MOMENTUM, ADX_MIN) if c[0] < c[1]]
    logger.info(f"Combinações válidas: {len(combos):,}")

    tasks = [(s, p, lado) for s in PROXY_SYMBOLS for p in combos]

    resultados = {}
    best_score = -999
    best_params = None

    with ProcessPoolExecutor(max_workers=os.cpu_count()-1) as pool:
        futures = {pool.submit(backtest, get_data_cached(sym), params, lado): (sym, params) for sym, params, lado in tasks}
        
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
                print(f"   → {i}/{len(tasks)} processados | Melhor atual: {best_score:.3f}", end="\r")

    # Salva
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

    print(f"{VERDE}OTIMIZAÇÃO TURBO CONCLUÍDA! 5 arquivos JSON gerados em tempo recorde!{RESET}")