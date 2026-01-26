"""
OTIMIZADOR XP3 - WINDOWS SAFE
Versão sem emojis para evitar erros de encoding
"""
import os
import sys
import time
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional, Dict, Any, List

# ===========================
# FIX ENCODING
# ===========================
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ===========================
# LOGGING SEM EMOJIS
# ===========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("optimizer.log", encoding='utf-8')
    ]
)
logger = logging.getLogger("otimizador")

# ===========================
# IMPORTS
# ===========================
try:
    import config
    logger.info("[OK] Config carregado")
except Exception as e:
    logger.error(f"[ERRO] Config: {e}")
    sys.exit(1)

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except:
    MT5_AVAILABLE = False
    mt5 = None

# ===========================
# CONFIGURAÇÕES
# ===========================
SANDBOX_MODE = bool(int(os.getenv("XP3_SANDBOX", "1")))
OUTPUT_DIR = getattr(config, "OPTIMIZER_OUTPUT", "optimizer_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "xrE09LEWJYBZfQcV57pCvsw4aqkOiqbz")

logger.info(f"Modo: {'SANDBOX' if SANDBOX_MODE else 'PRODUCAO'}")
logger.info(f"MT5: {'Disponivel' if MT5_AVAILABLE else 'Indisponivel'}")

# ===========================
# MT5 COM TIMEOUT
# ===========================
def connect_mt5_safe() -> bool:
    """Conecta ao MT5 com timeout"""
    if not MT5_AVAILABLE:
        return False
    
    try:
        path = getattr(config, 'MT5_TERMINAL_PATH', None)
        
        if path:
            result = mt5.initialize(path=path, timeout=5000)
        else:
            result = mt5.initialize(timeout=5000)
        
        if result:
            info = mt5.terminal_info()
            if info and info.connected:
                logger.info(f"[OK] MT5 conectado: {info.company}")
                return True
        
        logger.warning("[AVISO] MT5 nao conectado")
        return False
    
    except Exception as e:
        logger.error(f"[ERRO] MT5: {e}")
        return False

# ===========================
# CARREGAMENTO DE DADOS
# ===========================
def load_data_polygon(symbol: str, days: int = 730) -> Optional[pd.DataFrame]:
    """Carrega dados via Polygon"""
    try:
        import requests
        
        end = datetime.now().date()
        start = (end - timedelta(days=days)).strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")
        
        ticker = f"{symbol}.SA" if not symbol.startswith("^") else symbol
        
        # Tenta 15 minutos primeiro
        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/"
            f"15/minute/{start}/{end_str}?limit=5000&apiKey={POLYGON_API_KEY}"
        )
        
        resp = requests.get(url, timeout=15)
        
        if resp.status_code != 200:
            # Fallback para diário
            url = (
                f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/"
                f"1/day/{start}/{end_str}?limit=5000&apiKey={POLYGON_API_KEY}"
            )
            resp = requests.get(url, timeout=15)
        
        if resp.status_code != 200:
            return None
        
        data = resp.json().get("results", [])
        if not data:
            return None
        
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['t'], unit='ms')
        df = df.set_index('time')
        df = df.rename(columns={
            'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'
        })
        
        return df[['open', 'high', 'low', 'close', 'volume']]
    
    except Exception as e:
        logger.debug(f"Polygon erro para {symbol}: {e}")
        return None

def load_data_yahoo(symbol: str) -> Optional[pd.DataFrame]:
    """Carrega dados via Yahoo"""
    try:
        import yfinance as yf
        
        ticker = f"{symbol}.SA" if not symbol.startswith("^") else symbol
        df = yf.Ticker(ticker).history(period="730d", interval="15m")
        
        if df is None or df.empty:
            # Fallback para diário
            df = yf.Ticker(ticker).history(period="730d", interval="1d")
        
        if df is None or df.empty:
            return None
        
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        
        return df
    
    except Exception as e:
        logger.debug(f"Yahoo erro para {symbol}: {e}")
        return None

def load_data(symbol: str) -> Optional[pd.DataFrame]:
    """Carrega dados com fallback"""
    logger.info(f"[DATA] Carregando {symbol}...")
    
    # Polygon
    df = load_data_polygon(symbol)
    if df is not None and len(df) >= 100:
        logger.info(f"  [OK] Polygon: {len(df)} barras")
        return df
    
    # Yahoo
    df = load_data_yahoo(symbol)
    if df is not None and len(df) >= 100:
        logger.info(f"  [OK] Yahoo: {len(df)} barras")
        return df
    
    logger.warning(f"  [ERRO] {symbol}: Sem dados")
    return None

# ===========================
# INDICADORES
# ===========================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula indicadores básicos"""
    df = df.copy()
    
    # EMAs
    df['ema_9'] = df['close'].ewm(span=9).mean()
    df['ema_21'] = df['close'].ewm(span=21).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['atr'] = ranges.max(axis=1).rolling(14).mean()
    
    return df.fillna(0)

# ===========================
# BACKTEST
# ===========================
def backtest_simple(df: pd.DataFrame) -> Dict[str, Any]:
    """Backtest básico"""
    if df is None or len(df) < 100:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'total_return': 0,
            'max_drawdown': 1.0,
            'calmar': 0
        }
    
    df = calculate_indicators(df)
    
    # Sinais
    buy_signal = (df['ema_9'] > df['ema_21']) & (df['rsi'] < 40)
    sell_signal = (df['ema_9'] < df['ema_21']) | (df['rsi'] > 70)
    
    # Simulação
    position = 0
    equity = [100000.0]
    trades = 0
    wins = 0
    entry_price = 0
    
    for i in range(1, len(df)):
        if position == 0 and buy_signal.iloc[i]:
            # Compra
            position = equity[-1]
            entry_price = df['close'].iloc[i]
            trades += 1
        
        elif position > 0 and sell_signal.iloc[i]:
            # Venda
            exit_price = df['close'].iloc[i]
            pnl = position * (exit_price / entry_price - 1)
            
            if pnl > 0:
                wins += 1
            
            equity.append(equity[-1] + pnl)
            position = 0
        
        else:
            # Mantém
            if position > 0:
                current_value = position * (df['close'].iloc[i] / entry_price)
                equity.append(equity[-1] + (current_value - position))
            else:
                equity.append(equity[-1])
    
    # Métricas
    equity = np.array(equity)
    total_return = (equity[-1] / equity[0]) - 1
    
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    max_dd = np.max(drawdown) if len(drawdown) > 0 else 1.0
    
    win_rate = wins / trades if trades > 0 else 0
    calmar = total_return / max_dd if max_dd > 0 else 0
    
    return {
        'total_trades': trades,
        'wins': wins,
        'win_rate': win_rate,
        'total_return': total_return,
        'max_drawdown': max_dd,
        'calmar': calmar,
        'equity_curve': equity.tolist()
    }

# ===========================
# LIQUIDEZ
# ===========================
def check_liquidity(symbol: str) -> tuple:
    """Verifica liquidez do ativo"""
    try:
        df = load_data_polygon(symbol, days=30)
        
        if df is None or len(df) < 10:
            return False, "SEM_DADOS", {}
        
        avg_volume = df['volume'].tail(20).mean()
        avg_price = df['close'].tail(20).mean()
        avg_financial = avg_volume * avg_price
        
        MIN_FINANCIAL = 10_000_000 if not SANDBOX_MODE else 0
        
        is_liquid = avg_financial >= MIN_FINANCIAL
        reason = "OK" if is_liquid else "BAIXA_LIQUIDEZ"
        
        return is_liquid, reason, {"avg_fin": avg_financial}
    
    except Exception as e:
        return False, str(e), {}

# ===========================
# OTIMIZADOR
# ===========================
def run_optimizer():
    """Otimizador principal"""
    print("\n" + "="*60)
    print("OTIMIZADOR XP3 - Windows Safe")
    print(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print("="*60 + "\n")
    
    # 1. Símbolos
    logger.info("[1/4] Selecionando ativos...")
    
    if SANDBOX_MODE:
        symbols = ['PETR4', 'VALE3', 'ITUB4', 'BBDC4', 'ABEV3']
    else:
        try:
            symbols = list(getattr(config, 'SECTOR_MAP', {}).keys())[:20]
        except:
            symbols = ['PETR4', 'VALE3', 'ITUB4']
    
    logger.info(f"   Testando {len(symbols)} ativos")
    
    # 2. Liquidez
    logger.info("[2/4] Verificando liquidez...")
    valid_symbols = []
    
    for sym in tqdm(symbols, desc="Liquidez"):
        is_ok, reason, metrics = check_liquidity(sym)
        
        if is_ok or SANDBOX_MODE:
            valid_symbols.append(sym)
    
    logger.info(f"   [OK] {len(valid_symbols)} ativos aprovados")
    
    # 3. Backtest
    logger.info("[3/4] Executando backtests...")
    results = {}
    
    for sym in tqdm(valid_symbols, desc="Backtest"):
        df = load_data(sym)
        
        if df is not None and len(df) >= 100:
            metrics = backtest_simple(df)
            results[sym] = metrics
            
            logger.info(
                f"   {sym}: Trades={metrics['total_trades']} "
                f"WR={metrics['win_rate']:.1%} "
                f"Ret={metrics['total_return']:.2%}"
            )
    
    # 4. Relatório
    logger.info("[4/4] Gerando relatorio...")
    
    if results:
        summary = pd.DataFrame(results).T
        summary = summary.sort_values('calmar', ascending=False)
        
        output_file = os.path.join(OUTPUT_DIR, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        summary.to_csv(output_file)
        
        logger.info(f"   [OK] Salvo em: {output_file}")
        
        # Top 5
        print("\n" + "="*60)
        print("TOP 5 ATIVOS (por Calmar)")
        print("="*60)
        print(summary.head(5)[['total_trades', 'win_rate', 'total_return', 'calmar']])
    
    print("\n" + "="*60)
    print("OTIMIZACAO CONCLUIDA")
    print("="*60 + "\n")

if __name__ == "__main__":
    try:
        run_optimizer()
    except KeyboardInterrupt:
        logger.warning("\n[AVISO] Interrompido")
    except Exception as e:
        logger.exception(f"[ERRO] {e}")
        sys.exit(1)