import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["XP3_DISABLE_CONTRACT_RESOLVE"] = "1"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("KMP_WARNINGS", "0")
import functools
print = functools.partial(print, flush=True)
print("[DEBUG] Importando os...", flush=True); import os; print("[DEBUG] os importado.", flush=True)
print("[DEBUG] Importando json...", flush=True); import json; print("[DEBUG] json importado.", flush=True)
print("[DEBUG] Importando time...", flush=True); import time; print("[DEBUG] time importado.", flush=True)
print("[DEBUG] Importando logging...", flush=True); import logging; print("[DEBUG] logging importado.", flush=True)
logger = logging.getLogger(__name__)
print("[DEBUG] Importando sys...", flush=True); import sys; print("[DEBUG] sys importado.", flush=True)
print("[DEBUG] Importando io...", flush=True); import io; print("[DEBUG] io importado.", flush=True)
print("[DEBUG] Importando pathlib.Path...", flush=True); from pathlib import Path; print("[DEBUG] pathlib.Path importado.", flush=True)
print("[DEBUG] Importando typing...", flush=True); from typing import List, Dict, Any, Optional, Set; print("[DEBUG] typing importado.", flush=True)
print("[DEBUG] Importando concurrent.futures...", flush=True); from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError; print("[DEBUG] concurrent.futures importado.", flush=True)
print("[DEBUG] Importando dataclasses...", flush=True); from dataclasses import dataclass; print("[DEBUG] dataclasses importado.", flush=True)
print("[DEBUG] Importando datetime...", flush=True); from datetime import datetime, timedelta, timezone; print("[DEBUG] datetime importado.", flush=True)
print("[DEBUG] Importando collections...", flush=True); from collections import defaultdict, Counter; print("[DEBUG] collections importado.", flush=True)
print("[DEBUG] Importando requests...", flush=True); import requests; print("[DEBUG] requests importado.", flush=True)
print("[DEBUG] Importando config...", flush=True)
import config
print("[DEBUG] config importado.", flush=True)
print("[DEBUG] Importando numpy...", flush=True); import numpy as np; print("[DEBUG] numpy importado.", flush=True)
print("[DEBUG] Importando pandas...", flush=True); import pandas as pd; print("[DEBUG] pandas importado.", flush=True)
print("[DEBUG] Importando tqdm...", flush=True); from tqdm import tqdm; print("[DEBUG] tqdm importado.", flush=True)
print("[DEBUG] Importando ThreadPoolExecutor...", flush=True); from concurrent.futures import ThreadPoolExecutor; print("[DEBUG] ThreadPoolExecutor importado.", flush=True)
print("[DEBUG] Importando ta...", flush=True); import ta; print("[DEBUG] ta importado.", flush=True)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    print("[DEBUG] Encoding UTF-8 forçado no Windows")
os.environ["PYTHONUNBUFFERED"] = "1"
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True, write_through=True)
except Exception:
    pass
try:
    print("[DEBUG] Importando MetaTrader5...", flush=True)
    import MetaTrader5 as mt5
    print("[DEBUG] MetaTrader5 importado.", flush=True)
except Exception:
    mt5 = None
    print("[DEBUG] MetaTrader5 indisponível.", flush=True)
try:
    print("[DEBUG] Importando polygon.RESTClient...", flush=True)
    from polygon import RESTClient
    print("[DEBUG] polygon.RESTClient importado.", flush=True)
except Exception:
    RESTClient = None
    print("[DEBUG] polygon.RESTClient indisponível.", flush=True)
try:
    print("[DEBUG] Importando utils...", flush=True)
    import utils
    print("[DEBUG] utils importado.", flush=True)
except Exception:
    utils = None
print("[DEBUG] Importando tenacity...", flush=True); from tenacity import retry, stop_after_attempt, wait_fixed; print("[DEBUG] tenacity importado.", flush=True)
def _safe_import_minimize(timeout_seconds: float = 3.0):
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as _TErr
    def _do():
        from scipy.optimize import minimize
        return minimize
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_do)
        try:
            return fut.result(timeout=timeout_seconds)
        except _TErr:
            return None
        except Exception:
            return None

try:
    print("[DEBUG] Importando backfill...", flush=True)
    from backfill import ensure_history
    print("[DEBUG] backfill importado.", flush=True)
except Exception:
    ensure_history = None
    print("[DEBUG] backfill indisponível.", flush=True)
def atomic_write_json(path: str, data: Any, indent: Optional[int] = None) -> None:
    d = os.path.dirname(path) or "."
    tmp = os.path.join(d, f".tmp_{os.path.basename(path)}")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            import json as _json
            _json.dump(data, f, ensure_ascii=False, indent=indent)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass

# ============================================
# 🎯 IMPORTAÇÕES PARA MERCADO FUTURO
# ============================================
try:
    print("[DEBUG] Importando futures_core...", flush=True)
    import futures_core
    import config_futures
    FUTURES_MODE = True
    print("[DEBUG] ✅ Modo FUTUROS ativado (futures_core detectado)", flush=True)
except ImportError:
    FUTURES_MODE = False
    futures_core = None
    config_futures = None
    print("[DEBUG] ⚠️ futures_core não encontrado - modo ações ativo", flush=True)

try:
    from advanced_metrics_futures import (
        calculate_all_advanced_metrics,
        format_metrics_report,
        MIN_TRADES_REQUIRED,
        AdvancedMetrics
    )
    ADVANCED_METRICS_ENABLED = True
except Exception:
    ADVANCED_METRICS_ENABLED = False
    MIN_TRADES_REQUIRED = 20

def get_active_futures_symbols() -> List[str]:
    """Retorna lista de séries contínuas ($N) para cada base de futuros."""
    if not FUTURES_MODE:
        logger.error("❌ Modo futuros não disponível")
        return []
    
    futures_bases = list(config_futures.FUTURES_CONFIGS.keys())
    
    symbols = []
    for base in futures_bases:
        sym = base if "$N" in base else f"{base}$N"
        symbols.append(sym)
    return symbols


def normalize_futures_symbol(symbol: str) -> str:
    s = (symbol or "").upper().strip()
    try:
        import os as _os
        if _os.getenv("XP3_DISABLE_CONTRACT_RESOLVE", "0") == "1":
            return s
    except Exception:
        pass
    if "$N" in s:
        try:
            if mt5.symbol_select(s, True):
                return s
        except Exception:
            pass
        base = s.replace("$N", "")
        candidates = [s, base, f"{base}#", f"{base}!", f"{base}_C"]
        for name in candidates:
            try:
                if mt5.symbol_select(name, True):
                    logger.info(f"[MT5] Found futures symbol: {name} for input: {symbol}")
                    return name
            except Exception:
                pass
        try:
            syms = mt5.symbols_get(group=f"*{base}*") or []
        except Exception:
            syms = []
        if syms:
            try:
                sorted_syms = sorted(syms, key=lambda x: getattr(x, "volume", 0), reverse=True)
                logger.info(f"[MT5] Found futures symbol from group search: {sorted_syms[0].name} for input: {symbol}")
                return sorted_syms[0].name
            except Exception:
                pass
    return s

def load_futures_data_for_optimizer(symbol: str, bars: int, timeframe: str) -> Optional[pd.DataFrame]:
    if not FUTURES_MODE:
        logger.error(f"❌ {symbol}: Modo futuros não disponível")
        return None
    if FUTURES_MODE and mt5 is not None:
        try:
            logger.info(f"[{symbol}] Tentando carregar do MT5...")
            if not ensure_mt5_connection():
                raise ConnectionError("MT5 indisponível para futuros")
            try:
                if "$N" in (symbol or "") and os.getenv("XP3_FORCE_CONTINUOUS", "0") == "1":
                    import futures_core
                    tf_map_fc = {
                        "M5": mt5.TIMEFRAME_M5,
                        "M15": mt5.TIMEFRAME_M15,
                        "H1": mt5.TIMEFRAME_H1,
                        "D1": mt5.TIMEFRAME_D1
                    }
                    tfc = tf_map_fc.get(timeframe, mt5.TIMEFRAME_M15)
                    mgr = futures_core.get_manager()
                    base = (symbol or "").upper().strip().replace("$N", "")
                    dfc = mgr.concatenate_history(base, bars=max(bars, 2000), timeframe=tfc)
                    if is_valid_dataframe(dfc):
                        logger.info(f"✅ [{symbol}] Série contínua (concat) carregada: {len(dfc)} barras")
                        dfc = dfc.sort_index().tail(bars)
                        return dfc
            except Exception as e:
                logger.warning(f"[{symbol}] Falha em série contínua (concat): {e}")
            tf_map = {
                "M5": mt5.TIMEFRAME_M5,
                "M15": mt5.TIMEFRAME_M15,
                "H1": mt5.TIMEFRAME_H1,
                "D1": mt5.TIMEFRAME_D1
            }
            tf = tf_map.get(timeframe, mt5.TIMEFRAME_M15)
            try:
                import os as _os
                if _os.getenv("XP3_DISABLE_CONTRACT_RESOLVE", "0") == "1":
                    symbol_mt5 = (symbol or "").upper().strip()
                else:
                    symbol_mt5 = normalize_futures_symbol(symbol)
            except Exception:
                symbol_mt5 = normalize_futures_symbol(symbol)
            if not mt5.symbol_select(symbol_mt5, True):
                logger.warning(f"[{symbol}] Não foi possível ativar no Market Watch")
            rates = mt5.copy_rates_from_pos(symbol_mt5, tf, 0, max(bars, 2000))
            if rates is None:
                raise ValueError(f"MT5 retornou None para {symbol}")
            if len(rates) == 0:
                raise ValueError(f"MT5 retornou array vazio para {symbol}")
            if len(rates) < bars:
                try:
                    utc_to = datetime.utcnow()
                    days_back = int(bars / 28) + 60
                    utc_from = utc_to - timedelta(days=days_back)
                    alt = mt5.copy_rates_range(symbol_mt5, tf, utc_from, utc_to)
                    if alt is not None and len(alt) > len(rates):
                        rates = alt
                        logger.info(f"[{symbol}] copy_rates_range aumentou para {len(rates)} barras")
                    else:
                        logger.warning(f"[{symbol}] copy_rates_range não aumentou: {len(rates)} barras")
                except Exception as e:
                    logger.warning(f"[{symbol}] Falha em copy_rates_range: {e}")
            if len(rates) < bars * 0.5:
                logger.warning(f"[{symbol}] MT5 retornou apenas {len(rates)}/{bars} barras solicitadas")
            df = pd.DataFrame(rates)
            required_cols = ['time', 'open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Colunas faltando: {missing_cols}")
            df['time'] = pd.to_datetime(df['time'], unit='s', errors='coerce')
            if df['time'].isna().any():
                logger.warning(f"[{symbol}] Algumas timestamps inválidas foram removidas")
                df = df.dropna(subset=['time'])
            df.set_index('time', inplace=True)
            if 'volume' not in df.columns:
                if 'real_volume' in df.columns:
                    df['volume'] = df['real_volume'].astype(float)
                elif 'tick_volume' in df.columns:
                    df['volume'] = df['tick_volume'].astype(float)
                else:
                    logger.warning(f"[{symbol}] Volume não disponível, usando zeros")
                    df['volume'] = 0.0
            try:
                if utils and hasattr(utils, 'filter_trading_hours'):
                    base = symbol[:3] if len(symbol) >= 3 else symbol
                    df = utils.filter_trading_hours(df, base)
            except Exception as e:
                logger.warning(f"[{symbol}] Falha ao filtrar horário: {e}")
            df = df.sort_index().tail(bars)
            try:
                logger.info(f"[DADOS] {symbol}: {len(df)} barras MT5 pós-processamento | {df.index[0]} → {df.index[-1]} | TF={timeframe}")
            except Exception:
                pass
            if df.isna().sum().sum() > len(df) * 0.1:
                logger.warning(f"[{symbol}] Muitos NaNs detectados ({df.isna().sum().sum()} de {df.size})")
            logger.info(f"✅ [{symbol}] {len(df)} barras carregadas do MT5")
            return df
        except ConnectionError as e:
            logger.error(f"❌ [{symbol}] MT5 Connection Error: {e}")
        except ValueError as e:
            logger.error(f"❌ [{symbol}] MT5 Data Validation Error: {e}")
        except Exception as e:
            logger.error(f"❌ [{symbol}] MT5 Unexpected Error: {e}")
    else:
        logger.warning(f"[{symbol}] FUTURES_MODE={FUTURES_MODE}, mt5={'disponível' if mt5 else 'None'}")
    if RESTClient is not None:
        try:
            logger.info(f"[{symbol}] Tentando carregar da API Polygon...")
            polygon_symbol = symbol.replace("$N", "").upper()
            api_key = os.getenv("POLYGON_API_KEY", "")
            if not api_key:
                raise ValueError("POLYGON_API_KEY não configurada")
            client = RESTClient(api_key)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=bars // 28 + 30)
            multiplier_map = {"M5": (5, "minute"), "M15": (15, "minute"), "H1": (1, "hour")}
            multiplier, span = multiplier_map.get(timeframe, (15, "minute"))
            aggs = []
            for a in client.list_aggs(
                ticker=f"X:{polygon_symbol}",
                multiplier=multiplier,
                timespan=span,
                from_=start_date.strftime("%Y-%m-%d"),
                to=end_date.strftime("%Y-%m-%d"),
                limit=50000
            ):
                aggs.append(a)
            if not aggs:
                raise ValueError("Polygon retornou dados vazios")
            df = pd.DataFrame([{
                'time': pd.to_datetime(a.timestamp, unit='ms'),
                'open': a.open,
                'high': a.high,
                'low': a.low,
                'close': a.close,
                'volume': a.volume
            } for a in aggs])
            df.set_index('time', inplace=True)
            df = df.sort_index().tail(bars)
            logger.info(f"✅ [{symbol}] {len(df)} barras carregadas da Polygon")
            return df
        except ValueError as e:
            logger.error(f"❌ [{symbol}] Polygon Config Error: {e}")
        except Exception as e:
            logger.error(f"❌ [{symbol}] Polygon Error: {e}")
    cache_dir = Path("data_cache")
    cache_file = cache_dir / f"{symbol}_{timeframe}_{bars}.parquet"
    if cache_file.exists():
        try:
            logger.info(f"[{symbol}] Tentando carregar do cache local...")
            df = pd.read_parquet(cache_file)
            file_age_days = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days
            if file_age_days > 7:
                logger.warning(f"[{symbol}] Cache com {file_age_days} dias (recomendado: < 7)")
            logger.info(f"✅ [{symbol}] {len(df)} barras carregadas do cache (idade: {file_age_days}d)")
            return df
        except Exception as e:
            logger.error(f"❌ [{symbol}] Cache Error: {e}")
    logger.error(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║  FALHA CRÍTICA - DADOS INDISPONÍVEIS                         ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Símbolo:     {symbol:50s} ║
    ║  Timeframe:   {timeframe:50s} ║
    ║  Barras:      {bars:50d} ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Tentativas:                                                 ║
    ║    [X] MT5              (falhou ou indisponível)             ║
    ║    [X] Polygon API      (falhou ou indisponível)             ║
    ║    [X] Cache Local      (não existe ou expirado)             ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Ações Recomendadas:                                         ║
    ║  1. Verifique conexão do MT5                                 ║
    ║  2. Configure POLYGON_API_KEY                                ║
    ║  3. Execute backfill manual: python backfill.py {symbol:10s}  ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    return None


def safe_call_with_timeout(fn, timeout_seconds: float, *args, **kwargs):
    ex = ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(fn, *args, **kwargs)
    try:
        res = fut.result(timeout=timeout_seconds)
        try:
            ex.shutdown(cancel_futures=True, wait=False)
        except TypeError:
            ex.shutdown(wait=False)
        return True, res
    except TimeoutError:
        try:
            fut.cancel()
            ex.shutdown(cancel_futures=True, wait=False)
        except TypeError:
            ex.shutdown(wait=False)
        return False, {"error": "timeout"}
    except Exception as e:
        try:
            ex.shutdown(cancel_futures=True, wait=False)
        except TypeError:
            ex.shutdown(wait=False)
        return False, {"error": str(e)}

def is_valid_dataframe(df: Optional[pd.DataFrame], min_rows: int = 100) -> bool:
    """Valida se o DataFrame tem dados suficientes e é válido"""
    if df is None: return False
    if not isinstance(df, pd.DataFrame): return False
    if df.empty: return False
    if len(df) < min_rows: return False
    required_cols = {'open', 'high', 'low', 'close'}
    if not required_cols.issubset(set(df.columns)): return False
    return True

# ========================================
# 🔬 MODO DEBUG AVANÇADO
# ========================================
if __name__ == "__main__" and os.getenv("XP3_DEBUG", "0") == "1":
    os.environ["XP3_DISABLE_ML"] = "1"
    os.environ["XP3_RELAX_VOLATILITY"] = "1"
    os.environ["XP3_FORCE_ML_DIAG"] = "1"
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║            MODO DEBUG ATIVADO                                ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  TODOS OS FILTROS DESATIVADOS                                ║
    ║  - ML threshold: 0.50 (neutro)                               ║
    ║  - Volatilidade relaxada                                      ║
    ║  - Diagnóstico ML forçado                                    ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

# ========================================
# 📈 CONFIGURAÇÕES GLOBAIS
# ========================================
SANDBOX_MODE = os.getenv("XP3_SANDBOX", "0") == "1"
if SANDBOX_MODE:
    print("⏳ MODO SANDBOX ATIVADO - Execução em ambiente de teste.")

# ========================================
# 📦 CACHING
# ========================================
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
CACHE_TTL = timedelta(hours=int(os.getenv("XP3_CACHE_TTL_HOURS", "6")))

def is_cache_valid(path: Path) -> bool:
    if not path.exists(): return False
    try:
        mod_time = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        return (datetime.now(timezone.utc) - mod_time) < CACHE_TTL
    except Exception:
        return False

def load_from_cache(key: str) -> Optional[Any]:
    cache_file = CACHE_DIR / f"{key}.json"
    if is_cache_valid(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def save_to_cache(key: str, data: Any) -> None:
    cache_file = CACHE_DIR / f"{key}.json"
    try:
        atomic_write_json(str(cache_file), data)
    except Exception:
        pass

# ========================================
# 🌐 CONEXÕES EXTERNAS
# ========================================
_mt5_connected = False
def ensure_mt5_connection() -> bool:
    global _mt5_connected
    if _mt5_connected: return True
    if not mt5: return False
    try:
        if not mt5.initialize():
            logger.error(f"❌ MT5 initialize() falhou, código: {mt5.last_error()}")
            return False
        _mt5_connected = True
        return True
    except Exception as e:
        logger.error(f"❌ Exceção ao inicializar MT5: {e}")
        return False

_polygon_client = None
def get_polygon_client():
    global _polygon_client
    if _polygon_client: return _polygon_client
    if not RESTClient: return None
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key: return None
    try:
        _polygon_client = RESTClient(api_key)
        return _polygon_client
    except Exception:
        return None

# ========================================
# 🛠️ FUNÇÕES DE DADOS
# ========================================
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def _polygon_aggs_df(ticker: str, timespan: str, multiplier: int, from_dt: str, to_dt: str, limit: int, api_key: str) -> Optional[pd.DataFrame]:
    client = RESTClient(api_key)
    aggs = list(client.list_aggs(ticker, multiplier, timespan, from_dt, to_dt, limit=limit))
    if not aggs: return None
    df = pd.DataFrame(aggs)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.rename(columns={'timestamp': 'time', 'v': 'volume', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close'})
    df.set_index('time', inplace=True)
    return df

def _polygon_ticker(sym: str) -> str:
    return sym.replace(".SA", "")

def _yahoo_history(sym: str, interval: str = "1d", period: Optional[str] = None, start: Optional[str] = None, end: Optional[str] = None) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
        t = yf.Ticker(sym)
        df = t.history(period=period, interval=interval, start=start, end=end, auto_adjust=True, back_adjust=False)
        if df.empty: return None
        df.index = df.index.tz_convert('America/Sao_Paulo').tz_localize(None)
        df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}, inplace=True)
        return df
    except Exception:
        return None

def calculate_adx(high, low, close, period=14):
    adx_indicator = ta.trend.ADXIndicator(high=pd.Series(high), low=pd.Series(low), close=pd.Series(close), window=period)
    adx_values = adx_indicator.adx()
    atr_indicator = ta.volatility.AverageTrueRange(high=pd.Series(high), low=pd.Series(low), close=pd.Series(close), window=period)
    atr_values = atr_indicator.average_true_range()
    return adx_values.values, atr_values.values

def get_b3_market_hours():
    return {
        'pre_open_start': '09:30',
        'open': '10:00',
        'close': '17:55',
        'post_mkt_close': '18:15'
    }

def get_ibov_constituents(use_cache: bool = True) -> List[str]:
    cache_key = "ibov_constituents"
    if use_cache:
        cached_data = load_from_cache(cache_key)
        if cached_data:
            return cached_data
    
    try:
        url = "https://sistemaswebb3-listados.b3.com.br/indexPage/day/IBOV?language=pt-br"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        symbols = [f"{item['cod']}.SA" for item in data.get('results', [])]
        if symbols:
            save_to_cache(cache_key, symbols)
        return symbols
    except Exception as e:
        logger.error(f"Falha ao buscar constituintes do IBOV: {e}")
        # Fallback para uma lista estática em caso de falha
        return getattr(config, "IBOV_FALLBACK_LIST", [])


def get_sp500_constituents(use_cache: bool = True) -> List[str]:
    cache_key = "sp500_constituents"
    if use_cache:
        cached_data = load_from_cache(cache_key)
        if cached_data:
            return cached_data
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        df = pd.read_html(url, header=0)[0]
        symbols = df['Symbol'].tolist()
        if symbols:
            save_to_cache(cache_key, symbols)
        return symbols
    except Exception as e:
        logger.error(f"Falha ao buscar constituintes do S&P 500: {e}")
        return []

def get_b3_stocks(min_liquidity_usd: float = 1_000_000, use_cache: bool = True) -> List[str]:
    cache_key = f"b3_stocks_{min_liquidity_usd}"
    if use_cache:
        cached_data = load_from_cache(cache_key)
        if cached_data:
            return cached_data

    try:
        # Usando uma API pública para listar ações da B3
        url = "https://brapi.dev/api/quote/list"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        # Filtrando por liquidez (volume * preço)
        # A API da Brapi não fornece volume diretamente na lista, então vamos pegar as mais conhecidas
        # e depois o check_liquidez fará o trabalho pesado.
        stocks = [s['stock'] for s in data.get('stocks', []) if s.get('stock', '').endswith(('3', '4', '11'))]
        
        # Adicionando IBOV para garantir os principais
        stocks.extend(get_ibov_constituents())
        
        # Removendo duplicatas
        stocks = sorted(list(set(stocks)))
        
        # Adicionando sufixo .SA para compatibilidade com yfinance
        stocks_sa = [f"{s}.SA" for s in stocks if not s.endswith(".SA")]

        if stocks_sa:
            save_to_cache(cache_key, stocks_sa)
        
        return stocks_sa

    except Exception as e:
        logger.error(f"Falha ao buscar lista de ações da B3: {e}")
        return getattr(config, "B3_FALLBACK_LIST", [])


def check_liquidity_and_volatility(sym: str, period_days: int = 90, min_avg_vol: float = 1_000_000) -> (bool, str, dict):
    REJECT_SKIP = set()
    if sym in REJECT_SKIP:
        return False, "REJEITADO_CACHE", {}
    try:
        today = datetime.now()
        start = today - timedelta(days=period_days)
        api_key = os.getenv("POLYGON_API_KEY")
        df_d = None
        if mt5 and ensure_mt5_connection():
            try:
                mt5.symbol_select(sym, True)
                rates_d = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_D1, 0, period_days)
                if rates_d is not None and len(rates_d) > 0:
                    dfd = pd.DataFrame(rates_d)
                    dfd['time'] = pd.to_datetime(dfd['time'], unit='s')
                    dfd.set_index('time', inplace=True)
                    if 'real_volume' in dfd.columns and dfd['real_volume'].sum() > 0:
                        dfd = dfd.rename(columns={'real_volume': 'volume'})
                    elif 'tick_volume' in dfd.columns:
                        dfd = dfd.rename(columns={'tick_volume': 'volume'})
                    df_d = dfd[['open', 'high', 'low', 'close', 'volume']]
            except Exception:
                df_d = None
        if (df_d is None) or (not isinstance(df_d, pd.DataFrame)) or (df_d.empty):
            try:
                df_d = _polygon_aggs_df(_polygon_ticker(sym), "day", 1, start, today, period_days, api_key)
            except Exception:
                df_d = None
        if (df_d is None) or (not isinstance(df_d, pd.DataFrame)) or (df_d.empty):
            ok, res = safe_call_with_timeout(_yahoo_history, 2, sym, interval="1d", period="365d")
            df_a = res if ok else None
            if df_a is None or df_a.empty:
                REJECT_SKIP.add(sym)
                return False, "Sem dados diários", {}
            df_d = df_a
        avg_vol_shares = df_d['volume'].mean()
        avg_price = df_d['close'].mean()
        avg_fin_vol = avg_vol_shares * avg_price
        elite_blue = getattr(config, "ELITE_BLUE_CHIPS", []) if config else []
        MIN_FINANCEIRO = 100_000_000 if sym in elite_blue else 10_000_000
        if SANDBOX_MODE:
            MIN_FINANCEIRO = 0
        is_liquid = avg_fin_vol >= MIN_FINANCEIRO
        reason = "" if is_liquid else f"Baixa Liquidez"
        df_m15 = None
        if mt5 and ensure_mt5_connection():
            try:
                mt5.symbol_select(sym, True)
                rates15 = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M15, 0, 2000)
                if rates15 is not None and len(rates15) > 0:
                    dfm15 = pd.DataFrame(rates15)
                    dfm15['time'] = pd.to_datetime(dfm15['time'], unit='s')
                    dfm15.set_index('time', inplace=True)
                    if 'real_volume' in dfm15.columns and dfm15['real_volume'].sum() > 0:
                        dfm15 = dfm15.rename(columns={'real_volume': 'volume'})
                    elif 'tick_volume' in dfm15.columns:
                        dfm15 = dfm15.rename(columns={'tick_volume': 'volume'})
                    df_m15 = dfm15[['open', 'high', 'low', 'close', 'volume']]
            except Exception:
                df_m15 = None
        if (df_m15 is None) or (not isinstance(df_m15, pd.DataFrame)) or (df_m15.empty) or (len(df_m15) < 100):
            try:
                df_m15 = _polygon_aggs_df(_polygon_ticker(sym), "minute", 15, start, today, 5000, api_key)
            except Exception:
                df_m15 = None
        adx_threshold = 20

        if (df_m15 is None) or (not isinstance(df_m15, pd.DataFrame)) or (df_m15.empty) or (len(df_m15) < 100):
            is_fut = False
            if config_futures:
                 is_fut = sym.startswith(tuple(config_futures.FUTURES_CONFIGS.keys()))
            
            if not is_fut:
                import logging as _logging
                try:
                    _logging.getLogger("yfinance").setLevel(_logging.CRITICAL)
                except Exception:
                    pass
                ok2, res2 = safe_call_with_timeout(_yahoo_history, 2, sym, interval="15m", period="60d")
                df_m15 = res2 if ok2 else None
            else:
                 # Se for futuro e falhou MT5/Polygon, não tenta Yahoo
                 df_m15 = None

            if df_m15 is None:
                REJECT_SKIP.add(sym)
        if (df_m15 is not None) and isinstance(df_m15, pd.DataFrame) and (not df_m15.empty) and (len(df_m15) > 100):
            _, atr_vals = calculate_adx(df_m15["high"].values, df_m15["low"].values, df_m15["close"].values)
            avg_atr_pct = float(np.mean(atr_vals[-50:])) / max(float(np.mean(df_m15["close"].values[-50:])), 1e-9)
            adx_threshold = 18 if avg_atr_pct < 0.005 else 22
        return is_liquid, reason, {"avg_fin": avg_fin_vol, "adx_threshold": adx_threshold}
    except Exception as e:
        logger.error(f"❌ Erro no check_liquidez para {sym}: {e}")
        return False, f"ERRO_CHECK: {str(e)}", {}
# --- CÓDIGO DUPLICADO REMOVIDO PARA LIMPEZA ---
# Otimizador Principal consolidado no final do arquivo.

def scheduler():
    try:
        import schedule
    except Exception as e:
        print(f"[WARN] Scheduler indisponível: {e}. Rodando uma vez.")
        run_optimizer()
        return

    schedule.every().sunday.at("22:00").do(run_optimizer)
    while True:
        schedule.run_pending()
        time.sleep(30)


def _load_real_trade_overlay(lookback_days: int = 60, min_trades: int = 10) -> dict:
    try:
        db_path = "xp3_trades.db"
        if not os.path.exists(db_path):
            return {}

        import sqlite3
        cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(
            """
            SELECT
                symbol,
                COUNT(*) AS total,
                SUM(CASE WHEN pnl_money > 0 THEN 1 ELSE 0 END) AS wins,
                SUM(pnl_money) AS pnl
            FROM trades
            WHERE exit_price IS NOT NULL AND date(timestamp) >= date(?)
            GROUP BY symbol
            """,
            conn,
            params=(cutoff,)
        )
        conn.close()

        df = df[df['total'] >= min_trades]
        if df.empty:
            return {}

        df['win_rate'] = df['wins'] / df['total']
        
        # Normalização do PnL para um score entre -1 e 1
        # Usando uma transformação simples para o score, pode ser melhorado
        pnl_median = df['pnl'].median()
        pnl_std = df['pnl'].std()
        if pnl_std > 0:
            df['pnl_score'] = (df['pnl'] - pnl_median) / pnl_std
            df['pnl_score'] = np.clip(df['pnl_score'], -3, 3) / 3 # Clip e normaliza para [-1, 1]
        else:
            df['pnl_score'] = 0

        # Score final: combina win_rate e pnl_score
        # Peso maior para win_rate
        df['final_score'] = 0.7 * df['win_rate'] + 0.3 * df['pnl_score']
        
        overlay = {}
        for _, row in df.iterrows():
            overlay[row['symbol']] = {
                'score': row['final_score'],
                'win_rate': row['win_rate'],
                'trades': row['total']
            }
        return overlay

    except Exception as e:
        logger.warning(f"Falha ao carregar overlay de trades reais: {e}")
        return {}


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona indicadores técnicos necessários para as estratégias."""
    try:
        # Real volume (compatibilidade com Yahoo Finance e MT5)
        if 'volume' in df.columns:
            df['real_volume'] = df['volume']
        elif 'real_volume' not in df.columns:
            df['real_volume'] = 0.0
            
        # Volume MA (20 períodos)
        if 'real_volume' in df.columns:
            df['volume_ma'] = df['real_volume'].rolling(window=20).mean()
        else:
            df['volume_ma'] = 0.0
            
        # VWAP (calculado manualmente)
        if all(col in df.columns for col in ['high', 'low', 'close', 'real_volume']):
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            df['vwap'] = (typical_price * df['real_volume']).cumsum() / df['real_volume'].cumsum()
        else:
            df['vwap'] = df['close'] if 'close' in df.columns else 0.0
            
        # EMAs
        if 'close' in df.columns:
            df['ema_short'] = ta.trend.ema_indicator(df['close'], window=9)
            df['ema_long'] = ta.trend.ema_indicator(df['close'], window=21)
        else:
            df['ema_short'] = 0.0
            df['ema_long'] = 0.0
            
        # RSI
        if 'close' in df.columns:
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            df['rsi_2'] = ta.momentum.rsi(df['close'], window=2)
        else:
            df['rsi'] = 50.0
            df['rsi_2'] = 50.0
            
        # ADX
        if all(col in df.columns for col in ['high', 'low', 'close']):
            df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        else:
            df['adx'] = 25.0
            
        # SAR (Parabolic Stop and Reverse) - implementação simplificada
        if all(col in df.columns for col in ['high', 'low']):
            try:
                high = df['high'].values
                low = df['low'].values
                n = len(high)
                sar = np.zeros(n)
                
                # Inicialização
                af = 0.02  # Acceleration factor
                af_max = 0.2
                uptrend = True
                ep = high[0]  # Extreme point
                sar[0] = low[0]
                
                for i in range(1, n):
                    prev_sar = sar[i-1] + af * (ep - sar[i-1])
                    
                    if uptrend:
                        # Em tendência de alta
                        sar[i] = min(prev_sar, low[i-1], low[i-2] if i >= 2 else low[i-1])
                        if high[i] > ep:
                            ep = high[i]
                            af = min(af + 0.02, af_max)
                        if low[i] < sar[i]:
                            uptrend = False
                            sar[i] = ep
                            ep = low[i]
                            af = 0.02
                    else:
                        # Em tendência de baixa
                        sar[i] = max(prev_sar, high[i-1], high[i-2] if i >= 2 else high[i-1])
                        if low[i] < ep:
                            ep = low[i]
                            af = min(af + 0.02, af_max)
                        if high[i] > sar[i]:
                            uptrend = True
                            sar[i] = ep
                            ep = high[i]
                            af = 0.02
                
                df['sar'] = sar
            except:
                df['sar'] = df['close'] if 'close' in df.columns else 0.0
        else:
            df['sar'] = df['close'] if 'close' in df.columns else 0.0
            
        # ATR
        if all(col in df.columns for col in ['high', 'low', 'close']):
            df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        else:
            df['atr'] = 0.01
            
        # Momentum
        if 'close' in df.columns:
            df['momentum'] = ta.momentum.roc(df['close'], window=10)
        else:
            df['momentum'] = 0.0
            
        logger.info(f"✅ Indicadores técnicos adicionados: {len(df)} barras")
        return df
        
    except Exception as e:
        logger.error(f"❌ Erro ao adicionar indicadores técnicos: {e}")
        # Retorna o dataframe original mesmo com erro
        return df


def load_data_with_retry(symbol: str, bars: int, timeframe: str, is_future: bool) -> Optional[pd.DataFrame]:
    """Tenta carregar dados de várias fontes com retentativas."""
    
    # 1. Futuros: MT5 ou Polygon
    if is_future:
        try:
            df = load_futures_data_for_optimizer(symbol, bars, timeframe)
            if is_valid_dataframe(df, min_rows=bars*0.8):
                df = add_technical_indicators(df)
                return df
        except Exception as e:
            logger.warning(f"[{symbol}] Falha no load_futures_data_for_optimizer: {e}")

    # 2. Ações: MT5
    if not is_future and mt5 and ensure_mt5_connection():
        try:
            # ... (lógica de carregamento MT5 para ações)
            pass
        except Exception as e:
            logger.warning(f"[{symbol}] Falha no MT5 para ações: {e}")

    # 3. Ações: Polygon (se a chave estiver disponível)
    if not is_future and get_polygon_client():
        try:
            # ... (lógica de carregamento Polygon para ações)
            pass
        except Exception as e:
            logger.warning(f"[{symbol}] Falha no Polygon para ações: {e}")

    # 4. Ações: Yahoo Finance (como último recurso)
    if not is_future:
        try:
            logger.info(f"[{symbol}] Tentando Yahoo Finance como fallback...")
            df = _yahoo_history(symbol, interval="15m", period="60d") # Ajustar timeframe
            if is_valid_dataframe(df):
                df = add_technical_indicators(df)
                return df
        except Exception as e:
            logger.warning(f"[{symbol}] Falha no Yahoo Finance: {e}")
    
    logger.error(f"[{symbol}] DADOS INDISPONÍVEIS após todas as tentativas.")
    return None


def _ensure_non_generic(params: dict, symbol: str) -> dict:
    """Ajusta os parâmetros para o ativo específico, se necessário."""
    
    # Se for WIN$N, usar ranges específicos para VOLATILITY_BREAKOUT
    if "WIN" in symbol.upper() and params.get("strategy") == "VOLATILITY_BREAKOUT":
        p_copy = params.copy()
        
        # Exemplo de ajuste de range para o TP
        if "tp_mult" in p_copy:
            if isinstance(p_copy["tp_mult"], list) and len(p_copy["tp_mult"]) == 2:
                # Se for uma lista [min, max], ajusta para um range mais conservador
                p_copy["tp_mult"] = [max(1.5, p_copy["tp_mult"][0]), min(5.0, p_copy["tp_mult"][1])]
            else:
                # Se for um valor fixo, garante que está dentro de um limite
                p_copy["tp_mult"] = np.clip(p_copy["tp_mult"], 1.5, 5.0)

        # Exemplo de ajuste de range para o SL
        if "sl_atr_multiplier" in p_copy:
             if isinstance(p_copy["sl_atr_multiplier"], list) and len(p_copy["sl_atr_multiplier"]) == 2:
                p_copy["sl_atr_multiplier"] = [max(1.0, p_copy["sl_atr_multiplier"][0]), min(4.0, p_copy["sl_atr_multiplier"][1])]
             else:
                p_copy["sl_atr_multiplier"] = np.clip(p_copy["sl_atr_multiplier"], 1.0, 4.0)

        return p_copy

    return params

# ============================================
# 🧠 LÓGICA DE OTIMIZAÇÃO (WORKER)
# ============================================
@dataclass
class WorkerResult:
    symbol: str
    strategy_name: str
    params: Dict[str, Any]
    metrics: Dict[str, Any]
    exception: Optional[str] = None

def worker_wfo(symbol: str, strategy_name: str, params_config: Dict[str, Any], bars: int, timeframe: str, max_evals: int, metric_to_optimize: str) -> WorkerResult:
    """
    Função executada em um processo separado para otimizar uma estratégia para um único ativo.
    """
    pid = os.getpid()
    logger.info(f"[{pid}] Iniciando worker para {symbol} | {strategy_name} | {max_evals} evals")
    
    try:
        # 1. Carregar dados
        is_future = False
        if config_futures:
            is_future = symbol.startswith(tuple(config_futures.FUTURES_CONFIGS.keys()))
        
        df = load_data_with_retry(symbol, bars, timeframe, is_future)

        if not is_valid_dataframe(df, min_rows=bars * 0.8):
            raise ValueError(f"Dados insuficientes ou inválidos para {symbol} após todas as tentativas.")

        # 2. Ajustar parâmetros específicos do ativo (ex: WIN$N)
        params_config = _ensure_non_generic(params_config, symbol)

        from optimizer_optuna import run_optimization
        min_trades_required = int(os.getenv("OPT_MIN_TRADES", "10"))
        attempts = 0
        current_evals = max_evals
        best_params, best_metrics = {}, {}
        while True:
            best_params, best_metrics = run_optimization(
                strategy_name=strategy_name,
                params_config=params_config,
                metric_to_optimize=metric_to_optimize,
                data=df,
                symbol=symbol,
                asset_type='future' if is_future else 'stock',
                n_trials=current_evals,
                optimization_type='wfo',
                wfo_splits=5,
                wfo_train_size=0.7,
                wfo_test_size=0.3
            )
            trades_found = int(best_metrics.get("total_trades", 0) or 0)
            if trades_found >= min_trades_required:
                break
            attempts += 1
            if attempts >= 5:
                break
            current_evals = min(current_evals * 2, 1000)

        if not best_params:
            raise ValueError("Otimização não encontrou parâmetros válidos.")

        return WorkerResult(
            symbol=symbol,
            strategy_name=strategy_name,
            params=best_params,
            metrics=best_metrics
        )

    except Exception as e:
        logger.error(f"[{pid}] ❌ Erro no worker para {symbol}: {e}", exc_info=False)
        # Salvar o traceback para depuração
        import traceback
        err_msg = f"Exception: {e}\nTraceback: {traceback.format_exc()}"
        return WorkerResult(symbol=symbol, strategy_name=strategy_name, params={}, metrics={}, exception=err_msg)


# ============================================
#  orchestrator
# ============================================
def run_optimizer(symbols: Optional[List[str]] = None, strategies: Optional[List[str]] = None, max_evals: int = 100, bars: int = 2000, timeframe: str = "M15", workers: int = 4):
    start_time = time.time()
    
    # Carregar configurações de estratégias
    try:
        with open("strategies.json", "r", encoding="utf-8") as f:
            all_strategies_config = json.load(f)
    except Exception as e:
        logger.error(f"CRÍTICO: Não foi possível carregar 'strategies.json': {e}")
        return

    # Filtrar estratégias a serem otimizadas
    target_strategies = strategies or list(all_strategies_config.keys())
    
    # Determinar os símbolos-alvo
    if not symbols:
        # Modo padrão: busca ativos com liquidez
        logger.info("Nenhum símbolo especificado. Buscando ativos com liquidez...")
        
        # Adiciona futuros se o modo estiver ativo
        active_symbols = get_active_futures_symbols() if FUTURES_MODE else []
        
        # Adiciona ações da B3
        active_symbols.extend(get_b3_stocks(min_liquidity_usd=5_000_000))
        
        # Remove duplicatas
        symbols = sorted(list(set(active_symbols)))
        logger.info(f"Encontrados {len(symbols)} símbolos potenciais.")

    else:
        # Garante que os símbolos fornecidos estejam em uma lista
        symbols = [s.strip() for s in symbols if s.strip()]

    logger.info(f"🚀 Otimizador iniciado para {len(symbols)} símbolos e {len(target_strategies)} estratégias.")
    logger.info(f"Símbolos: {', '.join(symbols)}")
    logger.info(f"Estratégias: {', '.join(target_strategies)}")

    # Preparar tarefas para o pool de processos
    tasks = []
    for symbol in symbols:
        for strategy_name in target_strategies:
            if strategy_name in all_strategies_config:
                tasks.append({
                    "symbol": symbol,
                    "strategy_name": strategy_name,
                    "params_config": all_strategies_config[strategy_name]["params"],
                    "bars": bars,
                    "timeframe": timeframe,
                    "max_evals": max_evals,
                    "metric_to_optimize": "calmar" # Métrica padrão
                })

    # Executar otimização em paralelo
    results = []
    # Usar max_workers=1 para debug sequencial
    # workers = 1
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures_map = {executor.submit(worker_wfo, **task): task for task in tasks}
        
        with tqdm(total=len(tasks), desc="Otimizando Ativos") as pbar:
            for future in as_completed(futures_map):
                task_info = futures_map[future]
                try:
                    result = future.result()
                    if result and not result.exception:
                        results.append(result)
                        logger.info(f"✅ SUCESSO: {result.symbol} | {result.strategy_name}")
                    else:
                        logger.error(f"❌ FALHA: {task_info['symbol']} | {task_info['strategy_name']} | Erro: {result.exception[:200] if result else 'N/A'}")
                except Exception as e:
                    logger.error(f"❌ CRÍTICO: Exceção ao obter resultado para {task_info['symbol']}: {e}")
                pbar.update(1)

    # Processar e salvar resultados
    if not results:
        logger.warning("Nenhum sistema válido encontrado após a otimização.")
        return

    # Agrupar resultados por símbolo
    final_portfolio = defaultdict(list)
    for res in results:
        final_portfolio[res.symbol].append({
            "strategy": res.strategy_name,
            "params": res.params,
            "metrics": res.metrics
        })

    # Salvar o portfólio completo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("optimizer_output")
    output_dir.mkdir(exist_ok=True)
    
    full_portfolio_path = output_dir / f"full_portfolio_{timestamp}.json"
    atomic_write_json(str(full_portfolio_path), final_portfolio, indent=2)
    logger.info(f"Portfólio completo salvo em: {full_portfolio_path}")

    # Selecionar os "melhores" sistemas para um portfólio de elite
    # Critério simples: melhor 'calmar' por ativo
    elite_portfolio = {}
    min_trades_required = int(os.getenv("OPT_MIN_TRADES", "10"))
    for symbol, systems in final_portfolio.items():
        valid_systems = [s for s in systems if (s.get('metrics', {}).get('total_trades') or 0) >= min_trades_required]
        if not valid_systems:
            logger.warning(f"{symbol} sem sistema com pelo menos {min_trades_required} trades")
            continue
        best_system = max(valid_systems, key=lambda s: s['metrics'].get('calmar', -999))
        elite_portfolio[symbol] = best_system

    elite_path = output_dir / "elite_portfolio.json"
    atomic_write_json(str(elite_path), elite_portfolio, indent=2)
    logger.info(f"✅ Portfólio de elite salvo em: {elite_path}")

    # Gerar um relatório de diagnóstico
    report = []
    for symbol, system in elite_portfolio.items():
        metrics = system.get('metrics', {})
        report.append({
            "symbol": symbol,
            "strategy": system.get('strategy'),
            "calmar": metrics.get('calmar'),
            "win_rate": metrics.get('win_rate'),
            "trades": metrics.get('total_trades'),
            "avg_pnl": metrics.get('avg_trade_pnl'),
            "sharpe": metrics.get('sharpe_ratio')
        })
    
    report_df = pd.DataFrame(report).sort_values(by="calmar", ascending=False).set_index("symbol")
    report_path = output_dir / f"optimization_report_{timestamp}.csv"
    report_df.to_csv(str(report_path))
    logger.info(f"Relatório de otimização salvo em: {report_path}")

    end_time = time.time()
    logger.info(f"🏁 Otimização concluída em {((end_time - start_time) / 60):.2f} minutos.")


# ============================================
# CLI - Interface de Linha de Comando
# ============================================
def _setup_logging():
    """Configura o logging para arquivo e console."""
    global logger
    
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Usar UTC para logs de arquivo para consistência
    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
    log_file = log_dir / f"optimizer_{timestamp_utc}.log"

    # Configuração básica para o root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d [%(levelname)s] [%(module)s.%(funcName)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout) # Envia para o console
        ]
    )
    
    # Definir o logger global
    logger = logging.getLogger(__name__)
    
    # Silenciar loggers muito verbosos de bibliotecas de terceiros
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    
    logger.info(f"Logging configurado. Saída em console e no arquivo: {log_file}")


if __name__ == "__main__":
    import argparse
    
    _setup_logging()

    parser = argparse.ArgumentParser(description="XP3 - Otimizador de Estratégias de Trading")
    parser.add_argument("--symbols", type=str, help="Lista de símbolos para otimizar, separados por vírgula (ex: 'PETR4.SA,VALE3.SA,WIN$N')")
    parser.add_argument("--strategies", type=str, help="Lista de estratégias para otimizar, separadas por vírgula (ex: 'EMA_CROSS,VOLATILITY_BREAKOUT')")
    parser.add_argument("--maxevals", type=int, default=100, help="Número máximo de avaliações (trials) por ativo/estratégia no Optuna.")
    parser.add_argument("--bars", type=int, default=2000, help="Número de barras de dados históricos para carregar.")
    parser.add_argument("--timeframe", type=str, default="M15", help="Timeframe dos dados (ex: M5, M15, H1, D1).")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1), help="Número de processos paralelos para a otimização.")
    
    args = parser.parse_args()

    # Converter strings de argumentos em listas
    symbols_list = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None
    if symbols_list and FUTURES_MODE and config_futures:
        futures_configs = (config_futures.FUTURES_CONFIGS or {})
        root_to_continuous = {
            k.replace("$N", ""): k
            for k in futures_configs.keys()
            if isinstance(k, str) and k.endswith("$N")
        }
        symbols_list = [root_to_continuous.get(s, s) for s in symbols_list]
    strategies_list = [s.strip() for s in args.strategies.split(",")] if args.strategies else None

    try:
        run_optimizer(
            symbols=symbols_list,
            strategies=strategies_list,
            max_evals=args.maxevals,
            bars=args.bars,
            timeframe=args.timeframe,
            workers=args.workers
        )
    except Exception as e:
        logger.critical(f"Uma exceção não tratada encerrou o otimizador: {e}", exc_info=True)
        sys.exit(1)
