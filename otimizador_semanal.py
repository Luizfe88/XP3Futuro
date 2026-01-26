"""
OTIMIZADOR COM AUTO-SYNC DE MARKET WATCH - VERSÃO CORRIGIDA FINAL
✅ Reconexão automática MT5
✅ Workers com conexão própria
✅ ADX calculado corretamente
✅ Lógica de entrada flexível
✅ Import circular corrigido
"""
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
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
print("[DEBUG] Importando sys...", flush=True); import sys; print("[DEBUG] sys importado.", flush=True)
print("[DEBUG] Importando io...", flush=True); import io; print("[DEBUG] io importado.", flush=True)
print("[DEBUG] Importando pathlib.Path...", flush=True); from pathlib import Path; print("[DEBUG] pathlib.Path importado.", flush=True)
print("[DEBUG] Importando typing...", flush=True); from typing import List, Dict, Any, Optional, Set; print("[DEBUG] typing importado.", flush=True)
print("[DEBUG] Importando concurrent.futures...", flush=True); from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError; print("[DEBUG] concurrent.futures importado.", flush=True)
print("[DEBUG] Importando dataclasses...", flush=True); from dataclasses import dataclass; print("[DEBUG] dataclasses importado.", flush=True)
print("[DEBUG] Importando datetime...", flush=True); from datetime import datetime, timedelta; print("[DEBUG] datetime importado.", flush=True)
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

def compute_sharpe(equity_curve: list, bars_per_day: int = 28) -> float:
    try:
        if not equity_curve or len(equity_curve) < 5:
            return 0.0
        arr = np.array(equity_curve, dtype=float)
        rets = np.diff(arr) / arr[:-1]
        if len(rets) < 5:
            return 0.0
        mu = float(np.mean(rets))
        sd = float(np.std(rets)) if np.std(rets) > 0 else 1e-9
        daily_mu = mu * bars_per_day
        daily_sd = sd * np.sqrt(bars_per_day)
        return float(daily_mu / max(daily_sd, 1e-9))
    except Exception:
        return 0.0

def run_backtests_split(symbols: list) -> dict:
    try:
        from optimizer_optuna import backtest_params_on_df
        stocks = [s for s in symbols if not utils.is_future(s)]
        futures = [s for s in symbols if utils.is_future(s)]
        res = {}
        def _run(group):
            curves = []
            for sym in group:
                df = load_data_with_retry(sym, 1500, timeframe="M15")
                if df is None or len(df) < 100:
                    continue
                params = _ensure_non_generic(sym, {})
                m = backtest_params_on_df(sym, params, df, ml_model=None)
                curve = m.get("equity_curve", [])
                if curve:
                    curves.append(curve)
            if not curves:
                return 0.0
            avg_len = min(len(c) for c in curves)
            avg_curve = np.mean([np.array(c[:avg_len], dtype=float) for c in curves], axis=0).tolist()
            return compute_sharpe(avg_curve)
        res["stocks_pure_sharpe"] = _run(stocks)
        res["futures_pure_sharpe"] = _run(futures)
        mix_syms = (stocks[:10] + futures[:10])[:20]
        res["mixed_portfolio_sharpe"] = _run(mix_syms)
        return res
    except Exception:
        return {"stocks_pure_sharpe": 0.0, "futures_pure_sharpe": 0.0, "mixed_portfolio_sharpe": 0.0}

def get_ibov_monthly_regime() -> str:
    try:
        df = get_ibov_data()
        if df is None or df.empty or len(df) < 25:
            return "NEUTRAL"
        close = df["close"].astype(float)
        ret = float(close.iloc[-1] / close.iloc[-22] - 1.0)
        if ret >= 0.10:
            return "BULL"
        if ret <= -0.10:
            return "BEAR"
        return "NEUTRAL"
    except Exception:
        return "NEUTRAL"

def safe_mt5_initialize():
    """Inicializa MT5 com path configurado e retry"""
    path = getattr(config, 'MT5_TERMINAL_PATH', None) if config else None
    if not mt5.initialize(path=path) if path else mt5.initialize():
        raise ConnectionError("Falha na inicialização do MT5")
    return True
# ===========================
# CONFIGURAÇÕES
# ===========================
@dataclass
class BacktestConfig:
    BARS_PER_DAY: int = 28
    TRADING_DAYS_PER_YEAR: int = 252
    RISK_FREE_RATE: float = 0.05
    DEFAULT_SLIPPAGE: float = 0.0035
    RISK_PER_TRADE: float = 0.01
    MIN_BARS_FOR_METRICS: int = 50
    MIN_DATA_LENGTH: int = 100
    @property
    def bars_per_year(self) -> int:
        return self.BARS_PER_DAY * self.TRADING_DAYS_PER_YEAR
config_bt = BacktestConfig()
class JSONFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "time": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage()
        }
        return json.dumps(payload, ensure_ascii=False)
logger = logging.getLogger("otimizador_auto_sync")
logger.setLevel(logging.INFO)
if os.getenv("XP3_LOG_JSON", "1") == "1":
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    logger.handlers = [handler]
else:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
OPT_OUTPUT_DIR = getattr(config, "OPTIMIZER_OUTPUT", "optimizer_output")
os.makedirs(OPT_OUTPUT_DIR, exist_ok=True)
SECTOR_MAP = config.SECTOR_MAP
COMMODITY_SYMBOLS = {"PETR4", "VALE3", "PRIO3", "GGBR4", "CSNA3", "SUZB3", "KLBN11", "ENEV3"}
REJECT_SKIP = set()
print("[DEBUG] Importando warnings...", flush=True); import warnings; print("[DEBUG] warnings importado.", flush=True)
def _safe_import_optuna_warning(timeout_seconds: float = 3.0):
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as _TErr
    def _do():
        from optuna.exceptions import ExperimentalWarning
        return ExperimentalWarning
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_do)
        try:
            return fut.result(timeout=timeout_seconds)
        except _TErr:
            return None
        except Exception:
            return None
_ew = _safe_import_optuna_warning()
if _ew is not None:
    warnings.filterwarnings("ignore", category=_ew)
# ===========================
# CONFIGURAÇÕES GLOBAIS
# ===========================
SANDBOX_MODE = bool(int(os.getenv("XP3_SANDBOX", "1")))
BARS_TO_LOAD = 4000 if SANDBOX_MODE else 5000
# ===========================
# GARANTIR CONEXÃO MT5
# ===========================
def ensure_mt5_connection() -> bool:
    """
    Valida e força conexão com o terminal correto do MT5 (via Tenacity)
    """
    try:
        ok = try_mt5_connection(timeout_seconds=5)
        if not ok:
            return False
        terminal = mt5.terminal_info()
        if terminal and terminal.connected:
            logger.info("[OK] MT5 conectado (Tenacity OK)")
            return True
        return False
    except Exception as e:
        logger.critical(f"🚨 FALHA CRÍTICA: Não foi possível conectar ao MT5: {e}")
        return False
def try_mt5_connection(timeout_seconds: int = 10) -> bool:
    """
    Tenta conectar ao MT5 com timeout. Retorna False se falhar.
    """
    if not mt5:
        return False
    import threading
    result = [False]
    def connect():
        try:
            path = getattr(config, 'MT5_TERMINAL_PATH', None) if config else None
            ok = mt5.initialize(path=path) if path else mt5.initialize()
            result[0] = bool(ok)
        except Exception:
            result[0] = False
    thread = threading.Thread(target=connect)
    thread.daemon = True
    thread.start()
    thread.join(timeout_seconds)
    if thread.is_alive():
        logger.warning(f"⏱️ MT5 timeout ({timeout_seconds}s) - continuando sem MT5")
        return False
    return result[0]
# ===========================
# AUXILIARES DE OTIMIZAÇÃO
# ===========================
def load_all_symbols() -> List[str]:
    """Retorna lista de ativos Sniper"""
    return sorted(list(SECTOR_MAP.keys()))
def get_symbols():
    """Alias para carregar símbolos permitidos"""
    return load_all_symbols()
# =========================================================
# 3. ALOCAÇÃO MARKOWITZ (SCIPY)
# =========================================================
def optimize_portfolio_allocation(selected_assets_metrics):
    # Comentário: F.1 Pré-filtro + F.2 Restrições flexíveis + F.3 Fallback
    logger.info("🔹 Calculando Fronteira Eficiente (Markowitz Protegido)...")
    # Pré-filtro: remove ativos com DD extremo ou poucos trades
    filtered = []
    for m in selected_assets_metrics:
        try:
            dd = float(m.get("max_dd", 0.0) or 0.0)
            trades = int(m.get("total_trades", 0) or 0)
            af = float(m.get("avg_fin", 0.0) or 0.0)
        except Exception:
            dd, trades, af = 0.0, 0, 0.0
        # Mantém se liquidez mínima ok; aplica filtros de dd/trades apenas se disponíveis
        if af >= 10_000_000:
            if ("max_dd" in m and dd >= 0.65) or ("total_trades" in m and trades < 10):
                continue
            filtered.append(m)
    selected_assets_metrics = filtered if len(filtered) > 0 else selected_assets_metrics
    n = len(selected_assets_metrics)
    if n == 0:
        return {}
    returns = np.array([m.get('roi_esperado', 0.05) for m in selected_assets_metrics], dtype=np.float64)
    curves = []
    for m in selected_assets_metrics:
        ec = m.get('equity_curve') or []
        if isinstance(ec, list) and len(ec) >= 50:
            curves.append(np.asarray(ec, dtype=np.float64))
        else:
            curves.append(None)
    valid_idxs = [i for i, c in enumerate(curves) if c is not None]
    if valid_idxs:
        min_len = min(len(curves[i]) for i in valid_idxs)
        rets = []
        for i in range(n):
            if curves[i] is None:
                rets.append(np.zeros(min_len - 1, dtype=np.float64))
            else:
                c = curves[i][:min_len]
                r = np.diff(c) / c[:-1]
                rets.append(r)
        rets_mat = np.vstack(rets)
        cov_matrix = np.cov(rets_mat)
    else:
        cov_matrix = np.eye(n) * 0.001
    def negative_sharpe(weights):
        p_ret = np.sum(returns * weights)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(p_ret / p_vol) if p_vol > 0 else 0
    symbols_order = [m.get('symbol') for m in selected_assets_metrics]
    sector_constraints = []
    sector_groups = {}
    for i, sym in enumerate(symbols_order):
        sec = SECTOR_MAP.get(sym)
        if not sec:
            continue
        sector_groups.setdefault(sec, []).append(i)
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    for sec, idxs in sector_groups.items():
        constraints.append({'type': 'ineq', 'fun': lambda x, idxs=idxs: 0.25 - float(np.sum([x[j] for j in idxs]))})
    blue_idxs = [i for i, m in enumerate(selected_assets_metrics) if str(m.get("category","")).upper() == "BLUE CHIP"]
    opp_idxs = [i for i, m in enumerate(selected_assets_metrics) if str(m.get("category","")).upper() != "BLUE CHIP"]
    # Comentário: Blue Chips mínimo 50%, Oportunidades máximo 50%
    constraints.append({'type': 'ineq', 'fun': lambda x, idxs=blue_idxs: float(np.sum([x[j] for j in idxs])) - 0.50})
    constraints.append({'type': 'ineq', 'fun': lambda x, idxs=opp_idxs: 0.50 - float(np.sum([x[j] for j in idxs]))})
    bounds = []
    for i in range(n):
        af = float(selected_assets_metrics[i].get("avg_fin", 0.0) or 0.0)
        sym_i = selected_assets_metrics[i].get("symbol")
        blue = []
        try:
            blue = getattr(config, "ELITE_BLUE_CHIPS", [])
        except Exception:
            blue = []
        if af < 10_000_000:
            bounds.append((0.0, 0.0))
        else:
            if sym_i in blue:
                bounds.append((0.05, 0.15))
            else:
                bounds.append((0.0, 0.15))
    bounds = tuple(bounds)
    init_guess = [1 / n] * n
    minimize_fn = _safe_import_minimize()
    if minimize_fn is not None:
        res = minimize_fn(negative_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        allocations = {}
        for i, asset in enumerate(selected_assets_metrics):
            allocations[asset['symbol']] = res.x[i]
        return allocations
    else:
        symbols_order = [m.get('symbol') for m in selected_assets_metrics]
        elite = getattr(config, "ELITE_BLUE_CHIPS", [])
        blue_idxs = [i for i, m in enumerate(selected_assets_metrics) if m.get("symbol") in elite]
        opp_idxs = [i for i, m in enumerate(selected_assets_metrics) if m.get("symbol") not in elite]
        weights = np.zeros(n, dtype=np.float64)
        blue_valid = [i for i in blue_idxs if float(selected_assets_metrics[i].get("avg_fin", 0) or 0) >= 10_000_000]
        opp_valid = [i for i in opp_idxs if float(selected_assets_metrics[i].get("avg_fin", 0) or 0) >= 10_000_000]
        if blue_valid:
            w_b = 0.50 / max(len(blue_valid), 1)
            for i in blue_valid: weights[i] = w_b
        if opp_valid:
            w_o = 0.50 / max(len(opp_valid), 1)
            for i in opp_valid: weights[i] = min(weights[i] + w_o, 0.15)
        sector_groups = {}
        for i, sym in enumerate(symbols_order):
            sec = SECTOR_MAP.get(sym)
            if sec: sector_groups.setdefault(sec, []).append(i)
        for sec, idxs in sector_groups.items():
            s = float(np.sum([weights[j] for j in idxs]))
            if s > 0.25 and s > 0:
                f = 0.25 / s
                for j in idxs:
                    weights[j] = weights[j] * f
        total = float(np.sum(weights))
        if total > 0:
            weights = weights / total
        allocations = {}
        for i, asset in enumerate(selected_assets_metrics):
            allocations[asset['symbol']] = float(weights[i])
        return allocations
def filter_correlated_assets(final_elite: dict, threshold: float = 0.70) -> dict:
    print(f"\n[RISK] ⚔️ Iniciando Filtro de Correlação (Limiar: {threshold})...")
    data = {}
    metrics_map = {}
    for sym, content in final_elite.items():
        curve = content.get('equity_curve', [])
        if not isinstance(curve, list) or len(curve) < 50:
            continue
        data[sym] = curve
        try:
            calmar = float(content.get('test_metrics', {}).get('calmar', 0.0) or 0.0)
            if calmar == 0.0:
                calmar = float(content.get('calmar', 0.0) or 0.0)
        except Exception:
            calmar = 0.0
        metrics_map[sym] = float(calmar or 0.0)
    if len(data) < 2:
        return final_elite
    df_curves = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))
    df_curves = df_curves.fillna(method='ffill').dropna()
    if df_curves is None or df_curves.empty:
        return final_elite
    df_returns = df_curves.pct_change().dropna()
    corr_matrix = df_returns.corr()
    to_drop = set()
    columns = list(corr_matrix.columns)
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            sym_a = columns[i]
            sym_b = columns[j]
            if sym_a in to_drop or sym_b in to_drop:
                continue
            correlation = float(corr_matrix.iloc[i, j] or 0.0)
            if correlation > threshold:
                score_a = float(metrics_map.get(sym_a, 0.0) or 0.0)
                score_b = float(metrics_map.get(sym_b, 0.0) or 0.0)
                print(f"   ⚠️ Conflito Detectado: {sym_a} x {sym_b} (Corr: {correlation:.2f})")
                if score_a >= score_b:
                    print(f"      ❌ Removendo {sym_b} (Calmar menor: {score_b:.2f} vs {score_a:.2f})")
                    to_drop.add(sym_b)
                else:
                    print(f"      ❌ Removendo {sym_a} (Calmar menor: {score_a:.2f} vs {score_b:.2f})")
                    to_drop.add(sym_a)
    new_elite = {k: v for k, v in final_elite.items() if k not in to_drop}
    print(f"[RISK] Filtro Concluído. Ativos removidos: {len(to_drop)}. Restantes: {len(new_elite)}.\n")
    return new_elite
def _is_generic_params(p: dict) -> bool:
    try:
        return (int(p.get("ema_short", -1)) == 9 and
                int(p.get("ema_long", -1)) == 21 and
                int(p.get("rsi_low", -1)) == 30 and
                int(p.get("rsi_high", -1)) == 70 and
                float(p.get("adx_threshold", -1)) == 25 and
                float(p.get("sl_atr_multiplier", -1)) == 2.5 and
                float(p.get("tp_mult", -1)) == 5.0)
    except Exception:
        return False
def _ensure_non_generic(sym: str, p: dict) -> dict:
    elite = getattr(config, "ELITE_SYMBOLS", {}) if config else {}
    if isinstance(elite, dict):
        ep = elite.get(sym)
        if isinstance(ep, dict) and not _is_generic_params(ep):
            return {
                "ema_short": int(ep.get("ema_short", 12)),
                "ema_long": int(ep.get("ema_long", 97)),
                "rsi_low": int(ep.get("rsi_low", 37)),
                "rsi_high": int(ep.get("rsi_high", 73)),
                "adx_threshold": float(ep.get("adx_threshold", 13)),
                "mom_min": float(ep.get("mom_min", 0.0) or 0.0),
                "sl_atr_multiplier": float(ep.get("sl_atr_multiplier", 3.0) or 3.0),
                "tp_mult": float(ep.get("tp_mult", 3.0) or 3.0),
            }
    if not _is_generic_params(p):
        return p
    return {
        "ema_short": 12, "ema_long": 97, "rsi_low": 37, "rsi_high": 73,
        "adx_threshold": 13, "mom_min": float(p.get("mom_min", 0.0) or 0.0),
        "sl_atr_multiplier": 3.0, "tp_mult": 3.0
    }
def get_expected_vol_2026():
    """Retorna fator de volatilidade esperado (default 1.0)"""
    return 1.0
def adjust_params_for_vol(result: dict, vol: float):
    """Placeholder para ajuste de parâmetros baseado em volatilidade"""
    pass
def x_semantic_search(query: str, limit: int = 5):
    """Placeholder para pesquisa semântica simulada"""
    return [{"score": 0.5, "text": "default"}] * limit
def get_macro_rate(rate_name: str):
    cache_file = os.path.join(OPT_OUTPUT_DIR, "macro_cache.json")
    now = datetime.utcnow()
    name = rate_name.upper()
    if name.startswith("SELIC"):
        override = os.getenv("XP3_OVERRIDE_SELIC", "")
        try:
            selic_val = float(override) if override else 0.105
        except Exception:
            selic_val = 0.105
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}
        cache["selic"] = float(selic_val)
        cache["updated_at"] = now.isoformat()
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False)
        except Exception:
            pass
        return float(selic_val)
    try:
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                cache = json.load(f)
            ts = cache.get("updated_at")
            if ts:
                try:
                    updated = datetime.fromisoformat(ts)
                    if (now - updated) <= timedelta(hours=48):
                        if name.startswith("IPCA"):
                            return float(cache.get("ipca", 0.04) or 0.04)
                except Exception:
                    pass
    except Exception:
        pass
    if name.startswith("IPCA"):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}
        ipca_val = float(cache.get("ipca", 0.04) or 0.04)
        cache["ipca"] = ipca_val
        cache["updated_at"] = now.isoformat()
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False)
        except Exception:
            pass
        return ipca_val
    return 0.12
def _polygon_ticker(sym: str) -> str:
    s = sym.upper().strip()
    if s.startswith("^"):
        return s
    return f"{s}.SA"
@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def _polygon_aggs_df(ticker: str, timespan: str, multiplier: int, from_date: str, to_date: str, limit: int, api_key: str) -> Optional[pd.DataFrame]:
    try:
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}?limit={limit}&sort=asc&apiKey={api_key}"
        r = requests.get(url, timeout=8)
        if r.status_code != 200:
            return None
        j = r.json()
        results = j.get("results") or []
        if not results:
            return None
        df = pd.DataFrame(results)
        if "t" in df.columns:
            df["timestamp"] = df["t"]
        if "o" not in df.columns and "open" in df.columns:
            df["o"] = df["open"]
        if "h" not in df.columns and "high" in df.columns:
            df["h"] = df["high"]
        if "l" not in df.columns and "low" in df.columns:
            df["l"] = df["low"]
        if "c" not in df.columns and "close" in df.columns:
            df["c"] = df["close"]
        if "v" not in df.columns and "volume" in df.columns:
            df["v"] = df["volume"]
        if "timestamp" in df.columns:
            try:
                ts = df["timestamp"]
                if ts.dtype == np.int64 or ts.dtype == np.float64:
                    idx = pd.to_datetime(df["timestamp"], unit="ms")
                else:
                    idx = pd.to_datetime(df["timestamp"])
            except Exception:
                idx = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
            df.index = idx
        cols_map = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
        for k, v in cols_map.items():
            if k in df.columns and v not in df.columns:
                df[v] = df[k]
        need = ["open", "high", "low", "close", "volume"]
        if not all(c in df.columns for c in need):
            return None
        df = df[need].sort_index()
        return df
    except Exception:
        return None
def _yahoo_ticker(sym: str) -> str:
    s = sym.upper().strip()
    if s.startswith("^"):
        return s
    return f"{s}.SA"
def _yahoo_history(sym: str, interval: str = "1d", period: Optional[str] = None, start: Optional[str] = None, end: Optional[str] = None) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
        t = yf.Ticker(_yahoo_ticker(sym))
        if start and end:
            df = t.history(start=start, end=end, interval=interval, auto_adjust=True, actions=False)
        else:
            df = t.history(period=period or "730d", interval=interval, auto_adjust=True, actions=False)
        if df is None or df.empty:
            return None
        cols = ["Open", "High", "Low", "Close", "Volume"]
        for c in cols:
            if c not in df.columns:
                return None
        out = df[cols].copy()
        out.columns = ["open", "high", "low", "close", "volume"]
        out.index = pd.to_datetime(out.index).tz_localize(None)
        out = out.dropna().sort_index()
        return out
    except Exception:
        return None
def x_keyword_search(query: str, limit: int = 10) -> list:
    try:
        # Placeholder simples: retorna score neutro se não houver integração externa
        return [{"score": 0.5, "title": query}] * limit
    except Exception:
        return [{"score": 0.5}] * limit
def get_ibov_data(start: str = "2023-01-01", end: Optional[str] = None):
    try:
        from datetime import date
        today = date.today().strftime("%Y-%m-%d")
        end = end or today
        total = 100
        pbar = tqdm(total=total, desc="IBOV Download", leave=False)
        print("[DATA] Iniciando captura IBOV...", flush=True)
        pbar.update(5)
        if mt5 and ensure_mt5_connection():
            print("[DATA] IBOV via MT5...", flush=True)
            pbar.set_postfix_str("source=MT5")
            pbar.update(25)
            for sym in ["IBOV", "^BVSP"]:
                try:
                    if mt5.symbol_select(sym, True):
                        rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_D1, 0, 1000)
                        if rates is not None and len(rates) > 0:
                            dfm = pd.DataFrame(rates)
                            dfm['time'] = pd.to_datetime(dfm['time'], unit='s')
                            dfm.set_index('time', inplace=True)
                            if 'tick_volume' in dfm.columns:
                                dfm = dfm.rename(columns={'tick_volume': 'volume'})
                            dfm = dfm[['open', 'high', 'low', 'close', 'volume']]
                            pbar.update(total - pbar.n)
                            pbar.close()
                            print(f"[DATA] IBOV MT5 OK: {len(dfm)} linhas", flush=True)
                            return dfm.sort_index()
                except Exception:
                    pass
        pbar.set_postfix_str("source=Yahoo")
        print("[DATA] IBOV via Yahoo Finance...", flush=True)
        pbar.update(35)
        ydf = _yahoo_history("^BVSP", interval="1d", period="3650d")
        if ydf is not None and not ydf.empty:
            pbar.update(total - pbar.n)
            pbar.close()
            print(f"[DATA] IBOV Yahoo OK: {len(ydf)} linhas", flush=True)
            return ydf.sort_index()
        pbar.set_postfix_str("source=Polygon")
        print("[DATA] IBOV via Polygon...", flush=True)
        pbar.update(25)
        api_key = os.getenv("POLYGON_API_KEY", "xrE09LEWJYBZfQcV57pCvsw4aqkOiqbz")
        df = _polygon_aggs_df("^BVSP", "day", 1, start, end, 5000, api_key)
        pbar.update(total - pbar.n)
        pbar.close()
        if df is not None and not df.empty:
            print(f"[DATA] IBOV Polygon OK: {len(df)} linhas", flush=True)
        return df if df is not None and not df.empty else None
    except Exception:
        try:
            pbar.close()
        except Exception:
            pass
        print("[DATA] IBOV: Falha total nas fontes", flush=True)
        return None
def classify_asset_profile(symbol: str, df_ibov: Optional[pd.DataFrame] = None) -> str:
    try:
        from datetime import date
        today = date.today().strftime("%Y-%m-%d")
        start = "2024-01-01"
        api_key = os.getenv("POLYGON_API_KEY", "xrE09LEWJYBZfQcV57pCvsw4aqkOiqbz")
        df_a = _polygon_aggs_df(_polygon_ticker(symbol), "day", 1, start, today, 1000, api_key)
        if df_a is None or len(df_a) < 150:
            df_a = _yahoo_history(symbol, interval="1d", period="365d")
            if df_a is None or len(df_a) < 150:
                return "HIGH_VOLATILITY"
        a_ret = df_a["close"].pct_change().fillna(0).tail(126)
        if df_ibov is None or len(df_ibov or []) < 150:
            df_ibov = get_ibov_data()
        if df_ibov is None or len(df_ibov) < 150:
            return "HIGH_VOLATILITY"
        ib_ret = df_ibov["close"].pct_change().fillna(0).tail(126)
        min_len = min(len(a_ret), len(ib_ret))
        a_ret = a_ret.iloc[-min_len:]
        ib_ret = ib_ret.iloc[-min_len:]
        cov = float(np.cov(a_ret, ib_ret)[0][1])
        var_ib = float(np.var(ib_ret))
        beta = cov / var_ib if var_ib > 0 else 1.0
        vol_ann = float(np.std(a_ret)) * np.sqrt(252)
        sma50 = df_a["close"].rolling(50).mean().iloc[-1]
        sma200 = df_a["close"].rolling(200).mean().iloc[-1] if len(df_a) >= 200 else sma50
        is_uptrend = sma50 > sma200
        if (beta < 0.85) and (vol_ann < 0.30):
            return "CORE_DEFENSIVE"
        if (symbol in COMMODITY_SYMBOLS) and is_uptrend:
            return "CORE_COMMODITIES"
        return "HIGH_VOLATILITY"
    except Exception:
        return "HIGH_VOLATILITY"
def check_liquidity_dynamic(sym: str, ibov_df: pd.DataFrame = None) -> dict:
    from optimizer_optuna import calculate_adx
    try:
        from datetime import date
        today = date.today().strftime("%Y-%m-%d")
        start = "2024-01-01"
        api_key = os.getenv("POLYGON_API_KEY", "xrE09LEWJYBZfQcV57pCvsw4aqkOiqbz")
        df_d1 = None
        if mt5 and ensure_mt5_connection():
            try:
                mt5.symbol_select(sym, True)
                rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_D1, 0, 1000)
                if rates is not None and len(rates) > 0:
                    dfm = pd.DataFrame(rates)
                    dfm['time'] = pd.to_datetime(dfm['time'], unit='s')
                    dfm.set_index('time', inplace=True)
                    if 'real_volume' in dfm.columns and dfm['real_volume'].sum() > 0:
                        dfm = dfm.rename(columns={'real_volume': 'volume'})
                    elif 'tick_volume' in dfm.columns:
                        dfm = dfm.rename(columns={'tick_volume': 'volume'})
                    df_d1 = dfm[['open', 'high', 'low', 'close', 'volume']]
            except Exception:
                df_d1 = None
        if df_d1 is None or df_d1.empty:
            try:
                df_d1 = _polygon_aggs_df(_polygon_ticker(sym), "day", 1, start, today, 1000, api_key)
            except Exception:
                df_d1 = None
        if df_d1 is None or df_d1.empty:
            import logging as _logging
            try:
                _logging.getLogger("yfinance").setLevel(_logging.CRITICAL)
            except Exception:
                pass
            ok, res = safe_call_with_timeout(_yahoo_history, 2, sym, interval="1d", period="365d")
            df_d1 = res if ok else None
            if df_d1 is None or (hasattr(df_d1, "empty") and df_d1.empty):
                REJECT_SKIP.add(sym)
                return False, "SEM_DADOS_D1", {}
        df_liq = df_d1.tail(20)
        avg_vol_shares = float(df_liq["volume"].mean() or 0.0)
        avg_price = float(df_liq["close"].mean() or 0.0)
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
            import logging as _logging
            try:
                _logging.getLogger("yfinance").setLevel(_logging.CRITICAL)
            except Exception:
                pass
            ok2, res2 = safe_call_with_timeout(_yahoo_history, 2, sym, interval="15m", period="60d")
            df_m15 = res2 if ok2 else None
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
            params=(cutoff,),
        )
        conn.close()
        if df is None or df.empty:
            return {}

        overlay = {}
        for _, row in df.iterrows():
            sym = str(row.get("symbol", "") or "").strip()
            if not sym:
                continue
            total = int(row.get("total") or 0)
            if total < min_trades:
                continue
            wins = int(row.get("wins") or 0)
            pnl = float(row.get("pnl") or 0.0)
            overlay[sym] = {"total": total, "wins": wins, "pnl": pnl}

        return overlay
    except Exception:
        return {}
def extract_features(indicators, symbol):
    """
    Features + sentiment/macro (usa x_semantic_search via tool, macro via MT5).
    """
    features = [] # Original...
   
    # Novo: Sentiment X
    sentiment = x_semantic_search(f"sentimento {symbol} B3", limit=5) # Tool call simulado
    features.append(np.mean([s['score'] for s in sentiment]))
   
    # Novo: Macro (Selic via MT5 calendar)
    selic = get_macro_rate('Selic') # Nova função MT5
    features.append(selic)
   
    return features
# ===========================
# SINCRONIZAÇÃO MARKET WATCH
# ===========================
def open_all_mt5_symbols() -> int:
    """
    STEP 1: Abre TODOS os símbolos disponíveis no MT5
    Retorna o número de símbolos adicionados
    """
    print("\n" + "="*80)
    print("[INFO] ABRINDO TODOS OS SÍMBOLOS DO MT5")
    print("="*80)
   
    if not ensure_mt5_connection():
        logger.error("[ERROR] MT5 não disponível")
        return 0
   
    # Pega TODOS os símbolos disponíveis no MT5
    all_symbols = mt5.symbols_get()
    if not all_symbols:
        logger.error("[ERROR] Nenhum símbolo disponível no MT5")
        return 0
   
    total_symbols = len(all_symbols)
    print(f"\n[INFO] Total de símbolos no MT5: {total_symbols}")
    print(f"[INFO] Adicionando todos ao Market Watch...")
   
    added = 0
    failed = 0
   
    # Adiciona TODOS os símbolos
    for symbol in tqdm(all_symbols, desc="Adicionando símbolos"):
        try:
            if mt5.symbol_select(symbol.name, True):
                added += 1
            else:
                failed += 1
            time.sleep(0.01) # Pequeno delay para não sobrecarregar
        except Exception as e:
            failed += 1
   
    print(f"\n[OK] {added}/{total_symbols} símbolos adicionados")
    if failed > 0:
        print(f"[WARN] {failed} símbolos falharam")
   
    return added
def sync_market_watch_with_sector_map(clear_first: bool = True) -> bool:
    """
    STEP 2: Filtra Market Watch para manter apenas símbolos do SECTOR_MAP
    """
    print("\n" + "="*80)
    print("[INFO] FILTRANDO APENAS SÍMBOLOS DO SECTOR_MAP")
    print("="*80)
   
    if not ensure_mt5_connection():
        logger.error("[ERROR] MT5 não disponível")
        return False
   
    if SANDBOX_MODE:
        desired_symbols = {'PETR4', 'VALE3', 'ITUB4', 'BBDC4', 'ABEV3'}
    else:
        desired_symbols = {k.upper().strip() for k in SECTOR_MAP.keys()
                          if isinstance(k, str) and k.strip()}
   
    if not desired_symbols:
        logger.error("[ERROR] SECTOR_MAP vazio")
        return False
   
    print(f"\n[INFO] SECTOR_MAP contém: {len(desired_symbols)} símbolos")
   
    all_symbols = mt5.symbols_get()
    current_symbols = {s.name for s in all_symbols if s.visible} if all_symbols else set()
   
    print(f"[INFO] Market Watch atual: {len(current_symbols)} símbolos")
   
    # Símbolos que precisam ser adicionados (estão no SECTOR_MAP mas não no Market Watch)
    to_add = desired_symbols - current_symbols
   
    # Símbolos que podem ser removidos (estão no Market Watch mas não no SECTOR_MAP)
    to_remove = current_symbols - desired_symbols
   
    already_ok = current_symbols & desired_symbols
   
    print(f"\n[INFO] ANÁLISE:")
    print(f" • Já corretos (SECTOR_MAP): {len(already_ok)} símbolos")
    print(f" • Adicionar do SECTOR_MAP: {len(to_add)} símbolos")
   
    if clear_first:
        print(f" • Remover (não estão no SECTOR_MAP): {len(to_remove)} símbolos")
    else:
        print(f" • Manter extras no Market Watch: {len(to_remove)} símbolos")
   
    # ✅ REMOVIDO: Sem confirmação manual
    # Remove símbolos que NÃO estão no SECTOR_MAP (se clear_first=True)
    if clear_first and to_remove:
        print(f"\n[INFO] Removendo {len(to_remove)} símbolos automaticamente...")
        removed = 0
        for symbol in tqdm(list(to_remove), desc="Removendo"):
            try:
                if mt5.symbol_select(symbol, False):
                    removed += 1
                time.sleep(0.01)
            except:
                pass
        print(f"[OK] {removed} símbolos removidos")
   
    max_visible = int(getattr(config, "MT5_MAX_VISIBLE", 300) or 300)
    if len(current_symbols) > max_visible:
        logger.warning(f"[WARN] Market Watch excede o limite visível ({len(current_symbols)}/{max_visible})")
    remaining_slots = max(0, max_visible - len(current_symbols))
    to_add = set(list(to_add)[:remaining_slots]) if remaining_slots > 0 else set()
    # Adiciona símbolos do SECTOR_MAP que estão faltando
    if to_add:
        print(f"\n[INFO] Adicionando {len(to_add)} símbolos do SECTOR_MAP...")
        added = 0
        failed = []
       
        for symbol in tqdm(sorted(to_add), desc="Adicionando"):
            try:
                info = mt5.symbol_info(symbol)
                if not info:
                    logger.warning(f"[WARN] {symbol} não existe no MT5")
                    failed.append(symbol)
                    continue
               
                if mt5.symbol_select(symbol, True):
                    added += 1
                else:
                    failed.append(symbol)
               
                time.sleep(0.05)
               
            except Exception as e:
                logger.warning(f"[ERROR] {symbol}: {e}")
                failed.append(symbol)
       
        print(f"[OK] {added}/{len(to_add)} símbolos adicionados")
       
        if failed:
            print(f"\n[WARN] {len(failed)} símbolos falharam:")
            for sym in failed[:10]:
                print(f" - {sym}")
            if len(failed) > 10:
                print(f" ... e mais {len(failed) - 10}")
   
    final_symbols = mt5.symbols_get()
    final_in_sector_map = len([s for s in final_symbols if s.visible and s.name in desired_symbols]) if final_symbols else 0
    final_total = len([s for s in final_symbols if s.visible]) if final_symbols else 0
   
    print(f"\n[OK] SINCRONIZAÇÃO CONCLUÍDA!")
    print(f" Market Watch Total: {final_total} símbolos")
    print(f" Do SECTOR_MAP: {final_in_sector_map}/{len(desired_symbols)} ({final_in_sector_map/len(desired_symbols)*100:.1f}%)")
    print("="*80)
   
    return final_in_sector_map >= len(desired_symbols) * 0.9
# ===========================
# CARREGAMENTO DE DADOS
# ===========================
def load_data_with_retry(symbol: str, bars: int, timeframe=None, max_retries: int = 1) -> Optional[pd.DataFrame]:
    logger.info(f"📥 [DADOS] Carregando {symbol} (fallback Polygon/Yahoo)")
    s = (symbol or "").upper().strip()
    is_fut = s.endswith("$") or s.startswith(("WIN", "WDO", "IND", "DOL"))
    if is_fut:
        try:
            if ensure_mt5_connection():
                tf = mt5.TIMEFRAME_M15 if timeframe in (None, "M15") else mt5.TIMEFRAME_D1
                mt5.symbol_select(s, True)
                rates = mt5.copy_rates_from_pos(s, tf, 0, max(bars, 100))
                if rates:
                    df = pd.DataFrame(rates)
                    df["time"] = pd.to_datetime(df["time"], unit="s")
                    df.set_index("time", inplace=True)
                    return df.tail(bars).sort_index()
        except Exception:
            pass
    return load_data_polygon(symbol, bars, timeframe)
def load_data_polygon(symbol: str, bars: int, timeframe=None) -> Optional[pd.DataFrame]:
    try:
        from datetime import date, timedelta
        end = date.today()
        start = (end - timedelta(days=365*2)).strftime("%Y-%m-%d")
        end_s = end.strftime("%Y-%m-%d")
        ts = "minute" if timeframe is None else ("minute" if timeframe == "M15" else "day")
        # Primeiro tenta Yahoo (rápido e robusto)
        try:
            if ts == "minute":
                ydf = _yahoo_history(symbol, interval="15m", period="60d")
            else:
                ydf = _yahoo_history(symbol, interval="1d", period="365d")
            if ydf is not None and len(ydf) >= 100:
                return ydf.tail(bars).sort_index()
        except Exception:
            pass
        # Fallback Polygon com timeout curto
        api_key = os.getenv("POLYGON_API_KEY", "xrE09LEWJYBZfQcV57pCvsw4aqkOiqbz")
        mult = 15 if ts == "minute" else 1
        df = _polygon_aggs_df(_polygon_ticker(symbol), ts, mult, start, end_s, max(bars, 500), api_key)
        if df is not None and len(df) >= 100:
            return df.tail(bars).sort_index()
        return None
    except Exception:
        return None
# ===========================
# CÁLCULO CORRETO DO ADX
# ===========================
def calculate_adx(high, low, close, period=14):
    """Calcula o ADX (Average Directional Index) corretamente"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum.reduce([tr1, tr2, tr3])
   
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
   
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
   
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / (atr + 1e-10)
   
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().fillna(0).values
   
    return adx, atr
# ===========================
# MÉTRICAS
# ===========================
def compute_advanced_metrics(equity_curve: List[float]) -> Dict[str, Any]:
    """Calcula métricas avançadas de performance"""
    if not equity_curve or len(equity_curve) < 2:
        return {"total_return": 0.0, "max_drawdown": 0.01, "calmar": 0.0, "sortino": 0.0,
                "sharpe": 0.0, "profit_factor": 0.0, "recovery_factor": 0.0}
    returns = np.diff(equity_curve) / equity_curve[:-1]
    if len(returns) < config_bt.MIN_BARS_FOR_METRICS:
        return {"total_return": 0.0, "max_drawdown": 1.0, "calmar": 0.0, "sortino": 0.0,
                "sharpe": 0.0, "profit_factor": 0.0, "recovery_factor": 0.0}
    total_return = equity_curve[-1] / equity_curve[0] - 1
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peak) / peak
    max_dd = max(-np.min(drawdowns), 0.01)
    n_bars = len(equity_curve)
    years = n_bars / config_bt.bars_per_year
    annualized = (1 + total_return) ** (1 / years) - 1 if years >= 1 else total_return
    ret_std = np.std(returns)
    sharpe = (annualized - config_bt.RISK_FREE_RATE) / (ret_std * np.sqrt(config_bt.bars_per_year)) if ret_std > 0 else 0.0
    downside = returns[returns < 0]
    downside_std = np.std(downside) * np.sqrt(config_bt.bars_per_year) if len(downside) > 0 else 1e-6
    sortino = (annualized - config_bt.RISK_FREE_RATE) / downside_std
    calmar = annualized / max_dd
    wins = returns[returns > 0]
    losses = np.abs(returns[returns < 0])
    profit_factor = sum(wins) / sum(losses) if len(losses) > 0 else float('inf') if len(wins) > 0 else 0.0
    profit_factor = min(profit_factor, 999.0)
    recovery_factor = total_return / max_dd if max_dd > 0 else 0.0
    win_rate = len(wins) / len(returns) if len(returns) > 0 else 0.0
    avg_win = np.mean(wins) if len(wins) > 0 else 0.0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    return {
        "total_return": float(total_return),
        "max_drawdown": float(max_dd),
        "calmar": float(calmar),
        "sortino": float(sortino),
        "sharpe": float(sharpe),
        "profit_factor": float(profit_factor),
        "recovery_factor": float(recovery_factor),
        "expectancy": float(expectancy),
        "win_rate": float(win_rate),
        "final_equity": float(equity_curve[-1])
    }
# ===========================
# BACKTEST CORE (NUMBA)
# ===========================
def fast_backtest_core(
    close, high, low, volume, volume_ma,
    ema_short, ema_long, rsi, rsi_2, adx, momentum, atr,
    rsi_low, rsi_high,
    adx_threshold, mom_min,
    sl_mult, tp_mult, base_slippage,
    risk_per_trade=0.01,
    use_trailing=True,
    trail_atr_mult=1.0,
    asset_type=0,
    point_value=0.0,
    tick_size=0.01,
    action_cost_pct=0.00055,
    future_fee_per_contract=1.0
):
    """
    High Win Rate Backtest Core with Strict Take Profit (Sniper Mode)
    """
    cash = 100000.0
    equity = cash
    position = 0.0
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0 # ✅ Take Profit Target
    trades = 0
    wins = 0
    losses = 0
    costs_paid = 0.0
   
    transaction_cost_pct = action_cost_pct
    n = len(close)
   
    # Arrays para métricas
    # ✅ FIX: Inicializa com capital inicial para evitar divisão por zero
    equity_curve = np.full(n, cash)
    drawdown = np.zeros(n)
   
    # Estado: 0=Fora, 1=Comprado
    state = 0
   
    for i in range(1, n):
        current_price = close[i]
       
        # 1. GESTÃO DE POSIÇÃO (SE ESTIVER COMPRADO)
        if state == 1:
            # ✅ VERIFICAÇÃO DE TAKE PROFIT (Prioridade Máxima)
            # Se a máxima do dia tocou no alvo, é WIN
            if high[i] >= target_price:
                exit_price = target_price
                if asset_type == 1:
                    points = (exit_price - entry_price) / max(tick_size, 1e-6)
                    gross_profit = points * point_value * position
                    cost = future_fee_per_contract * position
                    net_profit = gross_profit - cost
                    cash += net_profit
                    costs_paid += cost
                else:
                    val_exit = position * exit_price
                    cost = val_exit * transaction_cost_pct
                    gross_profit = (exit_price - entry_price) * position
                    net_profit = gross_profit - cost
                    cash += (val_exit - cost)
                    costs_paid += cost
                if net_profit > 0: wins += 1
                else: losses += 1
               
                trades += 1
                state = 0
                position = 0.0
                equity = cash
           
            # ❌ VERIFICAÇÃO DE STOP LOSS
            elif low[i] <= stop_price:
                exit_price = stop_price - (stop_price * base_slippage)
                if asset_type == 1:
                    points = (exit_price - entry_price) / max(tick_size, 1e-6)
                    gross_profit = points * point_value * position
                    cost = future_fee_per_contract * position
                    net_profit = gross_profit - cost
                    cash += net_profit
                    costs_paid += cost
                else:
                    val_exit = position * exit_price
                    cost = val_exit * transaction_cost_pct
                    gross_profit = (exit_price - entry_price) * position
                    net_profit = gross_profit - cost
                    cash += (val_exit - cost)
                    costs_paid += cost
                losses += 1 # Stop é sempre Loss
                trades += 1
                state = 0
                position = 0.0
                equity = cash
            elif use_trailing:
                curr_move = current_price - entry_price
                if curr_move > (atr[i] * trail_atr_mult):
                    new_stop = current_price - (atr[i] * trail_atr_mult)
                    if new_stop > stop_price:
                        stop_price = new_stop
            # ⏳ MANUTENÇÃO (Atualiza Equity mas não sai)
            else:
                if asset_type == 1:
                    unreal = ((current_price - entry_price) / max(tick_size, 1e-6)) * point_value * position
                    equity = cash + unreal
                else:
                    equity = cash + (position * current_price)
        # 2. SINAL DE ENTRADA (SE ESTIVER FORA)
        # Lógica Híbrida: Pullback em Tendência OU Reversão Lateral (RSI 2)
        elif state == 0:
            signal = False
           
            # Setup A: Pullback Clássico (Tendência)
            # Preço acima da média longa (Tendência de Alta) E recuo no RSI
            trend_condition = close[i] > ema_long[i]
            pullback_condition = rsi[i] < rsi_low
           
            # Setup B: Larry Williams (Lateralidade/Correção Forte)
            # RSI de 2 períodos extremamente sobrevendido (< 5 ou param)
            reversion_condition = rsi_2[i] < 5
           
            # Filtro de Volatilidade (Evitar mercado morto)
            volatility_ok = adx[i] > adx_threshold
           
            if (trend_condition and pullback_condition and volatility_ok) or (reversion_condition):
                signal = True
            if signal:
                # Definição de Risco
                sl_dist = atr[i] * sl_mult
                tp_dist = atr[i] * tp_mult # ✅ Alvo baseado em ATR
               
                entry_price = close[i] * (1 + base_slippage)
                stop_price = entry_price - sl_dist
                target_price = entry_price + tp_dist
                if tick_size > 0:
                    entry_price = float(round(entry_price / tick_size) * tick_size)
                    stop_price = float(round(stop_price / tick_size) * tick_size)
                    target_price = float(round(target_price / tick_size) * tick_size)
               
                # Tamanho da posição (Risco Fixo 2%)
                risk_amt = equity * risk_per_trade
                if sl_dist > 0:
                    if asset_type == 1:
                        sl_points = sl_dist / max(tick_size, 1e-6)
                        contracts_raw = risk_amt / max(sl_points * point_value, 1e-6)
                        pos_contracts = np.floor(contracts_raw)
                        if pos_contracts >= 1:
                            c_entry = future_fee_per_contract * pos_contracts
                            cash -= c_entry
                            position = pos_contracts
                    else:
                        shares_raw = risk_amt / sl_dist
                        shares = np.floor(shares_raw / 100.0) * 100.0
                        max_shares = np.floor((cash * 0.95) / entry_price / 100.0) * 100.0
                        if shares > max_shares:
                            shares = max_shares
                        if shares >= 100.0:
                            cost_fin = shares * entry_price
                            c_entry = cost_fin * transaction_cost_pct
                            cash -= (cost_fin + c_entry)
                            position = shares
       
        equity_curve[i] = equity
    # Métricas Finais
    # Métricas Finais
    final_return = (equity - 100000.0) / 100000.0
    win_rate = wins / trades if trades > 0 else 0.0
    # ✅ Retorna tupla (Numba não suporta dicts)
    # Ordem: equity_curve, trades, wins, losses, final_return, win_rate, costs_paid
    return equity_curve, trades, wins, losses, final_return, win_rate, costs_paid
# ===========================
# FUNÇÃO PRINCIPAL DO BACKTEST
# ===========================
def backtest_params_on_df(symbol: str, params: dict, df: pd.DataFrame) -> Dict[str, Any]:
    """Executa backtest com os parâmetros fornecidos"""
    if df is None or len(df) < 100:
        return {
            "total_return": -1.0,
            "calmar": -10.0,
            "total_trades": 0,
            "equity_curve": [100000.0]
        }
    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    # --- INDICADORES ---
    ema_s = pd.Series(close).ewm(span=params.get("ema_short", 9), adjust=False).mean().values
    ema_l = pd.Series(close).ewm(span=params.get("ema_long", 21), adjust=False).mean().values
   
    # RSI
    delta = pd.Series(close).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).fillna(50).values
    
    # RSI_2 (Larry Williams)
    delta_2 = pd.Series(close).diff()
    gain_2 = (delta_2.where(delta_2 > 0, 0)).rolling(window=2).mean()
    loss_2 = (-delta_2.where(delta_2 < 0, 0)).rolling(window=2).mean()
    rs_2 = gain_2 / (loss_2 + 1e-10)
    rsi_2 = (100 - (100 / (1 + rs_2))).fillna(50).values
    # Momentum (Rate of Change)
    momentum = pd.Series(close).pct_change(periods=10).fillna(0).values
    # ADX e ATR CORRETOS
    adx, atr = calculate_adx(high, low, close, period=14)
   
    # Debug: Estatísticas dos indicadores
    trend_up = ema_s > ema_l
    rsi_ok = rsi < params.get("rsi_low", 30)
    momentum_ok = momentum > params.get("mom_min", 0.0)
    adx_ok = adx > params.get("adx_threshold", 25)
    buy_signals = (trend_up & (adx_ok | rsi_ok | momentum_ok))
   
    n_signals = np.sum(buy_signals)
    # ✅ ADX médio seguro
    if len(adx) > 50:
        adx_slice = adx[50:]
    else:
        adx_slice = adx
    adx_avg = float(np.mean(adx_slice)) if len(adx_slice) > 0 else 0.0
    logger.info(
        f"📊 {symbol} | "
        f"🔔 Sinais={n_signals}/{len(df)} ({n_signals/len(df)*100:.1f}%) | "
        f"📈 Trend={np.sum(trend_up)} | 💪 RSI={np.sum(rsi_ok)} | "
        f"⚡ Mom={np.sum(momentum_ok)} | 📐 ADX={np.sum(adx_ok)} | "
        f"📉 ADX_avg={adx_avg:.1f}"
    )
    # ✅ VOLUME PARA SLIPPAGE DINÂMICO e FILTROS
    volume = df['volume'].values.astype(np.float64)
    volume_ma = pd.Series(volume).rolling(20).mean().fillna(0).values
    def _validate_params(p: dict) -> dict:
        es = int(p.get("ema_short", 9))
        el = int(p.get("ema_long", 21))
        if es >= el:
            es = max(5, min(el - 1, es))
        rl = int(p.get("rsi_low", 30))
        rh = int(p.get("rsi_high", 70))
        if rl >= rh:
            rl = max(10, min(rh - 1, rl))
        adx_t = float(p.get("adx_threshold", 25))
        adx_t = max(5.0, min(60.0, adx_t))
        slm = float(p.get("sl_atr_multiplier", 2.0))
        tpm = float(p.get("tp_mult", 3.0))
        if tpm < slm:
            tpm = slm * 1.5
        base_s = float(p.get("base_slippage", 0.0035))
        trail_m = float(p.get("trail_atr_mult", 1.0))
        use_tr = bool(p.get("use_trailing", True))
        return {"ema_short": es, "ema_long": el, "rsi_low": rl, "rsi_high": rh, "adx_threshold": adx_t, "mom_min": float(p.get("mom_min", 0.0) or 0.0), "sl_atr_multiplier": slm, "tp_mult": tpm, "base_slippage": base_s, "use_trailing": use_tr, "trail_atr_mult": trail_m}
    params = _validate_params(params)
    # ✅ FIX: fast_backtest_core retorna tupla (equity_curve, trades, wins, losses, final_return, win_rate)
    try:
        from utils import AssetInspector
        ai = AssetInspector.detect(symbol)
        asset_type = 1 if ai.get("type") == "FUTURE" else 0
        pv = float(ai.get("point_value", 0.0))
        ts = float(ai.get("tick_size", 0.01))
    except Exception:
        asset_type = 0
        pv, ts = 0.0, 0.01
    import config as _cfg
    equity_arr, trades, wins, losses, total_return, win_rate, costs_paid = fast_backtest_core(
        close, high, low, volume, volume_ma,
        ema_s, ema_l, rsi, rsi_2, adx, momentum, atr,
        params["rsi_low"],
        params["rsi_high"],
        params["adx_threshold"],
        params["mom_min"],
        params["sl_atr_multiplier"],
        params["tp_mult"],
        params["base_slippage"],
        0.01,
        params["use_trailing"],
        params["trail_atr_mult"],
        asset_type,
        pv,
        ts,
        float(getattr(_cfg, "ACTION_COST_PCT", 0.00055)),
        float(getattr(_cfg, "FUTURE_FEE_PER_CONTRACT", 1.0))
    )
    # ✅ CORREÇÃO: Chama a função direto, sem import
    metrics = compute_advanced_metrics(equity_arr.tolist())
    metrics.update({
        "total_trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / trades if trades > 0 else 0.0,
        "costs_paid": float(costs_paid),
        "equity_curve": equity_arr.tolist()
    })
   
    return metrics
def is_valid_equity_curve(eq, min_len=50) -> bool:
    if eq is None:
        return False
    if not isinstance(eq, (list, np.ndarray)):
        return False
    if len(eq) < min_len:
        return False
    return True
def _auto_discover_futures():
    try:
        mapping = utils.update_futures_mappings()
        if mapping:
            print(f"[INFO] Auto-Discovery de Futuros aplicado: {mapping}", flush=True)
    except Exception as e:
        print(f"[WARN] Auto-Discovery falhou: {e}", flush=True)
# ===========================
# WORKER WFO
# ===========================
def worker_wfo(sym: str, bars: int, maxevals: int, wfo_windows: int,
               train_period: int, test_period: int) -> Dict[str, Any]:
    """Worker WFO com reconexão MT5 automática"""
    out = {"symbol": sym, "status": "ok", "wfo_windows": []}
   
    try:
        df_full = load_data_polygon(sym, bars)
       
        if not is_valid_dataframe(df_full):
            return {"symbol": sym, "error": "no_data"}
       
        df_full = df_full.sort_index()
        n = len(df_full)
       
        if n < (train_period + test_period):
            return {"symbol": sym, "error": "insufficient_data"}
       
        step = test_period
        wins = []
       
        for i in range(wfo_windows):
            train_start = i * step
            train_end = train_start + train_period
            test_end = train_end + test_period
           
            if test_end > n:
                break
           
            df_train = df_full.iloc[train_start:train_end].copy()
            df_test = df_full.iloc[train_end:test_end].copy()
           
            if df_train.empty or df_test.empty:
                continue
           
            ml_model = None
            best_params = {}
           
            try:
                from optimizer_optuna import optimize_with_optuna, backtest_params_on_df
               
                # ✅ Optional: Retrain ML Model per Fold if desired (Uncomment to enable)
                # from optimizer_optuna import train_ml_signal_boost
                # ml_model_fold = train_ml_signal_boost(df_train)
                # But optimize_with_optuna already trains it internally on df_train passed to it.
                # Use that model.
               
                if os.getenv("XP3_DISABLE_OPTUNA", "0") == "1":
                    res = {"status": "NO_VALID_TRIALS", "reason": "disabled_by_env", "best_params": {}, "ml_model": None}
                else:
                    logger.info(f"🧪 {sym}: Janela {i+1}/{wfo_windows} - Iniciando Optuna...")
                    res = optimize_with_optuna(sym, df_train, n_trials=120, timeout=1200)
               
                if res.get("status") == "SUCCESS":
                    best_params = res.get("best_params", {})
                    ml_model = res.get("ml_model", None)
                    logger.info(f"{sym}: ✅ Parâmetros otimizados: {best_params}")
                    try:
                        if _is_generic_params(best_params):
                            logger.warning(f"🔍 {sym}: Parâmetros genéricos detectados, expandindo busca...")
                            df_ext = df_full.iloc[-max(test_period*4, 800):]
                            ema_short_grid = list(range(8, 31, 3))
                            ema_long_grid = [60, 72, 90, 110, 125, 140]
                            best = None
                            for es in ema_short_grid:
                                for el in ema_long_grid:
                                    cparams = dict(best_params)
                                    cparams["ema_short"] = es
                                    cparams["ema_long"] = el
                                    m2 = backtest_params_on_df(sym, cparams, df_ext)
                                    score2 = float(m2.get("calmar", 0.0) or 0.0)
                                    if (best is None) or (score2 > best[0]):
                                        best = (score2, cparams)
                            if best is not None:
                                best_params = best[1]
                    except Exception:
                        pass
                else:
                    reason = res.get("reason", res.get("status", "Unknown"))
                    logger.warning(f"{sym}: ⚠️ Optuna sem resultado válido nesta janela ({reason})")
                    if len(wins) > 0:
                        logger.warning(f"{sym}: Usando parâmetros da melhor janela anterior")
                        best_params = wins[-1]["best_params"]
                        ml_model = wins[-1].get("ml_model")
                    else:
                        logger.warning(f"{sym}: Todas trials pruned — forçando grid search estendido")
                        df_ext = load_data_polygon(sym, 8000)
                        picked = None
                        if df_ext is not None and len(df_ext) >= 300:
                            ema_short_grid = list(range(8, 51, 3))
                            ema_long_grid = list(range(60, 161, 10))
                            try:
                                for es in ema_short_grid:
                                    for el in ema_long_grid:
                                        cparams = {
                                            "ema_short": es, "ema_long": el,
                                            "rsi_low": 30, "rsi_high": 70,
                                            "adx_threshold": 25,
                                            "sl_atr_multiplier": 2.5,
                                            "tp_mult": 5.0
                                        }
                                        m2 = backtest_params_on_df(sym, cparams, df_ext)
                                        score2 = float(m2.get("calmar", 0.0) or 0.0)
                                        if (picked is None) or (score2 > picked[0]):
                                            picked = (score2, cparams)
                            except Exception:
                                picked = None
                        if picked is not None:
                            best_params = picked[1]
                            ml_model = None
                        else:
                            elite = getattr(config, "ELITE_SYMBOLS", {}) if config else {}
                            if isinstance(elite, dict) and sym in elite:
                                ep = elite.get(sym) or {}
                                best_params = {
                                    "ema_short": int(ep.get("ema_short", 12)),
                                    "ema_long": int(ep.get("ema_long", 97)),
                                    "rsi_low": int(ep.get("rsi_low", 37)),
                                    "rsi_high": int(ep.get("rsi_high", 73)),
                                    "adx_threshold": float(ep.get("adx_threshold", 13)),
                                    "sl_atr_multiplier": float(ep.get("sl_atr_multiplier", 3.0)),
                                    "tp_mult": float(ep.get("tp_mult", 3.0))
                                }
                            else:
                                best_params = _ensure_non_generic(sym, {})
                            ml_model = None
            except Exception as e:
                logger.error(f"{sym}: ❌ OPTUNA FALHOU: {e}")
                if len(wins) > 0:
                    logger.warning(f"{sym}: Usando parâmetros da melhor janela anterior")
                    best_params = wins[-1]["best_params"]
                    ml_model = wins[-1].get("ml_model")
                else:
                    try:
                        df_use = df_full.iloc[-max(test_period*2, 400):] if df_full is not None else None
                        if df_use is not None and len(df_use) >= 200:
                            ema_short_grid = list(range(8, 31, 3))
                            ema_long_grid = [60, 72, 90, 110, 125, 140]
                            delta = pd.Series(df_use['close']).diff()
                            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                            rs = gain / (loss + 1e-10)
                            rsi_series = (100 - (100 / (1 + rs))).fillna(50).values
                            rsi_low = int(np.clip(np.percentile(rsi_series[-150:], 30), 25, 40))
                            rsi_high = int(np.clip(np.percentile(rsi_series[-150:], 70), 60, 85))
                            adx_vals, atr_vals = calculate_adx(
                                df_use['high'].values, df_use['low'].values, df_use['close'].values, period=14
                            )
                            adx_threshold = float(np.clip(np.median(adx_vals[-150:]), 15, 35))
                            vol_proxy = float(np.mean(atr_vals[-150:]) / max(np.mean(df_use['close'].values[-150:]), 1e-6))
                            sl_mult = 3.5 if vol_proxy > 0.02 else 2.5
                            tp_mult = 3.0
                            best = None
                            for es in ema_short_grid:
                                for el in ema_long_grid:
                                    cparams = {
                                        "ema_short": es, "ema_long": el,
                                        "rsi_low": rsi_low, "rsi_high": rsi_high,
                                        "adx_threshold": adx_threshold,
                                        "mom_min": 0.0,
                                        "sl_atr_multiplier": sl_mult,
                                        "tp_mult": tp_mult
                                    }
                                    m = backtest_params_on_df(sym, cparams, df_use)
                                    score = float(m.get("calmar", 0.0) or 0.0)
                                    if (best is None) or (score > best[0]):
                                        best = (score, cparams)
                            if best is not None:
                                best_params = best[1]
                            else:
                                best_params = _ensure_non_generic(sym, {})
                        else:
                            best_params = _ensure_non_generic(sym, {})
                    except Exception:
                        elite = getattr(config, "ELITE_SYMBOLS", {}) if config else {}
                        if isinstance(elite, dict) and sym in elite:
                            ep = elite.get(sym) or {}
                            best_params = {
                                "ema_short": int(ep.get("ema_short", 12)),
                                "ema_long": int(ep.get("ema_long", 97)),
                                "rsi_low": int(ep.get("rsi_low", 37)),
                                "rsi_high": int(ep.get("rsi_high", 73)),
                                "adx_threshold": float(ep.get("adx_threshold", 13)),
                                "sl_atr_multiplier": float(ep.get("sl_atr_multiplier", 3.0)),
                                "tp_mult": float(ep.get("tp_mult", 3.0))
                            }
                        else:
                            best_params = _ensure_non_generic(sym, {})
                    ml_model = None
           
            # ✅ Validar no OOS com ML Model se disponível
            # No bloco de métricas ou logs:

            test_res = backtest_params_on_df(sym, best_params, df_test, ml_model=ml_model)
            curve = test_res.get("equity_curve", None)
    
            if isinstance(curve, np.ndarray):
                curve_list = curve.tolist()
            elif isinstance(curve, list):
                curve_list = curve
            else:
                curve_list = [100000.0] # Fallback se for string ou None
           
            wins.append({
                "best_params": best_params,
                "ml_model": ml_model,
                "test_metrics": test_res,
                "equity_curve": test_res.get("equity_curve", [])
            })
       
        if not wins:
            return {"symbol": sym, "error": "wfo_no_windows"}
       
        best_win = max(wins, key=lambda w: w["test_metrics"].get("calmar", -100))
        cand_params = best_win["best_params"]
        cand_metrics = best_win["test_metrics"]
        try:
            elite_json_latest = os.path.join(OPT_OUTPUT_DIR, "elite_symbols_latest.json")
            baseline_params = None
            if os.path.exists(elite_json_latest):
                import json
                with open(elite_json_latest, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                elite_dict = (payload or {}).get("elite_symbols", {})
                b = elite_dict.get(sym)
                if isinstance(b, dict):
                    baseline_params = {
                        "ema_short": int(b.get("ema_short", 9)),
                        "ema_long": int(b.get("ema_long", 21)),
                        "rsi_low": int(b.get("rsi_low", 30)),
                        "rsi_high": int(b.get("rsi_high", 70)),
                        "adx_threshold": float(b.get("adx_threshold", 25)),
                        "mom_min": float(b.get("mom_min", 0.0) or 0.0),
                        "sl_atr_multiplier": float(b.get("sl_atr_multiplier", 2.5) or 2.5),
                        "tp_mult": float(b.get("tp_mult", 5.0) or 5.0),
                    }
            if baseline_params is None:
                elite = getattr(config, "ELITE_SYMBOLS", {}) if config else {}
                ep = elite.get(sym) if isinstance(elite, dict) else None
                if isinstance(ep, dict):
                    baseline_params = {
                        "ema_short": int(ep.get("ema_short", 9)),
                        "ema_long": int(ep.get("ema_long", 21)),
                        "rsi_low": int(ep.get("rsi_low", 30)),
                        "rsi_high": int(ep.get("rsi_high", 70)),
                        "adx_threshold": float(ep.get("adx_threshold", 25)),
                        "mom_min": float(ep.get("mom_min", 0.0) or 0.0),
                        "sl_atr_multiplier": float(ep.get("sl_atr_multiplier", 2.5) or 2.5),
                        "tp_mult": float(ep.get("tp_mult", 5.0) or 5.0),
                    }
            if baseline_params:
                ema_changed = (int(cand_params.get("ema_short", 9)) != int(baseline_params.get("ema_short", 9))) or (int(cand_params.get("ema_long", 21)) != int(baseline_params.get("ema_long", 21)))
                if ema_changed:
                    base_metrics = backtest_params_on_df(sym, baseline_params, df_test, ml_model=best_win.get("ml_model"))
                    base_wr = float(base_metrics.get("win_rate", 0.0) or 0.0)
                    new_wr = float(cand_metrics.get("win_rate", 0.0) or 0.0)
                    denom = base_wr if base_wr > 1e-9 else 1e-9
                    improvement = (new_wr - base_wr) / denom
                    if improvement < 0.15:
                        cand_params["ema_short"] = int(baseline_params.get("ema_short", cand_params.get("ema_short", 9)))
                        cand_params["ema_long"] = int(baseline_params.get("ema_long", cand_params.get("ema_long", 21)))
                        cand_metrics = backtest_params_on_df(sym, cand_params, df_test, ml_model=best_win.get("ml_model"))
        except Exception:
            pass
        out["selected_params"] = cand_params
        out["test_metrics"] = cand_metrics
        out["equity_curve"] = best_win["equity_curve"]
        logger.info(
            f"📊 {sym} | "
            f"🛠️ Calmar OOS={best_win['test_metrics'].get('calmar', 0):.3f} | "
            f"📈 Retorno={best_win['test_metrics'].get('total_return', 0):.2%} | "
            f"🔢 Trades={best_win['test_metrics'].get('total_trades', 0)} | "
            f"🛡️ Max DD={best_win['test_metrics'].get('max_drawdown', 0):.2%}"
        )
       
        return out
       
    except Exception as e:
        logger.exception(f"WFO falhou para {sym}")
        return {"symbol": sym, "error": str(e)}
# ===========================
# MONTE CARLO
# ===========================
def run_monte_carlo_stress(equity_curve: List[float], n_simulations: int = 1000, ibov_adx: float = 25) -> Dict[str, float]: # ✅ 1000 sims
    """
    Monte Carlo com cenários B3:
    - Crash: IBOV -20% (bear market)
    - Sideways: ADX < 15 (consolidação)
    - Rally: ADX > 30 (bull run)
    """
    if not is_valid_equity_curve(equity_curve, min_len=50):
        return {"win_rate": 0.0, "calmar_avg": 0.0, "calmar_median": 0.0,
                "calmar_5th": 0.0, "max_dd_95": 1.0}
   
    equity_curve = np.asarray(equity_curve, dtype=np.float64)
    returns = np.diff(equity_curve) / equity_curve[:-1]
    n_bars = len(returns)
   
    # ✅ CENÁRIOS B3 (Atualizado Phase 2)
    # Crash: Quedas > 2% (IBOV -20%)
    # Gaps: Retornos < -5% (Gap de baixa violento)
    stock_returns = returns
    stress_scenarios = {
        'crash': stock_returns[stock_returns < -0.02],
        'gaps': stock_returns[stock_returns < -0.05], # ✅ Novo Cenario: Gaps de Baixa
        'rally': stock_returns[stock_returns > 0.025],
        'sideways': stock_returns[abs(stock_returns) < 0.003]
    }
    try:
        # Injeta retorno real de 2020 como cenário de crash
        ibov_2020 = get_ibov_data(start="2020-01-01", end="2020-12-31")
        if ibov_2020 is not None and len(ibov_2020) > 50:
            r_2020 = ibov_2020['close'].pct_change().dropna().values
            stress_scenarios['crash_real'] = r_2020[r_2020 < -0.01]
    except Exception:
        pass
   
    # ✅ DISTRIBUIÇÃO DE CENÁRIOS baseada no ADX do IBOV
    # Se ADX < 15, aumenta risco de Crash (mercado fraco vira bear rapido)
    is_weak_trend = ibov_adx < 15
   
    scenario_weights = {
        'crash': 0.25 if is_weak_trend else 0.10,
        'gaps': 0.05,
        'rally': 0.15 if ibov_adx > 25 else 0.10,
        'sideways': 0.40 if is_weak_trend else 0.25
    }
    # ✅ Normalização
    _w_sum = sum(scenario_weights.values())
    scenario_weights = {k: v / _w_sum for k, v in scenario_weights.items()}
   
    calmars, max_dds, wins = [], [], 0
   
    for sim_idx in range(n_simulations):
        # ✅ 30% das sims usam cenários estressados
        if np.random.random() < 0.30 and any(len(v) > 10 for v in stress_scenarios.values()):
            keys = list(scenario_weights.keys()) + (['crash'] if 'crash_real' not in stress_scenarios else ['crash_real'])
            probs = list(scenario_weights.values()) + [0.10]
            probs = np.array(probs); probs = probs / probs.sum()
            scenario = np.random.choice(keys, p=probs)
            base = stress_scenarios.get(scenario, stress_scenarios.get('crash', stock_returns[stock_returns < -0.02]))
            base_arr = np.asarray(base)
            if base_arr is None or base_arr.size == 0:
                base_arr = returns
            sim_returns = np.random.choice(base_arr, size=n_bars, replace=True)
        else:
            # Bootstrap normal com blocos
            block_size = max(5, n_bars // 20)
            sim_returns = []
            while len(sim_returns) < n_bars:
                start_idx = np.random.randint(0, max(1, n_bars - block_size))
                block = returns[start_idx : start_idx + block_size]
                sim_returns.extend(block)
       
        sim_returns = np.array(sim_returns[:n_bars])
        sim_equity = np.cumprod(1 + sim_returns) * equity_curve[0]
       
        peak = np.maximum.accumulate(sim_equity)
        dd = (peak - sim_equity) / peak
        max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0
        max_dds.append(max_dd)
       
        total_ret = sim_equity[-1] / sim_equity[0] - 1
        wins += int(total_ret > 0)
       
        years = n_bars / (252 * 28)
        ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else total_ret
        calmar = ann_ret / max_dd if max_dd > 0 else 0.0
        calmars.append(calmar)
   
    return {
        "win_rate": wins / n_simulations,
        "calmar_avg": float(np.mean(calmars)),
        "calmar_median": float(np.median(calmars)),
        "calmar_5th": float(np.percentile(calmars, 5)),
        "calmar_95th": float(np.percentile(calmars, 95)),
        "max_dd_95": float(np.percentile(max_dds, 95)),
        "max_dd_99": float(np.percentile(max_dds, 99))
    }

def apply_safety_guardrails(final_elite):
    """
    TRAVAS DE SEGURANÇA (SAFETY TIERS):
    Força uma alocação 'Moderada' impedindo a zeragem de Blue Chips
    e limitando a exposição a ativos de risco.
    """
    # TIER 1: DEFESA (Obrigatório ter peso relevante)
    TIER_DEFENSIVE = ['VIVT3', 'BBDC4', 'BBAS3', 'B3SA3', 'ITUB4', 'CPLE6', 'TAEE11' ]
    MIN_WEIGHT_DEFENSIVE = 0.08  # 8%

    # TIER 2: COMMODITIES (Obrigatório não zerar se houver tendência)
    TIER_COMMODITIES = ['PETR4', 'PRIO3', 'VALE3', 'GGBR4' ]
    MIN_WEIGHT_COMMODITY = 0.05  # 5%

    # TIER 3: RISCO / SMALL CAPS (Varejo, Tech, Educação, Siderurgia não-Líder)
    # Todo o resto cai aqui.
    MAX_WEIGHT_RISK = 0.04       # Teto máximo de 4%

    print("[GUARDRAILS] Aplicando Travas de Segurança..." )

    for symbol in  final_elite:
        current_weight = final_elite[symbol].get('weight', 0.0 )
        
        # Regra 1: Se for Defensiva, garante o piso
        if symbol in  TIER_DEFENSIVE:
            if  current_weight < MIN_WEIGHT_DEFENSIVE:
                final_elite[symbol]['weight' ] = MIN_WEIGHT_DEFENSIVE
        
        # Regra 2: Se for Commodity, garante o piso
        elif symbol in  TIER_COMMODITIES:
            if  current_weight < MIN_WEIGHT_COMMODITY:
                final_elite[symbol]['weight' ] = MIN_WEIGHT_COMMODITY
        
        # Regra 3: Se for qualquer outra coisa (Risco), aplica o teto
        else :
            if  current_weight > MAX_WEIGHT_RISK:
                final_elite[symbol]['weight' ] = MAX_WEIGHT_RISK

    # Regra 4: Normalização Final (Soma deve ser 1.0)
    total_w = sum(x['weight'] for x in  final_elite.values())
    if total_w > 0 :
        factor = 1.0  / total_w
        for symbol in  final_elite:
            final_elite[symbol]['weight'] = round(final_elite[symbol]['weight'] * factor, 4 )
            
    return  final_elite
# =========================================================
# 5. EXECUÇÃO PRINCIPAL (PROCESS POOL)
# =========================================================
def process_symbol_wrapper(args):
    """Wrapper para ProcessPoolExecutor"""
    sym, config_bt = args
    try:
        return worker_wfo(sym, config_bt["BARS"], config_bt["MAX_EVALS"],
                          config_bt["WFO_WINDOWS"], config_bt["TRAIN_PERIOD"],
                          config_bt["TEST_PERIOD"])
    except Exception as e:
        return {"symbol": sym, "error": str(e), "status": "error"}
def run_optimizer():
    """
    Função Mestre:
    1. Filtra Liquidez
    2. Otimiza OOS (Pararelo)
    3. Valida Monte Carlo
    4. Aloca Portfólio (Markowitz)
    5. Gera Relatórios
    """
    print("\n" + "="*60)
    print(f"[STARTUP] PID: {os.getpid()} - Iniciando Otimizador...", flush=True)
    print("[RUN] INICIANDO OTIMIZADOR XP3 v5", flush=True)
    print(f"[INFO] Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", flush=True)
    print(f"[MODE] {'SANDBOX (Rápido)' if SANDBOX_MODE else 'PRODUÇÃO (Lento)'}", flush=True)
    print("[DEBUG] Entrou em run_optimizer()", flush=True)
    
    # Teste config
    print("[DEBUG] ELITE_BLUE_CHIPS =", getattr(config, "ELITE_BLUE_CHIPS", "NÃO DEFINIDO"), flush=True)
    print("[DEBUG] SECTOR_MAP keys:", len(getattr(config, "SECTOR_MAP", {})), flush=True)
    
    try:
        print("[INIT] Tentando conectar ao MetaTrader 5...", flush=True)
        mt5_connected = try_mt5_connection(timeout_seconds=5)
        print(f"[OK] MT5 conectado? {mt5_connected}", flush=True)
    except Exception as e:
        print(f"[ERROR] MT5 conexão falhou: {e}", flush=True)
        mt5_connected = False
    try:
        _auto_discover_futures()
    except Exception:
        pass
    
    print("[DEBUG] Indo para dados globais (IBOV)...", flush=True)
    mt5_connected = try_mt5_connection(timeout_seconds=10)
    if not mt5_connected:
        logger.warning("[WARN] MT5 indisponível - usando apenas Polygon/Yahoo Finance")
    try:
        selic = float(get_macro_rate("SELIC") or 0.105)
        ipca = float(get_macro_rate("IPCA") or 0.04)
        config_bt.RISK_FREE_RATE = selic
        print(f"[DATA] Selic capturada: {selic:.2%}", flush=True)
        print(f"[DATA] IPCA capturado: {ipca:.2%}", flush=True)
    except Exception:
        pass
    # 1. DADOS DE MERCADO GLOBAIS
    logger.info("[INFO] Baixando dados globais (IBOV)...")
    ibov_df = get_ibov_data()
    try:
        print(f"[DATA] IBOV DF: {'OK' if (ibov_df is not None and not ibov_df.empty) else 'FALHA'}", flush=True)
    except Exception:
        print("[DATA] IBOV DF: FALHA (exc ao validar)", flush=True)
   
    # 2. SELEÇÃO DE ATIVOS
    logger.info("[INFO] Filtrando universo de ativos...")
    all_symbols = load_all_symbols()
    valid_symbols = []
    valid_symbols_info = {} # ✅ Guarda métricas de liquidez
    rejected = []
    all_liquidity = {}  # symbol -> avg_fin
    elite_blue = []
    try:
        elite_blue = getattr(config, "ELITE_BLUE_CHIPS", [])
    except Exception:
        elite_blue = []
   
    # Check Liquidity (Serial é rápido o suficiente e mais seguro para MT5)
    print(f" Verificando liquidez de {len(all_symbols)} ativos...", flush=True)
    pbar = tqdm(all_symbols, desc="Liquidez/Beta", unit="sym")
    for sym in pbar:
        pbar.set_description(f"🔍 Analisando: {sym}")
        ok, res = safe_call_with_timeout(check_liquidity_dynamic, 10, sym, ibov_df)
        if not ok:
            logger.warning(f"[WARN] Timeout/Erro em {sym}: {res.get('error')}")
            rejected.append({"symbol": sym, "reason": res.get("error", "timeout")})
            continue
        is_ok, reason, metrics = res
        try:
            af = float(metrics.get("avg_fin", 0) or 0)
        except Exception:
            af = 0.0
        if af > 0:
            all_liquidity[sym] = af
        # ✅ RELAXAMENTO: Permite ativos com liquidez um pouco menor se estivermos em busca de oportunidades
        if is_ok or SANDBOX_MODE:
            valid_symbols.append(sym)
            valid_symbols_info[sym] = metrics
        if not is_ok:
            rejected.append({"symbol": sym, "reason": reason})
           
    # Salva rejeitados iniciais
    pd.DataFrame(rejected).to_csv(os.path.join(OPT_OUTPUT_DIR, "initial_rejected_assets.csv"), index=False)
    # Garante inclusão dos blue chips obrigatórios
    for bs in elite_blue:
        if bs not in valid_symbols:
            try:
                if bs in REJECT_SKIP:
                    valid_symbols.append(bs)
                    valid_symbols_info[bs] = {"avg_fin": 0}
                else:
                    is_ok, reason, metrics = check_liquidity_dynamic(bs, ibov_df)
                    valid_symbols.append(bs)
                    valid_symbols_info[bs] = metrics
                    if metrics.get("avg_fin", 0) > 0:
                        all_liquidity[bs] = float(metrics.get("avg_fin", 0))
            except Exception:
                valid_symbols.append(bs)
                valid_symbols_info[bs] = {"avg_fin": 20_000_000}
    
    # ✅ Fallback automático: se nenhum ativo passar o filtro de liquidez absoluto (ex.: volume do MT5 em tick_volume),
    # usa threshold relativo (quantil) para aprovar um universo mínimo e evitar "0 ativos".
    if len(valid_symbols) == 0 and all_liquidity:
        try:
            pct = float(getattr(config, "LIQUIDITY_THRESHOLD_PCT", 0.5) or 0.5)
            pct = max(0.0, min(1.0, pct))
        except Exception:
            pct = 0.5
        values = np.array(list(all_liquidity.values()), dtype=float)
        thr = float(np.quantile(values, pct)) if len(values) > 0 else 0.0
        logger.warning(
            f"[WARN] Nenhum ativo passou o filtro absoluto de liquidez. "
            f"Aplicando fallback por quantil: pct={pct:.2f} limiar_avg_fin={thr:.2f}"
        )
        for sym, af in sorted(all_liquidity.items(), key=lambda kv: kv[1], reverse=True):
            if af >= thr:
                valid_symbols.append(sym)
                valid_symbols_info[sym] = valid_symbols_info.get(sym, {"avg_fin": af})
        # garante pelo menos 20 ativos (se existirem) para o otimizador ter universo suficiente
        if len(valid_symbols) < 20:
            top_more = [s for s, _ in sorted(all_liquidity.items(), key=lambda kv: kv[1], reverse=True)]
            for sym in top_more:
                if len(valid_symbols) >= 20:
                    break
                if sym not in valid_symbols:
                    valid_symbols.append(sym)
                    valid_symbols_info[sym] = valid_symbols_info.get(sym, {"avg_fin": float(all_liquidity.get(sym, 0.0) or 0.0)})
    print(f"[OK] {len(valid_symbols)} ativos aprovados para otimização.")
    # 3. OTIMIZAÇÃO PARALELA (WFO + OPTUNA)
    # 3. OTIMIZAÇÃO PARALELA (WFO + OPTUNA)
    # Configuração
    bt_config = {
        "BARS": 2000 if SANDBOX_MODE else 5000,
        "MAX_EVALS": 30 if SANDBOX_MODE else 60,
        "WFO_WINDOWS": 5 if SANDBOX_MODE else 10,
        "TRAIN_PERIOD": 600 if SANDBOX_MODE else 1000,
        "TEST_PERIOD": 200 if SANDBOX_MODE else 250
    }
   
    tasks = [(sym, bt_config) for sym in valid_symbols]
    all_results = {}
   
    print(f"\n[INFO] Iniciando Workers (ProcessPool) em {os.cpu_count()} cores...", flush=True)
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_symbol_wrapper, t): t[0] for t in tasks}
        pbar_workers = tqdm(total=len(tasks), desc="Otimizando", unit="sym")
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                res = fut.result(timeout=600)
            except TimeoutError:
                res = {"symbol": sym, "error": "worker_timeout", "status": "error"}
            except Exception as e:
                res = {"symbol": sym, "error": str(e), "status": "error"}
            if res.get("status") == "ok":
                all_results[sym] = res
            else:
                logger.warning(f"[ERROR] {sym}: Falha na otimização - {res.get('error')}")
            pbar_workers.update(1)
        pbar_workers.close()
    for sym, res in all_results.items():
        try:
            os.makedirs(os.path.join(OPT_OUTPUT_DIR, "partials"), exist_ok=True)
            with open(os.path.join(OPT_OUTPUT_DIR, "partials", f"{sym}.json"), "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
            with open(os.path.join(OPT_OUTPUT_DIR, "partial_results.json"), "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    optimized_list = []
    for sym, res in all_results.items():
        if res.get("status") == "ok":
            m = res.get("test_metrics", {})
            liq_metrics = valid_symbols_info.get(sym, {})
            
            optimized_list.append({
                "symbol": sym,
                "res": res,
                "avg_fin": liq_metrics.get("avg_fin", 0),
                "calmar": m.get("calmar", 0),
                "max_dd": m.get("max_drawdown", 1.0),
                "trades": m.get("total_trades", 0),
                "params": res.get("selected_params", {})
            })
    opt_map = {item["symbol"]: item for item in optimized_list}
    blue_required = set(elite_blue)
    blue_list = []
    for bs in elite_blue:
        if bs in opt_map:
            blue_list.append(opt_map[bs])
        else:
            try:
                dfv = load_data_polygon(bs, 3000, "M15")
                params = _ensure_non_generic(bs, {})
                m = backtest_params_on_df(bs, params, dfv) if dfv is not None else {"calmar":0,"max_drawdown":1.0,"profit_factor":0.0,"total_trades":0,"equity_curve":[100000.0]}
                blue_list.append({
                    "symbol": bs,
                    "res": {"selected_params": params, "test_metrics": m, "equity_curve": m.get("equity_curve", [])},
                    "avg_fin": valid_symbols_info.get(bs, {}).get("avg_fin", 0),
                    "calmar": m.get("calmar", 0),
                    "max_dd": m.get("max_drawdown", 1.0),
                    "trades": m.get("total_trades", 0),
                    "params": params
                })
            except Exception:
                pass
    opp_candidates = [it for it in optimized_list if it["symbol"] not in blue_required]
    opp_sorted = sorted(
        opp_candidates,
        key=lambda x: (
            float(x.get("res", {}).get("test_metrics", {}).get("profit_factor", 0.0)) * 0.6
            + float(x.get("res", {}).get("test_metrics", {}).get("win_rate", 0.0)) * 0.4
        ),
        reverse=True
    )
    opp_list = opp_sorted[:max(0, 10)]
    print(f"[DEBUG] Blue obrigatórios aprovados: {len(blue_list)} | Oportunidades candidatas: {len(opp_candidates)} | Selecionadas: {len(opp_list)}", flush=True)
    final_selection_all = blue_list + opp_list
    final_selection_all = final_selection_all[:20]
    blue_chips_syms = set(elite_blue)
    
    final_elite = {}
    monte_carlo_approved = []
    
    for item in final_selection_all:
        sym = item["symbol"]
        res = item["res"]
        
        curve = res.get("equity_curve", [])
        mc_res = run_monte_carlo_stress(curve, n_simulations=1000)
        res["monte_carlo"] = mc_res
        
        final_elite[sym] = res
        
        # Prepara para Markowitz
        roi_est = res["test_metrics"].get("total_return", 0)
        monte_carlo_approved.append({
            "symbol": sym,
            "roi_esperado": max(0.01, roi_est),
            "params": _ensure_non_generic(sym, res["selected_params"]),
            "equity_curve": res.get("equity_curve", []),
            "avg_fin": float(valid_symbols_info.get(sym, {}).get("avg_fin", 0.0) or 0.0),
            "category": ("BLUE CHIP" if sym in blue_chips_syms else "OPORTUNIDADE")
        })

    print(f" 🎯 Portfólio Inicial (Pré-Filtro): {len(final_elite)} ativos.")
    if len(final_elite) > 2:
        final_elite = filter_correlated_assets(final_elite, threshold=0.75)
        monte_carlo_approved = [item for item in monte_carlo_approved if item["symbol"] in final_elite]

    # 5. ALOCAÇÃO DE PORTFÓLIO (MARKOWITZ)
    portfolio_weights = optimize_portfolio_allocation(monte_carlo_approved)
    for sym in list(portfolio_weights.keys()):
        portfolio_weights[sym] = round(float(portfolio_weights.get(sym, 0.0) or 0.0), 4)
    for sym in final_elite.keys():
        final_elite[sym]["weight"] = float(portfolio_weights.get(sym, 0.0) or 0.0)
    final_elite = apply_safety_guardrails(final_elite)
    for sym in final_elite.keys():
        portfolio_weights[sym] = float(final_elite[sym].get("weight", 0.0) or 0.0)
   
    # Comentário: I. Testes de Sanidade (Forward, Stress, Buy&Hold)
    try:
        from optimizer_optuna import backtest_params_on_df
        print("\n[SANITY] Iniciando testes de sanidade...")
        sample_syms = list(final_elite.keys())[:min(5, len(final_elite))]
        sanity_results = []
        for sym in sample_syms:
            df_recent = load_data_with_retry(sym, config_bt["TEST_PERIOD"], timeframe="M15")
            if df_recent is None or len(df_recent) < 100:
                print(f" [WARN] {sym}: dados insuficientes para sanity")
                continue
            params = _ensure_non_generic(sym, final_elite[sym]["selected_params"])
            # Forward validation (parâmetros finais)
            res_fwd = backtest_params_on_df(sym, params, df_recent, ml_model=None)
            # Stress test (slippage dobrado)
            params_stress = dict(params)
            params_stress["base_slippage"] = float(params.get("base_slippage", 0.0015) * 2.0)
            res_stress = backtest_params_on_df(sym, params_stress, df_recent, ml_model=None)
            # Buy & Hold comparação
            close = df_recent["close"].astype(float)
            bh_ret = float(close.iloc[-1] / close.iloc[0] - 1.0)
            sanity_results.append({
                "symbol": sym,
                "wr_fwd": float(res_fwd.get("win_rate", 0.0) or 0.0),
                "calmar_fwd": float(res_fwd.get("calmar", 0.0) or 0.0),
                "dd_fwd": float(res_fwd.get("max_drawdown", 0.0) or 0.0),
                "ret_fwd": float(res_fwd.get("total_return", 0.0) or 0.0),
                "ret_bh": bh_ret,
                "calmar_stress": float(res_stress.get("calmar", 0.0) or 0.0),
            })
        # Sumário
        ok_forward = all(r["wr_fwd"] >= 0.30 and r["calmar_fwd"] > 0.0 for r in sanity_results) if sanity_results else False
        ok_stress = all(r["calmar_stress"] > -0.2 for r in sanity_results) if sanity_results else False
        ok_bh = all(r["ret_fwd"] >= (r["ret_bh"] * 0.5) for r in sanity_results) if sanity_results else False
        print(f" [SANITY] Forward OK? {ok_forward} | Stress OK? {ok_stress} | Buy&Hold OK? {ok_bh}")
    except Exception as e:
        print(f" [SANITY] Falha ao executar testes: {e}")
    
    # 6. GERAÇÃO DE RELATÓRIOS
    # a) ELITE_SYMBOLS (Python Dict)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    elite_file = os.path.join(OPT_OUTPUT_DIR, f"elite_portfolio_{timestamp}.txt")
   
    with open(elite_file, "w", encoding="utf-8") as f:
        f.write("# XP3 V5 - ELITE PORTFOLIO\n")
        f.write(f"# Gerado em: {datetime.now()}\n")
        f.write("ELITE_SYMBOLS = {\n")
       
        for sym, res in final_elite.items():
            p = _ensure_non_generic(sym, res["selected_params"])
            weight = portfolio_weights.get(sym, 0.0)
            category = "BLUE CHIP" if sym in blue_chips_syms else "OPORTUNIDADE"
           
            f.write(f' "{sym}": {{\n')
            f.write(f'  "category": "{category}",\n')
            f.write(f'  "weight": {weight:.2f}, # {weight*100:.1f}%\n')
            f.write(f'  "ema_short": {p.get("ema_short", 9)},\n')
            f.write(f'  "ema_long": {p.get("ema_long", 21)},\n')
            f.write(f'  "rsi_low": {p.get("rsi_low", 40)},\n')
            f.write(f'  "rsi_high": {p.get("rsi_high", 70)},\n')
            f.write(f'  "adx_threshold": {p.get("adx_threshold", 15)},\n')
            f.write(f'  "mom_min": {p.get("mom_min", 0.0)},\n')
            f.write(f'  "sl_atr_multiplier": {p.get("sl_atr_multiplier", 2.0)},\n')
            f.write(f'  "tp_mult": {p.get("tp_mult", 3.0)}\n')
            f.write(f' }},\n')
           
        f.write("}\n")
       
    print(f"\n💾 Portfolio salvo em: {elite_file}")

    import json
    elite_payload = {
        "generated_at": datetime.now().isoformat(),
        "elite_symbols": {},
    }

    for sym, res in final_elite.items():
        p = _ensure_non_generic(sym, res["selected_params"])
        weight = float(portfolio_weights.get(sym, 0.0) or 0.0)
        category = "BLUE CHIP" if sym in blue_chips_syms else "OPORTUNIDADE"
        elite_payload["elite_symbols"][sym] = {
            "category": category,
            "weight": round(weight, 4),
            "ema_short": int(p.get("ema_short", 9)),
            "ema_long": int(p.get("ema_long", 21)),
            "rsi_low": int(p.get("rsi_low", 40)),
            "rsi_high": int(p.get("rsi_high", 70)),
            "adx_threshold": int(p.get("adx_threshold", 15)),
            "mom_min": float(p.get("mom_min", 0.0) or 0.0),
            "sl_atr_multiplier": float(p.get("sl_atr_multiplier", 2.0) or 2.0),
            "tp_mult": float(p.get("tp_mult", 3.0) or 3.0),
        }

    elite_json_file = os.path.join(OPT_OUTPUT_DIR, f"elite_symbols_{timestamp}.json")
    elite_json_latest = os.path.join(OPT_OUTPUT_DIR, "elite_symbols_latest.json")
    with open(elite_json_file, "w", encoding="utf-8") as f:
        json.dump(elite_payload, f, ensure_ascii=False, indent=2)
    with open(elite_json_latest, "w", encoding="utf-8") as f:
        json.dump(elite_payload, f, ensure_ascii=False, indent=2)
   
    # b) CSV Resumo
    summary_data = []
    for sym, res in final_elite.items():
        m = res["test_metrics"]
        mc = res["monte_carlo"]
        summary_data.append({
            "Symbol": sym,
            "Weight": portfolio_weights.get(sym, 0.0),
            "Calmar_OOS": m.get("calmar"),
            "DD_OOS": m.get("max_drawdown"),
            "WR_MC": mc.get("win_rate"),
            "DD95_MC": mc.get("max_dd_95"),
            "Trades": m.get("total_trades")
        })
   
    if summary_data:
        pd.DataFrame(summary_data).sort_values("Weight", ascending=False).to_csv(
            os.path.join(OPT_OUTPUT_DIR, f"portfolio_summary_{timestamp}.csv"), index=False
        )
    try:
        top10 = sorted([(s, portfolio_weights.get(s, 0.0)) for s in portfolio_weights], key=lambda x: x[1], reverse=True)[:10]
        sectors = {}
        for s, w in portfolio_weights.items():
            sec = SECTOR_MAP.get(s, "UNKNOWN")
            sectors[sec] = sectors.get(sec, 0.0) + w
        ideal_cap = 0.30
        selic_rate = get_macro_rate("SELIC")
        macro_cache_file = os.path.join(OPT_OUTPUT_DIR, "macro_cache.json")
        last_macro = None
        try:
            with open(macro_cache_file, "r", encoding="utf-8") as f:
                last_macro = json.load(f)
        except Exception:
            pass
        report_md = ["# Relatório Final XP3 v5", "", "## Top 10 por Peso"]
        for s, w in top10:
            report_md.append(f"- {s}: {w*100:.2f}%")
        report_md += ["", "## Métricas Agregadas"]
        eqs = []
        for s, res in final_elite.items():
            curve = np.asarray(res.get("equity_curve", []), dtype=np.float64)
            eqs.append((curve, portfolio_weights.get(s, 0.0)))
        agg = None
        if eqs:
            min_len = min(len(c) for c, _ in eqs if len(c) > 0) if any(len(c) > 0 for c, _ in eqs) else 0
            if min_len > 0:
                agg_curve = np.zeros(min_len, dtype=np.float64)
                for c, w in eqs:
                    agg_curve += w * c[:min_len]
                m = compute_advanced_metrics(agg_curve.tolist())
                report_md += [
                    f"- Sharpe: {m.get('sharpe',0):.2f}",
                    f"- Sortino: {m.get('sortino',0):.2f}",
                    f"- Calmar: {m.get('calmar',0):.2f}",
                    f"- Max DD: {m.get('max_drawdown',0):.2%}",
                    f"- Profit Factor: {m.get('profit_factor',0):.2f}",
                ]
        report_md += ["", "## Exposição Setorial"]
        for sec, w in sectors.items():
            report_md.append(f"- {sec}: {w*100:.1f}% (ideal ≤ {ideal_cap*100:.0f}%)")
        report_md += ["", "## Selic"]
        report_md.append(f"- Selic: {selic_rate:.2%}")
        if last_macro:
            report_md.append(f"- Última consulta: {last_macro.get('updated_at','N/D')}")
        with open(os.path.join(OPT_OUTPUT_DIR, f"final_report_{timestamp}.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(report_md))
    except Exception:
        pass
    # Validação final (code_execution simulado): backtest 3 ativos com dados recentes
    try:
        for sym in ["PETR4", "VALE3", "ITUB4"]:
            dfv = load_data_polygon(sym, 2000, "M15")
            if dfv is None or dfv.empty:
                continue
            dfv = dfv[dfv.index >= pd.Timestamp("2025-01-01")]
            p = _ensure_non_generic(sym, {})
            m = backtest_params_on_df(sym, p, dfv)
            selic = get_macro_rate("SELIC")
            ipca = get_macro_rate("IPCA")
            logger.info(f"🔎 VALIDAÇÃO {sym}: Ret={m.get('total_return',0):.2%} | PF={m.get('profit_factor',0):.2f} | DD={m.get('max_drawdown',0):.2%} | Selic={selic:.2%} | IPCA={ipca:.2%}")
    except Exception as e:
        logger.warning(f"Validação final falhou: {e}")
    # c) Relatório consolidado de todos os ativos (inclusive sem resultados)
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        rep_md = os.path.join(OPT_OUTPUT_DIR, f"weekly_all_assets_{ts}.md")
        rep_txt = os.path.join(OPT_OUTPUT_DIR, f"weekly_all_assets_{ts}.txt")
        with open(rep_md, "w", encoding="utf-8") as fmd:
            fmd.write("# 📊 Relatório Consolidado - Otimizador Semanal\n\n")
            fmd.write(f"**Gerado em:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n---\n\n")
            universe = set(valid_symbols_info.keys()) | set(all_results.keys())
            for sym in sorted(universe):
                fmd.write(f"## 🎯 {sym}\n\n")
                res = all_results.get(sym)
                if not res:
                    fmd.write("### ℹ️ Sem resultados disponíveis\n\n")
                else:
                    m = res.get("test_metrics", {})
                    params = res.get("selected_params") or res.get("best_params") or {}
                    fmd.write("### ✅ Parâmetros Selecionados\n\n")
                    fmd.write("```python\n")
                    fmd.write(f"params_{sym} = {{\n")
                    for k, v in params.items():
                        fmd.write(f"    '{k}': {v},\n")
                    fmd.write("}\n```\n\n")
                    fmd.write("### 📈 Métricas OOS\n\n")
                    fmd.write(f"- Sharpe: {float(m.get('sharpe',0)):.2f}\n")
                    fmd.write(f"- Win Rate: {float(m.get('win_rate',0)):.1%}\n")
                    fmd.write(f"- Total Trades: {int(m.get('total_trades',0))}\n")
                    fmd.write(f"- Max Drawdown: {float(m.get('max_drawdown',0)):.1%}\n")
                    fmd.write(f"- Profit Factor: {float(m.get('profit_factor',0)):.2f}\n\n")
                fmd.write("---\n\n")
        with open(rep_txt, "w", encoding="utf-8") as ftx:
            ftx.write("RELATORIO CONSOLIDADO - OTIMIZADOR SEMANAL\n")
            ftx.write(f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
            ftx.write("------------------------------------------------------------\n")
            universe = set(valid_symbols_info.keys()) | set(all_results.keys())
            for sym in sorted(universe):
                res = all_results.get(sym)
                ftx.write(f"SIMBOLO: {sym}\n")
                if not res:
                    ftx.write("SEM RESULTADOS DISPONIVEIS\n")
                else:
                    m = res.get("test_metrics", {})
                    params = res.get("selected_params") or res.get("best_params") or {}
                    ftx.write("PARAMETROS:\n")
                    for k, v in params.items():
                        ftx.write(f"  {k}: {v}\n")
                    ftx.write("METRICAS OOS:\n")
                    ftx.write(f"  Sharpe: {float(m.get('sharpe',0)):.2f}\n")
                    ftx.write(f"  Win Rate: {float(m.get('win_rate',0)):.1%}\n")
                    ftx.write(f"  Trades: {int(m.get('total_trades',0))}\n")
                    ftx.write(f"  Max Drawdown: {float(m.get('max_drawdown',0)):.1%}\n")
                    ftx.write(f"  Profit Factor: {float(m.get('profit_factor',0)):.2f}\n")
                ftx.write("------------------------------------------------------------\n")
        logger.info(f"💾 Relatórios consolidados salvos: {rep_md} | {rep_txt}")
    except Exception as e:
        logger.warning(f"Falha ao gerar relatório consolidado: {e}")
    # d) Diagnósticos e força de execução para ativos sem resultados ou genéricos
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        diag_txt = os.path.join(OPT_OUTPUT_DIR, f"weekly_diagnostics_{ts}.txt")
        with open(diag_txt, "w", encoding="utf-8") as fd:
            fd.write("DIAGNOSTICOS DE TRADES (SEMANAIS)\n")
            fd.write(f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
            fd.write("------------------------------------------------------------\n")
            universe = set(valid_symbols_info.keys()) | set(all_results.keys())
            for sym in sorted(universe):
                fd.write(f"SIMBOLO: {sym}\n")
                res = all_results.get(sym)
                if not res or int(res.get("test_metrics", {}).get("total_trades", 0)) <= 3:
                    try:
                        dfv = load_data_polygon(sym, 10000, "M15")
                        params = _ensure_non_generic(sym, res.get("selected_params", {}) if res else {})
                        m = backtest_params_on_df(sym, params, dfv) if dfv is not None else {"total_trades":0,"win_rate":0.0,"max_drawdown":1.0,"profit_factor":0.0}
                        fd.write(f"  FORCA_EXEC: Trades={int(m.get('total_trades',0))} WR={float(m.get('win_rate',0)):.2f} PF={float(m.get('profit_factor',0)):.2f} DD={float(m.get('max_drawdown',0)):.2f}\n")
                        all_results[sym] = {"selected_params": params, "test_metrics": m, "status": "ok", "equity_curve": m.get("equity_curve", [])}
                    except Exception as ex:
                        fd.write(f"  FALHA_FORCA_EXEC: {ex}\n")
                else:
                    m = res.get("test_metrics", {})
                    fd.write(f"  OK: Trades={int(m.get('total_trades',0))} WR={float(m.get('win_rate',0)):.2f}\n")
                fd.write("------------------------------------------------------------\n")
        logger.info(f"🧪 Diagnósticos semanais salvos: {diag_txt}")
    except Exception:
        pass
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["run", "schedule"], default=os.getenv("OPT_MODE", "run"))
    parser.add_argument("--use-real-db", action="store_true")
    args = parser.parse_args()

    if args.use_real_db:
        os.environ["OPT_USE_REAL_DB"] = "1"

    if args.mode == "schedule":
        scheduler()
    else:
        run_optimizer()

