"""
OTIMIZADOR COM AUTO-SYNC DE MARKET WATCH - VERSÃO CORRIGIDA FINAL
✅ Reconexão automática MT5
✅ Workers com conexão própria
✅ ADX calculado corretamente
✅ Lógica de entrada flexível
✅ Import circular corrigido
"""
import os
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import requests
import config
import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import njit
from concurrent.futures import ThreadPoolExecutor
import ta
try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None
try:
    import config
except Exception:
    config = None
try:
    import utils
except Exception:
    utils = None
from tenacity import retry, stop_after_attempt, wait_fixed
from scipy.optimize import minimize

try:
    from backfill import ensure_history
except Exception:
    ensure_history = None

def is_valid_dataframe(df: Optional[pd.DataFrame], min_rows: int = 100) -> bool:
    """Valida se o DataFrame tem dados suficientes e é válido"""
    if df is None: return False
    if not isinstance(df, pd.DataFrame): return False
    if df.empty: return False
    if len(df) < min_rows: return False
    required_cols = {'open', 'high', 'low', 'close'}
    if not required_cols.issubset(set(df.columns)): return False
    return True

# ✅ Retry automático com Tenacity
# ✅ Retry automático com Tenacity
@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
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
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("otimizador_auto_sync")
OPT_OUTPUT_DIR = getattr(config, "OPTIMIZER_OUTPUT", "optimizer_output")
os.makedirs(OPT_OUTPUT_DIR, exist_ok=True)
SECTOR_MAP = config.SECTOR_MAP
COMMODITY_SYMBOLS = {"PETR4", "VALE3", "PRIO3", "GGBR4", "CSNA3", "SUZB3", "KLBN11", "ENEV3"}
import warnings
try:
    from optuna.exceptions import ExperimentalWarning
    warnings.filterwarnings("ignore", category=ExperimentalWarning)
except ImportError:
    pass
# ===========================
# CONFIGURAÇÕES GLOBAIS
# ===========================
SANDBOX_MODE = False # Sniper Mode forced active
BARS_TO_LOAD = 5000
# ===========================
# GARANTIR CONEXÃO MT5
# ===========================
def ensure_mt5_connection() -> bool:
    """
    Valida e força conexão com o terminal correto do MT5 (via Tenacity)
    """
    try:
        safe_mt5_initialize()
        terminal = mt5.terminal_info()
        if terminal and terminal.connected:
            logger.info(f"✅ MT5 conectado (Tenacity OK)")
            return True
        return False
    except Exception as e:
        logger.critical(f"🚨 FALHA CRÍTICA: Não foi possível conectar ao MT5: {e}")
        return False
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
    logger.info("🔹 Calculando Fronteira Eficiente (Markowitz Puro)...")
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
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},)
    bounds = tuple([(0.0, 0.15) for _ in range(n)])
    init_guess = [1 / n] * n
    res = minimize(negative_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    allocations = {}
    for i, asset in enumerate(selected_assets_metrics):
        allocations[asset['symbol']] = res.x[i]
    return allocations
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
    """Busca taxa macro (ex: Selic) via API BCB (SGS)"""
    try:
        if rate_name.upper() in ("SELIC", "SELIC_D", "SELIC_DIARIA"):
            from datetime import date
            today = date.today()
            start = today.replace(year=today.year - 10)
            url = (
                "https://api.bcb.gov.br/dados/serie/bcdata.sgs.11/dados"
                f"?formato=json&dataInicial={start.strftime('%d/%m/%Y')}&dataFinal={today.strftime('%d/%m/%Y')}"
            )
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and len(data) > 0:
                    last = data[-1]
                    val = float(last.get("valor", "0").replace(",", "."))
                    return val / 100.0
        return 0.12
    except Exception:
        return 0.12
def get_ibov_data():
    """Baixa dados do IBOV para cálculo de Beta"""
    try:
        ibov = mt5.copy_rates_from_pos("IBOV", mt5.TIMEFRAME_D1, 0, 300)
        if ibov is None: return None
        return pd.DataFrame(ibov)
    except:
        return None
def classify_asset_profile(symbol: str, df_ibov: Optional[pd.DataFrame] = None) -> str:
    """
    Classifica o ativo dinamicamente por Beta, Volatilidade e Tendência (commodities):
    - CORE_DEFENSIVE: Beta < 0.85 e Vol anualizada < 30%
    - CORE_COMMODITIES: símbolo em COMMODITY_SYMBOLS e SMA50 > SMA200 (tendência de alta)
    - HIGH_VOLATILITY: caso contrário (inclui commodities em baixa)
    """
    try:
        if not ensure_mt5_connection():
            return "HIGH_VOLATILITY"
        mt5.symbol_select(symbol, True)
        d1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 200)
        if d1 is None or len(d1) < 100:
            return "HIGH_VOLATILITY"
        df_a = pd.DataFrame(d1)
        a_ret = df_a['close'].pct_change().fillna(0).tail(126)
        if df_ibov is None or len(df_ibov) < 150:
            df_ibov = get_ibov_data()
        if df_ibov is None or len(df_ibov) < 150:
            return "HIGH_VOLATILITY"
        ib_ret = df_ibov['close'].pct_change().fillna(0).tail(126)
        min_len = min(len(a_ret), len(ib_ret))
        a_ret = a_ret.iloc[-min_len:]
        ib_ret = ib_ret.iloc[-min_len:]
        cov = float(np.cov(a_ret, ib_ret)[0][1])
        var_ib = float(np.var(ib_ret))
        beta = cov / var_ib if var_ib > 0 else 1.0
        vol_ann = float(np.std(a_ret)) * np.sqrt(252)
        sma50 = df_a['close'].rolling(50).mean().iloc[-1]
        sma200 = df_a['close'].rolling(200).mean().iloc[-1] if len(df_a) >= 200 else sma50
        is_uptrend = sma50 > sma200
        if (beta < 0.85) and (vol_ann < 0.30):
            return "CORE_DEFENSIVE"
        if (symbol in COMMODITY_SYMBOLS) and is_uptrend:
            return "CORE_COMMODITIES"
        return "HIGH_VOLATILITY"
    except Exception:
        return "HIGH_VOLATILITY"
def check_liquidity_dynamic(sym: str, ibov_df: pd.DataFrame = None) -> dict:
    """
    Verifica liquidez dinâmica e Filtro de Beta (Blue Chips).
    """
    # Importação lazy para evitar erro se não estiver no topo
    from optimizer_optuna import calculate_adx
    try:
        if not ensure_mt5_connection():
            return False, "SEM_CONEXAO_MT5", {}
        # ✅ Infro: Garante que o símbolo está no Market Watch
        mt5.symbol_select(sym, True)
       
        # ✅ D1 para EMA80/200 e liquidez
        rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_D1, 1, 300)
        if rates is None or len(rates) == 0:
            logger.warning(f"⚠️ {sym}: Falha total ao capturar D1. Verifique corretora.")
            return False, "SEM_DADOS_D1", {}
        df_d1 = pd.DataFrame(rates)
       
        # ✅ TRATAMENTO DE VOLUME (B3)
        vol_col = None
        if 'real_volume' in df_d1.columns and df_d1['real_volume'].sum() > 0:
            vol_col = 'real_volume'
        elif 'tick_volume' in df_d1.columns:
            vol_col = 'tick_volume'
           
        if not vol_col:
            return False, "VOLUME_ZERO", {}
        # Média Financeira dos últimos 20 dias
        df_liq = df_d1.tail(20)
        avg_vol_shares = df_liq[vol_col].mean()
        avg_price = df_liq['close'].mean()
        avg_fin_vol = avg_vol_shares * avg_price
        # Critério de Liquidez
        MIN_FINANCEIRO = 1_000_000 # 1M Reais
        if SANDBOX_MODE:
            MIN_FINANCEIRO = 0
       
        is_liquid = avg_fin_vol >= MIN_FINANCEIRO
        reason = "" if is_liquid else f"Baixa Liquidez (< R$ {MIN_FINANCEIRO/1_000_000:.1f}M)"
       
        # Filtro Beta desativado
        # Filtro EMA80 desativado
        # ✅ CÁLCULO DO ADX THRESHOLD (Apenas métrica, não bloqueia aqui)
        adx_threshold = 20
        rates_m15 = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M15, 0, 500)
        if rates_m15 is not None and len(rates_m15) > 50:
             df_m15 = pd.DataFrame(rates_m15)
             df_m15 = pd.DataFrame(rates_m15)
             _, atr_vals = calculate_adx(df_m15['high'].values, df_m15['low'].values, df_m15['close'].values)
             avg_atr_pct = np.mean(atr_vals[-50:]) / np.mean(df_m15['close'].values[-50:])
             adx_threshold = 18 if avg_atr_pct < 0.005 else 22
       
        return is_liquid, reason, {
            "avg_fin": avg_fin_vol,
            "adx_threshold": adx_threshold
        }
    except Exception as e:
        logger.error(f"Erro no check_liquidity para {sym}: {e}")
        return False, f"ERRO_CHECK: {str(e)}", {}
# --- CÓDIGO DUPLICADO REMOVIDO PARA LIMPEZA ---
# Otimizador Principal consolidado no final do arquivo.

def scheduler():
    try:
        import schedule
    except Exception as e:
        print(f"⚠️ Scheduler indisponível: {e}. Rodando uma vez.")
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
    print("🔓 ABRINDO TODOS OS SÍMBOLOS DO MT5")
    print("="*80)
   
    if not ensure_mt5_connection():
        logger.error("❌ MT5 não disponível")
        return 0
   
    # Pega TODOS os símbolos disponíveis no MT5
    all_symbols = mt5.symbols_get()
    if not all_symbols:
        logger.error("❌ Nenhum símbolo disponível no MT5")
        return 0
   
    total_symbols = len(all_symbols)
    print(f"\n📊 Total de símbolos no MT5: {total_symbols}")
    print(f"⏳ Adicionando todos ao Market Watch...")
   
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
   
    print(f"\n✅ {added}/{total_symbols} símbolos adicionados")
    if failed > 0:
        print(f"⚠️ {failed} símbolos falharam")
   
    return added
def sync_market_watch_with_sector_map(clear_first: bool = True) -> bool:
    """
    STEP 2: Filtra Market Watch para manter apenas símbolos do SECTOR_MAP
    """
    print("\n" + "="*80)
    print("🎯 FILTRANDO APENAS SÍMBOLOS DO SECTOR_MAP")
    print("="*80)
   
    if not ensure_mt5_connection():
        logger.error("❌ MT5 não disponível")
        return False
   
    if SANDBOX_MODE:
        desired_symbols = {'PETR4', 'VALE3', 'ITUB4', 'BBDC4', 'ABEV3'}
    else:
        desired_symbols = {k.upper().strip() for k in SECTOR_MAP.keys()
                          if isinstance(k, str) and k.strip()}
   
    if not desired_symbols:
        logger.error("❌ SECTOR_MAP vazio")
        return False
   
    print(f"\n📊 SECTOR_MAP contém: {len(desired_symbols)} símbolos")
   
    all_symbols = mt5.symbols_get()
    current_symbols = {s.name for s in all_symbols if s.visible} if all_symbols else set()
   
    print(f"📊 Market Watch atual: {len(current_symbols)} símbolos")
   
    # Símbolos que precisam ser adicionados (estão no SECTOR_MAP mas não no Market Watch)
    to_add = desired_symbols - current_symbols
   
    # Símbolos que podem ser removidos (estão no Market Watch mas não no SECTOR_MAP)
    to_remove = current_symbols - desired_symbols
   
    already_ok = current_symbols & desired_symbols
   
    print(f"\n📋 ANÁLISE:")
    print(f" • Já corretos (SECTOR_MAP): {len(already_ok)} símbolos")
    print(f" • Adicionar do SECTOR_MAP: {len(to_add)} símbolos")
   
    if clear_first:
        print(f" • Remover (não estão no SECTOR_MAP): {len(to_remove)} símbolos")
    else:
        print(f" • Manter extras no Market Watch: {len(to_remove)} símbolos")
   
    # ✅ REMOVIDO: Sem confirmação manual
    # Remove símbolos que NÃO estão no SECTOR_MAP (se clear_first=True)
    if clear_first and to_remove:
        print(f"\n🗑️ Removendo {len(to_remove)} símbolos automaticamente...")
        removed = 0
        for symbol in tqdm(list(to_remove), desc="Removendo"):
            try:
                if mt5.symbol_select(symbol, False):
                    removed += 1
                time.sleep(0.01)
            except:
                pass
        print(f"✅ {removed} símbolos removidos")
   
    # Adiciona símbolos do SECTOR_MAP que estão faltando
    if to_add:
        print(f"\n➕ Adicionando {len(to_add)} símbolos do SECTOR_MAP...")
        added = 0
        failed = []
       
        for symbol in tqdm(sorted(to_add), desc="Adicionando"):
            try:
                info = mt5.symbol_info(symbol)
                if not info:
                    logger.warning(f" ⚠️ {symbol} não existe no MT5")
                    failed.append(symbol)
                    continue
               
                if mt5.symbol_select(symbol, True):
                    added += 1
                else:
                    failed.append(symbol)
               
                time.sleep(0.05)
               
            except Exception as e:
                logger.warning(f" ❌ {symbol}: {e}")
                failed.append(symbol)
       
        print(f"✅ {added}/{len(to_add)} símbolos adicionados")
       
        if failed:
            print(f"\n⚠️ {len(failed)} símbolos falharam:")
            for sym in failed[:10]:
                print(f" - {sym}")
            if len(failed) > 10:
                print(f" ... e mais {len(failed) - 10}")
   
    final_symbols = mt5.symbols_get()
    final_in_sector_map = len([s for s in final_symbols if s.visible and s.name in desired_symbols]) if final_symbols else 0
    final_total = len([s for s in final_symbols if s.visible]) if final_symbols else 0
   
    print(f"\n✅ SINCRONIZAÇÃO CONCLUÍDA!")
    print(f" Market Watch Total: {final_total} símbolos")
    print(f" Do SECTOR_MAP: {final_in_sector_map}/{len(desired_symbols)} ({final_in_sector_map/len(desired_symbols)*100:.1f}%)")
    print("="*80)
   
    return final_in_sector_map >= len(desired_symbols) * 0.9
# ===========================
# CARREGAMENTO DE DADOS
# ===========================
def load_data_with_retry(symbol: str, bars: int, timeframe=None, max_retries: int = 3) -> Optional[pd.DataFrame]:
    """Carrega dados com retry e verificação de conexão"""
    if not mt5:
        return None
   
    if timeframe is None:
        timeframe = mt5.TIMEFRAME_M15
   
    # 1. TENTA COPY_RATES PADRÃO
    for attempt in range(max_retries):
        if not ensure_mt5_connection():
            logger.error(f"❌ {symbol}: MT5 desconectado (tentativa {attempt+1})")
            time.sleep(2)
            continue
       
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
           
            if rates is not None and len(rates) >= 500: # Exigimos pelo menos 500 barras para qualidade
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
               
                # ✅ FIX: Renomeia tick_volume para volume (Padrão B3/MT5)
                if 'tick_volume' in df.columns:
                    df = df.rename(columns={'tick_volume': 'volume'})
                elif 'real_volume' in df.columns:
                    df = df.rename(columns={'real_volume': 'volume'})
               
                if 'volume' not in df.columns:
                    df['volume'] = 1.0
                df['volume'] = df['volume'].fillna(0)
                df = df[['open', 'high', 'low', 'close', 'volume']]
                df = df[~df.index.duplicated(keep='last')].sort_index()
               
                logger.debug(f"✅ {symbol}: {len(df)} barras")
                return df
            else:
                logger.warning(f"⚠️ {symbol}: Poucos dados ({len(rates) if rates is not None else 0}) - Tentando backfill...")
                break # Sai do loop normal para tentar backfill forçado
               
        except Exception as e:
            logger.warning(f"⚠️ {symbol}: {e} (tentativa {attempt+1})")
            time.sleep(1)
   
    # 2. SE FALHOU OU POUCOS DADOS, TENTA FORCE_BACKFILL
    if ensure_history:
        try:
            logger.info(f"🔄 {symbol}: Executando force_backfill (ensure_history)...")
            df = ensure_history(symbol, period_days=90, interval='15m')
            if df is not None and len(df) >= 500:
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
                    df = df.set_index('time')
               
                # Normaliza colunas
                if 'tick_volume' in df.columns: df = df.rename(columns={'tick_volume': 'volume'})
                if 'volume' not in df.columns: df['volume'] = 1.0
               
                df = df[['open', 'high', 'low', 'close', 'volume']]
                logger.info(f"✅ {symbol}: {len(df)} barras via backfill")
                return df.sort_index()
        except Exception as e:
            logger.warning(f"⚠️ {symbol}: force_backfill falhou: {e}")
   
    # 3. ÚLTIMA TENTATIVA VIA UTILS
    if utils and hasattr(utils, "safe_copy_rates"):
        try:
            df = utils.safe_copy_rates(symbol, timeframe, count=bars)
            if df is not None and len(df) >= 100:
                if "time" in df.columns:
                    try:
                        df["time"] = pd.to_datetime(df["time"])
                        df = df.set_index("time")
                    except Exception:
                        pass

                if "tick_volume" in df.columns:
                    df = df.rename(columns={"tick_volume": "volume"})
                elif "real_volume" in df.columns:
                    df = df.rename(columns={"real_volume": "volume"})

                if "volume" not in df.columns:
                    df["volume"] = 1.0
                df["volume"] = df["volume"].fillna(0)

                if all(c in df.columns for c in ("open", "high", "low", "close")):
                    df = df[["open", "high", "low", "close", "volume"]]
                return df.sort_index()
        except Exception:
            pass
    logger.error(f"❌ {symbol}: Falha crítica de dados (Falta de Histórico)")
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
@njit
def fast_backtest_core(
    close, high, low, volume, volume_ma,
    ema_short, ema_long, rsi, rsi_2, adx, momentum, atr,
    rsi_low, rsi_high,
    adx_threshold, mom_min,
    sl_mult, tp_mult, base_slippage,
    risk_per_trade=0.01,
    use_trailing=True # Trailing desligado para focar no Alvo Fixo
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
   
    # Custos B3
    transaction_cost_pct = 0.0003 + 0.00025
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
                gross_profit = (exit_price - entry_price) * (position / entry_price)
                cost = (position * transaction_cost_pct) + ((position + gross_profit) * transaction_cost_pct)
                net_profit = gross_profit - cost
               
                cash += position + net_profit
                costs_paid += cost
                if net_profit > 0: wins += 1
                else: losses += 1
               
                trades += 1
                state = 0
                position = 0.0
                equity = cash
           
            # ❌ VERIFICAÇÃO DE STOP LOSS
            elif low[i] <= stop_price:
                exit_price = stop_price - (stop_price * base_slippage) # Slippage no stop
                gross_profit = (exit_price - entry_price) * (position / entry_price)
                cost = (position * transaction_cost_pct) + ((position + gross_profit) * transaction_cost_pct)
                net_profit = gross_profit - cost
               
                cash += position + net_profit
                costs_paid += cost
                losses += 1 # Stop é sempre Loss
                trades += 1
                state = 0
                position = 0.0
                equity = cash
            # ⏳ MANUTENÇÃO (Atualiza Equity mas não sai)
            else:
                current_val = position * (current_price / entry_price)
                equity = cash + (current_val - position)
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
                target_price = entry_price + tp_dist # ✅ Define o Alvo Fixo
               
                # Tamanho da posição (Risco Fixo 2%)
                risk_amt = equity * risk_per_trade
                if sl_dist > 0:
                    shares_raw = risk_amt / sl_dist
                    cost_basis = shares_raw * entry_price
                    # Limita a 95% do caixa
                    if cost_basis > cash * 0.95:
                        cost_basis = cash * 0.95
                   
                    position = cost_basis
                    cash -= position
                    state = 1
       
        equity_curve[i] = equity
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
    f"{symbol}: Sinais={n_signals}/{len(df)} ({n_signals/len(df)*100:.1f}%) | "
    f"Trend={np.sum(trend_up)} | RSI={np.sum(rsi_ok)} | "
    f"Mom={np.sum(momentum_ok)} | ADX={np.sum(adx_ok)} | "
    f"ADX_avg={adx_avg:.1f}"
)
    # ✅ VOLUME PARA SLIPPAGE DINÂMICO e FILTROS
    volume = df['volume'].values.astype(np.float64)
    volume_ma = pd.Series(volume).rolling(20).mean().fillna(0).values
    # --- EXECUÇÃO DO BACKTEST ---
    # ✅ FIX: fast_backtest_core retorna tupla (equity_curve, trades, wins, losses, final_return, win_rate)
    equity_arr, trades, wins, losses, total_return, win_rate, costs_paid = fast_backtest_core(
        close, high, low, volume, volume_ma,
        ema_s, ema_l, rsi, rsi_2, adx, momentum, atr,
        params.get("rsi_low", 30),
        params.get("rsi_high", 70),
        params.get("adx_threshold", 25),
        params.get("mom_min", 0.0),
        params.get("sl_atr_multiplier", 2.5),
        params.get("tp_mult", 5.0),
        params.get("base_slippage", 0.0035),
        0.01,
        True # use_trailing
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
# ===========================
# WORKER WFO
# ===========================
def worker_wfo(sym: str, bars: int, maxevals: int, wfo_windows: int,
               train_period: int, test_period: int) -> Dict[str, Any]:
    """Worker WFO com reconexão MT5 automática"""
    out = {"symbol": sym, "status": "ok", "wfo_windows": []}
   
    try:
        if not mt5 or not mt5.initialize():
            logger.error(f"❌ {sym}: Falha ao inicializar MT5 no worker")
            return {"symbol": sym, "error": "mt5_init_failed"}
       
        time.sleep(0.5)
       
        df_full = load_data_with_retry(sym, bars)
       
        if not is_valid_dataframe(df_full):
            mt5.shutdown()
            return {"symbol": sym, "error": "no_data"}
       
        df_full = df_full.sort_index()
        n = len(df_full)
       
        if n < (train_period + test_period):
            mt5.shutdown()
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
               
                logger.info(f"{sym}: Janela {i+1}/{wfo_windows} - Iniciando Optuna...")
                res = optimize_with_optuna(sym, df_train, n_trials=60, timeout=600)
               
                if res.get("status") == "SUCCESS":
                    best_params = res.get("best_params", {})
                    ml_model = res.get("ml_model", None)
                    logger.info(f"{sym}: ✅ Parâmetros otimizados: {best_params}")
                else:
                    reason = res.get("reason", res.get("status", "Unknown"))
                    logger.warning(f"{sym}: ⚠️ Optuna sem resultado válido nesta janela ({reason})")
                    if len(wins) > 0:
                        logger.warning(f"{sym}: Usando parâmetros da melhor janela anterior")
                        best_params = wins[-1]["best_params"]
                        ml_model = wins[-1].get("ml_model")
                    else:
                        logger.warning(f"{sym}: Usando fallback genérico")
                        best_params = {
                            "ema_short": 9, "ema_long": 21, "rsi_low": 30, "rsi_high": 70,
                            "adx_threshold": 25, "sl_atr_multiplier": 2.5, "tp_mult": 5.0
                        }
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
                            ema_short_grid = [11, 13, 17, 21, 25]
                            ema_long_grid = [56, 72, 90, 97]
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
            mt5.shutdown()
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
            f"{sym}: Calmar OOS = {best_win['test_metrics'].get('calmar', 0):.3f} | "
            f"Retorno = {best_win['test_metrics'].get('total_return', 0):.2%} | "
            f"Trades = {best_win['test_metrics'].get('total_trades', 0)} | "
            f"Max DD = {best_win['test_metrics'].get('max_drawdown', 0):.2%}"
        )
       
        mt5.shutdown()
        return out
       
    except Exception as e:
        logger.exception(f"WFO falhou para {sym}")
        if mt5:
            mt5.shutdown()
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
        if np.random.random() < 0.30 and all(len(v) > 10 for v in stress_scenarios.values()):
            scenario = np.random.choice(list(scenario_weights.keys()), p=list(scenario_weights.values()))
            sim_returns = np.random.choice(stress_scenarios[scenario], size=n_bars, replace=True)
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
    print(f"🚀 INICIANDO OTIMIZADOR XP3 v5 (High Win Rate + Markowitz)")
    print(f"📅 Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"💻 Mode: {'SANDBOX (Rápido)' if SANDBOX_MODE else 'PRODUÇÃO (Lento)'}")
    print("="*60 + "\n")
    if not ensure_mt5_connection():
        logger.critical("❌ Falha crítica no MT5. Abortando.")
        return
    # 1. DADOS DE MERCADO GLOBAIS
    logger.info("🌍 Baixando dados globais (IBOV)...")
    ibov_df = get_ibov_data()
   
    # 2. SELEÇÃO DE ATIVOS
    logger.info("🔍 Filtrando universo de ativos...")
    all_symbols = load_all_symbols()
    valid_symbols = []
    valid_symbols_info = {} # ✅ Guarda métricas de liquidez
    rejected = []
    all_liquidity = {}  # symbol -> avg_fin
   
    # Check Liquidity (Serial é rápido o suficiente e mais seguro para MT5)
    print(f" Verificando liquidez de {len(all_symbols)} ativos...")
    for sym in tqdm(all_symbols, desc="Liquidez/Beta"):
        is_ok, reason, metrics = check_liquidity_dynamic(sym, ibov_df)
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
            f"⚠️ Nenhum ativo passou o filtro absoluto de liquidez. "
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
    print(f" ✅ {len(valid_symbols)} ativos aprovados para otimização.")
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
   
    print(f"\n🚀 Iniciando Workers (ProcessPool) em {os.cpu_count()} cores...")
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(executor.map(process_symbol_wrapper, tasks), total=len(tasks), desc="Otimizando"))
       
    for res in results:
        sym = res["symbol"]
        if res.get("status") == "ok":
            all_results[sym] = res
        else:
            logger.warning(f"❌ {sym}: Falha na otimização - {res.get('error')}")
    # 4. SELEÇÃO DE PORTFÓLIO (REGRA 10+10 GARANTIDA)
    print(f"\n🔍 Selecionando TOP 10 Blue Chips + TOP 10 Oportunidades...")
    
    # Prepara lista de sucessos para ranking
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

    # Overlay de trades reais desativado
    for item in optimized_list:
        item["calmar_adj"] = float(item.get("calmar", 0) or 0)

    final_selection_all = sorted(
        optimized_list,
        key=lambda x: (
            float(x.get("res", {}).get("test_metrics", {}).get("profit_factor", 0.0)) * 0.7
            - float(x.get("res", {}).get("test_metrics", {}).get("max_drawdown", 1.0)) * 0.3
        ),
        reverse=True
    )[:20]
    blue_chips_syms = set()
    for item in final_selection_all:
        if float(item.get("avg_fin", 0) or 0) >= 10_000_000:
            blue_chips_syms.add(item["symbol"])
    
    final_elite = {}
    monte_carlo_approved = []
    
    for item in final_selection_all:
        sym = item["symbol"]
        res = item["res"]
        
        # Monte Carlo (Rodado mas não bloqueia a inclusão no relatório se o usuário quiser o Top 20)
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
            "category": ("BLUE CHIP" if sym in blue_chips_syms else "OPORTUNIDADE")
        })

    print(f" 🎯 Portfólio de Elite composto por {len(final_elite)} ativos.")

    # 5. ALOCAÇÃO DE PORTFÓLIO (MARKOWITZ)
    portfolio_weights = optimize_portfolio_allocation(monte_carlo_approved)
    adjusted = {}
    for sym in final_elite.keys():
        w = float(portfolio_weights.get(sym, 0.0) or 0.0)
        label = classify_asset_profile(sym, ibov_df)
        if label == "CORE_DEFENSIVE":
            w = max(w, 0.10)
        elif label == "CORE_COMMODITIES":
            w = max(w, 0.09)
        else:
            w = min(w, 0.04)
        adjusted[sym] = w
    tw = sum(adjusted.values())
    if tw > 0:
        for sym in adjusted.keys():
            portfolio_weights[sym] = round(adjusted[sym] / tw, 4)
   
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

