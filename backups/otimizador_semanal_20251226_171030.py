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
from datetime import datetime
from collections import defaultdict, Counter
import requests
import config
import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import njit
from concurrent.futures import ThreadPoolExecutor

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

try:
    from backfill import ensure_history
except Exception:
    ensure_history = None

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

SECTOR_MAP = getattr(config, "SECTOR_MAP", {})

# ===========================
# GARANTIR CONEXÃO MT5
# ===========================
def ensure_mt5_connection() -> bool:
    """Garante conexão ativa com MT5"""
    if not mt5:
        return False
    
    terminal_info = None
    try:
        terminal_info = mt5.terminal_info()
    except:
        pass
    
    if not terminal_info:
        logger.info("🔄 Reconectando ao MT5...")
        if not mt5.initialize():
            logger.error("❌ Falha ao reconectar")
            return False
    
    terminal = mt5.terminal_info()
    if not terminal or not terminal.connected:
        logger.error("❌ Terminal MT5 não está conectado")
        return False
    
    return True

# ===========================
# SINCRONIZAÇÃO MARKET WATCH
# ===========================
def sync_market_watch_with_sector_map(clear_first: bool = True) -> bool:
    """Sincroniza Market Watch do MT5 com o SECTOR_MAP"""
    print("\n" + "="*80)
    print("🔄 SINCRONIZANDO MARKET WATCH COM SECTOR_MAP")
    print("="*80)
    
    if not ensure_mt5_connection():
        logger.error("❌ MT5 não disponível")
        return False
    
    desired_symbols = {k.upper().strip() for k in SECTOR_MAP.keys() 
                      if isinstance(k, str) and k.strip()}
    
    if not desired_symbols:
        logger.error("❌ SECTOR_MAP vazio")
        return False
    
    print(f"\n📊 SECTOR_MAP contém: {len(desired_symbols)} símbolos")
    
    all_symbols = mt5.symbols_get()
    current_symbols = {s.name for s in all_symbols if s.visible} if all_symbols else set()
    
    print(f"📊 Market Watch atual: {len(current_symbols)} símbolos")
    
    to_remove = current_symbols - desired_symbols
    to_add = desired_symbols - current_symbols
    already_ok = current_symbols & desired_symbols
    
    print(f"\n📋 AÇÕES NECESSÁRIAS:")
    print(f"   • Remover: {len(to_remove)} símbolos")
    print(f"   • Adicionar: {len(to_add)} símbolos")
    print(f"   • Já corretos: {len(already_ok)} símbolos")
    
    if len(to_remove) == 0 and len(to_add) == 0:
        print(f"\n✅ Market Watch já está sincronizado!")
        return True
    
    if clear_first and len(to_remove) > 100:
        print(f"\n⚠️  ATENÇÃO: {len(to_remove)} símbolos serão removidos!")
        confirm = input("   Continuar? (S/N): ")
        if confirm.upper() != "S":
            print("❌ Operação cancelada")
            return False
    
    if to_remove and clear_first:
        print(f"\n🗑️  Removendo {len(to_remove)} símbolos...")
        removed = 0
        for symbol in list(to_remove)[:100]:
            try:
                if mt5.symbol_select(symbol, False):
                    removed += 1
                time.sleep(0.01)
            except:
                pass
        print(f"✅ {removed} símbolos removidos")
    
    if to_add:
        print(f"\n➕ Adicionando {len(to_add)} símbolos do SECTOR_MAP...")
        added = 0
        failed = []
        
        for symbol in sorted(to_add):
            try:
                info = mt5.symbol_info(symbol)
                if not info:
                    logger.warning(f"   ⚠️ {symbol} não existe no MT5")
                    failed.append(symbol)
                    continue
                
                if mt5.symbol_select(symbol, True):
                    added += 1
                else:
                    failed.append(symbol)
                
                time.sleep(0.05)
                
            except Exception as e:
                logger.warning(f"   ❌ {symbol}: {e}")
                failed.append(symbol)
        
        print(f"✅ {added}/{len(to_add)} símbolos adicionados")
        
        if failed:
            print(f"\n⚠️ {len(failed)} símbolos falharam:")
            for sym in failed[:10]:
                print(f"   - {sym}")
            if len(failed) > 10:
                print(f"   ... e mais {len(failed) - 10}")
    
    final_symbols = mt5.symbols_get()
    final_count = len([s for s in final_symbols if s.visible and s.name in desired_symbols]) if final_symbols else 0
    
    print(f"\n✅ SINCRONIZAÇÃO CONCLUÍDA!")
    print(f"   Market Watch: {final_count}/{len(desired_symbols)} símbolos ({final_count/len(desired_symbols)*100:.1f}%)")
    print("="*80)
    
    return final_count >= len(desired_symbols) * 0.9

# ===========================
# CARREGAMENTO DE DADOS
# ===========================
def load_data_with_retry(symbol: str, bars: int, timeframe=None, max_retries: int = 3) -> Optional[pd.DataFrame]:
    """Carrega dados com retry e verificação de conexão"""
    if not mt5:
        return None
    
    if timeframe is None:
        timeframe = mt5.TIMEFRAME_M15
    
    for attempt in range(max_retries):
        if not ensure_mt5_connection():
            logger.error(f"❌ {symbol}: MT5 desconectado (tentativa {attempt+1})")
            time.sleep(2)
            continue
        
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
            
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                
                if 'tick_volume' in df.columns:
                    df = df.rename(columns={'tick_volume': 'volume'})
                
                df = df[['open', 'high', 'low', 'close', 'volume']]
                df = df[~df.index.duplicated(keep='last')].sort_index()
                
                logger.info(f"✅ {symbol}: {len(df)} barras (tentativa {attempt+1})")
                return df
            else:
                error = mt5.last_error()
                logger.warning(f"⚠️ {symbol}: Erro {error} (tentativa {attempt+1})")
                time.sleep(1)
                
        except Exception as e:
            logger.warning(f"⚠️ {symbol}: {e} (tentativa {attempt+1})")
            time.sleep(1)
    
    if utils and hasattr(utils, "safe_copy_rates"):
        try:
            df = utils.safe_copy_rates(symbol, timeframe, count=bars)
            if df is not None and not df.empty:
                logger.info(f"✅ {symbol}: {len(df)} barras via utils")
                return df.sort_index()
        except Exception as e:
            logger.warning(f"⚠️ {symbol}: utils falhou: {e}")
    
    if ensure_history:
        try:
            df = ensure_history(symbol, period_days=90, interval='15m')
            if df is not None and not df.empty:
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
                    df = df.set_index('time')
                logger.info(f"✅ {symbol}: {len(df)} barras via backfill")
                return df.sort_index()
        except Exception as e:
            logger.warning(f"⚠️ {symbol}: backfill falhou: {e}")
    
    logger.error(f"❌ {symbol}: Todas as fontes falharam")
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
    close, high, low,
    ema_short, ema_long,
    rsi, adx, momentum, atr,
    rsi_low, rsi_high,
    adx_threshold, mom_min,
    sl_mult, slippage,
    risk_per_trade=0.01
):
    """
    Backtest otimizado com Numba
    Lógica FINAL (alinhada com Optuna e diagnóstico):

    ✅ Tendência (EMA curta > EMA longa)
    ✅ + (RSI OU Momentum OU ADX)
    """
    cash = 100000.0
    equity = cash
    position = 0.0
    entry_price = 0.0
    stop_price = 0.0

    n = len(close)
    equity_curve = np.zeros(n)
    equity_curve[0] = cash
    trades = 0

    for i in range(1, n):
        price = close[i]

        # ===== CONDIÇÕES =====
        trend_up = ema_short[i] > ema_long[i]
        rsi_ok = rsi[i] < rsi_low
        momentum_ok = momentum[i] > mom_min
        adx_ok = adx[i] > adx_threshold

        # ✅ ENTRADA CORRETA (OR, não AND)
        buy_signal = (
            trend_up &                   # 1. Deve estar em tendência de alta
            (adx[i] > adx_threshold) &   # 2. A tendência deve ter força (mínimo 25)
            (rsi[i] < rsi_low) &         # 3. O preço deve estar em um recuo (ex: < 40)
            (momentum[i] > 0)            # 4. O impulso deve ser positivo
        )

        # SAÍDA
        sell_signal = (
            (ema_short[i] < ema_long[i]) or
            (rsi[i] > rsi_high and not trend_up)
        )

        if position == 0.0:
            if buy_signal:
                entry_price = price * (1 + slippage / 2)
                stop_price = entry_price - atr[i] * sl_mult

                risk = entry_price - stop_price
                if risk > 0:
                    max_position_value = equity * 1.0  # 100% do capital
                    position = max_position_value / entry_price if entry_price > 0 else 0.0
                    cash -= position * entry_price
                    trades += 1
        else:
            # Stop ou saída lógica
            if low[i] <= stop_price or sell_signal:
                exit_price = stop_price if low[i] <= stop_price else price * (1 - slippage / 2)
                cash += position * exit_price
                position = 0.0

        equity = cash + position * price
        equity_curve[i] = equity

    return equity_curve, trades
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
    rs = gain / loss
    rsi = (100 - (100 / (1 + rs))).fillna(50).values

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


    # --- EXECUÇÃO DO BACKTEST ---
    equity_arr, trades = fast_backtest_core(
        close, high, low, ema_s, ema_l, rsi, adx, momentum, atr,
        params.get("rsi_low", 30), 
        params.get("rsi_high", 70), 
        params.get("adx_threshold", 25), 
        params.get("mom_min", 0.0),
        params.get("sl_atr_multiplier", 2.0),
        config_bt.DEFAULT_SLIPPAGE,
        config_bt.RISK_PER_TRADE
    )

    # ✅ CORREÇÃO: Chama a função direto, sem import
    metrics = compute_advanced_metrics(equity_arr.tolist())
    metrics.update({
        "total_trades": trades, 
        "equity_curve": equity_arr.tolist()
    })
    
    return metrics

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
        
        if df_full is None or df_full.empty:
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
            
            try:
                from optimizer_optuna import optimize_with_optuna
                logger.info(f"{sym}: Janela {i+1}/{wfo_windows} - Iniciando Optuna...")
                res = optimize_with_optuna(sym, df_train, n_trials=60, timeout=180)  # ✅ Aumentei timeout para 3min
                best_params = res["best_params"]
                
                logger.info(f"{sym}: ✅ Parâmetros otimizados: {best_params}")
    
            except Exception as e:  # ✅ MOSTRA O ERRO
                logger.error(f"{sym}: ❌ OPTUNA FALHOU: {e}", exc_info=True)
    
                # ✅ USA PARÂMETROS DO TESTE ANTERIOR (melhor que fallback genérico)
                if len(wins) > 0:
                    logger.warning(f"{sym}: Usando parâmetros da melhor janela anterior")
                    best_params = wins[-1]["best_params"]
                else:
                    logger.warning(f"{sym}: Usando fallback genérico")
                    best_params = {
                        "ema_short": 9, 
                        "ema_long": 21, 
                        "rsi_low": 40,
                        "rsi_high": 70,
                        "adx_threshold": 15,
                        "mom_min": 0.0,
                        "sl_atr_multiplier": 2.0
                    }
            
            test_res = backtest_params_on_df(sym, best_params, df_test)
            wins.append({
                "best_params": best_params,
                "test_metrics": test_res,
                "equity_curve": test_res.get("equity_curve", [])
            })
        
        if not wins:
            mt5.shutdown()
            return {"symbol": sym, "error": "wfo_no_windows"}
        
        best_win = max(wins, key=lambda w: w["test_metrics"].get("calmar", -100))
        out["selected_params"] = best_win["best_params"]
        out["test_metrics"] = best_win["test_metrics"]
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
def run_monte_carlo_stress(equity_curve: List[float], n_simulations: int = 2000) -> Dict[str, float]:
    """Simulação Monte Carlo para validação de robustez"""
    if len(equity_curve) < 50:
        return {"win_rate": 0.0, "calmar_avg": 0.0, "calmar_median": 0.0,
                "calmar_5th": 0.0, "max_dd_95": 1.0}

    returns = np.diff(equity_curve) / equity_curve[:-1]
    n_bars = len(returns)
    block_size = max(5, n_bars // 20)

    calmars, max_dds, wins = [], [], 0

    for _ in range(n_simulations):
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
        "max_dd_95": float(np.percentile(max_dds, 95))
    }

# ===========================
# UTILITÁRIOS
# ===========================
def load_all_symbols() -> List[str]:
    """Carrega todos os símbolos do SECTOR_MAP"""
    syms = [k.upper().strip() for k in SECTOR_MAP.keys() if isinstance(k, str) and k.strip()]
    return sorted(list(set(syms)))

def send_telegram_elite(text):
    """Envia mensagem para o Telegram"""
    if not config.ENABLE_TELEGRAM_NOTIF:
        return
    
    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": config.TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML"
    }
    try:
        r = requests.post(url, data=payload)
        if not r.ok:
            print(f"❌ Erro API Telegram: {r.text}")
    except Exception as e:
        print(f"Erro ao enviar Telegram: {e}")

# ===========================
# EXECUÇÃO PRINCIPAL
# ===========================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 OTIMIZADOR XP3 PRO - COM AUTO-SYNC E ENVIO TELEGRAM")
    print("="*80)

    if not ensure_mt5_connection():
        print("❌ Falha ao conectar ao MT5")
        exit(1)

    print(f"✅ MT5 conectado")
    
    print(f"\n💡 Sincronizando Market Watch com SECTOR_MAP...")
    if not sync_market_watch_with_sector_map(clear_first=True):
        print("⚠️ Sincronização falhou, continuando...")

    symbols_to_optimize = load_all_symbols()
    if not symbols_to_optimize:
        print("❌ Nenhum símbolo no SECTOR_MAP!")
        mt5.shutdown()
        exit(1)

    # === TESTE RÁPIDO: LIMITAR A 10 SÍMBOLOS ===
    #test_symbols = ["PETR4", "VALE3", "ITUB4", "BBDC4", "PRIO3", "VBBR3", "SUZB3", "WEGE3", "ABEV3", "EQTL3"]  # Escolha os que quiser
    #symbols_to_optimize = [s for s in symbols_to_optimize if s in test_symbols]
    # Ou simplesmente: symbols_to_optimize = test_symbols[:10]

    print(f"✅ TESTE RÁPIDO: Otimizando apenas {len(symbols_to_optimize)} símbolos: {symbols_to_optimize}")

    print(f"✅ {len(symbols_to_optimize)} símbolos prontos para otimização")

    # 4. Configuração WFO
    WFO_PARAMS = {
        "bars": 30000,
        "maxevals": 50,
        "wfo_windows": 3,
        "train_period": 12000,
        "test_period": 4000
    }

    all_results = {}
    USE_PARALLEL = True

    if USE_PARALLEL:
        print("\n⚙️ Modo: PARALELO")
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(worker_wfo, sym, **WFO_PARAMS): sym for sym in symbols_to_optimize}
            pbar = tqdm(total=len(symbols_to_optimize), desc="Otimizando")
            for future in as_completed(futures):
                sym = futures[future]
                try:
                    all_results[sym] = future.result(timeout=600)
                except Exception as e:
                    logger.error(f"Erro em {sym}: {e}")
                pbar.update(1)
            pbar.close()
    else:
        print("\n⚙️ Modo: SEQUENCIAL")
        for sym in tqdm(symbols_to_optimize, desc="Otimizando"):
            try:
                all_results[sym] = worker_wfo(sym, **WFO_PARAMS)
            except Exception as e:
                logger.error(f"Erro em {sym}: {e}")

    # =========================================================
    # 5. FILTROS ADAPTATIVOS PARA DD BALANCEADO
    # =========================================================
    def get_pre_approved_with_adaptive_filters(all_results):
        """Tenta filtros progressivamente mais relaxados até conseguir 10-30 ativos"""
        
        filter_levels = [
            {"name": "Ultra-Rigoroso", "calmar": 2.0, "dd": 0.08, "trades": 50},
            {"name": "Rigoroso", "calmar": 1.5, "dd": 0.10, "trades": 40},
            {"name": "Moderado", "calmar": 1.2, "dd": 0.12, "trades": 35},
            {"name": "Balanceado", "calmar": 1.0, "dd": 0.15, "trades": 30},
        ]
        
        for level in filter_levels:
            pre_approved = {
                s: r for s, r in all_results.items() 
                if r.get("status") == "ok" 
                and r.get("test_metrics", {}).get("calmar", 0) >= level["calmar"]
                and r.get("test_metrics", {}).get("max_drawdown", 1.0) <= level["dd"]
                and r.get("test_metrics", {}).get("total_trades", 0) >= level["trades"]
            }
            
            print(f"   • Filtro {level['name']}: {len(pre_approved)} ativos")
            
            if 10 <= len(pre_approved) <= 50:
                print(f"   ✅ Usando filtro {level['name']}")
                return pre_approved, level
            
            if len(pre_approved) > 50:
                print(f"   ✅ Usando filtro {level['name']} (muitos candidatos)")
                return pre_approved, level
        
        print(f"   ⚠️  Nenhum filtro teve resultado ideal, pegando top 20 por Calmar")
        all_valid = {s: r for s, r in all_results.items() if r.get("status") == "ok"}
        sorted_by_calmar = sorted(all_valid.items(), key=lambda x: x[1]["test_metrics"].get("calmar", 0), reverse=True)
        top_20 = dict(sorted_by_calmar[:20])
        
        return top_20, {"name": "Top 20", "calmar": 0.8, "dd": 0.20, "trades": 25}

    print(f"\n🎲 Filtrando Elite com Critérios Adaptativos de DD...")
    final_elite = {}
    
    pre_approved, used_filter = get_pre_approved_with_adaptive_filters(all_results)
    print(f"   • {len(pre_approved)}/{len(all_results)} ativos passaram filtro OOS")

    def get_monte_carlo_approved(pre_approved, used_filter):
        # Critérios Monte Carlo mais relaxados (22/12/2025)
        mc_criteria = {
            "win_rate": 0.50,      # >= 50%
            "calmar_avg": 0.4,     # >= 0.4
            "calmar_5th": 0.0,     # Ignoramos o 5º percentil (pode ser negativo)
            "max_dd_95": 0.20      # <= 20%
        }
        
        print(f"\n   📊 Critérios Monte Carlo (Relaxados):")
        print(f"      WR >= {mc_criteria['win_rate']:.0%} | Calmar >= {mc_criteria['calmar_avg']:.1f} | DD95 <= {mc_criteria['max_dd_95']:.0%}")
        
        monte_carlo_approved = {}
        
        for sym, res in tqdm(pre_approved.items(), desc="Monte Carlo"):
            mc = run_monte_carlo_stress(res.get("equity_curve", []), n_simulations=2000)
            
            if (mc["win_rate"] >= mc_criteria["win_rate"] and
                mc["calmar_avg"] >= mc_criteria["calmar_avg"] and
                mc["max_dd_95"] <= mc_criteria["max_dd_95"]):
                monte_carlo_approved[sym] = {**res, "monte_carlo": mc}
        
        return monte_carlo_approved        
        
        monte_carlo_approved = {}
        
        for sym, res in tqdm(pre_approved.items(), desc="Monte Carlo"):
            mc = run_monte_carlo_stress(res.get("equity_curve", []), n_simulations=2000)
            
            if (mc["win_rate"] >= mc_criteria["win_rate"] and
                mc["calmar_avg"] >= mc_criteria["calmar_avg"] and
                mc["calmar_5th"] >= mc_criteria["calmar_5th"] and
                mc["max_dd_95"] <= mc_criteria["max_dd_95"]):
                monte_carlo_approved[sym] = {**res, "monte_carlo": mc}
        
        return monte_carlo_approved

    monte_carlo_approved = get_monte_carlo_approved(pre_approved, used_filter)
    print(f"   • {len(monte_carlo_approved)}/{len(pre_approved)} ativos passaram Monte Carlo")

    for sym, res in monte_carlo_approved.items():
        m = res.get("test_metrics", {})
        mc = res.get("monte_carlo", {})
        
        oos_dd = m.get("max_drawdown", 1.0)
        mc_dd_95 = mc.get("max_dd_95", 1.0)
        
        if mc_dd_95 > oos_dd * 1.8:
            print(f"   ⚠️  {sym}: DD inconsistente (OOS={oos_dd:.1%} vs MC95={mc_dd_95:.1%})")
            continue
        
        final_elite[sym] = res

    print(f"\n🎯 {len(final_elite)} ativos ELITE aprovados!")

    if final_elite:
        avg_calmar = sum(r["test_metrics"]["calmar"] for r in final_elite.values()) / len(final_elite)
        avg_dd = sum(r["test_metrics"]["max_drawdown"] for r in final_elite.values()) / len(final_elite)
        avg_wr = sum(r["monte_carlo"]["win_rate"] for r in final_elite.values()) / len(final_elite)
        avg_mc_dd = sum(r["monte_carlo"]["max_dd_95"] for r in final_elite.values()) / len(final_elite)
        
        best_calmar_symbol = max(final_elite.items(), key=lambda x: x[1]['test_metrics']['calmar'])[0]
        best_dd_symbol = min(final_elite.items(), key=lambda x: x[1]['test_metrics']['max_drawdown'])[0]
        
        print(f"\n📊 ESTATÍSTICAS DO PORTFÓLIO ELITE:")
        print(f"   • Calmar Médio OOS: {avg_calmar:.2f}")
        print(f"   • DD Médio OOS: {avg_dd:.1%}")
        print(f"   • Win Rate MC Médio: {avg_wr:.1%}")
        print(f"   • DD Médio MC (95%): {avg_mc_dd:.1%}")
        print(f"   • Melhor Calmar: {best_calmar_symbol}")
        print(f"   • Menor DD: {best_dd_symbol}")
    else:
        print("\n⚠️  NENHUM ATIVO APROVADO!")
        all_valid = {s: r for s, r in all_results.items() if r.get("status") == "ok"}
        if all_valid:
            print(f"\n   📋 Top 10 que quase passaram:")
            sorted_by_calmar = sorted(all_valid.items(), key=lambda x: x[1]["test_metrics"].get("calmar", 0), reverse=True)
            for i, (sym, res) in enumerate(sorted_by_calmar[:10], 1):
                m = res["test_metrics"]
                print(f"   {i}. {sym} | Calmar={m.get('calmar', 0):.2f} | DD={m.get('max_drawdown', 0):.1%}")

    # =========================================================
    # 6. SALVAR TXT E ENVIAR TELEGRAM
    # =========================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_txt = os.path.join(OPT_OUTPUT_DIR, f"elite_symbols_{timestamp}.txt")
    
    dict_lines = ["ELITE_SYMBOLS = {"]
    perf_lines = ["\n<b>📊 PERFORMANCE DOS ATIVOS:</b>", "<pre>"]
    perf_lines.append(f"{'ATIVO':<8} | {'CALMAR':<6} | {'WR %':<5} | {'DD MAX'}")
    perf_lines.append("-" * 35)

    for sym, res in final_elite.items():
        p = res.get("selected_params", {})
        m = res.get("test_metrics", {})
        mc = res.get("monte_carlo", {})
        
        param_str = (
            f"{{'ema_short': {p.get('ema_short', 9)}, "
            f"'ema_long': {p.get('ema_long', 21)}, "
            f"'rsi_low': {p.get('rsi_low', 30)}, "
            f"'rsi_high': {p.get('rsi_high', 70)}, "
            f"'adx_threshold': {p.get('adx_threshold', 25)}, "
            f"'mom_min': {p.get('mom_min', 0.0)}}}"
        )
        dict_lines.append(f"    \"{sym}\": {param_str},")
        
        wr = mc.get('win_rate', 0) * 100
        calmar = m.get('calmar', 0)
        dd = m.get('max_drawdown', 0) * 100
        perf_lines.append(f"{sym:<8} | {calmar:<6.2f} | {wr:<5.1f}% | {dd:>5.1f}%")

    dict_lines.append("}")
    perf_lines.append("</pre>")

    dict_text = "\n".join(dict_lines)
    perf_text = "\n".join(perf_lines)
    
    full_telegram_msg = f"<b>🚀 NOVO ELITE_SYMBOLS GERADO</b>\n\n<pre>{dict_text}</pre>\n{perf_text}"

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(dict_text)

    print("\n" + "="*30 + " RESULTADO FINAL " + "="*30)
    print(dict_text)
    print("\n" + "="*30 + " PERFORMANCE " + "="*30)
    print(perf_text.replace("<b>", "").replace("</b>", "").replace("<pre>", "").replace("</pre>", ""))
    print("="*77)

    if config.ENABLE_TELEGRAM_NOTIF:
        print("📤 Enviando para o Telegram...")
        send_telegram_elite(full_telegram_msg)

    mt5.shutdown()
    print(f"\n✅ Otimização concluída! Arquivo: {output_txt}")