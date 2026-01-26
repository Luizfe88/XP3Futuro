import os
import json
import time
import logging
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None

try:
    import config
except Exception:
    config = None

# ===========================
# LIMITES GLOBAIS (CONFIG)
# ===========================
MAX_SYMBOLS = None
MAX_PER_SECTOR = None
SECTOR_MAP = getattr(config, "SECTOR_MAP", {})

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("otimizador_final")

if mt5:
    if not mt5.initialize():
        logger.error("Falha ao inicializar o MT5.")
        mt5 = None
    else:
        logger.info("MT5 inicializado com sucesso.")

try:
    import utils
except Exception:
    utils = None

try:
    from backfill import ensure_history
except Exception:
    ensure_history = None

# Configurações
WFO_WINDOWS = int(getattr(config, "WFO_WINDOWS", 6))
TRAIN_PERIOD = int(getattr(config, "WFO_TRAIN_PERIOD", 800))
TEST_PERIOD = int(getattr(config, "WFO_TEST_PERIOD", 300))
OPT_OUTPUT_DIR = getattr(config, "OPTIMIZER_OUTPUT", "optimizer_output")
os.makedirs(OPT_OUTPUT_DIR, exist_ok=True)

def load_all_symbols() -> List[str]:
    secmap = getattr(config, "SECTOR_MAP", {}) or {}
    syms = [k.upper().strip() for k in secmap.keys() if isinstance(k, str) and k.strip()]
    if not syms:
        syms = list(getattr(config, "PROXY_SYMBOLS", []) or [])
    return sorted(list(set(syms)))

def safe_save_json(fp: str, data: dict):
    tmp = fp + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, fp)

def compute_basic_metrics(equity_curve: List[float]) -> Dict[str, Any]:
    # Verifica se a curva de equity é válida
    if not equity_curve or len(equity_curve) < 2:
        return {"total_return": 0.0, "max_drawdown": 0.01, "calmar": 0.0, "sortino": 0.0, "total_trades": 0}

    # Calcula os retornos barra a barra
    returns = np.diff(equity_curve) / equity_curve[:-1]
    
    # Se a amostra for muito pequena (menos de 50 barras ~ 2 dias), retorna zero para evitar distorções
    if len(returns) < 50:
         return {"total_return": 0.0, "max_drawdown": 1.0, "calmar": 0.0, "sortino": 0.0, "total_trades": len(returns)}

    total_return = equity_curve[-1] / equity_curve[0] - 1
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peak) / peak
    
    # Evita divisão por zero no drawdown
    max_dd = -np.minimum.reduce(drawdowns) if len(drawdowns) > 0 else 0.01
    if max_dd == 0: max_dd = 0.01

    # Ajuste de Bares/Ano para B3 (M15)
    # B3 tem ~7h a 8h de pregão. 7.5h * 4 candles/h = 30 candles/dia.
    # 252 dias * 30 candles = 7560 candles/ano.
    bars_per_year = 7560 
    years = len(equity_curve) / bars_per_year
    
    # Anualização do retorno
    if years < 1.0:
        # Projeção linear para períodos muito curtos (mais conservador)
        annualized = total_return
    else:
        # Juros compostos para períodos longos
        annualized = (1 + total_return) ** (1 / years) - 1

    # ===========================
    # SHARPE RATIO
    # ===========================
    ret_std = np.std(returns)
    if ret_std > 0:
        sharpe = annualized / (ret_std * np.sqrt(bars_per_year))
    else:
        sharpe = 0.0


    # Cálculo do Calmar
    calmar = annualized / max_dd

    # --- CORREÇÃO DO ERRO AQUI ---
    # Define os retornos negativos antes de verificar o tamanho
    downside_returns = returns[returns < 0]

    # Cálculo do Sortino
    if len(downside_returns) > 0:
        downside_std = np.std(downside_returns) * np.sqrt(bars_per_year)
        sortino = annualized / downside_std if downside_std > 0 else 0.0
    else:
        sortino = 0.0 # Sem retornos negativos, Sortino seria infinito (ou zero tecnicamente)

    return {
        "total_return": float(total_return),
        "max_drawdown": float(max_dd),
        "calmar": float(calmar),
        "sortino": float(sortino),
        "sharpe": float(sharpe),
        "final_equity": float(equity_curve[-1]),
        "total_trades": len(returns)
    }

# ===========================
# BACKTEST REALISTA (com ATR stop, risco 1%, slippage)
# ===========================
from numba import njit

@njit  # <--- O segredo da velocidade
def fast_backtest_core(close, high, low, ema_short, ema_long, atr, sl_mult, slippage):
    cash = 100000.0
    position = 0
    entry_price = 0.0
    stop_price = 0.0
    equity_curve = [cash] # Numba lida melhor com listas se pré-alocadas, mas append funciona
    # Pré-alocação é mais rápida, mas vamos simplificar:
    
    # Numba não gosta de listas dinâmicas complexas, arrays numpy são melhores
    # Mas para lógica simples, o loop abaixo voa:
    
    trades = 0
    n = len(close)
    
    # Precisamos criar um array para equity para ser rápido
    equity_arr = np.zeros(n)
    equity_arr[0] = cash
    current_equity = cash
    
    for i in range(1, n):
        price = close[i]
        curr_atr = atr[i]
        
        # Sinais
        buy_signal = ema_short[i] > ema_long[i]
        sell_signal = ema_short[i] < ema_long[i]
        
        if position == 0:
            if buy_signal:
                position = 1
                entry_price = price * (1 + slippage / 2)
                stop_price = entry_price - curr_atr * sl_mult
                risk = entry_price - stop_price
                if risk > 0:
                    vol = (current_equity * 0.01) / risk
                    trades += 1
            elif sell_signal:
                position = -1
                entry_price = price * (1 - slippage / 2)
                stop_price = entry_price + curr_atr * sl_mult
                risk = stop_price - entry_price
                if risk > 0:
                    vol = (current_equity * 0.01) / risk
                    trades += 1
                    
        elif position == 1:
            # Check Stop Loss ou Reversão
            if price <= stop_price or sell_signal:
                exit_price = price * (1 - slippage / 2)
                ret = (exit_price - entry_price) / entry_price
                current_equity *= (1 + ret)
                position = 0
                
        elif position == -1:
            if price >= stop_price or buy_signal:
                exit_price = price * (1 + slippage / 2)
                ret = (entry_price - exit_price) / entry_price
                current_equity *= (1 + ret)
                position = 0
        
        equity_arr[i] = current_equity
        
    return equity_arr, trades

# Wrapper para manter compatibilidade com seu código
def backtest_params_on_df(symbol, params, df) -> Dict[str, Any]:
    if df is None or len(df) < 100:
        return {"total_return": -1.0, "calmar": -10.0, "total_trades": 0, "equity_curve": [100000.0]}

    # Prepara dados numpy
    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    
    # Calcula indicadores (Pandas é rápido o suficiente aqui, ou pode converter para TA-Lib)
    ema_s_val = pd.Series(close).ewm(span=params.get("ema_short", 9), adjust=False).mean().values
    ema_l_val = pd.Series(close).ewm(span=params.get("ema_long", 21), adjust=False).mean().values
    
    # ATR Calculation
    tr = np.maximum.reduce([high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))])
    atr_val = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    
    from utils import get_real_slippage
    slippage = get_real_slippage(sym) if mt5 else 0.0035
    sl_mult = params.get("sl_atr_multiplier", 2.0)

    # CHAMA O NUMBA
    equity_curve_arr, trades = fast_backtest_core(close, high, low, ema_s_val, ema_l_val, atr_val, sl_mult, slippage)
    
    # Filtra zeros iniciais se houver start_idx
    equity_curve = equity_curve_arr.tolist()
    
    metrics = compute_basic_metrics(equity_curve)
    metrics["total_trades"] = trades
    metrics["equity_curve"] = equity_curve
    return metrics

def optimize_window(sym: str, df_train, maxevals: int):
    try:
        from optimizer_optuna import optimize_with_optuna
        res = optimize_with_optuna(sym, df_train, n_trials=80)
        return res["best_params"]
    except Exception as e:
        logger.warning(f"Optuna falhou para {sym}: {e}. Usando fallback.")
        return {"ema_short": 9, "ema_long": 21, "rsi_low": 30, "rsi_high": 70, "adx_threshold": 25, "mom_min": 0.0}

def load_series_with_backfill(sym: str, bars: int, timeframe=None):
    df = None
    if timeframe is None:
        timeframe = mt5.TIMEFRAME_M15 if mt5 else None

    # 1. Tentar via utils.safe_copy_rates (Sua principal fonte funcional atual)
    try:
        if utils and hasattr(utils, "safe_copy_rates"):
            df = utils.safe_copy_rates(sym, timeframe, count=bars)
            if df is not None and not df.empty:
                logger.info(f"{sym} carregado via utils.safe_copy_rates")
    except Exception as e:
        logger.warning(f"{sym} falha em utils.safe_copy_rates: {e}")

    # 2. Try direct MT5 copy_rates_from_pos (Fallback direto)
    try:
        if (df is None or (hasattr(df, "empty") and df.empty)) and mt5 and pd:
            if mt5.symbol_select(sym, True):
                rates = mt5.copy_rates_from_pos(sym, timeframe, 0, bars)
                if rates is not None and len(rates) > 0:
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    df.set_index('time', inplace=True)
                    # Padroniza colunas
                    if 'tick_volume' in df.columns:
                        df = df.rename(columns={'tick_volume': 'volume'})
                    cols_to_keep = ['open', 'high', 'low', 'close', 'volume']
                    df = df[[c for c in cols_to_keep if c in df.columns]]
                    logger.info(f"{sym} carregado diretamente via mt5.copy_rates_from_pos")
    except Exception as e:
        logger.warning(f"{sym} falha em mt5 direto: {e}")

    # 3. Try backfill.ensure_history (Fallback via CSV/Cache)
    try:
        if (df is None or (hasattr(df, "empty") and df.empty)) and ensure_history:
            df = ensure_history(sym, period_days=60, interval='15m')
            if df is not None and not df.empty:
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
                    df = df.set_index('time').sort_index()
                logger.info(f"{sym} carregado via backfill.ensure_history")
    except Exception as e:
        logger.warning(f"{sym} falha em backfill.ensure_history: {e}")

    # Verificação Final
    if df is None or (hasattr(df, "empty") and df.empty):
        logger.error(f"{sym} sem dados de nenhuma fonte")
        return None

    # Limpeza e Ordenação do DataFrame
    try:
        df = df[~df.index.duplicated(keep='last')] # Remove duplicatas de tempo
        df = df.sort_index()
    except Exception as e:
        logger.warning(f"Erro ao organizar índice de {sym}: {e}")

    return df

def worker_wfo(sym: str, bars: int, maxevals: int, wfo_windows: int, train_period: int, test_period: int) -> Dict[str, Any]:
    out = {"symbol": sym, "status": "ok", "wfo_windows": []}
    try:
        df_full = load_series_with_backfill(sym, bars)
        if df_full is None:
            return {"symbol": sym, "error": "no_data"}
        df_full = df_full.sort_index()
        n = len(df_full)
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
            best_params = optimize_window(sym, df_train, maxevals)
            test_res = backtest_params_on_df(sym, best_params, df_test)
            wins.append({
                "best_params": best_params,
                "test_metrics": test_res,
                "equity_curve": test_res.get("equity_curve", [])
            })

        if not wins:
            return {"symbol": sym, "error": "wfo_no_windows"}

        # Seleciona a melhor janela por Calmar
        best_win = max(wins, key=lambda w: w["test_metrics"].get("calmar", -100))
        out["selected_params"] = best_win["best_params"]
        out["test_metrics"] = best_win["test_metrics"]
        out["equity_curve"] = best_win["equity_curve"]

        fp = os.path.join(OPT_OUTPUT_DIR, f"WFO_{sym}.json")
        safe_save_json(fp, out)
        return out
    except Exception as e:
        logger.exception(f"WFO falhou para {sym}")
        return {"symbol": sym, "error": str(e)}

# ===========================
# MONTE CARLO COMPLETO
# ===========================
def run_monte_carlo_stress(equity_curve: List[float], n_simulations: int = 1000) -> Dict[str, float]:
    if len(equity_curve) < 50:
        return {
            "win_rate": 0.0,
            "calmar_avg": 0.0,
            "calmar_median": 0.0,
            "calmar_5th": 0.0,
            "max_dd_95": 1.0
        }

    returns = np.diff(equity_curve) / equity_curve[:-1]
    n_bars = len(returns)  # ✅ DEFINE AQUI
    block_size = max(5, n_bars // 20)  # ✅ DEFINE AQUI (5% dos dados)

    calmars, max_dds, wins = [], [], 0

    for _ in range(n_simulations):
        # Block Bootstrap
        sim_returns = []
        while len(sim_returns) < n_bars:
            start_idx = np.random.randint(0, max(1, n_bars - block_size))
            block = returns[start_idx : start_idx + block_size]
            sim_returns.extend(block)
        
        sim_returns = np.array(sim_returns[:n_bars])
        sim_equity = np.cumprod(1 + sim_returns) * equity_curve[0]
        
        # Métricas
        peak = np.maximum.accumulate(sim_equity)
        dd = (peak - sim_equity) / peak
        max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0
        max_dds.append(max_dd)
        
        total_ret = sim_equity[-1] / sim_equity[0] - 1
        wins += int(total_ret > 0)
        
        years = n_bars / (252 * 28)  # M15 na B3
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


import shutil
from collections import defaultdict

def enforce_sector_limits(elite_dict: dict) -> dict:
    """
    Aplica limites de setor e total apenas se configurados.
    Como agora o limite é controlado na carteira, aqui apenas retornamos todos os aprovados.
    """
    if not elite_dict:
        return {}

    # Se os limites estiverem desativados (None), retorna todos os aprovados
    if MAX_SYMBOLS is None and MAX_PER_SECTOR is None:
        print(f"🌍 Limites desativados no otimizador. Todos os {len(elite_dict)} aprovados serão salvos no ELITE_SYMBOLS.")
        return elite_dict

    # === MANTÉM O COMPORTAMENTO ANTIGO CASO ALGUÉM QUEIRA REATIVAR ===
    # (código original abaixo - mantido como fallback)

    sector_of = {}
    for sym, sector in SECTOR_MAP.items():
        if sym in elite_dict:
            sector_of[sym] = sector

    sector_groups = defaultdict(list)
    for sym, data in elite_dict.items():
        calmar = data.get("test_metrics", {}).get("calmar", 0)
        sector = sector_of.get(sym, "UNKNOWN")
        sector_groups[sector].append((sym, calmar, data))

    selected = {}
    for sector, items in sector_groups.items():
        items.sort(key=lambda x: x[1], reverse=True)
        limit = MAX_PER_SECTOR or len(items)
        for i, (sym, calmar, data) in enumerate(items):
            if i < limit:
                selected[sym] = data

    total_limit = MAX_SYMBOLS or len(selected)
    if len(selected) > total_limit:
        sorted_all = sorted(selected.items(), key=lambda x: x[1].get("test_metrics", {}).get("calmar", 0), reverse=True)
        selected = {sym: data for sym, data in sorted_all[:total_limit]}

    return selected

def update_elite_symbols(final_elite_params: dict, config_path="config.py"):
    """
    VERSÃO MAIS ROBUSTA: Apaga completamente o ELITE_SYMBOLS antigo e reescreve do zero.
    """
    if not final_elite_params:
        print("⚠️ Nenhum ativo elite. Não será alterado o config.py.")
        return

    # Garante que o arquivo existe
    if not os.path.exists(config_path):
        with open(config_path, "w", encoding="utf-8") as f:
            f.write("# config.py\n\n")
    
    # Lê todo o conteúdo
    with open(config_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Remove todas as linhas que pertencem ao ELITE_SYMBOLS antigo
    new_lines = []
    in_elite_block = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("ELITE_SYMBOLS"):
            in_elite_block = True
            # Pula todo o bloco antigo
            continue
        if in_elite_block:
            # Detecta fim do dicionário (linha que não começa com espaço/tabs ou fecha })
            if stripped == "}" or (not stripped.startswith((" ", "\t", "#")) and stripped):
                in_elite_block = False
                new_lines.append(line)  # mantém a linha que fechou ou próxima
            #否則 pula
        else:
            new_lines.append(line)

    # Remove ELITE_SYMBOLS completamente se não houver mais nada
    content = "".join(new_lines)
    if "ELITE_SYMBOLS" in content:
        # Limpa qualquer resquício
        import re
        content = re.sub(r"ELITE_SYMBOLS\s*=\s*\{.*?\}", "", content, flags=re.DOTALL)

    # Constrói o novo bloco
    new_block = "\nELITE_SYMBOLS = {\n"
    for sym, params in final_elite_params.items():
        new_block += f'    "{sym}": {params},\n'
    new_block += "}\n"

    # Adiciona ao final (ou após imports se preferir)
    content = content.rstrip() + "\n" + new_block + "\n"

    # Salva
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"✅ ELITE_SYMBOLS completamente reescrito com {len(final_elite_params)} ativos.")

# ===========================
# TELA BONITA DE RESULTADOS DO OTIMIZADOR
# ===========================
def print_optimizer_dashboard(results: dict):
    """
    Exibe uma tela colorida e didática com os melhores parâmetros e métricas por ativo.
    """
    from datetime import datetime
    
    print("\n\033[1m\033[96m╔" + "═" * 100 + "╗\033[0m")
    print("║ \033[1m\033[93m📊 RESULTADOS DO WALK-FORWARD OPTIMIZER - XP3 PRO B3\033[0m" + " " * 30 + f"\033[96mData: {datetime.now().strftime('%d/%m/%Y %H:%M')}\033[0m ║")
    print("║ \033[92mMelhores parâmetros encontrados por ativo (priorizando Calmar & Sortino)\033[0m" + " " * 20 + "║")
    print("\033[96m╠" + "═" * 100 + "╣\033[0m")
    print(f"║ {'ATIVO':<8} {'EMA S/L':<12} {'RSI L/H':<11} {'ADX TH':<8} {'CALMAR':<8} {'SORTINO':<9} {'RETORNO':<10} {'MAX DD':<10} {'STATUS'} ║")
    print("\033[96m╠" + "═" * 100 + "╣\033[0m")
    
    if not results:
        print("║ \033[93mNenhum resultado válido encontrado.\033[0m" + " " * 70 + "║")
    else:
        for sym, data in sorted(results.items()):
            if data.get("status") != "ok" or "selected_params" not in data:
                continue
            
            params = data["selected_params"]
            metrics = data.get("test_metrics", {})
            
            ema = f"{params.get('ema_short', '-')}/{params.get('ema_long', '-')}"
            rsi = f"{params.get('rsi_low', '-')}/{params.get('rsi_high', '-')}"
            adx = params.get('adx_threshold', '-')
            
            calmar = metrics.get('calmar', 0.0)
            sortino = metrics.get('sortino', 0.0)
            ret = metrics.get('total_return', 0.0)
            dd = metrics.get('max_drawdown', 0.0)
            sharpe = metrics.get("sharpe", 0.0)

            
            # Cores por performance
            calmar_color = "\033[92m" if calmar >= 1.5 else "\033[93m" if calmar >= 1.0 else "\033[91m"
            sortino_color = "\033[92m" if sortino >= 2.0 else "\033[93m"
            ret_color = "\033[92m" if ret > 0.3 else "\033[93m" if ret > 0 else "\033[91m"
            dd_color = "\033[92m" if dd < 0.20 else "\033[93m" if dd < 0.30 else "\033[91m"
            
            status = "🚀 ÓTIMO" if calmar >= 1.5 and sortino >= 2.0 else "✅ BOM" if calmar >= 1.0 else "⚠️ REVISAR"
            status_color = "\033[92m" if "ÓTIMO" in status else "\033[93m" if "BOM" in status else "\033[91m"
            
            print(f"║ \033[1m{sym:<8}\033[0m {ema:<12} {rsi:<11} {adx:<8} {calmar_color}{calmar:<8.2f}\033[0m {sortino_color}{sortino:<9.2f}\033[0m "
                  f"{ret_color}{ret:+8.1%}\033[0m {dd_color}{dd:<10.1%}\033[0m {status_color}{status}\033[0m ║")
    
    print("\033[96m╚" + "═" * 100 + "╝\033[0m")
    print("\033[93m✨ Legenda: Calmar ≥1.5 (Ótimo) | 1.0-1.5 (Bom) | <1.0 (Revisar)\033[0m")
    print("\033[92m🚀 Parâmetros atualizados automaticamente no config.py!\033[0m")
    print("\033[96mDica: Rode o bot agora — ele já está usando esses parâmetros otimizados! 🌟\033[0m\n")

def save_monte_carlo_log(final_elite: dict, elite_results: dict, mc_results: dict):
    """Salva um log detalhado dos resultados do Monte Carlo"""
    log_path = os.path.join(OPT_OUTPUT_DIR, "monte_carlo_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"MONTE CARLO STRESS TEST - {datetime.now().strftime('%d/%m/%Y %H:%M')}\n")
        f.write("="*80 + "\n")
        f.write(f"Total de ativos testados: {len(elite_results)}\n")
        f.write(f"Total aprovados como ELITE FINAL: {len(final_elite)}\n\n")
        
        f.write("ATIVOS APROVADOS (ELITE FINAL):\n")
        for sym in final_elite:
            mc = mc_results.get(sym, {})
            calmar = final_elite[sym].get("test_metrics", {}).get("calmar", 0)
            f.write(f"  ✅ {sym:8} | Calmar OOS: {calmar:6.2f} | "
                    f"WinRate: {mc.get('win_rate',0)*100:5.1f}% | "
                    f"Calmar 5%: {mc.get('calmar_5th',0):6.2f} | "
                    f"MaxDD 95%: {mc.get('max_dd_95',0)*100:5.1f}%\n")
        
        f.write("\nATIVOS REJEITADOS NO MONTE CARLO:\n")
        for sym, res in elite_results.items():
            if sym not in final_elite:
                mc = mc_results.get(sym, {})
                calmar = res.get("test_metrics", {}).get("calmar", 0)
                f.write(f"  ❌ {sym:8} | Calmar OOS: {calmar:6.2f} | "
                        f"WinRate: {mc.get('win_rate',0)*100:5.1f}% | "
                        f"Calmar 5%: {mc.get('calmar_5th',0):6.2f} | "
                        f"MaxDD 95%: {mc.get('max_dd_95',0)*100:5.1f}%\n")
    print(f"📄 Log completo do Monte Carlo salvo em: {log_path}")

# ===========================
# EXECUÇÃO PRINCIPAL
# ===========================
if __name__ == "__main__":
    try:
        from config import SECTOR_MAP
        symbols_to_optimize = list(SECTOR_MAP.keys())
        print(f"🔍 {len(symbols_to_optimize)} ativos carregados do SECTOR_MAP.")
    except Exception as e:
        print("❌ ERRO AO CARREGAR SECTOR_MAP:")
        raise e

    WFO_PARAMS = {
        "bars": 20000,
        "maxevals": 50,
        "wfo_windows": 4,
        "train_period": 4000,
        "test_period": 2000
    }

    all_results = {}
    print("🚀 Iniciando Otimização Walk-Forward...")

    with ProcessPoolExecutor(max_workers=4) as executor:
        future_to_symbol = {executor.submit(worker_wfo, sym, **WFO_PARAMS): sym for sym in symbols_to_optimize}
        for future in tqdm(as_completed(future_to_symbol), total=len(symbols_to_optimize), desc="Otimizando"):
            sym = future_to_symbol[future]
            try:
                all_results[sym] = future.result()
            except Exception as e:
                print(f"❌ Erro em {sym}: {e}")

    # Filtro inicial por Calmar + número mínimo de trades
    elite_results = {}
    for sym, res in all_results.items():
        if res.get("status") != "ok":
            continue

        metrics = res.get("test_metrics", {})
        calmar = metrics.get("calmar", 0.0)
        trades = metrics.get("total_trades", 0)

        if calmar >= 1.2 and trades >= 20:
            elite_results[sym] = res
            print(f"🟢 {sym:7} | Calmar OOS: {calmar:5.2f} | Trades: {trades:3d} | [PRÉ-APROVADO]")
        else:
            print(f"⚠️ {sym:7} | Calmar: {calmar:5.2f} | Trades: {trades:3d} | REJEITADO")

    # Monte Carlo Stress Test
    print("\n" + "═"*80)
    print("🧪 MONTE CARLO STRESS TEST COMPLETO (2000 simulações por ativo)")
    print("═"*80)

    final_elite = {}
    mc_results_dict = {}  # Armazena todos os resultados do MC para log posterior

    for sym, res in elite_results.items():
        eq_curve = res.get("equity_curve", [])
        mc = run_monte_carlo_stress(eq_curve, n_simulations=2000)
        mc_results_dict[sym] = mc  # Salva para o log

        print(
            f"{sym:8} | WinRate MC: {mc['win_rate']*100:5.1f}% | "
            f"Calmar Med: {mc['calmar_median']:5.2f} | "
            f"Calmar 5%: {mc['calmar_5th']:6.2f} | "
            f"DD 95%: {mc['max_dd_95']*100:5.1f}%"
        )

        if (mc["win_rate"] >= 0.60 and 
            mc["calmar_avg"] >= 1.0 and 
            mc["calmar_5th"] >= -3.25 and 
            mc["max_dd_95"] <= 0.25):
            final_elite[sym] = res
            print(f"✅ {sym} APROVADO COMO ELITE FINAL")
        else:
            print(f"❌ {sym} REJEITADO NO MONTE CARLO")

    # Monte Carlo Stress Test - EXECUÇÃO ÚNICA E DEFINITIVA
    print("\n" + "═"*80)
    print("🧪 MONTE CARLO STRESS TEST COMPLETO (2000 simulações por ativo)")
    print("═"*80)

    final_elite = {}
    mc_results_dict = {}

    for sym, res in elite_results.items():
        eq_curve = res.get("equity_curve", [])
        mc = run_monte_carlo_stress(eq_curve, n_simulations=2000)
        mc_results_dict[sym] = mc

        print(
            f"{sym:8} | WinRate MC: {mc['win_rate']*100:5.1f}% | "
            f"Calmar Med: {mc['calmar_median']:5.2f} | "
            f"Calmar 5%: {mc['calmar_5th']:6.2f} | "
            f"DD 95%: {mc['max_dd_95']*100:5.1f}%"
        )

        if (mc["win_rate"] >= 0.60 and 
            mc["calmar_avg"] >= 1.0 and 
            mc["calmar_5th"] >= -3.25 and 
            mc["max_dd_95"] <= 0.25):
            final_elite[sym] = res
            print(f"✅ {sym} APROVADO COMO ELITE FINAL")
        else:
            print(f"❌ {sym} REJEITADO NO MONTE CARLO")

    # Salva log completo
    save_monte_carlo_log(final_elite, elite_results, mc_results_dict)

    # ===========================
    # SELEÇÃO FINAL E ATUALIZAÇÃO DO CONFIG
    # ===========================
    candidates = final_elite
    source_method = "MONTE CARLO (Ultra-Robusto)"

    if not final_elite:
        print("\n⚠️ Nenhum ativo passou no Monte Carlo. Ativando Plano B (WFO puro)...")
        candidates = {}
        for sym, res in elite_results.items():
            metrics = res.get("test_metrics", {})
            if metrics.get("calmar", 0.0) >= 1.2 and metrics.get("total_trades", 0) >= 15:
                candidates[sym] = res
        source_method = "WFO (Alta Performance)" if candidates else "NENHUM"

    if candidates:
        limited_assets = enforce_sector_limits(candidates)

        # Distribuição por setor (opcional, mas útil!)
        from collections import Counter
        sectors = [SECTOR_MAP.get(sym, "UNKNOWN") for sym in limited_assets.keys()]
        print(f"\n🌍 Distribuição final por setor: {dict(Counter(sectors))}")

        clean_save_dict = {}
        for sym, data in limited_assets.items():
            raw_params = data.get("selected_params", {})
            clean_params = {
                k: v.item() if hasattr(v, 'item') else v
                for k, v in raw_params.items()
            }
            clean_save_dict[sym] = clean_params

        if clean_save_dict:
            update_elite_symbols(clean_save_dict)
            print(f"\n✨ SUCESSO ABSOLUTO! {len(clean_save_dict)} ativos ELITE salvos em config.py")
            print(f"📊 Fonte da seleção: {source_method}")
            print(f"🌍 Ativos selecionados: {', '.join(sorted(clean_save_dict.keys()))}")
        else:
            print("\n❌ Nenhum ativo após aplicação dos limites de setor. config.py não alterado.")
    else:
        print("\n❌ Nenhum ativo consistente encontrado hoje. config.py mantido inalterado.")

    # Dashboard final
    print_optimizer_dashboard(all_results)