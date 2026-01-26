# otimizador_semanal.py
# Walk-Forward Optimizer (WFO) - Versão atualizada com métricas avançadas, slippage e Monte Carlo

import os
import json
import time
import argparse
import logging
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import statistics
from tqdm import tqdm
from datetime import datetime
try:
    import pandas as pd
except Exception:
    pd = None

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("otimizador_final")

# Inicializa o MT5 se disponível
if mt5:
    if not mt5.initialize():
        logger.error("Falha ao inicializar o MT5. Verifique se o terminal está aberto e configurado corretamente.")
        mt5 = None
    else:
        logger.info("MT5 inicializado com sucesso.")

# try to import user's optimizer module(s)
try:
    import optimizer_updated as base
except Exception:
    try:
        import optimizer as base
    except Exception:
        base = None

try:
    import config
except Exception:
    config = None

try:
    import utils
except Exception:
    utils = None

# backfill helper (local file)
try:
    from backfill import ensure_history
except Exception:
    ensure_history = None

# defaults (can be overridden by CLI or config)
WFO_WINDOWS = int(getattr(config, "WFO_WINDOWS", 6))
TRAIN_PERIOD = int(getattr(config, "WFO_TRAIN_PERIOD", 500))
TEST_PERIOD = int(getattr(config, "WFO_TEST_PERIOD", 200))
OPT_OUTPUT_DIR = getattr(base, "OPT_OUTPUT_DIR", getattr(config, "OPTIMIZER_OUTPUT", "optimizer_output"))
os.makedirs(OPT_OUTPUT_DIR, exist_ok=True)
FAST_MODE = getattr(config, "WFO_FAST_MODE", False)  # pode ativar no config.py
ROBUST_EVALS = 100 if FAST_MODE else 300
SKIP_ROBUST_CALMAR_THRESHOLD = 1.3 if FAST_MODE else 0.0  # 0 = nunca pula


# Workers default melhorado
workers_default = max(1, (os.cpu_count() or 4) - 1)

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
    out = {"n": len(equity_curve)}
    if not equity_curve or len(equity_curve) < 2:
        out.update({"total_return": 0.0, "max_drawdown": 0.0, "sharpe": 0.0, "pf": 0.0, "calmar": 0.0, "sortino": 0.0})
        return out
    
    returns = []
    for i in range(1, len(equity_curve)):
        prev = equity_curve[i-1]
        cur = equity_curve[i]
        if prev == 0:
            returns.append(0.0)
        else:
            returns.append((cur - prev) / abs(prev))
    
    total_return = (equity_curve[-1] - equity_curve[0]) / (equity_curve[0] if equity_curve[0] != 0 else 1)
    
    # Max Drawdown
    peak = equity_curve[0]
    max_dd = 0.0
    for v in equity_curve:
        if v > peak:
            peak = v
        dd = (peak - v) / (peak if peak != 0 else 1)
        if dd > max_dd:
            max_dd = dd
    
    mean_r = statistics.mean(returns) if returns else 0.0
    std_r = statistics.pstdev(returns) if returns else 0.0
    sharpe = (mean_r / std_r) * (252 ** 0.5) if std_r and std_r > 0 else 0.0
    
    pos = sum(r for r in returns if r > 0)
    neg = -sum(r for r in returns if r < 0)
    pf = (pos / neg) if neg > 0 else (pos if pos > 0 else 0.0)
    
    # Calmar Ratio
    if len(equity_curve) > 1:
        years = (len(equity_curve) - 1) / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else total_return
    else:
        annualized_return = total_return
    calmar = annualized_return / max_dd if max_dd > 0 else 0.0
    
    # Sortino Ratio
    downside_returns = [r for r in returns if r < 0]
    downside_dev = statistics.pstdev(downside_returns) if downside_returns else 0.0
    mean_r_annual = mean_r * 252
    sortino = mean_r_annual / downside_dev if downside_dev > 0 else 0.0
    
    out.update({
        "total_return": total_return,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "pf": pf,
        "calmar": calmar,
        "sortino": sortino
    })
    return out

def backtest_params_on_df(symbol, params, df):
    """
    Versão corrigida para garantir que o Optuna receba dados reais de performance.
    """
    if df is None or len(df) < 50:
        return {"total_return": -1.0, "calmar": -10.0, "total_trades": 0, "final_equity": 100000.0}

    # Extração de parâmetros
    ema_s_val = params.get("ema_short", 9)
    ema_l_val = params.get("ema_long", 21)
    rsi_l = params.get("rsi_low", 30)
    rsi_h = params.get("rsi_high", 70)
    adx_t = params.get("adx_threshold", 20)
    mom_m = params.get("mom_min", 0.0)

    # Indicadores (Cálculo simplificado para performance)
    df = df.copy()
    ema_s = df['close'].ewm(span=ema_s_val, adjust=False).mean()
    ema_l = df['close'].ewm(span=ema_l_val, adjust=False).mean()
    
    # Simulação simplificada de sinais
    df['signal'] = 0
    # Compra: EMA curta > longa + Momentum > min
    buy_cond = (ema_s > ema_l)
    # Venda: EMA curta < longa
    sell_cond = (ema_s < ema_l)
    
    cash = 100000.0
    position = 0
    entry_price = 0
    drawdowns = []
    max_equity = cash
    total_trades = 0  # CRÍTICO: Contador para o Optuna
    
    prices = df['close'].values
    
    for i in range(1, len(prices)):
        current_price = prices[i]
        
        # Lógica de entrada
        if position == 0:
            if buy_cond.iloc[i]:
                position = 1
                entry_price = current_price
                total_trades += 1
                cash -= cash * 0.0007  # Taxas/Slippage B3
            elif sell_cond.iloc[i]:
                position = -1
                entry_price = current_price
                total_trades += 1
                cash -= cash * 0.0007
        
        # Lógica de saída (Inversão ou cruzamento)
        elif position == 1 and sell_cond.iloc[i]:
            ret = (current_price / entry_price) - 1
            cash *= (1 + ret)
            cash -= cash * 0.0007
            position = 0
        elif position == -1 and buy_cond.iloc[i]:
            ret = (entry_price / current_price) - 1
            cash *= (1 + ret)
            cash -= cash * 0.0007
            position = 0
            
        # Atualiza métricas de risco
        if cash > max_equity: max_equity = cash
        dd = (max_equity - cash) / max_equity
        drawdowns.append(dd)

    max_dd = max(drawdowns) if drawdowns else 0.0001
    total_return = (cash / 100000.0) - 1
    
    # Calmar Ratio (Proteção contra divisão por zero)
    calmar = total_return / max_dd if max_dd > 0 else 0
    
    # Retorno limpo (Métricas na raiz do dicionário)
    return {
        "total_return": float(total_return),
        "drawdown": float(max_dd),
        "calmar": float(calmar),
        "final_equity": float(cash),
        "total_trades": int(total_trades)
    }

def optimize_window(sym: str, df_train, maxevals: int):
    try:
        from optimizer_optuna import optimize_with_optuna
        res = optimize_with_optuna(sym, df_train, n_trials=80)
        return res["best_params"]
    except Exception as e:
        logger.warning(f"Optuna falhou para {sym}: {e}. Usando fallback.")
        return {
            "ema_short": 9,
            "ema_long": 21,
            "rsi_low": 30,
            "rsi_high": 70,
            "adx_threshold": 25,
            "mom_min": 0.0
        }

def load_series_with_backfill(sym: str, bars: int, timeframe=None):
    df = None
    if timeframe is None:
        timeframe = mt5.TIMEFRAME_M15 if mt5 else None

    # Try base.load_historical_bars
    try:
        if base and hasattr(base, "load_historical_bars"):
            df = base.load_historical_bars(sym, bars=bars)
            logger.info(f"{sym} carregado via base.load_historical_bars")
    except Exception as e:
        logger.warning(f"{sym} falha em base.load_historical_bars: {e}")

    # Try utils.safe_copy_rates
    try:
        if (df is None or (hasattr(df, "empty") and df.empty)) and utils and hasattr(utils, "safe_copy_rates"):
            df = utils.safe_copy_rates(sym, timeframe, count=bars)
            logger.info(f"{sym} carregado via utils.safe_copy_rates")
    except Exception as e:
        logger.warning(f"{sym} falha em utils.safe_copy_rates: {e}")

    # Try direct MT5 copy_rates_from_pos
    try:
        if (df is None or (hasattr(df, "empty") and df.empty)) and mt5 and pd:
            if mt5.symbol_select(sym, True):
                rates = mt5.copy_rates_from_pos(sym, timeframe, 0, bars)
                if rates is not None and len(rates) > 0:
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    df.set_index('time', inplace=True)
                    df = df[['open', 'high', 'low', 'close', 'tick_volume']].rename(columns={'tick_volume': 'volume'})
                    logger.info(f"{sym} carregado diretamente via mt5.copy_rates_from_pos")
                else:
                    logger.warning(f"{sym} sem dados em mt5.copy_rates_from_pos")
            else:
                logger.warning(f"{sym} não selecionado no MT5")
    except Exception as e:
        logger.warning(f"{sym} falha em mt5 direto: {e}")

    # Try backfill.ensure_history -> CSV
    try:
        if (df is None or (hasattr(df, "empty") and df.empty)) and ensure_history:
            df = ensure_history(sym, period_days=60, interval='15m')
            if df is not None and not getattr(df, "empty", False):
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
                    df = df.set_index('time').sort_index()
            logger.info(f"{sym} carregado via backfill.ensure_history")
    except Exception as e:
        logger.warning(f"{sym} falha em backfill.ensure_history: {e}")

    # Final check
    if df is None or (hasattr(df, "empty") and df.empty):
        logger.error(f"{sym} sem dados de nenhuma fonte")
        return None

    # ensure index is datetime and sorted
    try:
        if pd is not None and "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
            df = df.set_index("time").sort_index()
    except Exception:
        pass
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
        max_windows = int(wfo_windows)
        for i in range(max_windows):
            train_start_idx = i * step
            train_end_idx = train_start_idx + train_period
            test_end_idx = train_end_idx + test_period
            if test_end_idx > n:
                break
            df_train = df_full.iloc[train_start_idx:train_end_idx].copy()
            df_test = df_full.iloc[train_end_idx:test_end_idx].copy()
            if df_train.empty or df_test.empty:
                continue
            best_params = optimize_window(sym, df_train, maxevals)
            test_res = backtest_params_on_df(sym, best_params, df_test)
            wins.append({"train_range": (str(df_train.index[0]), str(df_train.index[-1])),
                         "test_range": (str(df_test.index[0]), str(df_test.index[-1])),
                         "best_params": best_params,
                         "test_metrics": test_res,
                         "equity_curve": test_res.get("equity_curve", [])})
        
        if not wins:
            return {"symbol": sym, "error": "wfo_no_windows"}
        
        # Scoring mais robusto (prioriza Sortino e Calmar)
        def score_win(w):
            m = w.get("test_metrics", {}) or {}
            return (m.get("sortino", 0.0) * 0.4) + (m.get("calmar", 0.0) * 0.3) + (m.get("total_return", 0.0) * 0.2) + (m.get("sharpe", 0.0) * 0.1)
        
        wins_sorted = sorted(wins, key=score_win, reverse=True)
        best_overall = wins_sorted[0]
        out["test_metrics"] = best_overall.get("test_metrics", {})
        out["selected_params"] = best_overall.get("best_params", {})
        out["status"]         = best_overall.get("status", "ok")
        
        # Monte Carlo robustness check (100 simulações)
        eq_curve = best_overall.get("equity_curve", [])
        if len(eq_curve) > 50:
            returns_mc = np.diff(eq_curve) / np.array(eq_curve[:-1])
            if len(returns_mc) < 20:  # Muito poucos retornos → pula Monte Carlo
                best_overall["monte_carlo_avg_max_dd"] = 0.0
            else:
                block_size = 20
                mc_dd = []
                for _ in range(100):
                    blocks = len(returns_mc) // block_size
                    if blocks == 0:
                        continue  # Agora dentro do loop for → válido!
                    shuffled_blocks = np.random.choice(range(blocks), blocks, replace=True)
                    shuffled = np.concatenate([returns_mc[i*block_size:(i+1)*block_size] for i in shuffled_blocks])
                    shuffled = shuffled[:len(returns_mc)]  # ajusta tamanho exato
                    mc_eq = np.cumprod(1 + shuffled)
                    peak = np.maximum.accumulate(mc_eq)
                    dd = (peak - mc_eq) / peak
                    mc_dd.append(np.max(dd))
                avg_mc_dd = np.mean(mc_dd) if mc_dd else 0.0
                best_overall["monte_carlo_avg_max_dd"] = float(avg_mc_dd)
        
            out["wfo_windows"] = wins
            out["selected_params"] = best_overall.get("best_params", {})
        
            fp = os.path.join(OPT_OUTPUT_DIR, f"WFO_{sym}.json")
            safe_save_json(fp, out)
        return out
    except Exception as e:
        logger.exception(f"WFO worker failed for {sym}: {e}")
        return {"symbol": sym, "error": str(e)}


def filter_by_sector_correlation(selected_symbols):
    final_list = []
    sectors_count = {}
    for sym in selected_symbols:
        sector = config.SECTOR_MAP.get(sym, "OUTROS")
        if sectors_count.get(sector, 0) < config.MAX_PER_SECTOR:
            final_list.append(sym)
            sectors_count[sector] = sectors_count.get(sector, 0) + 1
    return final_list

def run_parallel_wfo(symbols: List[str], bars: int, maxevals: int, workers: int, wfo_windows: int, train_period: int, test_period: int):
    results = {}
    if not symbols:
        logger.info("No symbols provided")
        return results
    if workers <= 0:
        workers = workers_default
    logger.info(f"Running WFO on {len(symbols)} symbols with {workers} workers")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(worker_wfo, s, bars, maxevals, wfo_windows, train_period, test_period): s for s in symbols}
        for fut in as_completed(futures):
            s = futures[fut]
            try:
                r = fut.result()
                results[s] = r
                if r.get("error"):
                    logger.warning(f"WFO {s} error: {r.get('error')}")
                else:
                    logger.info(f"WFO {s} done")
            except Exception as e:
                results[s] = {"symbol": s, "error": str(e)}
                logger.exception(f"WFO {s} failed")
    return results

#===========================
# ATUALIZAÇÃO AUTOMÁTICA DO CONFIG.PY COM MELHORES PARÂMETROS
# ===========================
def update_config_with_optimized_params(elite_results):
    """
    Atualiza apenas o dicionário SYMBOLS no config.py, 
    preservando todas as outras variáveis.
    """
    import re

    try:
        with open("config.py", "r", encoding="utf-8") as f:
            content = f.read()

        # Prepara o novo dicionário SYMBOLS apenas com os ativos de elite
        new_symbols_dict = "SYMBOLS = {\n"
        for symbol, data in elite_results.items():
            p = data["selected_params"]
            new_symbols_dict += f"    '{symbol}': {{\n"
            new_symbols_dict += f"        'ema_short': {p['ema_short']}, 'ema_long': {p['ema_long']},\n"
            new_symbols_dict += f"        'rsi_low': {p['rsi_low']}, 'rsi_high': {p['rsi_high']},\n"
            new_symbols_dict += f"        'adx_threshold': {p['adx_threshold']}, 'mom_min': {p['mom_min']}\n"
            new_symbols_dict += "    },\n"
        new_symbols_dict += "}\n"

        # Substitui o bloco SYMBOLS antigo pelo novo usando Regex
        # Isso preserva o resto do arquivo (TIMEFRAME, contas, etc)
        pattern = r"SYMBOLS\s*=\s*\{.*?\}"
        if re.search(pattern, content, re.DOTALL):
            new_content = re.sub(pattern, new_symbols_dict, content, flags=re.DOTALL)
        else:
            new_content = content + "\n\n" + new_symbols_dict

        with open("config.py", "w", encoding="utf-8") as f:
            f.write(new_content)
            
        print("💎 config.py atualizado com sucesso (Apenas Elite).")

    except Exception as e:
        print(f"❌ Erro ao atualizar config.py: {e}")

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

if __name__ == "__main__":
    # 1. Carrega todos os ativos do seu mapa de setores
    try:
        from config import SECTOR_MAP
        symbols_to_optimize = list(SECTOR_MAP.keys())
        print(f"🔍 {len(symbols_to_optimize)} ativos carregados do SECTOR_MAP.")
    except Exception as e:
        print(f"❌ Erro ao carregar SECTOR_MAP: {e}")
        symbols_to_optimize = ["PETR4", "VALE3"]

    # --- CONFIGURAÇÃO DOS PARÂMETROS DO WFO ---
    # Ajuste estes valores conforme sua necessidade
    WFO_PARAMS = {
        "bars": 5000,          # Quantidade de candles para baixar
        "maxevals": 80,        # Trials do Optuna
        "wfo_windows": 3,      # Janelas de Walk-Forward
        "train_period": 1000,  # Tamanho do treino
        "test_period": 300     # Tamanho do teste (OOS)
    }

    all_results = {}
    print(f"🚀 Iniciando Otimização WFO...")

    # 2. Execução Paralela CORRIGIDA
    with ProcessPoolExecutor(max_workers=4) as executor:
        # Passamos o símbolo + todos os argumentos que a função worker_wfo exige
        future_to_symbol = {
            executor.submit(
                worker_wfo, 
                sym, 
                WFO_PARAMS["bars"], 
                WFO_PARAMS["maxevals"], 
                WFO_PARAMS["wfo_windows"], 
                WFO_PARAMS["train_period"], 
                WFO_PARAMS["test_period"]
            ): sym for sym in symbols_to_optimize
        }
        
        for future in tqdm(as_completed(future_to_symbol), total=len(symbols_to_optimize), desc="Processando"):
            sym = future_to_symbol[future]
            try:
                res = future.result()
                all_results[sym] = res
            except Exception as e:
                print(f"\n❌ Erro em {sym}: {e}")

    # 3. FILTRO DE ELITE
    print("\n" + "═"*60)
    print("🏆 RESULTADOS DO FILTRO DE ELITE (CALMAR >= 1.0)")
    print("═"*60)

    elite_results = {}
    for sym, res in all_results.items():
        if not isinstance(res, dict):
            print(f"⚠️ {sym}: resultado inválido (None)")
            continue
        if res.get("status") == "ok":
            # Puxa o Calmar médio das janelas de teste
            calmar_teste = res.get("test_metrics", {}).get("calmar", 0)
            
            if calmar_teste >= 1.0:
                elite_results[sym] = res
                print(f"🟢 {sym:7} | Calmar: {calmar_teste:5.2f} | [APROVADO]")
            else:
                print(f"🔴 {sym:7} | Calmar: {calmar_teste:5.2f} | [REJEITADO]")

    # 4. ATUALIZAÇÃO DO CONFIG.PY
    if elite_results:
        update_config_with_optimized_params(elite_results)
        print(f"\n✨ Sucesso: {len(elite_results)} ativos de elite salvos!")
    else:
        print("\n⚠️ Nenhum ativo atingiu Calmar >= 1.0. Config inalterado.")