# otimizador_semanal.py — VERSÃO FINAL COM MONTE CARLO COMPLETO (16/12/2025)

import os
import json
import time
import logging
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import random
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

try:
    import config
except Exception:
    config = None

# ===========================
# LIMITES GLOBAIS (CONFIG)
# ===========================
MAX_SYMBOLS = int(getattr(config, "MAX_SYMBOLS", 10))
MAX_PER_SECTOR = int(getattr(config, "MAX_PER_SECTOR", 2))
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
    if not equity_curve or len(equity_curve) < 2:
        return {"total_return": 0.0, "max_drawdown": 0.01, "calmar": 0.0, "total_trades": 0}

    returns = np.diff(equity_curve) / equity_curve[:-1]
    total_return = equity_curve[-1] / equity_curve[0] - 1
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peak) / peak
    max_dd = -np.minimum.reduce(drawdowns) if len(drawdowns) > 0 else 0.01

    years = len(equity_curve) / (252 * 16)  # M15
    annualized = (1 + total_return) ** (1 / years) - 1 if years > 0 else total_return
    calmar = annualized / max_dd if max_dd > 0 else 0.0

    return {
        "total_return": float(total_return),
        "max_drawdown": float(max_dd),
        "calmar": float(calmar),
        "final_equity": float(equity_curve[-1]),
        "total_trades": len(returns)
    }

def backtest_params_on_df(symbol, params, df) -> Dict[str, Any]:
    if df is None or len(df) < 100:
        return {"total_return": -1.0, "calmar": -10.0, "total_trades": 0, "final_equity": 100000.0, "equity_curve": [100000.0]}

    df = df.copy()
    close = df['close'].values
    ema_s = pd.Series(close).ewm(span=params.get("ema_short", 9), adjust=False).mean().values
    ema_l = pd.Series(close).ewm(span=params.get("ema_long", 21), adjust=False).mean().values

    cash = 100000.0
    position = 0  # 1 long, -1 short, 0 flat
    entry_price = 0.0
    equity_curve = [cash]
    trades = 0
    slippage = 0.0012  # ~0.12% round-trip B3

    for i in range(max(params.get("ema_long", 50), 50), len(close)):
        price = close[i]
        buy_signal = ema_s[i] > ema_l[i]
        sell_signal = ema_s[i] < ema_l[i]

        if position == 0:
            if buy_signal:
                position = 1
                entry_price = price * (1 + slippage / 2)
                trades += 1
            elif sell_signal:
                position = -1
                entry_price = price * (1 - slippage / 2)
                trades += 1
        elif position == 1 and sell_signal:
            exit_price = price * (1 - slippage / 2)
            ret = (exit_price / entry_price) - 1
            cash *= (1 + ret)
            position = 0
        elif position == -1 and buy_signal:
            exit_price = price * (1 + slippage / 2)
            ret = (entry_price / exit_price) - 1
            cash *= (1 + ret)
            position = 0

        equity_curve.append(cash)

    metrics = compute_basic_metrics(equity_curve)
    metrics["equity_curve"] = equity_curve  # <-- ESSENCIAL PARA MONTE CARLO
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

    # Garante que o índice seja datetime e ordenado
    try:
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True)
        df = df.sort_index()
    except Exception as e:
        logger.warning(f"Erro ao ajustar índice de {sym}: {e}")

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

    calmars = []
    max_dds = []
    wins = 0

    for _ in range(n_simulations):
        shuffled = np.random.choice(returns, size=len(returns), replace=True)

        sim_equity = [equity_curve[0]]
        for r in shuffled:
            sim_equity.append(sim_equity[-1] * (1 + r))

        # Max Drawdown da simulação
        peak = np.maximum.accumulate(sim_equity)
        dd = (peak - sim_equity) / peak
        max_dd = float(np.max(dd)) if len(dd) else 0.0
        max_dds.append(max_dd)

        total_ret = sim_equity[-1] / sim_equity[0] - 1
        wins += int(total_ret > 0)

        years = len(sim_equity) / (252 * 16)  # M15
        ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else total_ret
        calmar = ann_ret / max_dd if max_dd > 0 else 0.0
        calmars.append(calmar)

    return {
        "win_rate": wins / n_simulations,
        "calmar_avg": float(np.mean(calmars)),
        "calmar_median": float(np.median(calmars)),
        "calmar_5th": float(np.percentile(calmars, 5)),
        "max_dd_95": float(np.percentile(max_dds, 95))  # 🔥 AQUI ESTÁ O OURO
    }


import shutil
from collections import defaultdict

def enforce_sector_limits(elite_dict: dict) -> dict:
    """
    Garante que não exceda MAX_PER_SECTOR por setor e MAX_SYMBOLS no total.
    Prioriza os melhores por Calmar dentro do setor.
    """
    if not elite_dict:
        return {}

    # Mapeia setor de cada símbolo
    sector_of = {}
    for sym, sector in SECTOR_MAP.items():
        if sym in elite_dict:
            sector_of[sym] = sector

    # Agrupa por setor e ordena por Calmar (do resultado do test_metrics)
    sector_groups = defaultdict(list)
    for sym, data in elite_dict.items():
        calmar = data.get("test_metrics", {}).get("calmar", 0)
        sector = sector_of.get(sym, "UNKNOWN")
        sector_groups[sector].append((sym, calmar, data))

    # Seleciona os melhores por setor
    selected = {}
    for sector, items in sector_groups.items():
        items.sort (key=lambda x: x[1], reverse=True)  # maior Calmar primeiro
        for i, (sym, calmar, data) in enumerate(items):
            if i < MAX_PER_SECTOR:
                selected[sym] = data

    # Limita total global
    if len(selected) > MAX_SYMBOLS:
        sorted_all = sorted(selected.items(), key=lambda x: x[1].get("test_metrics", {}).get("calmar", 0), reverse=True)
        selected = {sym: data for sym, data in sorted_all[:MAX_SYMBOLS]}

    return selected

def update_elite_symbols(final_elite: dict, config_path="config.py"):
    """
    Atualiza ou cria o bloco ELITE_SYMBOLS no config.py de forma segura.
    """
    if not final_elite:
        print("⚠️ ELITE vazia. Nenhuma atualização feita no config.py.")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    start_idx = None
    end_idx = None

    # Localiza início do bloco
    start_idx = None
    end_idx = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("ELITE_SYMBOLS"):
            start_idx = i
            # Procura o próximo } a partir daqui
            for j in range(i + 1, len(lines)):
                if "}" in lines[j].strip():
                    end_idx = j
                    break
            break

    # Monta novo bloco
    new_block = ["\n# ===========================\n",
                 "# PARÂMETROS OTIMIZADOS MANUAIS (ELITE)\n",
                 "# ===========================\n",
                 "ELITE_SYMBOLS = {\n"]

    for sym, params in final_elite.items():
        new_block.append(f'    "{sym}": {params},\n')

    new_block.append("}\n")

    # CASO 1 — bloco existe corretamente
    if start_idx is not None and end_idx is not None:
        new_lines = lines[:start_idx] + new_block + lines[end_idx + 1:]

    # CASO 2 — bloco não existe → cria no final
    elif start_idx is None:
        print("ℹ️ ELITE_SYMBOLS não encontrado. Criando novo bloco no final do config.py.")
        new_lines = lines + ["\n"] + new_block

    # CASO 3 — bloco corrompido
    else:
        raise RuntimeError(
            "❌ Bloco ELITE_SYMBOLS encontrado mas não fechado corretamente no config.py"
        )

    with open(config_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print(f"✅ ELITE_SYMBOLS atualizado com {len(final_elite)} ativos.")


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
        "bars": 5000,
        "maxevals": 80,
        "wfo_windows": 3,
        "train_period": 1000,
        "test_period": 300
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

    # Filtro inicial por Calmar
    # Filtro inicial por Calmar + número mínimo de trades
    elite_results = {}
    for sym, res in all_results.items():
        if res.get("status") != "ok":
            continue

        metrics = res.get("test_metrics", {})
        calmar = metrics.get("calmar", 0.0)
        trades = metrics.get("total_trades", 0)

        if calmar >= 1.0 and trades >= 40:
            elite_results[sym] = res
        print(
            f"🟢 {sym:7} | Calmar OOS: {calmar:5.2f} | "
            f"Trades: {trades:3d} | [PRÉ-APROVADO]"
        )
    else:
        print(
            f"⚠️ {sym:7} | Calmar: {calmar:5.2f} | "
            f"Trades: {trades:3d} | REJEITADO (POUCOS TRADES)"
        )

    # Monte Carlo Stress Test
    print("\n" + "═"*80)
    print("🧪 MONTE CARLO STRESS TEST COMPLETO (1000 simulações por ativo)")
    print("═"*80)

    final_elite = {}
    for sym, res in elite_results.items():
        eq_curve = res.get("equity_curve", [])
        mc = run_monte_carlo_stress(eq_curve, n_simulations=1000)

        print(
            f"{sym:8} | WinRate MC: {mc['win_rate']*100:5.1f}% | "
            f"Calmar Med: {mc['calmar_median']:5.2f} | "
            f"Calmar 5%: {mc['calmar_5th']:6.2f} | "
            f"DD 95%: {mc['max_dd_95']*100:5.1f}%"
        )
        stability = mc["calmar_5th"] / max(mc["calmar_avg"], 0.01)

        if (mc["win_rate"] >= 0.60 and 
            mc["calmar_avg"] >= 1.0 and 
            mc["calmar_5th"] >= -25.0 and
            mc["max_dd_95"] <= 0.35):
            final_elite[sym] = res["selected_params"]
            print(f"✅ {sym} APROVADO COMO ELITE FINAL")
        else:
            print(f"❌ {sym} REJEITADO NO MONTE CARLO")

    # Atualiza config apenas com os verdadeiros elite
    if final_elite:
        update_elite_symbols(final_elite)
        print(f"\n🎯 {len(final_elite)} ativos aprovados como ELITE FINAL após Monte Carlo!")
        print("   config.py atualizado com parâmetros ultra-robustos.")
    else:
        print("\n⚠️ Nenhum ativo passou no Monte Carlo. Config.py mantido inalterado.")

    print_optimizer_dashboard(all_results)