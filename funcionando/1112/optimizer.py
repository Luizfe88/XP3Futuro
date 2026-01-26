
"""
Optimizer Turbo — WFO + Progress multi-nível + Save JSON per símbolo
Gerado (modificado) para seu projeto XP3-B3
Modificações principais:
- Limita análise a 6 símbolos (config.PROXY_SYMBOLS[:6])
- Progresso por símbolo (nível 1)
- Progresso por janela WFO (nível 2)
- Progresso por grid dentro da janela (nível 3)
- Logs de indicadores finais (EMA, RSI, ADX, MOM, ATR)
- Salva resumo detalhado em JSON por símbolo
"""

import os
import json
import math
import logging
import uuid
import multiprocessing
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
import itertools

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

try:
    import pandas_ta as ta
    _HAS_TA = True
except Exception:
    _HAS_TA = False

import config
import utils

# ========================= LOG ============================
GREEN = "\033[92m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
YELLOW = "\033[93m"
RESET = "\033[0m"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("optimizer_turbo")

# ===========================================================
# MÉTRICAS AVANÇADAS (mesmas funções)
# ===========================================================
def max_drawdown(returns: pd.Series) -> float:
    if returns is None or len(returns) == 0:
        return 0.0
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    dd = (peak - wealth) / peak
    return float(dd.max())

def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    if len(returns) == 0:
        return 0.0
    total_return = (1 + returns).prod() - 1
    years = len(returns) / periods_per_year
    if years <= 0:
        return 0.0
    return float((1 + total_return)**(1 / years) - 1)

def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    mu = returns.mean() * periods_per_year
    sigma = returns.std() * math.sqrt(periods_per_year)
    if sigma == 0:
        return 0.0
    return float(mu / sigma)

def calmar_ratio(returns: pd.Series) -> float:
    ann_ret = annualized_return(returns)
    mdd = max_drawdown(returns)
    if mdd == 0:
        return float("inf") if ann_ret > 0 else 0.0
    return float(ann_ret / mdd)

def expectancy_per_trade(returns: pd.Series, signals: pd.Series) -> float:
    if signals is None or returns is None:
        return 0.0
    trades_mask = signals.diff().abs() > 0
    trade_returns = []
    sig = signals.fillna(0)
    pos = sig.iloc[0]
    start_idx = None

    for i in range(1, len(sig)):
        if sig.iloc[i] != pos:
            if start_idx is not None:
                r = (1 + returns.iloc[start_idx:i]).prod() - 1
                trade_returns.append(r)
            pos = sig.iloc[i]
            start_idx = i

    if start_idx is not None:
        r = (1 + returns.iloc[start_idx:]).prod() - 1
        trade_returns.append(r)

    if not trade_returns:
        return 0.0

    wins = [r for r in trade_returns if r > 0]
    losses = [r for r in trade_returns if r <= 0]
    win_rate = len(wins) / len(trade_returns)
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = -np.mean(losses) if losses else 0.0
    return float(win_rate * avg_win - (1 - win_rate) * avg_loss)

def prob_of_ruin(expectancy: float, std_trade: float, capital=1.0, target_drawdown=0.5) -> float:
    if expectancy <= 0 or std_trade <= 0:
        return 1.0
    try:
        steps = math.log(1 - target_drawdown) / math.log(1 + expectancy)
        z = expectancy / std_trade * math.sqrt(max(1, steps))
        from math import erf
        p = 0.5 * (1 - math.erf(z / math.sqrt(2)))
        return float(max(0.0, min(1.0, p)))
    except Exception:
        return 1.0

# ===========================================================
# BACKTEST
# ===========================================================
def compute_indicators(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=int(params["ema_short"]), adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=int(params["ema_long"]), adjust=False).mean()

    rsi_p = int(params.get("rsi_period", 14))
    if _HAS_TA:
        df["rsi"] = ta.rsi(df["close"], length=rsi_p)
    else:
        delta = df["close"].diff()
        up = delta.clip(lower=0).rolling(rsi_p).mean()
        down = -delta.clip(upper=0).rolling(rsi_p).mean()
        rs = up / down
        df["rsi"] = 100 - (100 / (1 + rs))

    adx_p = int(params.get("adx_period", 14))
    if _HAS_TA:
        adx_df = ta.adx(df["high"], df["low"], df["close"], length=adx_p)
        df = pd.concat([df, adx_df], axis=1)
    else:
        df["ADX_14"] = 25

    df["mom"] = df["close"].pct_change(int(params.get("mom_period", 10)))
    return df.dropna()

def backtest_for_params(df: pd.DataFrame, params: Dict[str, Any], side: str) -> Dict[str, float]:
    if df.empty:
        return {"sharpe": 0.0}

    df2 = compute_indicators(df, params)
    df2["signal"] = 0

    if side.upper() in ("BUY", "COMPRA"):
        cond = (
            (df2["ema_fast"] > df2["ema_slow"]) &
            (df2["rsi"] < params.get("rsi_overbought", 70)) &
            (df2["ADX_14"] >= params.get("adx_threshold", 20)) &
            (df2["mom"] > 0)
        )
        df2.loc[cond, "signal"] = 1

    else:
        cond = (
            (df2["ema_fast"] < df2["ema_slow"]) &
            (df2["rsi"] > params.get("rsi_oversold", 30)) &
            (df2["ADX_14"] >= params.get("adx_threshold", 20)) &
            (df2["mom"] < 0)
        )
        df2.loc[cond, "signal"] = -1

    df2["pos"] = df2["signal"].shift(1).fillna(0)
    df2["ret"] = df2["close"].pct_change() * df2["pos"]
    df2["ret"] = df2["ret"].fillna(0)

    r = df2["ret"]

    metrics = {
        "sharpe": sharpe_ratio(r),
        "ann_return": annualized_return(r),
        "max_dd": max_drawdown(r),
        "calmar": calmar_ratio(r),
        "expectancy": expectancy_per_trade(r, df2["signal"]),
        "prob_ruin": prob_of_ruin(expectancy_per_trade(r, df2["signal"]), r.std() if r.std() is not None else 0.0)
    }

    # Indicadores finais (para relatório)
    try:
        last = df2.iloc[-1]
        indicators = {
            "ema_fast": float(last["ema_fast"]),
            "ema_slow": float(last["ema_slow"]),
            "rsi": float(last["rsi"]) if not pd.isna(last["rsi"]) else None,
            "adx": float(last.get("ADX_14", float("nan"))) if "ADX_14" in df2.columns else None,
            "mom": float(last["mom"]) if not pd.isna(last["mom"]) else None
        }
    except Exception:
        indicators = {}

    metrics["indicators"] = indicators
    return metrics

# ===========================================================
# WALK FORWARD
# ===========================================================
def generate_wfo_windows(df: pd.DataFrame, in_sample: int, oos: int, n_windows: int):
    df = df.sort_index()
    end = df.index.max()
    windows = []

    for w in range(n_windows):
        oos_end = end - timedelta(days=w * oos)
        oos_start = oos_end - timedelta(days=oos) + timedelta(days=1)
        is_end = oos_start - timedelta(days=1)
        is_start = is_end - timedelta(days=in_sample) + timedelta(days=1)

        is_df = df.loc[is_start:is_end]
        oos_df = df.loc[oos_start:oos_end]

        if len(is_df) > 100 and len(oos_df) > 30:
            windows.append((is_df, oos_df))

    return windows

def evaluate_grid_on_window(is_df, oos_df, grid_params, side, top_k=5):
    is_scores = []
    total_grid = len(grid_params)

    for i, p in enumerate(grid_params, start=1):
        # progresso interno do grid (nível 3) - log a cada 10%
        if total_grid > 0 and i % max(1, total_grid // 10) == 0:
            logger.info(f"Grid progress: { (i/total_grid)*100:.1f}%  ({i}/{total_grid})")

        m = backtest_for_params(is_df, p, side)
        is_scores.append((float(m["sharpe"]), p))

    is_scores.sort(reverse=True, key=lambda x: x[0])
    top_params = [p for _, p in is_scores[:top_k]]

    result = {}
    # para cada top param, testa OOS e guarda métricas
    for p in top_params:
        m = backtest_for_params(oos_df, p, side)
        key = json.dumps(p, sort_keys=True)
        result[key] = {
            "oos_sharpe": float(m["sharpe"]),
            "oos_metrics": m
        }

        logger.info(
            f"OOS test → sharpe={m['sharpe']:.4f} | params={p}"
        )

    return result

def wfo_for_symbol(symbol: str, grid_params, side: str, top_k=5):
    df = utils.safe_copy_rates(symbol, mt5.TIMEFRAME_D1, 2000)
    if df is None or df.empty:
        return {}

    windows = generate_wfo_windows(
        df,
        config.WFO_IN_SAMPLE_DAYS,
        config.WFO_OOS_DAYS,
        config.WFO_WINDOWS
    )

    if not windows:
        return {}

    results = {}
    total_wfo = len(windows)

    # use pool for parallel evaluation across windows
    with multiprocessing.Pool(processes=max(1, multiprocessing.cpu_count() - 1)) as pool:
        tasks = []
        for idx, (is_df, oos_df) in enumerate(windows, start=1):
            logger.info(f"{BLUE}[{symbol}] WFO janela {idx}/{total_wfo} → { (idx/total_wfo)*100:.1f}%{RESET}")
            task = pool.apply_async(
                evaluate_grid_on_window,
                (is_df, oos_df, grid_params, side, top_k)
            )
            tasks.append((idx, task))

        # coleta resultados na ordem que terminarem
        for idx, task in tasks:
            try:
                res = task.get()
                logger.info(f"{GREEN}[{symbol}] WFO janela concluída {idx}/{total_wfo}{RESET}")
                # res é dict key-> { "oos_sharpe":..., "oos_metrics":... }
                for k, v in res.items():
                    results.setdefault(k, []).append(v["oos_sharpe"])
            except Exception as e:
                logger.exception(f"wfo_for_symbol: janela {idx} task failed: {e}")

    return results

# ===========================================================
# AGREGAÇÃO E SALVAMENTO
# ===========================================================
def aggregate_results(results: List[Dict[str, List[float]]]):
    agg = {}
    count = {}
    for d in results:
        for k, vals in d.items():
            agg[k] = agg.get(k, 0.0) + np.mean(vals)
            count[k] = count.get(k, 0) + 1
    for k in agg:
        agg[k] /= count[k]
    return agg

def optimize_regime_turbo(regime: str, side: str, top_k=5):
    logger.info(f"{MAGENTA}>>> Otimizando TURBO regime={regime} lado={side}{RESET}")

    # monta grid
    grid_params = []
    g = config.GRID
    keys = list(g.keys())
    combos = itertools.product(*(g[k] for k in keys))

    for combo in combos:
        p = dict(zip(keys, combo))
        if p["ema_short"] >= p["ema_long"]:
            continue
        p.update({
            "rsi_overbought": config.DEFAULT_PARAMS["rsi_overbought"],
            "rsi_oversold": config.DEFAULT_PARAMS["rsi_oversold"],
            "adx_period": config.DEFAULT_PARAMS["adx_period"],
            "mom_period": config.DEFAULT_PARAMS["mom_period"],
        })
        grid_params.append(p)

    logger.info(f"Grid total: {len(grid_params)} parâmetros")

    # WFO por símbolo
    partial_dir = os.path.join(config.OPTIMIZER_OUTPUT, "partial")
    os.makedirs(partial_dir, exist_ok=True)

    results = []

    # Limitar análise para apenas 6 ações
    symbols = config.PROXY_SYMBOLS[:6]
    total = len(symbols)

    for i, sym in enumerate(symbols, start=1):

        # Barra de progresso em porcentagem (nível 1)
        progress = (i / total) * 100
        logger.info(f"{YELLOW}Progresso: {progress:.1f}% ({i}/{total}){RESET}")

        logger.info(f"{BLUE}Processando símbolo {sym}{RESET}")

        res = wfo_for_symbol(sym, grid_params, side, top_k)
        results.append(res)

        # salva parcial por símbolo (raw)
        fname = os.path.join(
            partial_dir,
            f"{regime}_{sym}_{uuid.uuid4().hex[:8]}.json"
        )
        try:
            with open(fname, "w", encoding="utf-8") as f:
                json.dump({"symbol": sym, "results": res}, f, indent=2, ensure_ascii=False)
        except Exception:
            logger.exception("Erro salvando partial file")

        # resumo por símbolo: média sharpe por param, top params, indicador snapshot para o melhor param
        try:
            # calcular média de cada param (res: key-> [oos_sharpe,...])
            mean_scores = {k: float(np.mean(v)) for k, v in res.items()} if res else {}
            sorted_params = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)
            top_params = [json.loads(k) for k, _ in sorted_params[:top_k]]

            # indicador snapshot usando o melhor param (se houver)
            indicator_snapshot = {}
            if top_params:
                best_param = top_params[0]
                try:
                    full_df = utils.safe_copy_rates(sym, mt5.TIMEFRAME_D1, 2000)
                    if full_df is not None and not full_df.empty:
                        ind_df = compute_indicators(full_df, best_param)
                        last = ind_df.iloc[-1]
                        atr = utils.get_atr(full_df, best_param.get("adx_period", 14))
                        indicator_snapshot = {
                            "ema_fast": float(last["ema_fast"]),
                            "ema_slow": float(last["ema_slow"]),
                            "rsi": float(last["rsi"]) if not pd.isna(last["rsi"]) else None,
                            "adx": float(last.get("ADX_14", float("nan"))) if "ADX_14" in ind_df.columns else None,
                            "mom": float(last["mom"]) if not pd.isna(last["mom"]) else None,
                            "atr": float(atr) if atr is not None else None
                        }
                except Exception:
                    logger.exception("Erro gerando indicador snapshot")

            summary = {
                "symbol": sym,
                "mean_scores": mean_scores,
                "top_params": top_params,
                "indicator_snapshot": indicator_snapshot,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            summary_fname = os.path.join(config.OPTIMIZER_OUTPUT, f"summary_{regime}_{sym}_{uuid.uuid4().hex[:8]}.json")
            with open(summary_fname, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            logger.info(f"{GREEN}Resumo salvo: {summary_fname}{RESET}")

            # log de sharpe médio do símbolo
            if mean_scores:
                mean_sharpe = float(np.mean(list(mean_scores.values())))
            else:
                mean_sharpe = 0.0
            logger.info(f"{GREEN}{sym} — Sharpe médio geral: {mean_sharpe:.4f}{RESET}")

        except Exception:
            logger.exception("Erro criando summary por símbolo")

    # agregação final de todos os símbolos processados
    agg = aggregate_results(results)
    if not agg:
        logger.error("Nenhum parâmetro válido encontrado!")
        return

    best_k, best_score = max(agg.items(), key=lambda x: x[1])
    best_params = json.loads(best_k)

    # salva melhor param no arquivo apropriado
    mapping = {
        "STRONG_BULL": config.PARAMS_STRONG_BULL,
        "BULL": config.PARAMS_BULL,
        "SIDEWAYS": config.PARAMS_SIDEWAYS,
        "BEAR": config.PARAMS_BEAR,
        "CRISIS": config.PARAMS_CRISIS
    }

    out_file = mapping.get(regime, config.PARAMS_SIDEWAYS)

    try:
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=2, ensure_ascii=False)
        logger.info(f"{GREEN}Melhor parâmetro salvo em {out_file} (Sharpe={best_score:.4f}){RESET}")
    except Exception:
        logger.exception("Erro salvando best params")

    # histórico
    hist_file = config.OPTIMIZER_HISTORY_FILE
    history = []
    if os.path.exists(hist_file):
        try:
            history = json.load(open(hist_file, "r", encoding="utf-8"))
        except:
            history = []

    history.append({
        "regime": regime,
        "side": side,
        "score": best_score,
        "params": best_params,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
    })

    try:
        with open(hist_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception:
        logger.exception("Erro atualizando histórico")

# ===========================================================
# MAIN
# ===========================================================
def main():
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

    for regime, side in regimes:
        optimize_regime_turbo(regime, side, top_k=5)
        print("\n" + "="*80 + "\n")

    mt5.shutdown()
    logger.info("Otimização TURBO concluída!")

# ==================== EXECUÇÃO ====================
if __name__ == "__main__":
    print("Inicializando MT5 usando instância já aberta...")

    # tenta conectar sem login, usando terminal já aberto
    ok = mt5.initialize()

    if not ok:
        erro = mt5.last_error()
        logger.critical(f"ERRO: Não foi possível conectar ao MT5! Detalhe: {erro}")
        exit()

    print("MT5 conectado com sucesso!")
    main()
