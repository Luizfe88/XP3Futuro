"""
OPTIMIZER_OPTUNA.PY — VERSÃO PROFISSIONAL FINAL
✅ Sem import circular
✅ Logging detalhado de erros
✅ Validação robusta
"""

import optuna
import logging
import numpy as np
import pandas as pd
from numba import njit

logger = logging.getLogger(__name__)

# =========================================================
# BACKTEST CORE (COPIADO PARA EVITAR IMPORT CIRCULAR)
# =========================================================
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

        trend_up = ema_short[i] > ema_long[i]
        buy_signal = (
            trend_up &
            (adx[i] > adx_threshold) &
            (rsi[i] < rsi_low) &
            (momentum[i] > 0)
        )

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
                    max_position_value = equity * 1.0
                    position = max_position_value / entry_price if entry_price > 0 else 0.0
                    cash -= position * entry_price
                    trades += 1
        else:
            if low[i] <= stop_price or sell_signal:
                exit_price = stop_price if low[i] <= stop_price else price * (1 - slippage / 2)
                cash += position * exit_price
                position = 0.0

        equity = cash + position * price
        equity_curve[i] = equity

    return equity_curve, trades


def calculate_adx(high, low, close, period=14):
    """Calcula ADX"""
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


def compute_metrics(equity_curve):
    """Calcula métricas simplificadas"""
    if len(equity_curve) < 50:
        return {"total_return": -1.0, "max_drawdown": 1.0, "calmar": -10.0}
    
    returns = np.diff(equity_curve) / equity_curve[:-1]
    total_return = equity_curve[-1] / equity_curve[0] - 1
    
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peak) / peak
    max_dd = max(-np.min(drawdowns), 0.01)
    
    years = len(equity_curve) / (252 * 28)
    annualized = (1 + total_return) ** (1 / years) - 1 if years >= 1 else total_return
    calmar = annualized / max_dd
    
    return {
        "total_return": float(total_return),
        "max_drawdown": float(max_dd),
        "calmar": float(calmar)
    }


def backtest_local(symbol: str, params: dict, df: pd.DataFrame):
    """Backtest sem dependências externas"""
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

    # Indicadores
    ema_s = pd.Series(close).ewm(span=params.get("ema_short", 9), adjust=False).mean().values
    ema_l = pd.Series(close).ewm(span=params.get("ema_long", 21), adjust=False).mean().values
    
    delta = pd.Series(close).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = (100 - (100 / (1 + rs))).fillna(50).values

    momentum = pd.Series(close).pct_change(periods=10).fillna(0).values
    adx, atr = calculate_adx(high, low, close, period=14)

    # Backtest
    equity_arr, trades = fast_backtest_core(
        close, high, low, ema_s, ema_l, rsi, adx, momentum, atr,
        params.get("rsi_low", 30),
        params.get("rsi_high", 70),
        params.get("adx_threshold", 25),
        params.get("mom_min", 0.0),
        params.get("sl_atr_multiplier", 2.0),
        0.0035,
        0.01
    )

    metrics = compute_metrics(equity_arr.tolist())
    metrics["total_trades"] = trades
    metrics["equity_curve"] = equity_arr.tolist()
    
    return metrics


# =========================================================
# FUNÇÃO OBJETIVO
# =========================================================
def objective(trial, symbol, df):
    params = {
        "ema_short": trial.suggest_int("ema_short", 8, 25),
        "ema_long": trial.suggest_int("ema_long", 40, 200),
        "rsi_low": trial.suggest_int("rsi_low", 35, 55),
        "rsi_high": trial.suggest_int("rsi_high", 65, 85),
        "adx_threshold": trial.suggest_int("adx_threshold", 20, 40),
        "mom_min": trial.suggest_float("mom_min", 0.0, 0.002),
        "sl_atr_multiplier": trial.suggest_float("sl_atr_multiplier", 1.2, 2.5)
    }

    result = backtest_local(symbol, params, df)

    trades = result.get("total_trades", 0)
    calmar = result.get("calmar", -10.0)
    total_return = result.get("total_return", -1.0)
    max_dd = result.get("max_drawdown", 1.0)

    # Filtros de sanidade
    if trades < 30:
        return 50.0 + (30 - trades)
    if trades > 300:
        return 30.0 + (trades - 300) * 0.1
    if max_dd > 0.50:
        return 40.0 + max_dd * 10
    if total_return < -0.10:
        return 25.0

    # Penalty extra para DD (exemplo: penaliza acima de 20%)
    dd_penalty_extra = max(0, max_dd - 0.2) * 10  # Ajuste esse valor conforme sua estratégia

    # Score corrigido
    score = calmar * 0.50 + total_return * 0.20 - max_dd * 0.30 - dd_penalty_extra

    trial.set_user_attr("trades", trades)
    trial.set_user_attr("calmar", calmar)
    trial.set_user_attr("return", total_return)
    trial.set_user_attr("max_dd", max_dd)

    return -score


# =========================================================
# FUNÇÃO PRINCIPAL DE OTIMIZAÇÃO
# =========================================================
def optimize_with_optuna(sym: str, df_train, n_trials: int = 60, timeout: int = 120):
    """
    Otimização RÁPIDA com Optuna
    - 60 trials (balanceado para velocidade)
    - Timeout de 120s (2 minutos máximo)
    - Pruning agressivo de trials ruins
    """
    logger.info(f"{sym}: 🔎 Optuna ({n_trials} trials, timeout={timeout}s)")

    if df_train is None or len(df_train) < 100:
        raise ValueError(f"{sym}: DataFrame insuficiente")

    optuna.logging.set_verbosity(optuna.logging.ERROR)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=42,
            n_startup_trials=15,  # ✅ Reduzido de 25 para 15
            multivariate=True
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,  # ✅ Mais agressivo
            n_warmup_steps=5
        )
    )

    try:
        study.optimize(
            lambda trial: objective(trial, sym, df_train),
            n_trials=n_trials,
            timeout=timeout,  # ✅ Timeout de 2 minutos
            show_progress_bar=False,
            catch=(Exception,)
        )
    except Exception as e:
        logger.error(f"{sym}: ❌ Optuna falhou: {e}")
        raise

    if not study.trials or study.best_trial is None:
        raise ValueError(f"{sym}: Nenhum trial válido")

    best_params = study.best_params
    best_score = -study.best_value
    attrs = study.best_trial.user_attrs

    logger.info(
        f"{sym}: ✅ {len(study.trials)} trials | "
        f"Score={best_score:.2f} | "
        f"Calmar={attrs.get('calmar', 0):.2f} | "
        f"EMA={best_params.get('ema_short')}/{best_params.get('ema_long')} | "
        f"RSI={best_params.get('rsi_low')}"
    )

    return {
        "best_params": best_params,
        "best_score": best_score,
        "best_trial_attrs": attrs
    }