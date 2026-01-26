# optimizer_optuna.py — VERSÃO SINCRONIZADA COM OTIMIZADOR_SEMANAL FINAL (16/12/2025)

import optuna
import logging
from otimizador_semanal import backtest_params_on_df

logger = logging.getLogger(__name__)

def objective(trial, sym: str, df_train):
    params = {
        "ema_short": trial.suggest_int("ema_short", 5, 35),
        "ema_long": trial.suggest_int("ema_long", 40, 120),
        "rsi_low": trial.suggest_int("rsi_low", 20, 45),
        "rsi_high": trial.suggest_int("rsi_high", 55, 80),
        "adx_threshold": trial.suggest_int("adx_threshold", 15, 40),
        "mom_min": trial.suggest_float("mom_min", -0.15, 0.15, step=0.01),
    }

    result = backtest_params_on_df(sym, params, df_train)

    # Proteção contra poucos trades
    if result["total_trades"] < 15:
        return 10.0  # Penalidade alta

    calmar = result.get("calmar", -10.0)
    total_return = result.get("total_return", -1.0)

    # Limitar retornos negativos extremos para evitar bias
    total_return = max(total_return, -0.9)

    # Score priorizando Calmar (robustez) + retorno
    # Como não temos Sortino mais, focamos no que importa: Calmar alto e retorno positivo
    score = (calmar * 0.75) + (total_return * 0.25)

    return -score  # Optuna minimiza, então invertemos

def optimize_with_optuna(sym: str, df_train, n_trials: int = 80):
    logger.info(f"{sym}: iniciando otimização Optuna ({n_trials} trials)")
    
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20)
    )
    
    study.optimize(lambda trial: objective(trial, sym, df_train), n_trials=n_trials, timeout=None)

    if len(study.trials) == 0:
        raise ValueError("Nenhum trial completado")

    best_params = study.best_params
    best_score = -study.best_value
    logger.info(f"{sym}: concluído! Melhor score: {best_score:.3f} → {best_params}")

    return {
        "best_params": best_params,
        "best_score": best_score
    }