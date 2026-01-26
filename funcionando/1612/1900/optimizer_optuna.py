# optimizer_optuna.py
import optuna
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Importe sua função de backtest do otimizador_semanal
from otimizador_semanal import backtest_params_on_df

def objective(trial, sym: str, df_train):
    # Espaço de busca (mantenha como está)
    params = {
        "ema_short": trial.suggest_int("ema_short", 5, 30),
        "ema_long": trial.suggest_int("ema_long", 35, 100),
        "rsi_low": trial.suggest_int("rsi_low", 20, 40),
        "rsi_high": trial.suggest_int("rsi_high", 60, 80),
        "adx_threshold": trial.suggest_int("adx_threshold", 15, 40),
        "mom_min": trial.suggest_float("mom_min", -0.1, 0.1, step=0.01),
    }

    try:
        result = backtest_params_on_df(sym, params, df_train)
        
        # CORREÇÃO 1: O backtest não retorna 'total_trades', usamos 'final_equity' para checar se houve algo
        if result.get("final_equity") == 100000.0: # Se o capital não mudou, não houve trades
            return 10.0

        # CORREÇÃO 2: Acessar métricas da raiz do dicionário 'result'
        calmar       = result.get("calmar", -10.0)
        total_return = result.get("total_return", -2.0)
        # O sortino não está sendo calculado no seu otimizador_semanal, usamos 0 ou removemos
        sortino      = 0.0 

        # Score ponderado
        score = (calmar * 0.7) + (total_return * 0.3)
        return -score 

    except Exception as e:
        logger.warning(f"Trial falhou para {sym}: {e}")
        return 10.0

def optimize_with_optuna(sym: str, df_train, n_trials: int = 80):
    logger.info(f"{sym}: iniciando otimização com Optuna ({n_trials} trials)")

    # DESATIVAMOS O PRUNER AUTOMÁTICO por enquanto (causa do erro)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    study.optimize(lambda trial: objective(trial, sym, df_train), n_trials=n_trials, timeout=None)

    if len(study.trials) == 0 or all(trial.state != optuna.trial.TrialState.COMPLETE for trial in study.trials):
        raise ValueError("No trials completed successfully")

    best_params = study.best_params
    best_score = -study.best_value

    logger.info(f"{sym}: otimização concluída! Melhor score: {best_score:.3f} → {best_params}")

    return {"best_params": best_params, "best_score": best_score}