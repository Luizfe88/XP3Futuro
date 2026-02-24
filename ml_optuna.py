import time
import logging
from typing import Dict, Any, Optional

import numpy as np

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False

try:
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import make_scorer, accuracy_score
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

from ml_optimizer import OptimizationResult

logger = logging.getLogger("ml_optuna")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class OptunaOptimizer:
    def __init__(self, random_state: int = 42, scoring=None, cv_splits: int = 5, n_trials: int = 30):
        if not (OPTUNA_AVAILABLE and SKLEARN_AVAILABLE):
            raise RuntimeError("Optuna ou scikit-learn indisponível.")
        self.random_state = int(random_state)
        self.n_trials = int(n_trials)
        self.scoring = scoring or make_scorer(accuracy_score)
        self.cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)

    def run(self, pipeline: Pipeline, X, y, param_space: Dict[str, Any]) -> OptimizationResult:
        start = time.time()
        sampler = TPESampler(seed=self.random_state)
        pruner = MedianPruner()
        history = []
        convergence = []

        def objective(trial: optuna.Trial) -> float:
            params = {}
            for name, spec in param_space.items():
                if isinstance(spec, dict) and all(k in spec for k in ("min", "max")):
                    mn = float(spec["min"]); mx = float(spec["max"])
                    if spec.get("log"):
                        v = trial.suggest_float(name, mn, mx, log=True)
                    else:
                        v = trial.suggest_float(name, mn, mx)
                    params[name] = float(v)
                elif isinstance(spec, (list, tuple)):
                    params[name] = trial.suggest_categorical(name, list(spec))
                else:
                    params[name] = spec
            try:
                scores = cross_val_score(pipeline.set_params(**params), X, y, cv=self.cv, scoring=self.scoring)
                score = float(np.mean(scores))
                history.append({"trial": trial.number, "score": score, "params": dict(params)})
                convergence.append((trial.number, score))
                return score
            except Exception as e:
                logger.debug(f"Falha no trial {trial.number}: {e}")
                return float("-inf")

        try:
            study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
            study.optimize(objective, n_trials=self.n_trials, gc_after_trial=True)
            best_params = study.best_params if study.best_params else {}
            best_score = float(study.best_value) if study.best_value is not None else float("-inf")
        except Exception as e:
            logger.error(f"Optuna falhou: {e}")
            best_params, best_score = {}, float("-inf")
        runtime = time.time() - start
        return OptimizationResult("Optuna", best_params, best_score, runtime, convergence=convergence, history=history)
