import time
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional, Callable
import os
import json
from datetime import datetime

import numpy as np

try:
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import make_scorer, accuracy_score
    SKLEARN_AVAILABLE = True
except Exception as e:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger("ml_optimizer")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )


@dataclass
class OptimizationResult:
    method: str
    best_params: Dict[str, Any]
    best_score: float
    runtime_seconds: float
    convergence: List[Tuple[int, float]] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "best_params": self.best_params,
            "best_score": self.best_score,
            "runtime_seconds": self.runtime_seconds,
            "convergence": self.convergence,
            "history": self.history,
        }


def _ensure_sklearn():
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn não está disponível. Instale 'scikit-learn' para usar o otimizador.")


def _is_numeric_space(space_item: Any) -> bool:
    return isinstance(space_item, dict) and all(k in space_item for k in ("min", "max"))


def _clamp(v: float, mn: float, mx: float) -> float:
    return max(mn, min(mx, v))


def _sample_params(param_space: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
    params = {}
    for name, spec in param_space.items():
        if _is_numeric_space(spec):
            mn, mx = float(spec["min"]), float(spec["max"])
            if "log" in spec and spec["log"]:
                v = float(np.exp(rng.uniform(np.log(mn), np.log(mx))))
            else:
                v = float(rng.uniform(mn, mx))
            params[name] = v
        elif isinstance(spec, (list, tuple)):
            params[name] = spec[int(rng.integers(0, len(spec)))]
        else:
            params[name] = spec
    return params


def _finite_diff_gradient(pipeline: Pipeline, X, y, params: Dict[str, Any], cv, scoring, eps: float = 1e-3) -> Dict[str, float]:
    grad = {}
    base_score = float(np.mean(cross_val_score(pipeline.set_params(**params), X, y, cv=cv, scoring=scoring)))
    for k, v in list(params.items()):
        if not isinstance(v, (int, float)):
            grad[k] = 0.0
            continue
        perturb = dict(params)
        perturb[k] = float(v) + eps
        try:
            s_plus = float(np.mean(cross_val_score(pipeline.set_params(**perturb), X, y, cv=cv, scoring=scoring)))
        except Exception:
            s_plus = base_score
        grad[k] = (s_plus - base_score) / eps
    return grad


class OptimizerBase:
    def __init__(self, random_state: int = 42, scoring: Optional[Callable] = None, cv_splits: int = 5):
        _ensure_sklearn()
        self.random_state = int(random_state)
        self.rng = np.random.default_rng(self.random_state)
        self.scoring = scoring or make_scorer(accuracy_score)
        self.cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)

    def _evaluate(self, pipeline: Pipeline, X, y, params: Dict[str, Any]) -> float:
        try:
            scores = cross_val_score(pipeline.set_params(**params), X, y, cv=self.cv, scoring=self.scoring)
            return float(np.mean(scores))
        except Exception as e:
            logger.debug(f"Falha ao avaliar params={params}: {e}")
            return float("-inf")


class GridSearchOptimizer(OptimizerBase):
    def run(self, pipeline: Pipeline, X, y, param_grid: Dict[str, Any]) -> OptimizationResult:
        start = time.time()
        try:
            logger.info("Iniciando GridSearchCV")
            gs = GridSearchCV(pipeline, param_grid=param_grid, cv=self.cv, scoring=self.scoring, n_jobs=-1, refit=False)
            gs.fit(X, y)
            best_params = gs.best_params_
            best_score = float(gs.best_score_)
            logger.info(f"GridSearchCV concluído: score={best_score:.4f}")
        except Exception as e:
            logger.error(f"GridSearch falhou: {e}")
            best_params, best_score = {}, float("-inf")
        runtime = time.time() - start
        return OptimizationResult("GridSearchCV", best_params, best_score, runtime, convergence=[], history=[])


class RandomSearchOptimizer(OptimizerBase):
    def run(self, pipeline: Pipeline, X, y, param_distributions: Dict[str, Any], n_iter: int = 32) -> OptimizationResult:
        start = time.time()
        try:
            logger.info("Iniciando RandomizedSearchCV")
            rs = RandomizedSearchCV(
                pipeline, param_distributions=param_distributions, n_iter=n_iter, cv=self.cv,
                scoring=self.scoring, random_state=self.random_state, n_jobs=-1, refit=False
            )
            rs.fit(X, y)
            best_params = rs.best_params_
            best_score = float(rs.best_score_)
            logger.info(f"RandomizedSearchCV concluído: score={best_score:.4f}")
        except Exception as e:
            logger.error(f"RandomizedSearch falhou: {e}")
            best_params, best_score = {}, float("-inf")
        runtime = time.time() - start
        return OptimizationResult("RandomizedSearchCV", best_params, best_score, runtime, convergence=[], history=[])


class GradientDescentOptimizer(OptimizerBase):
    def run(self, pipeline: Pipeline, X, y, param_space: Dict[str, Any], max_iters: int = 50, lr: float = 0.1, tol: float = 1e-4, patience: int = 5) -> OptimizationResult:
        start = time.time()
        logger.info(f"Iniciando GradientDescent iters={max_iters} lr={lr}")
        params = _sample_params(param_space, self.rng)
        history = []
        convergence = []
        best_params = dict(params)
        best_score = self._evaluate(pipeline, X, y, best_params)
        no_improve = 0
        for it in range(1, max_iters + 1):
            try:
                grad = _finite_diff_gradient(pipeline, X, y, params, self.cv, self.scoring, eps=1e-3)
                for k, g in grad.items():
                    if k in param_space and _is_numeric_space(param_space[k]):
                        mn = float(param_space[k]["min"]); mx = float(param_space[k]["max"])
                        step = lr / math.sqrt(it)
                        params[k] = _clamp(float(params[k]) + step * g, mn, mx)
                score = self._evaluate(pipeline, X, y, params)
                history.append({"iter": it, "score": score, "params": dict(params)})
                convergence.append((it, score))
                if it % max(1, (max_iters // 5)) == 0:
                    logger.info(f"GD it={it} score={score:.4f}")
                if score > best_score + tol:
                    best_score = score
                    best_params = dict(params)
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        logger.info(f"GD early stopping em it={it} (sem melhora por {patience})")
                        break
            except Exception as e:
                logger.debug(f"GD falhou na iteração {it}: {e}")
                continue
        runtime = time.time() - start
        logger.info(f"GradientDescent concluído: best_score={best_score:.4f} tempo={runtime:.2f}s")
        return OptimizationResult("GradientDescent", best_params, best_score, runtime, convergence=convergence, history=history)


class GeneticAlgorithmOptimizer(OptimizerBase):
    def run(self, pipeline: Pipeline, X, y, param_space: Dict[str, Any], population: int = 16, generations: int = 20, elite_frac: float = 0.25, mutation_prob: float = 0.2, mutation_scale: float = 0.1, tol: float = 1e-4, patience: int = 5) -> OptimizationResult:
        start = time.time()
        logger.info(f"Iniciando GeneticAlgorithm pop={population} gens={generations}")
        pop = [_sample_params(param_space, self.rng) for _ in range(population)]
        history = []
        convergence = []
        best_params, best_score = {}, float("-inf")
        no_improve = 0
        for gen in range(1, generations + 1):
            try:
                fitness = [self._evaluate(pipeline, X, y, ind) for ind in pop]
                order = np.argsort(fitness)[::-1]
                pop = [pop[i] for i in order]
                fitness = [fitness[i] for i in order]
                if fitness[0] > best_score + tol:
                    best_score = float(fitness[0])
                    best_params = dict(pop[0])
                    no_improve = 0
                else:
                    no_improve += 1
                history.append({"gen": gen, "best": best_score})
                convergence.append((gen, best_score))
                if gen % max(1, (generations // 5)) == 0:
                    logger.info(f"GA gen={gen} best={best_score:.4f}")
                if no_improve >= patience:
                    logger.info(f"GA early stopping em gen={gen} (sem melhora por {patience})")
                    break
                elites = pop[:max(1, int(elite_frac * population))]
                children = []
                while len(children) + len(elites) < population:
                    p1, p2 = self.rng.choice(elites, size=2, replace=True)
                    child = {}
                    for k in param_space.keys():
                        v = p1[k] if self.rng.random() < 0.5 else p2[k]
                        if self.rng.random() < mutation_prob:
                            spec = param_space[k]
                            if _is_numeric_space(spec):
                                rng_scale = mutation_scale * (float(spec["max"]) - float(spec["min"]))
                                v = _clamp(float(v) + self.rng.normal(0, rng_scale), float(spec["min"]), float(spec["max"]))
                            elif isinstance(spec, (list, tuple)):
                                v = spec[int(self.rng.integers(0, len(spec)))]
                        child[k] = v
                    children.append(child)
                pop = elites + children
            except Exception as e:
                logger.debug(f"GA falhou na geração {gen}: {e}")
                continue
        runtime = time.time() - start
        logger.info(f"GeneticAlgorithm concluído: best_score={best_score:.4f} tempo={runtime:.2f}s")
        return OptimizationResult("GeneticAlgorithm", best_params, best_score, runtime, convergence=convergence, history=history)


class SimulatedAnnealingOptimizer(OptimizerBase):
    def run(self, pipeline: Pipeline, X, y, param_space: Dict[str, Any], max_iters: int = 200, T_start: float = 1.0, T_end: float = 0.01, tol: float = 1e-4, patience: int = 10) -> OptimizationResult:
        start = time.time()
        logger.info(f"Iniciando SimulatedAnnealing iters={max_iters}")
        current = _sample_params(param_space, self.rng)
        current_score = self._evaluate(pipeline, X, y, current)
        best_params, best_score = dict(current), float(current_score)
        history = []
        convergence = []
        no_improve = 0
        for it in range(1, max_iters + 1):
            try:
                t = it / max_iters
                T = T_start * ((T_end / T_start) ** t)
                neighbor = dict(current)
                k = self.rng.choice(list(param_space.keys()))
                spec = param_space[k]
                if _is_numeric_space(spec):
                    rng_scale = 0.1 * (float(spec["max"]) - float(spec["min"]))
                    neighbor[k] = _clamp(float(neighbor[k]) + self.rng.normal(0, rng_scale), float(spec["min"]), float(spec["max"]))
                elif isinstance(spec, (list, tuple)):
                    neighbor[k] = spec[int(self.rng.integers(0, len(spec)))]
                s_neighbor = self._evaluate(pipeline, X, y, neighbor)
                accept = (s_neighbor > current_score) or (math.exp((s_neighbor - current_score) / max(T, 1e-8)) > self.rng.random())
                if accept:
                    current, current_score = neighbor, s_neighbor
                if current_score > best_score + tol:
                    best_params, best_score = dict(current), float(current_score)
                    no_improve = 0
                else:
                    no_improve += 1
                history.append({"iter": it, "score": current_score, "T": T})
                convergence.append((it, best_score))
                if it % max(1, (max_iters // 5)) == 0:
                    logger.info(f"SA it={it} current={current_score:.4f} best={best_score:.4f} T={T:.4f}")
                if no_improve >= patience:
                    logger.info(f"SA early stopping em it={it} (sem melhora por {patience})")
                    break
            except Exception as e:
                logger.debug(f"SA falhou na iteração {it}: {e}")
                continue
        runtime = time.time() - start
        logger.info(f"SimulatedAnnealing concluído: best_score={best_score:.4f} tempo={runtime:.2f}s")
        return OptimizationResult("SimulatedAnnealing", best_params, best_score, runtime, convergence=convergence, history=history)


def build_default_pipeline(estimator, use_scaler: bool = True) -> Pipeline:
    _ensure_sklearn()
    steps = []
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    try:
        steps.append(("pca", PCA(n_components=None, whiten=False, random_state=42)))
    except Exception:
        pass
    steps.append(("model", estimator))
    return Pipeline(steps)


def compare_methods(pipeline: Pipeline, X, y, param_space_grid: Dict[str, Any], param_space_random: Dict[str, Any], param_space_numeric: Dict[str, Any], random_state: int = 42) -> Dict[str, OptimizationResult]:
    gd = GradientDescentOptimizer(random_state=random_state)
    ga = GeneticAlgorithmOptimizer(random_state=random_state)
    sa = SimulatedAnnealingOptimizer(random_state=random_state)
    gs = GridSearchOptimizer(random_state=random_state)
    rs = RandomSearchOptimizer(random_state=random_state)
    results = {}
    try:
        results["GridSearchCV"] = gs.run(pipeline, X, y, param_space_grid)
    except Exception as e:
        logger.warning(f"GridSearchCV indisponível: {e}")
    try:
        results["RandomizedSearchCV"] = rs.run(pipeline, X, y, param_space_random, n_iter=32)
    except Exception as e:
        logger.warning(f"RandomizedSearchCV indisponível: {e}")
    try:
        results["GradientDescent"] = gd.run(pipeline, X, y, param_space_numeric, max_iters=50, lr=0.2)
    except Exception as e:
        logger.warning(f"GradientDescent indisponível: {e}")
    try:
        results["GeneticAlgorithm"] = ga.run(pipeline, X, y, param_space_numeric, population=16, generations=20)
    except Exception as e:
        logger.warning(f"GeneticAlgorithm indisponível: {e}")
    try:
        results["SimulatedAnnealing"] = sa.run(pipeline, X, y, param_space_numeric, max_iters=200)
    except Exception as e:
        logger.warning(f"SimulatedAnnealing indisponível: {e}")
    try:
        from ml_optuna import OptunaOptimizer
        opt = OptunaOptimizer(random_state=random_state, n_trials=20)
        results["Optuna"] = opt.run(pipeline, X, y, param_space_numeric)
    except Exception as e:
        logger.warning(f"Optuna indisponível: {e}")
    return results


class WeeklyOptimizerRunner:
    def __init__(self, random_state: int = 42, scoring: Optional[Callable] = None, output_dir: Optional[str] = None):
        _ensure_sklearn()
        self.random_state = int(random_state)
        self.scoring = scoring or make_scorer(accuracy_score)
        self.output_dir = output_dir or "optimizer_output"
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except Exception:
            pass

    def _persist(self, tag: str, results: Dict[str, OptimizationResult]) -> Optional[str]:
        try:
            payload = {k: v.to_dict() for k, v in results.items()}
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.output_dir, f"weekly_{tag}_{ts}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            return path
        except Exception as e:
            logger.warning(f"Falha ao persistir resultados: {e}")
            return None

    def run_weekly(self, estimator, data_loader: Callable[[], Tuple[np.ndarray, np.ndarray]], param_grid: Dict[str, Any], param_distributions: Dict[str, Any], param_numeric_space: Dict[str, Any], use_scaler: bool = True) -> Dict[str, Any]:
        try:
            X, y = data_loader()
        except Exception as e:
            raise RuntimeError(f"Falha ao carregar dados: {e}")
        pipe = build_default_pipeline(estimator, use_scaler=use_scaler)
        try:
            results = compare_methods(pipe, X, y, param_grid, param_distributions, param_numeric_space, random_state=self.random_state)
        except Exception as e:
            logger.error(f"Falha na comparação de métodos: {e}")
            raise
        out_path = self._persist(estimator.__class__.__name__, results)
        summary = {
            "best": sorted([(k, v.best_score) for k, v in results.items()], key=lambda x: -x[1])[0],
            "output_path": out_path,
            "count_methods": len(results),
        }
        return {"results": results, "summary": summary}


class EnsembleOptimizer:
    def __init__(self, storage_path: Optional[str] = None, max_history: int = 5000):
        self.storage_path = storage_path or os.path.join(os.getcwd(), "ml_ensemble_history.json")
        self.max_history = int(max_history)
        self.history: List[Dict[str, Any]] = []
        self.history_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
        self.last_train_at: Optional[float] = None
        self._load()

    def _load(self) -> None:
        try:
            if not os.path.exists(self.storage_path):
                return
            with open(self.storage_path, "r", encoding="utf-8") as f:
                payload = json.load(f) or {}
            history = payload.get("history", [])
            if isinstance(history, list):
                self.history = [h for h in history if isinstance(h, dict)]
            hbs = payload.get("history_by_symbol", {})
            if isinstance(hbs, dict):
                out: Dict[str, List[Dict[str, Any]]] = {}
                for sym, arr in hbs.items():
                    if isinstance(arr, list):
                        out[str(sym)] = [h for h in arr if isinstance(h, dict)]
                self.history_by_symbol = out
        except Exception:
            self.history = []
            self.history_by_symbol = {}

    def force_save(self) -> None:
        try:
            payload = {
                "history": self.history[-self.max_history :],
                "history_by_symbol": {
                    sym: arr[-self.max_history :] for sym, arr in (self.history_by_symbol or {}).items()
                },
                "last_train_at": self.last_train_at,
                "saved_at": time.time(),
            }
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            return

    def record_trade(self, symbol: str, pnl_pct: float, indicators: Optional[Dict[str, Any]] = None) -> None:
        try:
            item = {
                "ts": time.time(),
                "symbol": str(symbol),
                "pnl_pct": float(pnl_pct),
                "indicators": indicators if isinstance(indicators, dict) else {},
            }
            self.history.append(item)
            self.history_by_symbol.setdefault(str(symbol), []).append(item)
            if len(self.history) > self.max_history:
                self.history = self.history[-self.max_history :]
            if len(self.history_by_symbol.get(str(symbol), [])) > self.max_history:
                self.history_by_symbol[str(symbol)] = self.history_by_symbol[str(symbol)][-self.max_history :]
        except Exception:
            return

    def train_ensemble(self) -> None:
        self.last_train_at = time.time()
        self.force_save()

    def optimize(self, df, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Otimiza limiares de trading (ADX, RSI, EMAs) usando Random Search simples.
        Maximiza: Total Profit * Win Rate
        """
        try:
            if df is None or len(df) < 100:
                return None
                
            # Prepara dados (indicadores)
            # Assume que df já tem OHLCV
            # Recalcula indicadores básicos se não existirem
            # Para otimização rápida, calculamos indicadores 'on-the-fly' ou pré-calculamos faixas?
            # Melhor pré-calcular colunas fixas
            
            import pandas as pd
            import pandas_ta as ta
            
            data = df.copy()
            
            # Garante colunas de indicadores base
            # EMAs variadas para seleção
            for p in [9, 21, 50, 200]:
                data[f"EMA_{p}"] = ta.ema(data['close'], length=p)
                
            data["RSI"] = ta.rsi(data['close'], length=14)
            adx_df = ta.adx(data['high'], data['low'], data['close'], length=14)
            if adx_df is not None and not adx_df.empty:
                data["ADX"] = adx_df[adx_df.columns[0]] # ADX_14
            else:
                data["ADX"] = 0
                
            data.dropna(inplace=True)
            if len(data) < 50:
                return None
                
            # Define espaço de busca
            param_grid = {
                "adx_threshold": [15, 20, 25, 30, 35],
                "rsi_low": [25, 30, 35, 40],
                "rsi_high": [60, 65, 70, 75],
                "ema_short": [9, 21], # Simplificado
                "ema_long": [50, 200], # Simplificado
                "tp_ratio": [1.5, 2.0, 3.0],
                "sl_atr_mult": [1.5, 2.0, 2.5]
            }
            
            best_score = -float('inf')
            best_params = {}
            
            # Random Search (30 iterações)
            import random
            for _ in range(30):
                # Amostra parâmetros
                curr_params = {k: random.choice(v) for k, v in param_grid.items()}
                
                # Simula estratégia (Vetorizada seria ideal, mas iterativa é mais fácil de implementar rápido)
                # Vamos fazer uma simulação simplificada vetorizada
                
                # Sinais
                ema_s = data[f"EMA_{curr_params['ema_short']}"]
                ema_l = data[f"EMA_{curr_params['ema_long']}"]
                
                trend_up = ema_s > ema_l
                trend_down = ema_s < ema_l
                
                adx_ok = data["ADX"] > curr_params['adx_threshold']
                rsi_buy = data["RSI"] < curr_params['rsi_high'] # Não comprado
                rsi_sell = data["RSI"] > curr_params['rsi_low'] # Não vendido
                
                # Entradas (Close > Open para simplificar 'candle verde')
                # Na verdade, o bot entra no tick. Aqui simulamos no fechamento do candle.
                
                # Shift(1) para não olhar o futuro (entra na abertura do candle seguinte ao sinal)
                entries_long = (trend_up & adx_ok & rsi_buy).shift(1).fillna(False)
                entries_short = (trend_down & adx_ok & rsi_sell).shift(1).fillna(False)
                
                # Retornos (simplificado: Close - Open do candle seguinte)
                # Real: (Close_t+1 - Open_t+1)
                # Aqui usamos log returns do candle onde estaríamos posicionados
                
                returns = np.log(data['close'] / data['close'].shift(1))
                
                # PnL da estratégia
                # Se entry_long no candle T, pegamos retorno do candle T
                # entries_long já tem shift, então alinha com returns
                
                strategy_returns = pd.Series(0.0, index=data.index)
                strategy_returns[entries_long] = returns[entries_long]
                strategy_returns[entries_short] = -returns[entries_short]
                
                # Métricas
                total_return = strategy_returns.sum()
                wins = (strategy_returns > 0).sum()
                trades = (strategy_returns != 0).sum()
                
                if trades < 5:
                    continue
                    
                win_rate = wins / trades
                score = total_return * win_rate # Score simples
                
                if score > best_score:
                    best_score = score
                    best_params = curr_params
            
            if not best_params:
                # Retorna padrões se falhar
                return {
                    "adx_threshold": 20,
                    "rsi_low": 30,
                    "rsi_high": 70,
                    "ema_short": 9,
                    "ema_long": 21,
                    "tp_ratio": 2.0,
                    "sl_atr_mult": 2.0
                }
                
            return best_params
            
        except Exception as e:
            logger.error(f"Erro na otimização de {symbol}: {e}")
            return None
