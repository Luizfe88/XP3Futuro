import pytest
import numpy as np

sk = pytest.importorskip("sklearn")

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, make_scorer

from ml_optimizer import (
    build_default_pipeline,
    compare_methods,
    GridSearchOptimizer,
    RandomSearchOptimizer,
    GradientDescentOptimizer,
    GeneticAlgorithmOptimizer,
    SimulatedAnnealingOptimizer,
    WeeklyOptimizerRunner,
)


def _dataset(seed=42):
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=8, n_redundant=4,
        n_clusters_per_class=2, class_sep=1.2, random_state=seed
    )
    return X, y


def test_wrappers_grid_and_random_search():
    X, y = _dataset(7)
    pipe = build_default_pipeline(LogisticRegression(max_iter=1000, random_state=7))
    grid = {"model__C": [0.1, 1.0, 10.0]}
    dist = {"model__C": np.logspace(-2, 2, 20)}

    gs = GridSearchOptimizer(random_state=7, scoring=make_scorer(accuracy_score))
    rs = RandomSearchOptimizer(random_state=7, scoring=make_scorer(accuracy_score))

    r1 = gs.run(pipe, X, y, grid)
    r2 = rs.run(pipe, X, y, dist, n_iter=16)

    assert 0.0 <= r1.best_score <= 1.0
    assert 0.0 <= r2.best_score <= 1.0
    assert isinstance(r1.best_params, dict)
    assert isinstance(r2.best_params, dict)


def test_population_optimizers_reproducibility():
    X, y = _dataset(123)
    pipe = build_default_pipeline(LogisticRegression(max_iter=1000, random_state=123))
    space_num = {"model__C": {"min": 0.01, "max": 100.0, "log": True}}

    ga1 = GeneticAlgorithmOptimizer(random_state=123)
    ga2 = GeneticAlgorithmOptimizer(random_state=123)
    r1 = ga1.run(pipe, X, y, space_num, population=12, generations=8)
    r2 = ga2.run(pipe, X, y, space_num, population=12, generations=8)

    assert abs(r1.best_score - r2.best_score) < 1e-6


def test_compare_methods_smoke():
    X, y = _dataset(42)
    pipe = build_default_pipeline(LogisticRegression(max_iter=1000, random_state=42))
    grid = {"model__C": [0.1, 1.0, 10.0]}
    dist = {"model__C": np.logspace(-2, 2, 20)}
    space_num = {"model__C": {"min": 0.01, "max": 100.0, "log": True}}

    results = compare_methods(pipe, X, y, grid, dist, space_num, random_state=42)
    assert "GridSearchCV" in results
    assert "RandomizedSearchCV" in results
    assert "GeneticAlgorithm" in results
    assert "SimulatedAnnealing" in results
    assert "GradientDescent" in results

    for k, res in results.items():
        assert 0.0 <= float(res.best_score) <= 1.0


def test_weekly_runner_smoke(tmp_path):
    X, y = _dataset(21)
    def loader():
        return X, y
    est = LogisticRegression(max_iter=500, random_state=21)
    grid = {"model__C": [0.1, 1.0]}
    dist = {"model__C": np.logspace(-2, 2, 10)}
    num = {"model__C": {"min": 0.01, "max": 50.0, "log": True}}
    runner = WeeklyOptimizerRunner(random_state=21, output_dir=str(tmp_path))
    out = runner.run_weekly(est, loader, grid, dist, num, use_scaler=True)
    assert out["summary"]["count_methods"] >= 3
    assert isinstance(out["summary"]["output_path"], str)
