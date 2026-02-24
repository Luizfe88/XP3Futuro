import pytest

optuna = pytest.importorskip("optuna")
sk = pytest.importorskip("sklearn")

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from ml_optimizer import build_default_pipeline
from ml_optuna import OptunaOptimizer

def test_optuna_optimizer_smoke():
    X, y = make_classification(n_samples=600, n_features=12, n_informative=5, n_redundant=3, random_state=33)
    pipe = build_default_pipeline(LogisticRegression(max_iter=800, random_state=33))
    space = {"model__C": {"min": 0.01, "max": 50.0, "log": True}}
    opt = OptunaOptimizer(random_state=33, n_trials=10)
    res = opt.run(pipe, X, y, space)
    assert 0.0 <= float(res.best_score) <= 1.0
    assert isinstance(res.best_params, dict)
