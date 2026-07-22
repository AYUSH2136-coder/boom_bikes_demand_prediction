import pytest
import numpy as np
from src.models import get_model_by_name

def test_model_registry_retrieval():
    models = ["linear_regression", "ridge", "lasso", "decision_tree", "random_forest", "xgboost", "lightgbm", "catboost"]
    for name in models:
        model = get_model_by_name(name)
        assert model is not None
        
def test_model_fit_predict():
    X = np.random.rand(20, 5)
    y = np.random.rand(20)
    
    model = get_model_by_name("linear_regression")
    model.fit(X, y)
    preds = model.predict(X)
    
    assert preds.shape == (20,)

def test_compute_metrics():
    from src.evaluation.metrics import compute_metrics
    y_true = np.array([100.0, 150.0, 200.0])
    y_pred = np.array([110.0, 140.0, 210.0])
    metrics = compute_metrics(y_true, y_pred)
    assert "R2" in metrics
    assert "RMSE" in metrics
    assert "MAE" in metrics
    assert "MAPE" in metrics

def test_check_normality():
    from src.evaluation.residual_analysis import check_normality
    residuals = np.random.normal(0, 1, 100)
    check_normality(residuals)
