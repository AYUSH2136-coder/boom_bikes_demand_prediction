import pytest
import os
import numpy as np
import pandas as pd
from src.evaluation.plots import plot_actual_vs_predicted, plot_residuals, plot_error_by_segment
from src.evaluation.feature_importance import plot_builtin_importance

def test_plots(tmp_path):
    y_true = np.array([100.0, 150.0, 200.0])
    y_pred = np.array([110.0, 140.0, 210.0])
    dates = pd.to_datetime(['2018-01-01', '2018-01-02', '2018-01-03'])
    
    # 1. actual vs predicted plot
    path1 = tmp_path / "plot1.png"
    plot_actual_vs_predicted(y_true, y_pred, dates, save_path=str(path1))
    assert path1.exists()
    
    # 2. residuals plot
    path2 = tmp_path / "plot2.png"
    plot_residuals(y_true, y_pred, save_path=str(path2))
    assert path2.exists()
    
    # 3. error by segment plot
    df_raw_test = pd.DataFrame({
        'season': [1, 2, 3],
        'weathersit': [1, 2, 1],
        'temp': [10.0, 20.0, 30.0],
        'hum': [50.0, 60.0, 70.0],
        'windspeed': [10.0, 12.0, 15.0],
        'yr': [0, 1, 0],
        'mnth': [1, 6, 12],
        'workingday': [0, 1, 0]
    })
    residuals = y_true - y_pred
    path3 = tmp_path / "plot3.png"
    plot_error_by_segment(df_raw_test, residuals, y_true, y_pred, save_path=str(path3))
    assert path3.exists()

def test_feature_importance_plot(tmp_path):
    from sklearn.ensemble import RandomForestRegressor
    X = np.random.rand(20, 3)
    y = np.random.rand(20)
    model = RandomForestRegressor(n_estimators=5, random_state=42)
    model.fit(X, y)
    
    path = tmp_path / "importance.png"
    plot_builtin_importance(model, ["col1", "col2", "col3"], save_path=str(path))
    assert path.exists()
