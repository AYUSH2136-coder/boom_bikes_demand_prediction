import pytest
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from src.training.hyperparameter_tuning import tune_hyperparameters

def test_hyperparameter_tuning():
    X = np.random.rand(30, 4)
    y = np.random.rand(30)
    
    model = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [5, 10],
        'max_depth': [3, 5]
    }
    
    search = tune_hyperparameters(
        model=model,
        param_grid=param_grid,
        X_train=X,
        y_train=y,
        n_iter=2,
        cv=2,
        random_state=42
    )
    
    assert search is not None
    assert search.best_estimator_ is not None
    assert 'n_estimators' in search.best_params_

def test_set_seed():
    from src.utils.seed import set_seed
    import random
    import numpy as np
    
    set_seed(42)
    val1 = random.random()
    np_val1 = np.random.rand()
    
    set_seed(42)
    val2 = random.random()
    np_val2 = np.random.rand()
    
    assert val1 == val2
    assert np_val1 == np_val2
