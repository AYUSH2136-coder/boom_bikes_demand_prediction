import pandas as pd
from typing import Dict, Any
from sklearn.model_selection import RandomizedSearchCV
from src.utils.logger import get_logger

logger = get_logger(__name__)

def tune_hyperparameters(
    model: Any,
    param_grid: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 50,
    cv: int = 5,
    random_state: int = 42,
    scoring: str = "neg_root_mean_squared_error"
) -> RandomizedSearchCV:
    """Performs RandomizedSearchCV to tune hyperparameters of a given model."""
    logger.info(f"Initializing hyperparameter tuning (n_iter={n_iter}, cv={cv})...")
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        random_state=random_state,
        verbose=1,
        return_train_score=True
    )
    search.fit(X_train, y_train)
    logger.info(f"Tuning complete. Best CV score: {search.best_score_:.4f}")
    return search
