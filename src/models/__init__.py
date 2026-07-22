from .catboost_model import get_model as get_catboost
from .decision_tree import get_model as get_decision_tree
from .lasso_regression import get_model as get_lasso
from .lightgbm_model import get_model as get_lightgbm
from .linear_regression import get_model as get_linear_regression
from .random_forest import get_model as get_random_forest
from .ridge_regression import get_model as get_ridge
from .xgboost_model import get_model as get_xgboost

MODEL_REGISTRY = {
    "catboost": get_catboost,
    "decision_tree": get_decision_tree,
    "lasso": get_lasso,
    "lightgbm": get_lightgbm,
    "linear_regression": get_linear_regression,
    "random_forest": get_random_forest,
    "ridge": get_ridge,
    "xgboost": get_xgboost,
}

def get_model_by_name(name: str, **kwargs):
    """Retrieves a model instance by name from the registry."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)
