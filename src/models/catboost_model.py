from catboost import CatBoostRegressor

def get_model(**kwargs) -> CatBoostRegressor:
    """Returns a CatBoostRegressor model instance."""
    return CatBoostRegressor(**kwargs)
