import lightgbm as lgb

def get_model(**kwargs) -> lgb.LGBMRegressor:
    """Returns an LGBMRegressor model instance."""
    return lgb.LGBMRegressor(**kwargs)
