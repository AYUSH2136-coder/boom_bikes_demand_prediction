import xgboost as xgb

def get_model(**kwargs) -> xgb.XGBRegressor:
    """Returns an XGBRegressor model instance."""
    return xgb.XGBRegressor(**kwargs)
