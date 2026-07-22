from sklearn.ensemble import RandomForestRegressor

def get_model(**kwargs) -> RandomForestRegressor:
    """Returns a RandomForestRegressor model instance."""
    return RandomForestRegressor(**kwargs)
