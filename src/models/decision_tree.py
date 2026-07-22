from sklearn.tree import DecisionTreeRegressor

def get_model(**kwargs) -> DecisionTreeRegressor:
    """Returns a DecisionTreeRegressor model instance."""
    return DecisionTreeRegressor(**kwargs)
