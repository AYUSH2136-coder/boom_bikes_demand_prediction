from sklearn.linear_model import Ridge

def get_model(**kwargs) -> Ridge:
    """Returns a Ridge Regression model instance."""
    return Ridge(**kwargs)
