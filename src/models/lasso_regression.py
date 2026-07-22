from sklearn.linear_model import Lasso

def get_model(**kwargs) -> Lasso:
    """Returns a Lasso Regression model instance."""
    return Lasso(**kwargs)
