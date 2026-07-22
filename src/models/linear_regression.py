from sklearn.linear_model import LinearRegression

def get_model(**kwargs) -> LinearRegression:
    """Returns a LinearRegression model instance."""
    return LinearRegression(**kwargs)
