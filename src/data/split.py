import pandas as pd
from typing import Tuple
from src.utils.logger import get_logger

logger = get_logger(__name__)

def split_data(
    df: pd.DataFrame, 
    target_column: str, 
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Performs time-based split of features and target. Keeps chronological order."""
    logger.info(f"Splitting data with test_size={test_size} (time-based)...")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    split_idx = int(len(df) * (1 - test_size))
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    logger.info(f"Split complete. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test
