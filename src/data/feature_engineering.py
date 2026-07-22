import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates interaction and calendar features: comfort_index, is_weekend, quarter, is_warm_season, is_bad_weather, and temp_yr."""
    logger.info("Performing feature engineering...")
    df = df.copy()
    
    # 1. Comfort Index
    df['comfort_index'] = df['temp'] - 0.5 * df['hum'] / 100
    
    # 2. Weekend flag
    df['is_weekend'] = df['weekday'].isin([0, 6]).astype(int)
    
    # 3. Quarter
    df['quarter'] = pd.cut(df['mnth'], bins=[0, 3, 6, 9, 12], labels=[1, 2, 3, 4]).astype(int)
    
    # 4. Warm season flag
    df['is_warm_season'] = df['season'].isin([2, 3]).astype(int)
    
    # 5. Bad weather flag
    df['is_bad_weather'] = (df['weathersit'] >= 3).astype(int)
    
    # 6. Temperature x Year interaction
    df['temp_yr'] = df['temp'] * df['yr']
    
    logger.info(f"Feature engineering complete. Current shape: {df.shape}")
    return df
