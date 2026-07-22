import pandas as pd
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from src.utils.logger import get_logger

logger = get_logger(__name__)

def preprocess_features(
    df: pd.DataFrame, 
    drop_columns: List[str], 
    numeric_columns: List[str], 
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = False
) -> Tuple[pd.DataFrame, Optional[StandardScaler]]:
    """Drops metadata/leakage columns and scales numeric columns. Returns the processed DataFrame and the scaler."""
    logger.info("Starting data preprocessing...")
    df = df.copy()
    
    # Parse date helper to keep track of day of year if dteday is present
    if "dteday" in df.columns:
        dteday_parsed = pd.to_datetime(df["dteday"], dayfirst=True)
        df["day_of_year"] = dteday_parsed.dt.dayofyear
        
    # Drop unwanted columns
    cols_to_drop = [c for c in drop_columns if c in df.columns]
    if "dteday" in df.columns:
        cols_to_drop.append("dteday")
        
    if cols_to_drop:
        logger.info(f"Dropping columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
        
    # Scale numerical columns
    if fit_scaler:
        logger.info(f"Fitting new StandardScaler on numeric columns: {numeric_columns}")
        scaler = StandardScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    elif scaler is not None:
        logger.info(f"Transforming numeric columns using provided scaler: {numeric_columns}")
        # Only scale the columns that exist in the DataFrame
        existing_numeric = [col for col in numeric_columns if col in df.columns]
        df[existing_numeric] = scaler.transform(df[existing_numeric])
    else:
        logger.info("No scaler provided or requested to fit. Numerical features left unscaled.")
        
    return df, scaler
