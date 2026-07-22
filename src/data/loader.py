import pandas as pd
from pathlib import Path
from typing import Union
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_raw_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """Loads raw data from path, validates required columns, and returns a DataFrame."""
    logger.info(f"Loading raw data from {file_path}")
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Raw data file not found at: {file_path}")
    
    df = pd.read_csv(file_path)
    
    required_cols = ["season", "yr", "mnth", "holiday", "weekday", "workingday", 
                     "weathersit", "temp", "atemp", "hum", "windspeed", "cnt"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")
        
    logger.info(f"Raw data loaded successfully. Shape: {df.shape}")
    return df
