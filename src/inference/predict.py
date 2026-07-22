import os
import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional

from src.utils.logger import get_logger
from src.utils.helpers import load_yaml
from src.data.feature_engineering import add_engineered_features
from src.data.preprocessing import preprocess_features

logger = get_logger(__name__)

def predict(
    input_df: pd.DataFrame,
    model_path: Union[str, Path],
    scaler_path: Optional[Union[str, Path]] = None,
    config_dir: str = "configs"
) -> np.ndarray:
    """Loads model and optional scaler, applies feature engineering and preprocessing, and returns predictions."""
    logger.info("Starting prediction pipeline...")
    
    # Load config
    cfg = load_yaml(os.path.join(config_dir, "config.yaml"))
    
    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = joblib.load(model_path)
    logger.info(f"Loaded model from {model_path}")
    
    # Load scaler if scaling is enabled
    scaler = None
    if cfg["preprocessing"]["scale_numeric"]:
        if scaler_path is None or not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaling enabled in config, but scaler file not found or path not provided: {scaler_path}")
        scaler = joblib.load(scaler_path)
        logger.info(f"Loaded scaler from {scaler_path}")
    
    # 1. Feature Engineering
    df_fe = add_engineered_features(input_df)
    
    # 2. Preprocessing (dropping columns and scaling)
    drop_columns = cfg["features"]["drop_columns"]
    numeric_columns = cfg["features"]["numeric"]
    
    df_preprocessed, _ = preprocess_features(
        df_fe,
        drop_columns=drop_columns,
        numeric_columns=numeric_columns,
        scaler=scaler
    )
    
    # Drop target column if present in input
    target = cfg["project"]["target"]
    if target in df_preprocessed.columns:
        df_preprocessed = df_preprocessed.drop(columns=[target])
        
    # Get predictions
    logger.info("Running model inference...")
    preds = model.predict(df_preprocessed)
    logger.info(f"Inference complete. Generated {len(preds)} predictions.")
    return preds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on BoomBikes raw data.")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save predictions CSV.")
    parser.add_argument("--model", type=str, default="models/tuned_model.pkl", help="Path to model file.")
    parser.add_argument("--scaler", type=str, default="models/scaler.pkl", help="Path to scaler file.")
    parser.add_argument("--config-dir", type=str, default="configs", help="Path to config folder.")
    
    args = parser.parse_args()
    
    # Read input data
    df_input = pd.read_csv(args.input)
    
    # Predict
    preds = predict(
        input_df=df_input,
        model_path=args.model,
        scaler_path=args.scaler,
        config_dir=args.config_dir
    )
    
    # Save predictions
    df_output = df_input.copy()
    cfg = load_yaml(os.path.join(args.config_dir, "config.yaml"))
    target = cfg["project"]["target"]
    df_output[f"predicted_{target}"] = preds
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_output.to_csv(args.output, index=False)
    print(f"Predictions successfully saved to: {args.output}")
