import numpy as np
import pandas as pd
from typing import Dict, Union
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def compute_metrics(
    y_true: Union[np.ndarray, pd.Series], 
    y_pred: Union[np.ndarray, pd.Series]
) -> Dict[str, float]:
    """Computes R2, RMSE, MAE, and MAPE between true and predicted values."""
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    
    mae = mean_absolute_error(y_true_arr, y_pred_arr)
    rmse = np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))
    r2 = r2_score(y_true_arr, y_pred_arr)
    
    # Handle division by zero for MAPE
    y_true_safe = np.where(y_true_arr == 0, 1e-5, y_true_arr)
    mape = np.mean(np.abs((y_true_arr - y_pred_arr) / y_true_safe)) * 100
    
    return {
        "R2": float(r2),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "MAPE": float(mape)
    }
