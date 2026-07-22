import pytest
import os
import pandas as pd
from src.inference.predict import predict

@pytest.fixture
def mock_inference_input():
    data = {
        'dteday': ['01-01-2018', '01-06-2018'],
        'season': [1, 2],
        'yr': [0, 1],
        'mnth': [1, 6],
        'holiday': [0, 0],
        'weekday': [6, 3],
        'workingday': [0, 1],
        'weathersit': [1, 2],
        'temp': [14.1, 28.5],
        'atemp': [18.0, 32.2],
        'hum': [80.5, 60.1],
        'windspeed': [10.7, 15.3]
    }
    return pd.DataFrame(data)

def test_inference_pipeline(mock_inference_input):
    model_path = "models/tuned_model.pkl"
    scaler_path = "models/scaler.pkl"
    
    # Run prediction if model files exist
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        preds = predict(
            input_df=mock_inference_input,
            model_path=model_path,
            scaler_path=scaler_path
        )
        assert len(preds) == 2
    else:
        pytest.skip("Model or scaler files not found, skipping inference test.")
