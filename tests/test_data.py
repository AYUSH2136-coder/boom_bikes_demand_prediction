import pytest
import os
import pandas as pd
import numpy as np
from src.data.feature_engineering import add_engineered_features
from src.data.split import split_data
from src.data.preprocessing import preprocess_features

@pytest.fixture
def sample_raw_data():
    # Construct a small mock DataFrame mimicking the raw dataset
    data = {
        'instant': range(1, 11),
        'dteday': pd.date_range(start='2018-01-01', periods=10).strftime('%d-%m-%Y'),
        'season': [1, 1, 1, 2, 2, 2, 3, 3, 4, 4],
        'yr': [0]*5 + [1]*5,
        'mnth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'holiday': [0]*10,
        'weekday': [0, 1, 2, 3, 4, 5, 6, 0, 1, 2],
        'workingday': [0, 1, 1, 1, 1, 1, 0, 0, 1, 1],
        'weathersit': [1, 2, 1, 3, 1, 2, 1, 1, 2, 1],
        'temp': [10.0, 12.0, 14.0, 20.0, 22.0, 25.0, 28.0, 27.0, 21.0, 15.0],
        'atemp': [11.0, 13.0, 15.0, 21.0, 23.0, 26.0, 29.0, 28.0, 22.0, 16.0],
        'hum': [50.0, 55.0, 60.0, 70.0, 65.0, 60.0, 55.0, 50.0, 58.0, 62.0],
        'windspeed': [10.0, 12.0, 8.0, 15.0, 14.0, 10.0, 11.0, 9.0, 13.0, 11.0],
        'casual': [10]*10,
        'registered': [90]*10,
        'cnt': [100]*10
    }
    return pd.DataFrame(data)

def test_feature_engineering(sample_raw_data):
    df_fe = add_engineered_features(sample_raw_data)
    expected_cols = ['comfort_index', 'is_weekend', 'quarter', 'is_warm_season', 'is_bad_weather', 'temp_yr']
    for col in expected_cols:
        assert col in df_fe.columns
    # Check values
    assert df_fe['is_weekend'].iloc[0] == 1  # weekday 0 is Sunday
    assert df_fe['is_weekend'].iloc[1] == 0  # weekday 1 is Monday

def test_data_split(sample_raw_data):
    X_train, X_test, y_train, y_test = split_data(sample_raw_data, target_column='cnt', test_size=0.3)
    
    assert len(X_train) == 7
    assert len(X_test) == 3
    assert len(y_train) == 7
    assert len(y_test) == 3
    
    # Ensure chronological index ordering (no leakage)
    assert X_train.index.max() < X_test.index.min()

def test_preprocessing(sample_raw_data):
    df_fe = add_engineered_features(sample_raw_data)
    drop_cols = ['instant', 'dteday', 'casual', 'registered']
    num_cols = ['temp', 'atemp', 'hum', 'windspeed']
    
    df_preprocessed, scaler = preprocess_features(
        df_fe,
        drop_columns=drop_cols,
        numeric_columns=num_cols,
        fit_scaler=True
    )
    
    for col in drop_cols:
        assert col not in df_preprocessed.columns
        
    assert scaler is not None
    # Verify that scaling was performed (mean should be close to 0)
    assert abs(df_preprocessed['temp'].mean()) < 1e-7

def test_load_raw_data():
    from src.data.loader import load_raw_data
    path = "data/raw/BoomBikes_dataset.csv"
    if os.path.exists(path):
        df = load_raw_data(path)
        assert len(df) > 0
        assert "cnt" in df.columns
