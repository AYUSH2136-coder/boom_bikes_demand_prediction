# BoomBikes Demand Prediction Pipeline — Deployment Guide

This guide outlines the procedure for deploying, running, and testing the modularised BoomBikes demand prediction pipeline in production environments.

---

## 1. Prerequisites and Installation

The codebase supports **Python >= 3.12** and utilizes **`uv`** as its package manager to guarantee rapid, deterministic environments.

### Step 1: Install `uv`
If not already installed, install `uv` on your server/local machine:
- **Windows (PowerShell)**:
  ```powershell
  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
- **Linux / macOS**:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### Step 2: Synchronize Environment
Navigate to the project root and run `uv sync` to set up the virtualenv and install all dependencies:
```bash
# Install core and optional dependencies (including dev and notebooks)
uv sync --all-extras
```

---

## 2. Configuration Management

The pipeline is configuration-driven to ensure code remains decoupling from hyperparameter changes or path mutations.

All configurations are located in the `configs/` directory:
- **`paths.yaml`**: Governs paths to raw datasets, interim storage, final output predictions, model folders, and reports.
- **`config.yaml`**: Governs train/test chronological split ratio, features to use or drop, target column name (`cnt`), random seed, and whether numerical scaling is enabled.
- **`model.yaml`**: Governs initial baseline parameters for model wrappers and the randomized cross-validation space (`n_iter`, `cv`, `scoring`) used for champion optimization.

---

## 3. CLI Orchestration

All pipelines are run using the root orchestration script `main.py`.

### A. Model Training, Comparison & Tuning
To run the end-to-end training pipeline:
1. Loads raw data from the configured path.
2. Applies feature engineering and preprocessing (normalizes columns if enabled).
3. Evaluates all 8 baseline regressors.
4. Identifies the model with the highest $R^2$ score (the "champion").
5. Performs randomized search CV hyperparameter optimization on the champion.
6. Saves the optimized model to `models/champion_model.pkl` and generating metric charts under `outputs/`.

```bash
uv run python main.py train
```

*To skip the baseline model comparison and proceed directly to training/tuning the default champion:*
```bash
uv run python main.py train --no-comparison
```

### B. Batch Inference
To run batch predictions on a new raw CSV file:
1. Validates the raw input schema.
2. Automatically derives engineered features (e.g. `comfort_index`, `is_weekend`, `is_warm_season`).
3. Loads the serialized `scaler.pkl` to normalize numerical columns.
4. Loads `champion_model.pkl` and returns prediction counts.
5. Saves the final predictions as `predicted_cnt` alongside the original input features.

```bash
uv run python main.py predict --input data/raw/BoomBikes_dataset.csv --output outputs/batch_predictions.csv
```

---

## 4. Model Artifacts

After running the training pipeline, the following production artifacts are generated in the `models/` directory:
- **`scaler.pkl`**: The fitted `StandardScaler` used to transform numeric columns (`temp`, `atemp`, `hum`, `windspeed`).
- **`champion_model.pkl` / `tuned_model.pkl`**: The serialized champion estimator (optimized XGBoost).
- **`best_params.json`**: JSON log of the hyperparameter search results.

---

## 5. Verification and Quality Assurance

Run the test suite using `pytest` to verify the stability of the feature engineering transformation pipelines, split boundaries, and model outputs:

```bash
uv run pytest
```
*Note: Test coverage must meet a minimum of **60%** as defined in `pyproject.toml`.*

---

## 6. Production Deployment Patterns

### Pattern A: Scheduled Batch Prediction (Cron / Airflow)
If bike-sharing demand predictions are run as a batch once a day (e.g., at midnight for the coming week's fleet planning), trigger the prediction command via a Cron job:
```cron
0 0 * * * cd /path/to/boom-bikes-demand-prediction && uv run python main.py predict --input new_incoming_day_data.csv --output predictions/today.csv
```

### Pattern B: REST API Deployment (FastAPI)
For dynamic pricing or live fleet optimization apps, the inference logic can be wrapped inside a lightweight FastAPI app. Example code:

```python
import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.inference.predict import predict

app = FastAPI(title="BoomBikes Demand Prediction Service")

class DayFeatures(BaseModel):
    dteday: str
    season: int
    yr: int
    mnth: int
    holiday: int
    weekday: int
    workingday: int
    weathersit: int
    temp: float
    atemp: float
    hum: float
    windspeed: float

@app.post("/predict")
def get_prediction(payload: DayFeatures):
    try:
        # Convert incoming JSON payload to DataFrame
        df_input = pd.DataFrame([payload.model_dump()])
        
        # Invoke inference module
        preds = predict(
            input_df=df_input,
            model_path="models/champion_model.pkl",
            scaler_path="models/scaler.pkl"
        )
        return {"predicted_cnt": int(round(preds[0]))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```
