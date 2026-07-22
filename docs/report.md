# BoomBikes daily rental demand prediction — Project Report

**Author**: Ayush Pradhan  
**Date**: 2026-07  

---

## 1. Executive Summary
This project delivers a robust, production-ready machine learning pipeline to forecast daily bike rental demand for BoomBikes. By predicting total daily rentals (`cnt`) from calendar, meteorological, and engineered features, the model assists BoomBikes in:
- **Supply planning**: Fleet right-sizing and maintenance scheduling.
- **Dynamic pricing**: Designing promotions during predicted low-demand periods.
- **Recovery strategy**: Analyzing post-pandemic demand shifts.

Our final tuned XGBoost regressor model achieves an **$R^2$ score of 0.7094** and an **RMSE of 1012.10** on the chronologically partitioned test set, showing strong forecasting capabilities.

---

## 2. Exploratory Data Analysis & Feature Engineering
Through data analysis, we identified key relationships driving demand:
- **Temperature**: Shows a strong positive correlation with rental counts up to ~25°C, after which it plateaus.
- **Year-over-Year Growth**: 2019 showed a massive demand increase compared to 2018, indicating strong organic growth.
- **Weather Severity**: Days with light rain/snow saw rental drops of over 50% compared to clear days.

### Engineered Features
To capture these non-linear physical interactions, we engineered several variables:
- `comfort_index`: Interaction between temperature and humidity.
- `temp_yr`: Captures how temperature effects scale year-over-year.
- `day_of_year`: Extracted temporal progression to model smooth seasonal waves.
- Flags: `is_weekend`, `is_warm_season`, and `is_bad_weather`.

---

## 3. Modeling and Evaluation

We established baseline performances across 8 algorithms:
- **Linear Models**: Linear Regression, Ridge, and Lasso.
- **Tree-Based Models**: Decision Tree, Random Forest.
- **Gradient Boosting Models**: XGBoost, LightGBM, CatBoost.

### Model Comparison Table
On the 30% chronological test set, the models scored as follows:

| Model | Test $R^2$ | RMSE | MAE | MAPE |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost (champion)** | **0.6982** | **1031.45** | **824.24** | **158.61%** |
| CatBoost | 0.6845 | 1054.56 | 875.56 | 110.79% |
| LightGBM | 0.6599 | 1095.03 | 859.87 | 128.81% |
| Random Forest | 0.6506 | 1109.88 | 894.31 | 166.97% |
| Ridge | 0.5967 | 1192.33 | 911.45 | 129.77% |
| Lasso | 0.5856 | 1208.71 | 920.65 | 130.37% |
| Linear Regression | 0.5855 | 1208.88 | 920.71 | 130.38% |
| Decision Tree | 0.5465 | 1264.36 | 1018.18 | 167.11% |

### Hyperparameter Tuning
Optimizing the XGBoost champion model via 5-fold cross-validation yielded:
- **Tuned Test $R^2$**: **0.7094**
- **Tuned Test RMSE**: **1012.10**
- **Selected Hyperparameters**: `n_estimators=500`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.6`, `colsample_bytree=0.6`, `gamma=0.5`.

---

## 4. Diagnostics and Normality Checks
- **Homoscedasticity**: Standard homoscedasticity is confirmed via random residual dispersion plots.
- **Normality**: The Shapiro-Wilk test yielded a statistic of `0.9367` ($p < 0.001$). While the residuals are not perfectly normal due to weather extremes, the error distribution is centered around zero with no critical bias.

---

## 5. Execution Guide
The pipeline is managed via the root `main.py` entrypoint.

### Train the model:
```bash
uv run python main.py train
```

### Run batch predictions:
```bash
uv run python main.py predict --input data/raw/BoomBikes_dataset.csv --output outputs/batch_predictions.csv
```
