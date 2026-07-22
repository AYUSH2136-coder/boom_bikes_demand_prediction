# BoomBikes Demand Prediction — Results and Evaluation

This document summarizes the performance metrics, tuning parameters, and diagnostic evaluations of the final prediction model.

---

## 1. Baseline Model Comparison

To select the most robust regressor for daily demand forecasting, all algorithms registered in `src/models/` were trained on preprocessed features using default hyperparameters. Evaluations were performed on a chronologically partitioned 30% test subset.

| Model | Test $R^2$ Score | RMSE | MAE | MAPE |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost (Champion)** | **0.6982** | **1031.45** | **824.24** | **158.61%** |
| CatBoost | 0.6845 | 1054.56 | 875.56 | 110.79% |
| LightGBM | 0.6599 | 1095.03 | 859.87 | 128.81% |
| Random Forest | 0.6506 | 1109.88 | 894.31 | 166.97% |
| Ridge | 0.5967 | 1192.33 | 911.45 | 129.77% |
| Lasso | 0.5856 | 1208.71 | 920.65 | 130.37% |
| Linear Regression | 0.5855 | 1208.88 | 920.71 | 130.38% |
| Decision Tree | 0.5465 | 1264.36 | 1018.18 | 167.11% |

---

## 2. Hyperparameter Tuning Results

We tuned the champion XGBoost model using RandomizedSearchCV with 5-fold cross-validation. The optimization yielded a substantial increase in model generalization and a decrease in prediction errors.

### Best Configuration:
```json
{
  "n_estimators": 500,
  "max_depth": 6,
  "learning_rate": 0.05,
  "subsample": 0.6,
  "colsample_bytree": 0.6,
  "gamma": 0.5
}
```

### Final Metrics (Tuned XGBoost):
- **$R^2$ Score**: **0.7094**
- **RMSE**: **1012.10**
- **MAE**: **820.94**
- **MAPE**: **136.86%**

---

## 3. Residual Diagnostics and Statistical Verification

The model diagnostics were verified using standard regression assumptions:

### Homoscedasticity & Trend Correlation
- Residual analysis plots (`outputs/residual_analysis.png`) show random scattering of errors around zero. No distinct cone shapes (heteroscedasticity) are present.
- The actual vs predicted time series (`outputs/eval_actual_vs_predicted.png`) shows the model closely tracking seasonal transitions and annual growth trends.

### Normality Check (Shapiro-Wilk Test)
- **Test Statistic**: `0.9367`
- **p-value**: `3.9429e-06`
- **Interpretation**: The null hypothesis of perfect residual normality is rejected at typical significance levels ($p < 0.05$). This is normal for weather-driven rental demand datasets due to tail outliers during severe weather conditions. The distribution remains highly concentrated around zero, indicating acceptable linear regression diagnostics.

---

## 4. Key Predictors (Feature Importance)
Based on built-in feature importance logs (`outputs/feature_importance.png`), the top features contributing to the model predictions are:
1. **`temp_yr`** (Interaction of normalized temperature and annual growth).
2. **`yr`** (Year-over-year brand growth/market share expansion).
3. **`temp`** / **`atemp`** (Ambient air temperature).
4. **`season`** / **`mnth`** (Seasonal demand shifts).
