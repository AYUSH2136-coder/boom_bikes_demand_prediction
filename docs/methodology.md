# BoomBikes Demand Prediction — Project Methodology

This document details the scientific and engineering methodology applied to preprocess, model, evaluate, and interpret the BoomBikes daily bike sharing rental demand prediction task.

---

## 1. Data Partitioning Strategy

We applied a **chronological (time-based) split** instead of a random train-test split.
- **Rationale**: Daily demand forecasting is inherently temporal. Random splitting would result in temporal data leakage, where the model utilizes future dates to predict past demand, inflating test metrics artificially.
- **Split Ratio**: 70% of the chronologically ordered rows (~1.4 years) are assigned to training, and the final 30% (~0.6 years) are reserved for testing.
- **Split Boundary**: The training set covers the first 511 days (2018-01-01 to 2019-05-26), and the test set covers the final 219 days (2019-05-27 to 2019-12-31).

---

## 2. Preprocessing & Leakage Mitigation

### Leakage Column Dropping
- The raw columns `casual` (count of casual users) and `registered` (count of registered users) sum up exactly to the target column `cnt`. Utilizing them during training would create an identity mapping, rendering the model useless for forecasting future demand. Both columns are strictly dropped.
- The `instant` column (sequential row ID) is dropped to prevent the model from learning from index-driven patterns.

### Numerical Scaling
- Continuous features (`temp`, `atemp`, `hum`, `windspeed`) are normalized using a `StandardScaler` fitted exclusively on the training partition:
  $$z = \frac{x - \mu}{\sigma}$$
- The fitted scaler is saved to `models/scaler.pkl` and applied to test and inference datasets to prevent data leakage from test statistics.

---

## 3. Feature Engineering

To capture physical and environmental context, we derived several interaction and flags features:
1. **`comfort_index`**: Captures the combined effect of temperature and humidity on outdoor physical comfort.
   $$\text{comfort\_index} = \text{temp} - 0.5 \times \left(\frac{\text{hum}}{100}\right)$$
2. **`is_weekend`**: Indicator flag mapping whether `weekday` is Sunday (0) or Saturday (6).
3. **`quarter`**: Calendar quarter mapped from `mnth` (1=Q1, 2=Q2, 3=Q3, 4=Q4).
4. **`is_warm_season`**: Indicator flag if the season is Summer (2) or Fall (3).
5. **`is_bad_weather`**: Indicator flag if the weather situation (`weathersit`) is Light Rain/Snow (3) or Heavy Rain/Snow (4).
6. **`temp_yr`**: Multiplicative interaction of temperature and year (`temp` * `yr`). This models whether demand grows faster/slower relative to ambient heat as the brand gains popularity in Year 2.
7. **`day_of_year`**: Extracted day of the calendar year (1-365) from `dteday` to model continuous seasonal progression.

---

## 4. Modeling and Hyperparameter Search

We registered 8 regression algorithms across multiple model classes:
- **Linear Models**: Ordinary Least Squares (OLS), Ridge ($L_2$ regularized), and Lasso ($L_1$ regularized).
- **Tree-Based Ensembles**: Decision Trees, Random Forests, XGBoost, LightGBM, and CatBoost.

### Tuning Framework
We selected the top-performing baseline model (XGBoost) and applied `RandomizedSearchCV` over 5-fold cross-validation. The optimization objective maximized the $R^2$ coefficient:
$$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
The search evaluated 50 randomized iterations of hyperparameter configurations, tuning learning rate, estimators, tree depth, and column/row subsampling ratios.
