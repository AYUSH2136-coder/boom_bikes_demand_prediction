# Business Understanding

**Project:** BoomBikes Demand Prediction  
**Author:** Ayush Pradhan  
**Date:** 2026-07  

---

## Objective

BoomBikes is a US-based bike-sharing provider that has suffered considerable revenue loss during the Covid-19 pandemic.  
The company wants to understand which factors drive daily bike demand so it can:

1. **Plan supply** — right-size the fleet and staffing on a given day.
2. **Price dynamically** — offer promotional rates when demand is predicted to be low.
3. **Accelerate post-pandemic recovery** — distinguish pre-pandemic patterns from emerging new habits.

We will build a regression model that predicts the total daily bike count (`cnt`) from weather, calendar, and seasonal features.

---

## Problem Framing

| Dimension | Detail |
|---|---|
| **Task type** | Supervised regression |
| **Target variable** | `cnt` — total daily rentals (casual + registered) |
| **Leakage columns** | `casual`, `registered` — must be dropped because they sum to `cnt` |
| **Granularity** | One row = one calendar day |
| **Dataset size** | 730 rows (2 years of daily data) |
| **Primary metric** | R² and RMSE on held-out test set |

---

## Feature Glossary

| Column | Type | Description |
|---|---|---|
| `instant` | ID | Row index — drop before modelling |
| `dteday` | Date | Calendar date (DD-MM-YYYY) — may derive features |
| `season` | Categorical | 1=Spring 2=Summer 3=Fall 4=Winter |
| `yr` | Binary | 0=2018, 1=2019 |
| `mnth` | Categorical | 1–12 |
| `holiday` | Binary | 1 if public holiday |
| `weekday` | Categorical | 0=Sunday … 6=Saturday |
| `workingday` | Binary | 1 if neither holiday nor weekend |
| `weathersit` | Categorical | 1=Clear, 2=Mist, 3=Light snow/rain, 4=Heavy rain |
| `temp` | Numeric | Normalised temperature (°C) |
| `atemp` | Numeric | Normalised 'feels like' temperature (°C) |
| `hum` | Numeric | Normalised humidity (%) |
| `windspeed` | Numeric | Normalised wind speed |
| `casual` | ⚠️ Leakage | Casual user count — **do not use** |
| `registered` | ⚠️ Leakage | Registered user count — **do not use** |
| `cnt` | **TARGET** | Total daily rentals |

---

## Success Criteria

| Metric | Minimum acceptable | Stretch goal |
|---|---|---|
| R² (test) | ≥ 0.80 | ≥ 0.90 |
| RMSE (test) | ≤ 900 | ≤ 600 |
| MAE (test) | ≤ 700 | ≤ 500 |

---

## Proposed Notebook Workflow

```text
01_Business_Understanding   ← this notebook
02_Data_Understanding       ← schema inspection, basic stats
03_EDA                      ← distributions, correlations, time trends
04_Data_Preprocessing       ← encoding, scaling, cleaning
05_Feature_Engineering      ← derived features, interaction terms
06_Model_Building           ← baseline + multi-model comparison
07_Hyperparameter_Tuning    ← tune champion model
08_Model_Evaluation         ← final metrics, residual analysis
09_Model_Interpretation     ← SHAP values, feature importance
10_Inference                ← load artifacts, predict on new data
```

---

## Key Business Hypotheses

1. **Temperature** — Warmer days attract more riders; demand peaks in summer/fall.
2. **Weather severity** — Rain and snow significantly suppress demand.
3. **Day type** — Working days may have a consistent commuter base; weekends may have more casual leisure riders.
4. **Year trend** — 2019 likely shows higher demand than 2018 as brand awareness grows.
5. **Holiday effect** — Public holidays may reduce demand if commuters dominate the user base.

These hypotheses will be tested during EDA (Notebook 03).
