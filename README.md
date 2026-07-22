# BoomBikes Demand Prediction Pipeline

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![uv](https://img.shields.io/badge/packager-uv-pink.svg)](https://github.com/astral-sh/uv)
[![tests](https://img.shields.io/badge/tests-passing-green.svg)](tests/)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A production-grade, modularized machine learning pipeline to forecast daily bike sharing rental demand for **BoomBikes**. The pipeline transitions the experimental Jupyter notebooks into structured Python packages featuring clean preprocessing, configuration-driven hyperparameter tuning, statistical model evaluations, and automated unit tests.

---

## 🚀 Key Features

* **Modular ML Pipeline**: Fully decoupled modules for data loading, preprocessing, chronological splitting, model registry, tuning, evaluation, and batch inference under `src/`.
* **Dynamic Model Registry**: Easily toggle between Linear/Ridge/Lasso models, Decision Trees, and Gradient Boosted trees (XGBoost, LightGBM, CatBoost) via single configuration settings.
* **Config-Driven Architecture**: All parameters, paths, and modeling details are controlled via configuration YAML files in `configs/`.
* **Tuning Optimization**: Out-of-the-box Randomized Search CV over 5-fold cross-validation to select and optimize the champion model.
* **Robust Test Suite**: Integrated unit testing using `pytest` achieving >60% code coverage.
* **Interactive Evaluation**: Generates diagnostic and normality plots (actual vs. predicted, residual homoscedasticity, and segment-wise errors) under `outputs/`.

---

## 📁 Project Structure

```text
boom-bikes-demand-prediction/
├── configs/                  # Pipeline and model configuration YAMLs
│   ├── config.yaml           # Features, targets, and data parameters
│   ├── model.yaml            # Hyperparameters and tuning search spaces
│   └── paths.yaml            # Data and artifact paths
├── data/                     # Raw and engineered datasets (ignored from git)
│   ├── processed/
│   └── raw/
├── docs/                     # Technical documentation & project methodology
│   ├── business_understanding.md
│   ├── data_dictionary.md
│   ├── deployment.md
│   ├── methodology.md
│   └── report.md
├── models/                   # Saved model pkl artifacts (ignored from git)
├── notebooks/                # Experimental Jupyter notebooks (01 to 10)
├── outputs/                  # Diagnostic figures and batch prediction exports
├── src/                      # Source package modules
│   ├── data/                 # Loader, preprocessor, engineering, and split
│   ├── evaluation/           # Metrics, plots, features, and residual tests
│   ├── inference/            # Prediction wrappers and CLI utilities
│   ├── models/               # Model wraps and centralized registry
│   ├── training/             # Hyperparameter tuning and train pipeline
│   └── utils/                # Loggers, configs, and seeds
├── tests/                    # Pytest unit tests
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_plots.py
│   ├── test_prediction.py
│   └── test_training.py
├── main.py                   # Central orchestrator CLI
├── pyproject.toml            # Dependencies, pytest config, and build system
└── README.md                 # Project landing page (this document)
```

---

## ⚙️ Installation and Setup

We recommend using **`uv`** for managing dependencies and virtual environments.

### 1. Install `uv`
```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Synchronize Project Dependencies
```bash
# Clone the repository and navigate inside
git clone https://github.com/your-username/boom-bikes-demand-prediction.git
cd boom-bikes-demand-prediction

# Synchronize venv and all optional extras (dev & notebook tooling)
uv sync --all-extras
```

---

## 🎯 Quick Start Guide

### 1. End-to-End Model Training
To run the training pipeline, compile baselines across all registered models, tune the champion (XGBoost), and save the diagnostic evaluation plots:
```bash
uv run python main.py train
```
*To skip baseline comparison and train the champion directly:*
```bash
uv run python main.py train --no-comparison
```

### 2. Batch Inference
To run predictions on a raw CSV file and export results:
```bash
uv run python main.py predict --input data/raw/BoomBikes_dataset.csv --output outputs/batch_predictions.csv
```

---

## 📊 Performance Benchmark

Chronological splitting (70% train / 30% test) was used to validate models without temporal leakage.

### Baseline Model Comparisons ($R^2$ on Test Set):
* **XGBoost**: **0.6982** *(Champion)*
* **CatBoost**: **0.6845**
* **LightGBM**: **0.6599**
* **Random Forest**: **0.6506**
* **Linear Regression**: **0.5855**
* **Decision Tree**: **0.5465**

### Optimized Tuned Champion (XGBoost):
* **$R^2$ Score**: **0.7094** (improved from 0.6982)
* **RMSE**: **1012.10**
* **MAE**: **820.94**
* **MAPE**: **136.86%**

*Tuned Hyperparameters: `n_estimators=500`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.6`, `colsample_bytree=0.6`, `gamma=0.5`.*

---

## 🧪 Running Unit Tests

Our tests validate data transformations, scaling, model prediction shapes, and tuning steps. Run them via `pytest`:
```bash
uv run pytest
```
*Test coverage details are printed automatically. The suite is configured to fail if total code coverage falls below **60%**.*

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
