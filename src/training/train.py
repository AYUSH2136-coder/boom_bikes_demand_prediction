import os
import argparse
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.logger import get_logger
from src.utils.helpers import load_yaml
from src.utils.seed import set_seed
from src.data.loader import load_raw_data
from src.data.feature_engineering import add_engineered_features
from src.data.preprocessing import preprocess_features
from src.data.split import split_data
from src.models import get_model_by_name
from src.training.hyperparameter_tuning import tune_hyperparameters
from src.evaluation.metrics import compute_metrics
from src.evaluation.plots import plot_actual_vs_predicted, plot_residuals, plot_error_by_segment
from src.evaluation.feature_importance import plot_builtin_importance
from src.evaluation.residual_analysis import check_normality

logger = get_logger(__name__)

def train_pipeline(config_dir: str = "configs", run_comparison: bool = True) -> None:
    """End-to-end model training, hyperparameter tuning, and evaluation pipeline."""
    logger.info("Starting training pipeline...")
    
    # Load configuration
    paths = load_yaml(os.path.join(config_dir, "paths.yaml"))
    cfg = load_yaml(os.path.join(config_dir, "config.yaml"))
    mcfg = load_yaml(os.path.join(config_dir, "model.yaml"))
    
    # Set reproducibility seed
    set_seed(cfg["project"]["random_state"])
    
    # Create directories for outputs
    models_dir = Path(paths["artifacts"]["models"])
    outputs_dir = Path(paths["artifacts"]["outputs"])
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load raw data
    df_raw = load_raw_data(paths["data"]["raw"])
    
    # 2. Feature engineering
    df_engineered = add_engineered_features(df_raw)
    
    # Save engineered dataset
    processed_dir = Path(paths["data"]["processed"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    df_engineered.to_csv(processed_dir / "data_engineered.csv", index=False)
    logger.info(f"Saved engineered dataset to {processed_dir / 'data_engineered.csv'}")
    
    # 3. Split data (time-based chronological split)
    target = cfg["project"]["target"]
    test_size = cfg["project"]["test_size"]
    X_train_raw, X_test_raw, y_train, y_test = split_data(df_engineered, target_column=target, test_size=test_size)
    
    # 4. Preprocessing (Drop columns and scale numeric features)
    drop_columns = cfg["features"]["drop_columns"]
    numeric_columns = cfg["features"]["numeric"]
    scale_numeric = cfg["preprocessing"]["scale_numeric"]
    
    X_train_processed, scaler = preprocess_features(
        X_train_raw,
        drop_columns=drop_columns,
        numeric_columns=numeric_columns,
        fit_scaler=scale_numeric
    )
    
    X_test_processed, _ = preprocess_features(
        X_test_raw,
        drop_columns=drop_columns,
        numeric_columns=numeric_columns,
        scaler=scaler
    )
    
    # Save scaler if fitted
    if scaler is not None:
        scaler_path = models_dir / "scaler.pkl"
        joblib.dump(scaler, scaler_path)
        logger.info(f"Saved scaler to {scaler_path}")
        
    # Save processed training dataset for reference
    train_df = X_train_processed.copy()
    train_df[target] = y_train.values
    train_df.to_csv(processed_dir / "data_processed.csv", index=False)
    
    # 5. Model Comparison (Optional)
    champion_name = "xgboost"
    
    if run_comparison:
        logger.info("Running baseline comparison across all algorithms...")
        comparison_results = []
        
        # Compare all configured models
        model_names = ["linear_regression", "ridge", "lasso", "decision_tree", "random_forest", "xgboost", "lightgbm", "catboost"]
        
        for name in model_names:
            try:
                params = mcfg.get(name, {})
                model = get_model_by_name(name, **params)
                
                model.fit(X_train_processed, y_train)
                preds = model.predict(X_test_processed)
                
                metrics = compute_metrics(y_test, preds)
                metrics["Model"] = name
                comparison_results.append(metrics)
                
                logger.info(f"{name:<20} | R2: {metrics['R2']:.4f} | RMSE: {metrics['RMSE']:.2f} | MAE: {metrics['MAE']:.2f}")
            except Exception as e:
                logger.warning(f"Failed to train baseline model {name}: {e}")
                
        if comparison_results:
            results_df = pd.DataFrame(comparison_results).sort_values("R2", ascending=False).reset_index(drop=True)
            logger.info("\nBaseline Model Comparison Results:\n" + results_df.to_string(index=False))
            
            # Select champion (highest R² on test set)
            champion_name = results_df.iloc[0]["Model"]
            logger.info(f"Selected Champion Model: {champion_name}")
            
            # Save baseline comparison plot
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('Model Comparison on Test Set', fontsize=14)
            palette = ['#38bdf8', '#fb7185', '#34d399', '#fbbf24', '#a78bfa', '#f97316', '#e879f9', '#22d3ee']
            
            for ax, metric, title in zip(axes, ['R2', 'RMSE', 'MAE'], ['R² Score', 'RMSE', 'MAE']):
                data = results_df.sort_values(metric, ascending=(metric != 'R2'))
                colors = [palette[i % len(palette)] for i in range(len(data))]
                ax.barh(data['Model'], data[metric], color=colors, edgecolor='#0f172a')
                ax.set_title(title)
                ax.set_xlabel(metric)
                ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(outputs_dir / 'model_comparison.png', dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved model comparison plot to {outputs_dir / 'model_comparison.png'}")
            
    # 6. Hyperparameter Tuning on Champion
    logger.info(f"Tuning champion model: {champion_name}")
    
    param_grids = {
        'random_forest': {
            'n_estimators':    [100, 200, 300, 500],
            'max_depth':       [None, 6, 8, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf':  [1, 2, 4],
            'max_features':    ['sqrt', 'log2', 0.5],
        },
        'xgboost': {
            'n_estimators':   [200, 300, 500],
            'learning_rate':  [0.01, 0.05, 0.1],
            'max_depth':      [4, 6, 8],
            'subsample':      [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma':          [0, 0.1, 0.5],
        },
        'lightgbm': {
            'n_estimators':  [200, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves':    [20, 31, 50, 100],
            'max_depth':     [-1, 6, 10],
            'min_child_samples': [5, 10, 20],
        },
        'catboost': {
            'iterations':    [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'depth':         [4, 6, 8],
        }
    }
    
    base_params = mcfg.get(champion_name, {})
    base_model = get_model_by_name(champion_name, **base_params)
    
    param_grid = param_grids.get(champion_name, None)
    if param_grid is None:
        logger.info(f"No tuning grid defined for {champion_name}. Training with default/config parameters...")
        tuned_model = base_model
        tuned_model.fit(X_train_processed, y_train)
        best_params = base_params
    else:
        tuning_cfg = mcfg["tuning"]
        search = tune_hyperparameters(
            model=base_model,
            param_grid=param_grid,
            X_train=X_train_processed,
            y_train=y_train,
            n_iter=tuning_cfg["n_iter"],
            cv=tuning_cfg["cv"],
            random_state=tuning_cfg["random_state"],
            scoring=tuning_cfg["scoring"]
        )
        tuned_model = search.best_estimator_
        best_params = search.best_params_
        logger.info(f"Tuning complete. Best params: {best_params}")
        
    # Save best parameters
    with open(models_dir / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2, default=str)
        
    # Save tuned model
    joblib.dump(tuned_model, models_dir / "tuned_model.pkl")
    logger.info(f"Tuned model saved to {models_dir / 'tuned_model.pkl'}")
    
    # Save champion model (alias to tuned model)
    joblib.dump(tuned_model, models_dir / "champion_model.pkl")
    
    # 7. Model Evaluation
    preds_test = tuned_model.predict(X_test_processed)
    residuals = y_test.values - preds_test
    
    test_metrics = compute_metrics(y_test, preds_test)
    logger.info(f"\nTuned Champion Model — Test Set Metrics:")
    logger.info(f"  R²  : {test_metrics['R2']:.4f}")
    logger.info(f"  RMSE: {test_metrics['RMSE']:.2f}")
    logger.info(f"  MAE : {test_metrics['MAE']:.2f}")
    logger.info(f"  MAPE: {test_metrics['MAPE']:.2f}%")
    
    # Save test predictions
    test_dates = pd.to_datetime(df_raw['dteday'].iloc[X_train_processed.shape[0]:].values, dayfirst=True)
    predictions_df = pd.DataFrame({
        "date": test_dates,
        "actual": y_test.values,
        "predicted": np.round(preds_test).astype(int),
        "error": residuals,
        "abs_error": np.abs(residuals)
    })
    predictions_df.to_csv(outputs_dir / "test_predictions.csv", index=False)
    logger.info(f"Test predictions saved to: {outputs_dir / 'test_predictions.csv'}")
    
    # 8. Plot generation
    logger.info("Generating evaluation and diagnostic plots...")
    plot_actual_vs_predicted(
        y_true=y_test.values,
        y_pred=preds_test,
        dates=test_dates,
        save_path=str(outputs_dir / "eval_actual_vs_predicted.png")
    )
    
    plot_residuals(
        y_true=y_test.values,
        y_pred=preds_test,
        save_path=str(outputs_dir / "residual_analysis.png")
    )
    
    df_raw_test = df_raw.iloc[X_train_processed.shape[0]:].copy()
    plot_error_by_segment(
        df_raw_test=df_raw_test,
        residuals=residuals,
        y_true=y_test.values,
        y_pred=preds_test,
        save_path=str(outputs_dir / "error_by_segment.png")
    )
    
    feature_names = X_train_processed.columns.tolist()
    plot_builtin_importance(
        model=tuned_model,
        feature_names=feature_names,
        save_path=str(outputs_dir / "feature_importance.png")
    )
    
    check_normality(residuals)
    logger.info("Training and evaluation pipeline complete successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BoomBikes demand prediction model.")
    parser.add_argument("--config-dir", type=str, default="configs", help="Path to config folder.")
    parser.add_argument("--no-comparison", action="store_true", help="Skip baseline model comparison.")
    args = parser.parse_args()
    
    train_pipeline(config_dir=args.config_dir, run_comparison=not args.no_comparison)
