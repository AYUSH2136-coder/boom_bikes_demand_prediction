import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Optional, Union

PALETTE = ['#38bdf8', '#fb7185', '#34d399', '#fbbf24', '#a78bfa', '#f97316']

# Set style parameters
plt.rcParams.update({
    'figure.facecolor': '#0f172a', 'axes.facecolor': '#1e293b',
    'axes.edgecolor': '#334155', 'axes.labelcolor': '#e2e8f0',
    'xtick.color': '#94a3b8', 'ytick.color': '#94a3b8', 'text.color': '#e2e8f0',
    'grid.color': '#334155', 'figure.titlesize': 16, 'axes.titlesize': 13
})

def plot_actual_vs_predicted(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    dates: Optional[np.ndarray], 
    save_path: str
) -> None:
    """Saves a plot comparing actual vs predicted demand as a scatter plot and time series overlay."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Model Evaluation — Actual vs Predicted')
    
    # Scatter
    axes[0].scatter(y_true, y_pred, alpha=0.6, color=PALETTE[0], s=30)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    axes[0].plot([lo, hi], [lo, hi], 'r--', linewidth=2, label='Perfect prediction')
    axes[0].set(title='Scatter: Actual vs Predicted', xlabel='Actual cnt', ylabel='Predicted cnt')
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    
    # Time Series Overlay
    if dates is not None:
        dates_dt = pd.to_datetime(dates)
        axes[1].plot(dates_dt, y_true, color=PALETTE[0], linewidth=1.5, label='Actual', alpha=0.9)
        axes[1].plot(dates_dt, y_pred, color=PALETTE[1], linewidth=1.5, label='Predicted', linestyle='--')
        axes[1].set(title='Test Period: Actual vs Predicted', xlabel='Date', ylabel='cnt')
        axes[1].legend()
        axes[1].grid(alpha=0.2)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, save_path: str) -> None:
    """Saves residual diagnostic plots (residuals vs predicted, distribution hist, Q-Q plot)."""
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Residual Analysis')
    
    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5, color=PALETTE[0], s=25)
    axes[0].axhline(0, color='white', linestyle='--', linewidth=2)
    axes[0].set(title='Residuals vs Predicted', xlabel='Predicted', ylabel='Residual (Actual − Predicted)')
    axes[0].grid(alpha=0.2)
    
    # Histogram
    axes[1].hist(residuals, bins=25, color=PALETTE[2], edgecolor='#0f172a', alpha=0.9)
    axes[1].axvline(0, color='white', linestyle='--', linewidth=2)
    axes[1].set(title='Residual Distribution', xlabel='Residual', ylabel='Frequency')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Q-Q plot
    stats.probplot(residuals, plot=axes[2])
    axes[2].get_lines()[0].set(markerfacecolor=PALETTE[0], markersize=4, alpha=0.6)
    axes[2].get_lines()[1].set(color=PALETTE[1], linewidth=2)
    axes[2].set_title('Q-Q Plot of Residuals')
    axes[2].grid(alpha=0.2)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_error_by_segment(
    df_raw_test: pd.DataFrame, 
    residuals: np.ndarray, 
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str
) -> None:
    """Saves MAE breakdown by Season, Weather condition, and Month."""
    test_meta = df_raw_test[['season', 'weathersit', 'mnth', 'yr', 'workingday']].copy()
    test_meta['abs_error'] = np.abs(residuals)
    test_meta['pred'] = y_pred
    test_meta['actual'] = y_true
    
    test_meta['season_label']  = test_meta['season'].map({1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'})
    test_meta['weather_label'] = test_meta['weathersit'].map({1: 'Clear', 2: 'Mist', 3: 'Light Rain', 4: 'Heavy Rain'})
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Mean Absolute Error by Segment')
    
    for ax, col, order, title in [
        (axes[0], 'season_label',  ['Spring', 'Summer', 'Fall', 'Winter'],    'By Season'),
        (axes[1], 'weather_label', ['Clear', 'Mist', 'Light Rain', 'Heavy Rain'], 'By Weather'),
        (axes[2], 'mnth',          None,                                   'By Month'),
    ]:
        if col == 'mnth':
            seg = test_meta.groupby(col)['abs_error'].mean()
            ax.bar(seg.index.astype(str), seg.values, color=PALETTE[0], edgecolor='#0f172a')
        else:
            seg = test_meta.groupby(col)['abs_error'].mean()
            if order:
                seg = seg.reindex([o for o in order if o in seg.index])
            ax.bar(seg.index, seg.values, color=PALETTE[0], edgecolor='#0f172a')
        ax.set(title=title, xlabel=col, ylabel='MAE')
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=20)
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
