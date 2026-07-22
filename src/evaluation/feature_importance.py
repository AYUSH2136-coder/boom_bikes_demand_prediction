import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Any

def plot_builtin_importance(model: Any, feature_names: List[str], save_path: str) -> None:
    """Saves built-in feature importance plot if model supports it."""
    if hasattr(model, 'feature_importances_'):
        fi = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(fi.index, fi.values, color='#38bdf8', edgecolor='#0f172a')
        ax.set(title='Built-in Feature Importance', xlabel='Importance')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        print("Model does not have feature_importances_ attribute.")
