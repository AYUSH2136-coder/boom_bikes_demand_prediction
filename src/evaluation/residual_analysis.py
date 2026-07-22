import numpy as np
import scipy.stats as stats
from src.utils.logger import get_logger

logger = get_logger(__name__)

def check_normality(residuals: np.ndarray) -> float:
    """Performs Shapiro-Wilk test for normality of residuals. Returns the p-value."""
    # Shapiro-Wilk is limited to N <= 5000, which fits our dataset size (730)
    stat, p_val = stats.shapiro(residuals)
    logger.info(f"Shapiro-Wilk test on residuals: stat={stat:.4f}, p-value={p_val:.4e}")
    return p_val
