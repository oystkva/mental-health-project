import os
import numpy as np
from statsmodels.stats.multitest import fdrcorrection

from config import (
    DATA_DIR,
    LOG_DIR,
)
from utils import log_message

def fdr_correction_fc(p_values: np.ndarray, alpha: float = 0.05) -> (tuple[np.ndarray, np.ndarray]):
    """
    Perform FDR correction on a list of p-values.

    Parameters:
    p_values (list or np.array): List of p-values to correct.
    alpha (float): Significance level for FDR correction.

    Returns:
    np.array: Array of boolean values indicating which hypotheses are rejected.
    np.array: Array of adjusted p-values after FDR correction.
    """

    if p_values.ndim != 2 or p_values.shape[0] != p_values.shape[1]:
        raise ValueError("Input p-values must be a square matrix (n x n).")

    n = p_values.shape[0]
    triu_idx = np.triu_indices(n, k=0)
    p_flat = p_values[triu_idx]

    reject_flat, p_adjusted_flat = fdrcorrection(p_flat, alpha=alpha)

    reject_fdr = np.zeros((n, n), dtype=bool)
    p_fdr = np.zeros((n, n), dtype=float)

    reject_fdr[triu_idx] = reject_flat
    reject_fdr[(triu_idx[1], triu_idx[0])] = reject_flat  # Symmetric

    p_fdr[triu_idx] = p_adjusted_flat
    p_fdr[(triu_idx[1], triu_idx[0])] = p_adjusted_flat  # Symmetric

    return reject_fdr, p_fdr


def fdr_correct_pipeline(alpha: float = 0.05):
    """
    Pipeline to perform FDR correction on all p-value matrices which havent been corrected yet.
    """

    test_result_dir = os.path.join(DATA_DIR, "permutation_test_results")
    
    log_path = os.path.join(LOG_DIR, "fdr_correction_log.txt")
    log_message("Starting FDR correction pipeline...", log_path)

    for root, dirs, files in os.walk(test_result_dir):
        for file in files:
            if file.startswith("p_values") and file.endswith(".npy"):
                p_values_path = os.path.join(root, file)
                p_fdr_values_path = os.path.join(root, file.replace("p_values", "p_fdr_values"))
                fdr_reject_path = os.path.join(root, file.replace("p_values", "fdr_reject"))

                if not os.path.exists(p_fdr_values_path) or not os.path.exists(fdr_reject_path):
                    log_message(f"Performing FDR correction for: {p_values_path}", log_path)
                    p_values = np.load(p_values_path)
                    reject_fdr, p_fdr = fdr_correction_fc(p_values, alpha=alpha)
                    np.save(p_fdr_values_path, p_fdr)
                    np.save(fdr_reject_path, reject_fdr)