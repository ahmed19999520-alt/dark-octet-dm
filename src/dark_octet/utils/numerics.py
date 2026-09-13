from __future__ import annotations
import numpy as np
 
 
def safe_log(x: float | np.ndarray, floor: float = 1e-300) -> float | np.ndarray:
    return np.log(np.maximum(x, floor))
 
 
def safe_exp(x: float | np.ndarray, ceiling: float = 700.0) -> float | np.ndarray:
    return np.exp(np.minimum(x, ceiling))
 
 
def trapz_log(
    y: np.ndarray,
    x: np.ndarray,
) -> float:
    log_x = np.log(x)
    return float(np.trapz(y * x, log_x))
 
 
def relative_error(computed: float, reference: float) -> float:
    if reference == 0.0:
        return float("inf")
    return abs(computed - reference) / abs(reference)