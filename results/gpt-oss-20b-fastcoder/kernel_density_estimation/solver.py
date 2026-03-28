import numpy as np
from numbers import Real
from typing import Any, Dict, List

# Mapping kernel names to functions that compute log kernel values
# for a given squared distance array and bandwidth h
def _gaussian_logkernel(dist2: np.ndarray, h: float) -> np.ndarray:
    return -0.5 * dist2 / h**2

def _tophat_logkernel(dist2: np.ndarray, h: float) -> np.ndarray:
    mask = dist2 <= h**2
    log_vals = np.full_like(dist2, -np.inf, dtype=np.float64)
    log_vals[mask] = 0.0
    return log_vals

def _epanechnikov_logkernel(dist2: np.ndarray, h: float) -> np.ndarray:
    mask = dist2 <= h**2
    log_vals = np.full_like(dist2, -np.inf, dtype=np.float64)
    log_vals[mask] = 0.5 * (1 - dist2[mask] / h**2)
    return log_vals

def _exponential_logkernel(dist2: np.ndarray, h: float) -> np.ndarray:
    return -np.sqrt(dist2) / h

def _linear_logkernel(dist2: np.ndarray, h: float) -> np.ndarray:
    mask = np.sqrt(dist2) <= h
    log_vals = np.full_like(dist2, -np.inf, dtype=np.float64)
    log_vals[mask] = 1 - np.sqrt(dist2[mask]) / h
    return log_vals

def _cosine_logkernel(dist2: np.ndarray, h: float) -> np.ndarray:
    mask = np.sqrt(dist2) <= h
    log_vals = np.full_like(dist2, -np.inf, dtype=np.float64)
    angle = np.sqrt(dist2[mask]) / h
    log_vals[mask] = np.log(np.cos(angle / 2)) + np.log(2)  # normalization
    return log_vals

_KERNEL_MAP = {
    "gaussian": _gaussian_logkernel,
    "tophat": _tophat_logkernel,
    "epanechnikov": _epanechnikov_logkernel,
    "exponential": _exponential_logkernel,
    "linear": _linear_logkernel,
    "cosine": _cosine_logkernel,
}

class Solver:
    """Fast KDE solver for 2‑D data using NumPy broadcasting."""

    def __init__(self):
        self.available_kernels = list(_KERNEL_MAP.keys())

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Return log density estimates for query points.

        Parameters
        ----------
        problem : dict
            Must contain:
                data_points  : 2‑D array of shape (n_samples, n_features)
                query_points : 2‑D array of shape (m_samples, n_features)
                kernel       : one of the supported kernel names
                bandwidth    : positive float

        Returns
        -------
        dict
            {'log_density': list} on success,
            {'error': str} on error.
        """
        try:
            # Input validation
            X = np.asarray(problem["data_points"], dtype=np.float64)
            X_q = np.asarray(problem["query_points"], dtype=np.float64)
            kernel = problem["kernel"]
            bandwidth = problem["bandwidth"]

            if X.ndim != 2 or X_q.ndim != 2:
                raise ValueError("Data points and query points must be 2‑D arrays.")
            if X.shape[0] == 0:
                raise ValueError("No data points provided.")
            if X_q.shape[0] == 0:
                return {"log_density": []}
            if X.shape[1] != X_q.shape[1]:
                raise ValueError("Data points and query points have different dimensions.")

            if not isinstance(bandwidth, Real) or bandwidth <= 0:
                raise ValueError("Bandwidth must be a positive real number.")
            h = float(bandwidth)

            if kernel not in self.available_kernels:
                raise ValueError(f"Unknown kernel: {kernel}")

            # Compute squared Euclidean distances: (n_samples, m_samples)
            # Using broadcasting: ||x_i - y_j||^2 = ||x_i||^2 + ||y_j||^2 - 2 x_i·y_j
            X_norm2 = np.sum(X * X, axis=1).reshape(-1, 1)
            Xq_norm2 = np.sum(X_q * X_q, axis=1).reshape(1, -1)
            dist2 = X_norm2 + Xq_norm2 - 2 * X @ X_q.T
            dist2 = np.maximum(dist2, 0.0)  # numerical safety

            # Kernel log-values matrix
            logkernel_func = _KERNEL_MAP[kernel]
            log_kernel_vals = logkernel_func(dist2, h)

            # Numerical stable log-sum-exp over data points
            eps = 1e-300
            max_log = np.max(log_kernel_vals, axis=0)
            max_log[~np.isfinite(max_log)] = -np.inf  # avoid NaNs

            # Sum of exponentials
            stable = np.exp(log_kernel_vals - max_log)
            sum_exp = np.sum(stable, axis=0)
            log_density = max_log + np.log(sum_exp + eps) - np.log(X.shape[0])

            solution = {"log_density": log_density.tolist()}
            return solution

        except KeyError as e:
            return {"error": f"Missing key: {e}"}
        except (ValueError, TypeError) as e:
            return {"error": f"Computation error: {e}"}
        except Exception as e:  # pragma: no cover
            return {"error": f"Unexpected error: {e}"}