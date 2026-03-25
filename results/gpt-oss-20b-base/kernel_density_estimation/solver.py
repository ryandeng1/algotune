import numpy as np
from numbers import Real
from typing import Any, Dict, Iterable, List, Tuple, Union

class Solver:
    """Fast KDE implementation supporting gaussian, tophat and epanechnikov kernels."""

    kernel_docs = ("gaussian", "tophat", "epanechnikov")

    def _validate_problem(self, problem: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, str, float, int]:
        """Parse and validate problem data, raising ValueError on failure."""
        # Extract required keys
        try:
            data = np.asarray(problem["data_points"], dtype=float)
            query = np.asarray(problem["query_points"], dtype=float)
            kernel = problem["kernel"]
            bandwidth = problem["bandwidth"]
        except KeyError as e:
            raise ValueError(f"Missing key: {e}") from None
        # Shape checks
        if data.ndim != 2 or query.ndim != 2:
            raise ValueError("Data points and query points must be 2‑D arrays.")
        if data.shape[0] == 0:
            raise ValueError("No data points provided.")
        if data.shape[1] != query.shape[1]:
            raise ValueError("Dimension mismatch between data and query points.")
        # Bandwidth sanity
        if not isinstance(bandwidth, Real) or bandwidth <= 0:
            raise ValueError("Bandwidth must be a positive real number.")
        # Kernel name
        if kernel not in self.kernel_docs:
            raise ValueError(f"Unknown kernel: {kernel}")
        return data, query, kernel, float(bandwidth), data.shape[1]

    # Helper function to compute log density for gaussian kernel
    _gaussian_log = staticmethod(
        lambda diff_norm2, n, d, h:  # noqa: E741
        -0.5 * d * np.log(2 * np.pi) - d * np.log(h)
        - 0.5 * diff_norm2 / h**2
    )

    # Helper for tophat
    @staticmethod
    def _tophat_density(diff_norm, n, d, h) -> np.ndarray:
        """Counts points within radius h."""
        mask = diff_norm <= h
        count = np.sum(mask, axis=1, dtype=np.float64)
        volume = h**d * np.pi**(d/2) / np.math.gamma(d/2 + 1)
        density = count / (n * volume)
        return density

    # Epanechnikov
    @staticmethod
    def _epane_density(diff_norm2, n, d, h) -> np.ndarray:
        """Sum of kernel weights for epanechnikov."""
        inside = diff_norm2 <= h**2
        w = (1 - diff_norm2 / h**2) * inside
        weight_sum = np.sum(w, axis=1, dtype=np.float64)
        volume = h**d * np.pi**(d/2) / np.math.gamma(d/2 + 1)
        density = weight_sum / (n * volume)
        return density

    def solve(self, problem: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        """Compute logarithm of kernel density estimates at query points."""
        try:
            X, Xq, kernel, h, d = self._validate_problem(problem)
        except ValueError as e:
            return {"error": str(e)}

        n = X.shape[0]
        if Xq.shape[0] == 0:
            return {"log_density": []}

        # Diff matrix: (q, n, d)
        diff = Xq[:, None, :] - X[None, :, :]
        diff_norm2 = np.sum(diff ** 2, axis=2)  # (q, n)
        diff_norm = np.sqrt(diff_norm2)

        try:
            if kernel == "gaussian":
                # Compute log of kernel values
                logk = self._gaussian_log(diff_norm2, n, d, h)[None, :]
                # log density = log(1/n) + logsumexp(logk)
                # Use stable logsumexp
                max_logk = np.max(logk, axis=1, keepdims=True)
                sum_exp = np.exp(logk - max_logk).sum(axis=1)
                log_density = -np.log(n) + np.squeeze(max_logk + np.log(sum_exp))
                return {"log_density": log_density.tolist()}

            elif kernel == "tophat":
                density = self._tophat_density(diff_norm, n, d, h)
                log_density = np.log(density, where=density > 0)
                # Handle zero density -> -inf (log(0))
                log_density[density == 0] = -np.inf
                return {"log_density": log_density.tolist()}

            elif kernel == "epanechnikov":
                density = self._epane_density(diff_norm2, n, d, h)
                log_density = np.log(density, where=density > 0)
                log_density[density == 0] = -np.inf
                return {"log_density": log_density.tolist()}

            else:
                return {"error": f"Unsupported kernel: {kernel}"}
        except Exception as exc:
            return {"error": f"Computation error: {exc}"}
