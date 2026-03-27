import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[float]:
        """
        Solve a linear regression problem using NumPy's least‑squares routine.
        The function accepts a dictionary containing:
            * problem["X"] – a 2‑D NumPy array of shape (n_samples, n_features)
            * problem["y"] – a 1‑D NumPy array of shape (n_samples,)
        and returns a list of coefficients of length n_features. If the
        computation fails for any reason, a zero vector is returned.
        """
        try:
            X: np.ndarray = problem["X"]
            y: np.ndarray = problem["y"]

            # ``np.linalg.lstsq`` is very fast and works for dense arrays.
            # We request no regularisation and the solution that minimises
            # the Euclidean norm of the residuals.
            coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)

            # ``coeffs`` is a 1‑D array of shape (n_features,).
            return coeffs.tolist()

        except Exception:
            # In case of any error return a zero vector of the appropriate
            # dimensionality.  It is implied that ``X`` is at least a 2‑D
            # array, so ``shape[1]`` gives the number of features.
            n_features = problem["X"].shape[1]
            return [0.0] * n_features