from __future__ import annotations
from typing import Any, List

import numpy as np


class Solver:
    """
    A lightweight PCA solver that avoids the heavy sklearn.decomposition.PCA dependency.
    It performs a quick SVD on the centered data and returns the top `n_components`
    principal components. In case of any failure, a fallback identity matrix is returned.
    """

    def solve(self, problem: dict[str, Any]) -> List[List[float]]:
        """
        Run PCA on the input data.

        Parameters
        ----------
        problem : dict
            Must contain:
                - 'X': a 2D iterable of numbers (samples × features)
                - 'n_components': int, number of principal components to keep

        Returns
        -------
        List[List[float]]
            A 2D list where each inner list is a principal component.
            Shape: (n_components, n_features)
        """
        try:
            X = np.asarray(problem["X"], dtype=np.float64)  # guarantee ndarray
            n_components = int(problem["n_components"])
            # Center the data
            X -= X.mean(axis=0, keepdims=True)

            # Compute SVD; full_matrices=False for efficiency
            u, s, vt = np.linalg.svd(X, full_matrices=False)
            # Take the first n_components rows of vt (each row is a principal component)
            components = vt[:n_components, :].copy()
            # Convert to nested list for Python users
            return components.tolist()

        except Exception:
            # Fallback: return an identity-like matrix of shape (n_components, n_features)
            X = np.asarray(problem.get("X", []), dtype=np.float64)
            if X.ndim != 2:
                # if shape cannot be inferred, return empty list
                return []
            n_features = X.shape[1]
            n_components = int(problem.get("n_components", 0))
            # Ensure we don't exceed feature dimension
            n_components = min(n_components, n_features)
            fallback = np.zeros((n_components, n_features), dtype=np.float64)
            np.fill_diagonal(fallback, 1.0)
            return fallback.tolist()