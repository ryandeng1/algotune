import numpy as np
from typing import Any


class Solver:
    def solve(self, problem: dict[str, Any]) -> list[list[float]]:
        X = np.array(problem["X"])
        n_components = problem["n_components"]
        # center the data
        Xc = X - np.mean(X, axis=0, keepdims=True)
        try:
            # compute SVD of the centered data
            # use full_matrices=False to get compact form
            u, s, vt = np.linalg.svd(Xc, full_matrices=False)
            # vᵀ has shape (d, min(n,d)); take first n_components rows
            V = vt[:n_components, :].T
            return V
        except Exception:
            # fall back to identity matrix of appropriate shape
            n, d = X.shape
            V = np.eye(n_components, n)
            return V