import math
from typing import Any, Dict

import numpy as np


class Solver:
    @staticmethod
    def _simplex_projection(v: np.ndarray) -> np.ndarray:
        """Project vector v onto the probability simplex {x >= 0, sum(x)=1}."""
        n = v.shape[0]
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, n + 1) > cssv - 1)[0]
        rho = rho[-1] if rho.size > 0 else 0
        theta = (cssv[rho] - 1) / (rho + 1.0)
        return np.maximum(v - theta, 0.0)

    def solve(self, problem: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        """Fast projected gradient ascent solver for channel capacity."""
        P = np.asarray(problem.get("P", []), dtype=np.float64)
        if P.ndim != 2 or P.size == 0:
            return {"x": [], "C": None}

        m, n = P.shape
        # Validate channel
        if not np.allclose(P.sum(axis=0), 1.0, atol=1e-6):
            return {"x": [], "C": None}

        # Precompute c = sum_i P_ij log2(P_ij) ; treat 0*log0 as 0
        with np.errstate(divide="ignore", invalid="ignore"):
            logP = np.where(P > 0, np.log2(P), 0.0)
        c = np.sum(P * logP, axis=0)  # shape (n,)

        # Initialize uniform distribution
        x = np.full(n, 1.0 / n)
        ln2 = math.log(2.0)

        # Projected gradient ascent
        max_iter = 2000
        step = 0.05
        for _ in range(max_iter):
            y = P @ x  # output distribution
            # avoid log(0) by clipping
            y_clip = np.where(y > 1e-12, y, 1e-12)
            grad = c - P.T @ (np.log2(y_clip) + 1.0) / ln2
            x_new = x + step * grad
            x_new = self._simplex_projection(x_new)
            # If change is tiny break
            if np.linalg.norm(x_new - x, ord=1) < 1e-10:
                x = x_new
                break
            x = x_new
            # Reduce step size if oscillation
            step *= 0.99

        # Compute capacity
        y = P @ x
        with np.errstate(divide="ignore", invalid="ignore"):
            H = -np.sum(np.where(y > 0, y * np.log2(y), 0.0))
        C = np.dot(c, x) + H

        return {"x": x.tolist(), "C": float(C)}
