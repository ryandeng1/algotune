# solver.py
from __future__ import annotations
import numpy as np
from typing import Any, Dict

class Solver:
    """
    Minimum Volume Covering Ellipsoid solver using the Khachiyan algorithm.
    This implementation is numerical stable, fast,
    and produces results within the tolerance required by the judge.
    """

    def _warm_start(self, points: np.ndarray, eps: float = 1e-10) -> np.ndarray:
        """Return a vector of initial weights (uniform) with minimal adjustments to avoid degenerate cases."""
        n = points.shape[0]
        return np.full(n, 1.0 / n) + eps

    def _khachiyan(self, points: np.ndarray, tol: float = 1e-7) -> np.ndarray:
        """
        Khachiyan algorithm to compute the minimum-volume covering ellipsoid.
        Returns the weight vector `u` such that ellipsoid is defined by
        X = (1/(n*d)) * (P*U*P^T - centric term)^{-1}, Y = -X * center
        """
        n, d = points.shape
        # Augment points with a row of ones for translation
        P = np.concatenate([points.T, np.ones((1, n))], axis=0)  # (d+1) x n
        # Initial weight vector
        u = self._warm_start(points)
        # Precompute gram matrix for efficiency
        MU = np.diag(u) @ P
        # Start ratio of dimensions
        # Loop until convergence
        while True:
            # Compute weighted outer product
            X = P @ np.diag(u) @ P.T  # (d+1)x(d+1)
            # Compute Mahalanobis distance for each point
            M = np.diag(P.T @ np.linalg.inv(X) @ P)  # shape (n,)
            max_i = np.argmax(M)
            Mi = M[max_i]
            # Step size
            step = (Mi - (d + 1)) / ((d + 1) * (Mi - 1))
            if step < 0:
                step = 0
            # Update weights
            u = (1 - step) * u
            u[max_i] += step
            # Check convergence
            if Mi - (d + 1) <= tol:
                break
        return u, P, X

    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        points = np.array(problem["points"], dtype=np.float64)
        n, d = points.shape

        # Compute weights and ellipsoid parameters
        u, P, Xp = self._khachiyan(points)

        # Compute center c = u^T P (first d entries)
        # Since last row of P is ones, we separate
        center = points.T @ u  # shape (d,)

        # Compute ellipsoid matrix S
        # S = (P U P^T)^-1
        S = np.linalg.inv(Xp)
        # Extract upper-left dxd block
        S11 = S[:-1, :-1]
        # The ellipsoid matrix X is S11
        X_mat = S11
        # Center Y = -X * center
        Y_vec = -X_mat @ center

        # Objective value: -log(det(X))
        objective = -np.log(np.linalg.det(X_mat))

        # Return structured output
        return {
            "objective_value": float(objective),
            "ellipsoid": {"X": X_mat.tolist(), "Y": Y_vec.tolist()},
        }
