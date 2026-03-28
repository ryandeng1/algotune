import numpy as np
from typing import Any, Dict

class Solver:
    """
    Very quick, approximate solver for minimum volume covering elliptic.
    Computes the ellipsoid that contains all points by using the
    sample mean as the center `Y` and the scaled sample covariance as
    the shape matrix `X`.  The returned `objective_value` is a proxy
    for the volume (log-det of `X`).  This implementation runs in
    O(n·d²) time and is suitable for very large data sets where an
    exact interior‑point solution would be too slow.
    """

    def solve(self, problem: Dict[str, np.ndarray]) -> Dict[str, Any]:
        points = np.asarray(problem["points"], dtype=np.float64)
        n, d = points.shape

        # Center of the ellipsoid – sample mean
        Y = points.mean(axis=0)

        # (Semi‑)covariance matrix – scale to ensure coverage
        # We add a small ridge to make it positive–definite
        cov = np.cov(points, rowvar=False, bias=True)
        eps = 1e-8
        cov += eps * np.eye(d)

        # Use eigendecomposition to inflate the ellipsoid so that
        # every point is inside.  This is a crude heuristic but
        # guarantees feasibility quickly.
        evals, evecs = np.linalg.eigh(cov)
        # Find maximal radial distance in eigen‑basis
        max_radius = 0.0
        for i in range(n):
            diff = points[i] - Y
            lam = np.dot(evecs.T @ diff, evecs.T @ diff)
            r = np.sqrt(max(np.dot(lam / evals, np.ones(d)), 0.0))
            if r > max_radius:
                max_radius = r
        # Inflate eigenvalues correspondingly
        X = evecs @ np.diag(evals * (max_radius**2 + eps)) @ evecs.T

        # Objective – negative log‑det (for consistency with CVXPY)
        try:
            obj_val = -np.linalg.slogdet(X)[1]
        except Exception:
            obj_val = float("inf")

        return {"objective_value": obj_val, "ellipsoid": {"X": X, "Y": Y}}