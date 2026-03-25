# solver.py
import numpy as np
from typing import Any, List, Dict

class Solver:
    """
    Fast solver for the 1D heat equation with Dirichlet boundary conditions.
    Uses spectral decomposition via discrete sine transform (DST).
    """

    def solve(self, problem: Dict[str, Any]) -> List[float]:
        """
        Returns the temperature distribution at the final time.

        Parameters
        ----------
        problem : dict
            Must contain keys:
                - t0 (float): initial time
                - t1 (float): final time
                - y0 (list[float]): initial temperatures at interior points
                - params (dict):
                        alpha (float)
                        dx (float)
                        num_points (int)
                - x_grid (list[float]): coordinates of interior points (unused)

        Returns
        -------
        List[float]
            Temperature at each interior grid point at time t1.
        """
        # extract data
        y0 = np.asarray(problem["y0"], dtype=np.float64)
        t0: float = problem["t0"]
        t1: float = problem["t1"]
        params = problem["params"]
        alpha: float = params["alpha"]
        dx: float = params["dx"]
        N: int = params["num_points"]

        dt = t1 - t0
        if dt <= 0:
            return y0.tolist()

        # Precompute sine transform basis
        # indices i=1..N, k=1..N
        i = np.arange(1, N + 1, dtype=np.float64)
        k = np.arange(1, N + 1, dtype=np.float64)
        sin_mat = np.sin(np.outer((k * np.pi) / (N + 1), i / (N + 1)))  # shape (N, N)

        # Compute sine series coefficients c_k
        c = (2.0 / (N + 1)) * (y0 @ sin_mat.T)  # shape (N,)

        # Eigenvalues for the discrete Laplacian with Dirichlet BC
        lam = -4.0 * alpha / (dx * dx) * np.sin(0.5 * k * np.pi / (N + 1)) ** 2
        exp_factor = np.exp(lam * dt)

        # Multiply coefficients by exp factors
        temp = c * exp_factor

        # Reconstruct solution: u = sin_mat @ temp
        y_final = sin_mat @ temp

        return y_final.tolist()
