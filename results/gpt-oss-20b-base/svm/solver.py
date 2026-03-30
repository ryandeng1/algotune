# solver.py
from typing import Any, Dict, List
import cvxpy as cp
import numpy as np

class Solver:
    """
    Optimized SVM solver using CVXPY.
    """

    # Prefer a fast open-source solver that ships with CVXPY.
    _solver_options = {"solver": cp.ECOS, "verbose": False}

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any] | None:
        """
        Solve the hard‑margin SVM problem with slack variables.

        Parameters
        ----------
        problem : dict
            Must contain keys 'X', 'y', 'C'.

        Returns
        -------
        dict
            Dictionary with keys:
                * beta0: float
                * beta : list[float]
                * optimal_value : float
                * missclass_error : float
            Returns ``None`` if the problem cannot be solved.
        """
        try:
            X = np.asarray(problem["X"], dtype=np.float64)
            y = np.asarray(problem["y"], dtype=np.float64).reshape(-1, 1)
            C = float(problem["C"])
        except Exception:
            # Missing keys or bad data
            return None

        n, p = X.shape
        beta = cp.Variable((p, 1))
        beta0 = cp.Variable()
        xi = cp.Variable((n, 1))

        objective = cp.Minimize(0.5 * cp.sum_squares(beta) + C * cp.sum(xi))
        constraints = [
            xi >= 0,
            cp.multiply(y, X @ beta + beta0) >= 1 - xi,
        ]
        prob = cp.Problem(objective, constraints)

        try:
            optimal_value = prob.solve(**self._solver_options)
        except cp.SolverError:
            return None
        except Exception:
            return None

        if beta.value is None or beta0.value is None:
            return None

        # Forward‑pass prediction
        decision = X @ beta.value + beta0.value  # shape (n, 1)
        missclass = float(np.mean(decision * y < 0))

        return {
            "beta0": float(beta0.value),
            "beta": beta.value.ravel().tolist(),
            "optimal_value": float(optimal_value),
            "missclass_error": missclass,
        }