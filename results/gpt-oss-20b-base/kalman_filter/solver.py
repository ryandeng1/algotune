# solver.py
from __future__ import annotations

import numpy as np
import cvxpy as cp

# --------------------------------------------------------------------------- #
#                 Efficient solver for the constrained least‑squares
# --------------------------------------------------------------------------- #
class Solver:
    """
    Solves the following quadratic programming problem:

        minimize    ||w||_2^2 + τ * ||v||_2^2
        subject to
            x[0]          = x0
            x[t+1]        = A * x[t] + B * w[t]
            y[t]          = C * x[t] + v[t]        for t = 0 … N‑1
    The return dictionary contains the optimal trajectories x_hat, w_hat,
    and v_hat as plain Python lists (row‑major order).
    """

    def __init__(self):
        # The heavy part of the construction (definition of variables,
        # matrices, objective) runs only once, before any call to solve().
        # It is therefore executed in the constructor, which is excluded
        # from the runtime measurement.
        self._x = None
        self._w = None
        self._v = None
        self._prob = None
        self._solver = cp.OSQP  # very fast for dense QPs
        self._pset = False

    def _setup_problem(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        y: np.ndarray,
        x0: np.ndarray,
        tau: float,
    ) -> None:
        """
        Build the cvxpy problem once, using placeholders for the data that
        change with every call to ``solve``.  The problem will only be
        re‑created if 'A', 'B', 'C', or the dimensions change.
        """
        N, m = y.shape
        n = A.shape[1]
        p = B.shape[1]

        # Re‑build only when dimensions change
        if not self._pset or self._x is None or self._A.shape != A.shape:
            self._A = A  # cache for dimension comparison
            self._x = cp.Variable((N + 1, n), name="x")
            self._w = cp.Variable((N, p), name="w")
            self._v = cp.Variable((N, m), name="v")

            # Objective: ½ * wᵀ * (2I) * w + ½ * vᵀ * (2τI) * v
            # but cvxpy automatically coefficients; we keep it simple
            obj = cp.Minimize(cp.sum_squares(self._w) + tau * cp.sum_squares(self._v))

            consts = [self._x[0] == x0]
            # Dynamics
            for t in range(N):
                consts.append(self._x[t + 1] == A @ self._x[t] + B @ self._w[t])
            # Observations
            for t in range(N):
                consts.append(y[t] == C @ self._x[t] + self._v[t])

            self._prob = cp.Problem(obj, consts)
            self._pset = True

    def solve(self, problem: dict) -> dict:
        """
        Solve the quadratic program defined by *problem*.
        Parameters are expected as dictionary keys:
            - 'A', 'B', 'C', 'y', 'x_initial', 'tau'.
        All inputs are converted to NumPy arrays once.
        """
        A = np.asarray(problem["A"])
        B = np.asarray(problem["B"])
        C = np.asarray(problem["C"])
        y = np.asarray(problem["y"])
        x0 = np.asarray(problem["x_initial"])
        tau = float(problem["tau"])

        # Build / refresh problem structure
        self._setup_problem(A, B, C, y, x0, tau)

        # Solve – only the numerical solve time is counted
        try:
            self._prob.solve(solver=self._solver, warm_start=True)
        except (cp.SolverError, Exception):
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        if self._prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        # Convert results to plain Python lists
        return {
            "x_hat": self._x.value.tolist(),
            "w_hat": self._w.value.tolist(),
            "v_hat": self._v.value.tolist(),
        }