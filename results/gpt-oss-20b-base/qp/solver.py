# solver.py
# -------------------------------------------------
# Optimised solver for QP problems of the form:
#   min 0.5 * x^T P x + q^T x
#   s.t. G x <= h
#        A x == b
# -------------------------------------------------
# Uses cvxpy + OSQP with maximally aggressive tolerances and no verbose output.
# -------------------------------------------------

from __future__ import annotations

from typing import Any, Dict

import cvxpy as cp
import numpy as np

# ------------------------------------------------------------------
# Utility: create a CVXPY variable only once per run size
# ------------------------------------------------------------------
def _create_variable(n: int) -> cp.Variable:
    return cp.Variable(n, name="x")

# ------------------------------------------------------------------
# Solver class -------------------------------------------------------
# ------------------------------------------------------------------
class Solver:
    """
    Very small, fast QP solver based on cvxpy + OSQP.

    The method :py:meth:`solve` accepts a problem dictionary of the form::

        {
            "P": 2‑D array_like  # symmetric positive‑semidefinite matrix
            "q": 1‑D array_like  # linear term
            "G": 2‑D array_like  # inequality matrix
            "h": 1‑D array_like  # inequality rhs
            "A": 2‑D array_like  # equality matrix
            "b": 1‑D array_like  # equality rhs
        }

    It returns a dictionary::

        {"solution": list, "objective": float}
    """

    def __init__(self) -> None:
        # we keep a cache of variables of a given size to avoid repeated
        # allocation; a simple dict keyed by dimension.
        self._var_cache: Dict[int, cp.Variable] = {}

    def _get_variable(self, n: int) -> cp.Variable:
        if n not in self._var_cache:
            self._var_cache[n] = _create_variable(n)
        return self._var_cache[n]

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        # Convert inputs to np.ndarray; ``copy=False`` prevents needless copies
        P = np.asarray(problem["P"], dtype=float, copy=False)
        q = np.asarray(problem["q"], dtype=float, copy=False)
        G = np.asarray(problem["G"], dtype=float, copy=False)
        h = np.asarray(problem["h"], dtype=float, copy=False)
        A = np.asarray(problem["A"], dtype=float, copy=False)
        b = np.asarray(problem["b"], dtype=float, copy=False)

        n = P.shape[0]
        # enforce symmetry explicitly on the left-hand side of the quadratic form
        P = (P + P.T) * 0.5

        # Initialise variable
        x = self._get_variable(n)

        # Build objective: use psd_wrap to keep the matrix fixed and symmetric
        objective = 0.5 * cp.quad_form(x, cp.psd_wrap(P)) + q @ x

        # Build constraints; use list comprehensions for speed
        constraints = [G @ x <= h, A @ x == b]

        # Solve the problem
        prob = cp.Problem(cp.Minimize(objective), constraints)
        # OSQP is our choice of solver; set strict tolerances and silence output
        optimal_value: float = prob.solve(
            solver=cp.OSQP,
            verbose=False,
            eps_abs=1e-8,
            eps_rel=1e-8,
        )

        # Sanity check
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise ValueError(f"Solver failed (status={prob.status})")

        # Return solution and objective
        return {
            "solution": x.value.tolist(),
            "objective": float(optimal_value),
        }