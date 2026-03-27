#!/usr/bin/env python3
"""
Optimised projection of a point onto a CVaR constraint set.

The implementation keeps the same public API – the function `Solver.solve`
takes a problem dictionary and returns a dictionary with the projected
point.  Internally the routine:
  * works purely with NumPy arrays (no repeated conversion between lists and
    NumPy).
  * uses CVXPY with the OSQP solver (which is fast for small–medium
    quadratic problems).
  * pre‑computes the number `k` of scenarios to trigger a CVaR constraint
    and the corresponding threshold `alpha` once.

This is a drop‑in replacement that should run noticeably faster than the
reference implementation on typical data sizes (hundreds of scenarios and
tens of variables).
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
from typing import Dict, List, Any


class Solver:
    """
    Wrapper around a CVXPY quadratic program that projects a point onto a
    CVaR‑constraint set.

    The problem is:
        minimize   ||x - x0||_2^2
        subject to sumLargest(Ax, k) <= alpha

    with:
        k     = floor((1 - beta) * ns)
        alpha = kappa * k
    """

    def __init__(self, beta: float = 0.9, kappa: float = 1.0) -> None:
        self.beta = beta
        self.kappa = kappa

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Compute the projection of x0 onto the CVaR set defined by the
        loss scenarios in `problem["loss_scenarios"]`.

        Parameters
        ----------
        problem : dict
            Must contain the keys:
                - "x0" : List[float] or np.ndarray
                - "loss_scenarios" : List[List[float]] or np.ndarray
            Optional keys:
                - "beta" : float  (use instance default if missing)
                - "kappa" : float (use instance default if missing)

        Returns
        -------
        dict
            A dictionary with the single key "x_proj" containing the
            projected point as a list of floats.  In case of an
            infeasible/unsolved problem an empty list is returned.
        """
        # --- 1. Input handling ---------------------------------------------
        x0 = np.asarray(problem["x0"], dtype=np.float64)
        A = np.asarray(problem["loss_scenarios"], dtype=np.float64)

        beta = float(problem.get("beta", self.beta))
        kappa = float(problem.get("kappa", self.kappa))

        ns, nd = A.shape

        # --- 2. Problem definition -----------------------------------------
        x = cp.Variable(nd)

        # Objective: half the squared Euclidean distance
        objective = cp.Minimize(cp.sum_squares(x - x0))

        k = int(np.floor((1 - beta) * ns))
        if k == 0:
            # The CVaR constraint is vacuous; return x0 directly.
            return {"x_proj": x0.tolist()}

        alpha = kappa * k
        constraints = [cp.sum_largest(A @ x, k) <= alpha]

        prob = cp.Problem(objective, constraints)

        # --- 3. Solve -------------------------------------------------------
        try:
            # OSQP is fast for quadratic problems and does not require a
            # problem conversion that CP depends on (at least for CP 1.2+).
            prob.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-8, eps_rel=1e-8)
        except cp.SolverError:
            return {"x_proj": []}
        except Exception:
            return {"x_proj": []}

        # --- 4. Result extraction -----------------------------------------
        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x.value is None:
            return {"x_proj": []}

        return {"x_proj": x.value.ravel().tolist()}