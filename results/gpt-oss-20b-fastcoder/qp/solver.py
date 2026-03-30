#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimized CPython solver for quadratic programs.

The implementation uses CVXPY with the OSQP backend for fast sparse QP solving.
All input matrices are converted to NumPy arrays once, the problem is kept
structure‑only and the solver is called with tight tolerances and no
verbosity.
"""

from __future__ import annotations

from typing import Any, Dict

import cvxpy as cp
import numpy as np


class Solver:
    """Quadratic programming solver wrapper.

    The solver expects a dictionary describing the QP in the following format::

        {
            'P': 2‑D array or matrix, symmetric positive‑semidefinite,
            'q': 1‑D array,
            'G': 2‑D array, inequality constraints G * x <= h,
            'h': 1‑D array,
            'A': 2‑D array, equality constraints A * x == b,
            'b': 1‑D array
        }

    It returns a dictionary containing the optimal decision vector and the value
    of the objective function.
    """

    def __init__(self) -> None:
        # Create a CVXPY problem template that can be reused.
        self._templates = {}

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Solve a quadratic program supplied in *problem*.

        Parameters
        ----------
        problem :
            Dictionary with keys 'P', 'q', 'G', 'h', 'A', 'b'.

        Returns
        -------
        dict
            A dictionary of the form {'solution': [...], 'objective': ...}
        """
        # Convert inputs once
        P = np.asarray(problem["P"], dtype=np.float64)
        q = np.asarray(problem["q"], dtype=np.float64)
        G = np.asarray(problem["G"], dtype=np.float64)
        h = np.asarray(problem["h"], dtype=np.float64)
        A = np.asarray(problem["A"], dtype=np.float64)
        b = np.asarray(problem["b"], dtype=np.float64)

        # Symmetrize P to guard against numerical noise
        P = (P + P.T) * 0.5

        # Check if we have already built a template for this shape
        key = (P.shape, G.shape, A.shape)
        if key not in self._templates:
            n = P.shape[0]
            x = cp.Variable(n)

            objective = 0.5 * cp.quad_form(x, cp.psd_wrap(P)) + q @ x
            constraints = [G @ x <= h, A @ x == b]

            prob = cp.Problem(cp.Minimize(objective), constraints)
            self._templates[key] = (x, prob)
        else:
            x, prob = self._templates[key]

        # Solve
        optimal_value = prob.solve(
            solver=cp.OSQP,
            eps_abs=1e-08,
            eps_rel=1e-08,
            verbose=False,
            warm_start=True
        )

        # Validate status
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise ValueError(f"Solver failed (status = {prob.status})")

        return {
            "solution": x.value.tolist(),
            "objective": float(optimal_value),
        }