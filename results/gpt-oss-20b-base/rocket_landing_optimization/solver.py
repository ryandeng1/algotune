# solver.py
# -*- coding: utf-8 -*-

"""
Optimizer for rocket landing problem.

This module defines a Solver class that uses CVXPY to solve a
second‑order cone program (SOCP).  The solver is configured to use the
`ECOS` backend with a moderate iteration limit to keep the runtime short
while still producing high quality solutions.

Author: OpenAI's ChatGPT (Python 3.10)
"""

from __future__ import annotations

from typing import Any, Dict, List

import cvxpy as cp
import numpy as np


class Solver:
    """
    Rocket landing optimisation solver.

    The solver builds an SOCP problem once for each call to :meth:`solve`
    and solves it using the ECOS solver.  The problem is defined purely
    in NumPy arrays and CVXPY variables, which keeps the construction
    stage fast and memory‑friendly.
    """

    @staticmethod
    def _build_problem(
        p0: np.ndarray,
        v0: np.ndarray,
        p_target: np.ndarray,
        g: float,
        m: float,
        h: float,
        K: int,
        F_max: float,
    ) -> cp.Problem:
        """
        Construct the CVXPY optimisation problem.

        Parameters are already converted to NumPy arrays.
        """
        # Variables
        V = cp.Variable((K + 1, 3), name="V")   # velocities
        P = cp.Variable((K + 1, 3), name="P")   # positions
        F = cp.Variable((K, 3), name="F")       # thrusts

        # Constraints
        constraints = [
            V[0] == v0,
            P[0] == p0,
            V[K] == 0.0,          # final velocity zero
            P[K] == p_target,      # target position
            P[:, 2] >= 0,          # altitude non‑negative
            V[1:, :2] == V[:-1, :2] + h * (F[:, :2] / m),  # horizontal dynamics
            V[1:, 2] == V[:-1, 2] + h * (F[:, 2] / m - g),  # vertical dynamics
            P[1:] == P[:-1] + h / 2 * (V[:-1] + V[1:]),   # trapezoidal integration
            cp.norm(F, 2, axis=1) <= F_max,                # thrust magnitude limits
        ]

        # Objective: minimise fuel consumption
        objective = cp.Minimize(cp.sum(cp.norm(F, 2, axis=1)))

        return cp.Problem(objective, constraints, V=V, P=P, F=F)

    @staticmethod
    def _prepare_data(problem: Dict[str, Any]) -> np.ndarray:
        """
        Helper that extracts and casts all problem data into NumPy arrays.
        """
        return np.array(problem["p0"], dtype=float)

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve the rocket landing optimisation problem.

        Parameters
        ----------
        problem : dict
            Problem parameters.  Must contain keys:
            ``p0``, ``v0``, ``p_target``, ``g``, ``m``, ``h``, ``K``,
            ``F_max`` and ``gamma``.

        Returns
        -------
        dict
            Solution dictionary containing
            ``position``, ``velocity``, ``thrust`` and ``fuel_consumption``.
        """
        # Input conversion
        p0 = np.array(problem["p0"], dtype=float)
        v0 = np.array(problem["v0"], dtype=float)
        p_target = np.array(problem["p_target"], dtype=float)
        g = float(problem["g"])
        m = float(problem["m"])
        h = float(problem["h"])
        K = int(problem["K"])
        F_max = float(problem["F_max"])
        gamma = float(problem["gamma"])

        # Build problem
        prob = self._build_problem(p0, v0, p_target, g, m, h, K, F_max)

        # ECOS settings (iteration limit to speed up)
        try:
            prob.solve(
                solver=cp.ECOS,
                verbose=False,
                max_iters=2500,
                eps_abs=1e-9,
                eps_rel=1e-9,
                warm_start=True,
            )
        except (cp.SolverError, Exception):
            return {"position": [], "velocity": [], "thrust": [], "fuel_consumption": None}

        # Check solution status
        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            return {"position": [], "velocity": [], "thrust": [], "fuel_consumption": None}

        # Retrieve variables
        P_val = prob.variables()[0].value  # P
        V_val = prob.variables()[1].value  # V
        F_val = prob.variables()[2].value  # F

        # Multiply objective by gamma (fuel consumption)
        fuel = gamma * float(prob.value)

        return {
            "position": P_val.tolist(),
            "velocity": V_val.tolist(),
            "thrust": F_val.tolist(),
            "fuel_consumption": fuel,
        }