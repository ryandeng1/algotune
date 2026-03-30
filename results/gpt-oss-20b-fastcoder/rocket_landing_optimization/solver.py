# solver.py
"""
An aggressive CVXPY formulation for the rocket landing problem.
All computations are vectorised, and the problem is solved with
OSQP – a fast QP solver – which usually outperforms the default
solvers for this structure.
"""

from __future__ import annotations
from typing import Any

import cvxpy as cp
import numpy as np


class Solver:

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Fast CVXPY optimisation for the rocket landing problem."""

        # ------------------------------------------------------------------
        # 1. Extract problem data as tensors for immediate use
        # ------------------------------------------------------------------
        p0 = np.asarray(problem["p0"], dtype=np.float64)
        v0 = np.asarray(problem["v0"], dtype=np.float64)
        p_target = np.asarray(problem["p_target"], dtype=np.float64)

        g = float(problem["g"])
        m = float(problem["m"])
        h = float(problem["h"])
        K = int(problem["K"])
        F_max = float(problem["F_max"])
        gamma = float(problem["gamma"])

        # ------------------------------------------------------------------
        # 2. CVXPY variables
        # ------------------------------------------------------------------
        V = cp.Variable((K + 1, 3), name="V")  # velocity
        P = cp.Variable((K + 1, 3), name="P")  # position
        F = cp.Variable((K, 3), name="F")      # thrust

        # ------------------------------------------------------------------
        # 3. Constraints – fully vectorised, no python loops
        # ------------------------------------------------------------------
        constraints = [
            V[0] == v0,
            P[0] == p0,
            V[K] == np.zeros(3, dtype=np.float64),
            P[K] == p_target,
            P[:, 2] >= 0,  # keep altitude non‑negative
            # Dynamics for horizontal velocity
            V[1:, :2] == V[:-1, :2] + h * (F[:, :2] / m),
            # Dynamics for vertical velocity (including gravity)
            V[1:, 2] == V[:-1, 2] + h * (F[:, 2] / m - g),
            # Trapezoidal integration to update positions
            P[1:] == P[:-1] + h / 2 * (V[:-1] + V[1:]),
            # Thrust magnitude limits
            cp.norm(F, 2, axis=1) <= F_max,
        ]

        # ------------------------------------------------------------------
        # 4. Objective – minimise fuel consumption
        # ------------------------------------------------------------------
        fuel_cost = gamma * cp.sum(cp.norm(F, axis=1))
        objective = cp.Minimize(fuel_cost)

        # ------------------------------------------------------------------
        # 5. Problem definition
        # ------------------------------------------------------------------
        prob = cp.Problem(objective, constraints)

        # ------------------------------------------------------------------
        # 6. Solve – explicit OSQP gives fast QP solving
        # ------------------------------------------------------------------
        try:
            prob.solve(
                solver=cp.OSQP,
                verbose=False,
                warm_start=True,
                max_iter=200000,       # generous but realistic limit
                eps_abs=1e-8,
                eps_rel=1e-8,
                alpha=1.8,             # over-relaxation parameter
            )
        except Exception:
            # Any failure – return empty solution
            return {"position": [], "velocity": [], "thrust": [], "fuel_consumption": None}

        # ------------------------------------------------------------------
        # 7. Validation & result extraction
        # ------------------------------------------------------------------
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or P.value is None:
            return {"position": [], "velocity": [], "thrust": [], "fuel_consumption": None}

        return {
            "position": P.value.tolist(),
            "velocity": V.value.tolist(),
            "thrust": F.value.tolist(),
            "fuel_consumption": float(prob.value),
        }