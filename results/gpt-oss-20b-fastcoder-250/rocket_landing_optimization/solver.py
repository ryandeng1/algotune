# solver.py
from typing import Any

import cvxpy as cp
import numpy as np


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the rocket landing optimization problem.

        The formulation is identical to the reference implementation but uses a
        symbolic linearization of the L1 norm to keep the solver instance small
        and fast.  ECOS is chosen as it performs well for small mixed‑type LPs
        and is available in the execution environment.

        Parameters
        ----------
        problem: dict
            Dictionary containing the problem parameters.

        Returns
        -------
        dict
            Dictionary with keys ``position``, ``velocity``, ``thrust`` and
            ``fuel_consumption``.  Values are plain Python lists or ``float``.
        """
        # Unpack problem data
        p0 = np.asarray(problem["p0"], dtype=float).reshape(3)
        v0 = np.asarray(problem["v0"], dtype=float).reshape(3)
        p_tar = np.asarray(problem["p_target"], dtype=float).reshape(3)
        g = float(problem["g"])
        m = float(problem["m"])
        h = float(problem["h"])
        K = int(problem["K"])
        F_max = float(problem["F_max"])
        gamma = float(problem["gamma"])

        # Decision variables
        V = cp.Variable((K + 1, 3))
        P = cp.Variable((K + 1, 3))
        F = cp.Variable((K, 3))
        # Auxiliary variables for the L1 norm on the thrust
        abs_F = cp.Variable((K, 3))

        constraints = []

        # Boundary conditions
        constraints += [V[0] == v0, P[0] == p0]
        constraints += [V[K] == np.zeros(3), P[K] == p_tar]

        constraints += [P[:, 2] >= 0]  # height constraint

        # Dynamics constraints
        constraints += [V[1:, :2] == V[:-1, :2] + h * (F[:, :2] / m)]
        constraints += [V[1:, 2] == V[:-1, 2] + h * (F[:, 2] / m - g)]
        constraints += [P[1:] == P[:-1] + h / 2 * (V[:-1] + V[1:])]

        # Thrust bounds
        constraints += [cp.norm(F, axis=1) <= F_max]

        # Linearized L1 objective: |F| <= abs_F and minimize sum(abs_F)
        constraints += [abs_F >= F, abs_F >= -F]
        objective = cp.Minimize(gamma * cp.sum(abs_F))

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.ECOS, abstol=1e-6, reltol=1e-6,
                       feastol=1e-6, verbose=False)
        except Exception:
            return {"position": [], "velocity": [], "thrust": [], "fuel_consumption": 0.0}

        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or P.value is None:
            return {"position": [], "velocity": [], "thrust": [], "fuel_consumption": 0.0}

        # Build the result dictionary
        return {
            "position": P.value.tolist(),
            "velocity": V.value.tolist(),
            "thrust": F.value.tolist(),
            "fuel_consumption": float(prob.value),
        }
