# solver.py
from typing import Any
import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Solve the rocket landing optimisation using a fast quadratic solver."""
        # --- Unpack problem parameters ------------------------------------
        p0 = np.array(problem["p0"], dtype=np.float64)
        v0 = np.array(problem["v0"], dtype=np.float64)
        p_target = np.array(problem["p_target"], dtype=np.float64)
        g = float(problem["g"])
        m = float(problem["m"])
        h = float(problem["h"])
        K = int(problem["K"])
        F_max = float(problem["F_max"])
        gamma = float(problem["gamma"])

        # --- Variables ----------------------------------------------------
        V = cp.Variable((K + 1, 3))   # Velocity
        P = cp.Variable((K + 1, 3))   # Position
        F = cp.Variable((K, 3))      # Thrust

        # --- Constraints -----------------------------------------------
        constraints = []

        # Initial conditions
        constraints.append(V[0] == v0)
        constraints.append(P[0] == p0)

        # Terminal conditions
        constraints.append(V[K] == 0)
        constraints.append(P[K] == p_target)

        # Height constraint
        constraints.append(P[:, 2] >= 0)

        # Velocity dynamics
        constraints.append(V[1:, :2] == V[:-1, :2] + h * (F[:, :2] / m))
        constraints.append(V[1:, 2]  == V[:-1, 2]  + h * (F[:, 2]  / m - g))

        # Position dynamics
        constraints.append(P[1:] == P[:-1] + 0.5 * h * (V[:-1] + V[1:]))

        # Max thrust
        constraints.append(cp.norm(F, axis=1) <= F_max)

        # --- Objective --------------------------------------------------
        obj = cp.Minimize(gamma * cp.sum(cp.norm(F, axis=1)))

        # --- Problem & Solver --------------------------------------------
        prob = cp.Problem(obj, constraints)

        try:
            prob.solve(solver=cp.OSQP,
                       eps_abs=1e-9,
                       eps_rel=1e-9,
                       max_iter=50000,
                       warm_start=True)
        except Exception:
            return {"position": [], "velocity": [], "thrust": []}

        if prob.status not in {"optimal", "optimal_inaccurate"} or V.value is None:
            return {"position": [], "velocity": [], "thrust": []}

        # --- Build solution ----------------------------------------------
        return {
            "position": P.value.tolist(),
            "velocity": V.value.tolist(),
            "thrust": F.value.tolist(),
            "fuel_consumption": float(prob.value)
        }
