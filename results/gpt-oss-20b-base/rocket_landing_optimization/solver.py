import numpy as np
import cvxpy as cp
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        # Convert inputs to numpy arrays / scalars
        p0 = np.asarray(problem["p0"])
        v0 = np.asarray(problem["v0"])
        p_target = np.asarray(problem["p_target"])
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

        # Constraints
        cons: List[cp.Expression] = [
            V[0] == v0,
            P[0] == p0,
            V[K] == np.zeros(3),
            P[K] == p_target,
            P[:, 2] >= 0,
            V[1:, :2] == V[:-1, :2] + h * (F[:, :2] / m),
            V[1:, 2] == V[:-1, 2] + h * (F[:, 2] / m - g),
            P[1:] == P[:-1] + h / 2 * (V[:-1] + V[1:]),
            cp.norm(F, 2, axis=1) <= F_max,
        ]

        # Objective: minimise fuel consumption
        fuel = gamma * cp.sum(cp.norm(F, axis=1))
        prob = cp.Problem(cp.Minimize(fuel), cons)

        # Solve
        try:
            prob.solve(solver=cp.OSQP, warm_start=True, max_iter=50000)
        except Exception:
            return {"position": [], "velocity": [], "thrust": []}

        # Result handling
        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or P.value is None:
            return {"position": [], "velocity": [], "thrust": []}

        return {
            "position": P.value.tolist(),
            "velocity": V.value.tolist(),
            "thrust": F.value.tolist(),
            "fuel_consumption": float(prob.value),
        }