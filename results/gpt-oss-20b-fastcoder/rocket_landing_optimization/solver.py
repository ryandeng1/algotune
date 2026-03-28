from typing import Any
import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the rocket landing optimization problem using CVXPY.

        :param problem: Dictionary with problem parameters
        :return: Dictionary with position, velocity, and thrust trajectories
        """
        p0 = np.asarray(problem["p0"])
        v0 = np.asarray(problem["v0"])
        p_target = np.asarray(problem["p_target"])
        g = float(problem["g"])
        m = float(problem["m"])
        h = float(problem["h"])
        K = int(problem["K"])
        F_max = float(problem["F_max"])
        gamma = float(problem["gamma"])

        V = cp.Variable((K + 1, 3))
        P = cp.Variable((K + 1, 3))
        F = cp.Variable((K, 3))

        constraints = [
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

        objective = cp.Minimize(gamma * cp.sum(cp.norm(F, axis=1)))
        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.OSQP, verbose=False)
        except Exception:
            return {"position": [], "velocity": [], "thrust": [], "fuel_consumption": None}

        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or P.value is None:
            return {"position": [], "velocity": [], "thrust": [], "fuel_consumption": None}

        return {
            "position": P.value.tolist(),
            "velocity": V.value.tolist(),
            "thrust": F.value.tolist(),
            "fuel_consumption": float(prob.value),
        }