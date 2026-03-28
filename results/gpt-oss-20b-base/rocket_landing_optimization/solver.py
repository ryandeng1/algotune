import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem: dict[str, any]) -> dict[str, any]:
        # Extract problem data
        p0 = np.asarray(problem["p0"], dtype=float)
        v0 = np.asarray(problem["v0"], dtype=float)
        p_target = np.asarray(problem["p_target"], dtype=float)
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
        constraints = [
            V[0] == v0,
            P[0] == p0,
            V[K] == np.zeros(3),
            P[K] == p_target,
            P[:, 2] >= 0,
            V[1:, :2] == V[:-1, :2] + h * (F[:, :2] / m),
            V[1:, 2] == V[:-1, 2] + h * (F[:, 2] / m - g),
            P[1:] == P[:-1] + h / 2 * (V[:-1] + V[1:]),
            cp.norm(F, 2, axis=1) <= F_max
        ]

        # Objective
        fuel_consumption = gamma * cp.sum(cp.norm(F, axis=1))
        objective = cp.Minimize(fuel_consumption)

        # Solve using OSQP (fast for convex problems with linear constraints)
        try:
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-4, eps_rel=1e-4)
        except Exception:
            return {"position": [], "velocity": [], "thrust": [], "fuel_consumption": None}

        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or P.value is None:
            return {"position": [], "velocity": [], "thrust": [], "fuel_consumption": None}

        return {
            "position": P.value.tolist(),
            "velocity": V.value.tolist(),
            "thrust": F.value.tolist(),
            "fuel_consumption": float(prob.value)
        }