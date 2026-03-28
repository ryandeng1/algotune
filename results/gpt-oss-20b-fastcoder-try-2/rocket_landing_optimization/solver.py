from typing import Any
import numpy as np
import cvxpy as cp

class Solver:
    """
    The solver has been rewritten to avoid unnecessary Python
    overhead and to exploit CVXPY's OSQP backend for high performance.
    """

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        # Convert inputs to NumPy arrays for fast access
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
            V[0] == v0,          # initial velocity
            P[0] == p0,          # initial position
            V[K] == 0,           # final velocity zero
            P[K] == p_target,    # final position
            P[:, 2] >= 0,        # altitude non‑negative
            V[1:, :2] == V[:-1, :2] + h * (F[:, :2] / m),   # horizontal dynamics
            V[1:, 2] == V[:-1, 2] + h * (F[:, 2] / m - g),   # vertical dynamics
            P[1:] == P[:-1] + h / 2 * (V[:-1] + V[1:]),     # trapezoidal integration
            cp.norm(F, 2, axis=1) <= F_max,  # thrust limits
        ]

        # Objective: minimise fuel consumption
        fuel_consumption = gamma * cp.sum(cp.norm(F, axis=1))
        objective = cp.Minimize(fuel_consumption)

        # Build and solve the problem using OSQP (fast QP solver)
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.OSQP, eps_abs=1e-8, eps_rel=1e-8, max_iter=20000, verbose=False)
        except Exception:
            return {"position": [], "velocity": [], "thrust": [], "fuel_consumption": None}

        # Check solution status
        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            return {"position": [], "velocity": [], "thrust": [], "fuel_consumption": None}

        # Prepare output
        return {
            "position": P.value.tolist(),
            "velocity": V.value.tolist(),
            "thrust": F.value.tolist(),
            "fuel_consumption": float(prob.value),
        }