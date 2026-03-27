from typing import Any
import cvxpy as cp
import numpy as np


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the rocket landing optimization problem using CVXPY.

        :param problem: Dictionary with problem parameters
        :return: Dictionary with position, velocity, and thrust trajectories
        """
        # Extract problem parameters and convert to NumPy types
        p0 = np.asarray(problem["p0"], dtype=float)
        v0 = np.asarray(problem["v0"], dtype=float)
        p_target = np.asarray(problem["p_target"], dtype=float)
        g = float(problem["g"])
        m = float(problem["m"])
        h = float(problem["h"])
        K = int(problem["K"])
        F_max = float(problem["F_max"])
        gamma = float(problem["gamma"])

        # Variables
        V = cp.Variable((K + 1, 3))  # Velocity
        P = cp.Variable((K + 1, 3))  # Position
        F = cp.Variable((K, 3))      # Thrust

        # Constraints
        constraints = []

        # Initial and terminal conditions
        constraints.append(V[0] == v0)
        constraints.append(P[0] == p0)
        constraints.append(V[K] == np.zeros(3))
        constraints.append(P[K] == p_target)

        # Height constraint (always positive)
        constraints.append(P[:, 2] >= 0)

        # Dynamics for velocity
        constraints.append(V[1:, :2] == V[:-1, :2] + h * (F[:, :2] / m))
        constraints.append(V[1:, 2] == V[:-1, 2] + h * (F[:, 2] / m - g))

        # Dynamics for position
        constraints.append(P[1:] == P[:-1] + h / 2 * (V[:-1] + V[1:]))

        # Maximum thrust constraint
        constraints.append(cp.norm(F, 2, axis=1) <= F_max)

        # Objective: minimize fuel consumption
        fuel_consumption = gamma * cp.sum(cp.norm(F, axis=1))
        objective = cp.Minimize(fuel_consumption)

        # Solve the problem
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve()  # default solver selection
        except (cp.SolverError, Exception):
            return {"position": [], "velocity": [], "thrust": []}

        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or P.value is None:
            return {"position": [], "velocity": [], "thrust": []}

        # Return solution
        return {
            "position": P.value.tolist(),
            "velocity": V.value.tolist(),
            "thrust": F.value.tolist(),
            "fuel_consumption": float(prob.value),
        }