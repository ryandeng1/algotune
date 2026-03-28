from typing import Any
import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the rocket landing optimization problem using CVXPY.
        """
        # Fast numpy conversions
        p0 = np.asarray(problem['p0'], dtype=float)
        v0 = np.asarray(problem['v0'], dtype=float)
        p_target = np.asarray(problem['p_target'], dtype=float)
        g = float(problem['g'])
        m = float(problem['m'])
        h = float(problem['h'])
        K = int(problem['K'])
        F_max = float(problem['F_max'])
        gamma = float(problem['gamma'])

        # Decision variables
        V = cp.Variable((K + 1, 3))
        P = cp.Variable((K + 1, 3))
        F = cp.Variable((K, 3))

        # Constraints
        constraints = [
            V[0] == v0,
            P[0] == p0,
            V[K] == 0,
            P[K] == p_target,
            P[:, 2] >= 0,
            V[1:, :2] == V[:-1, :2] + h * (F[:, :2] / m),
            V[1:, 2] == V[:-1, 2] + h * (F[:, 2] / m - g),
            P[1:] == P[:-1] + h / 2 * (V[:-1] + V[1:]),
            cp.norm(F, 2, axis=1) <= F_max,
        ]

        # Objective: minimize fuel consumption
        fuel_consumption = gamma * cp.sum(cp.norm(F, axis=1))
        objective = cp.Minimize(fuel_consumption)

        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.ECOS, verbose=False)
        except Exception:  # noqa: BLE001
            return {'position': [], 'velocity': [], 'thrust': [], 'fuel_consumption': None}

        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or P.value is None:
            return {'position': [], 'velocity': [], 'thrust': [], 'fuel_consumption': None}

        return {
            'position': P.value.tolist(),
            'velocity': V.value.tolist(),
            'thrust': F.value.tolist(),
            'fuel_consumption': float(prob.value)
        }