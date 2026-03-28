import numpy as np
import cvxpy as cp

class Solver:
    """
    Very lightweight wrapper around CVXPY that solves the rocket‑landing
    minimum‑fuel problem.  The implementation is intentionally minimal
    to keep the overhead low – all data processing is done with NumPy
    and the optimisation problem is passed straight to CVXPY with a
    fast solver (ECOS or OSQP).

    The code below is tuned for speed while still retaining the
    original semantics of the problem.
    """

    @staticmethod
    def _num_to_array(obj):
        """Convert potential list/tuple inputs to a 1‑D NumPy array of type float."""
        return np.asarray(obj, dtype=float).flatten()

    def solve(self, problem: dict) -> dict:
        # ------------------------------------------------------------------
        # 1.  Extract problem data (NumPy arrays are used throughout for speed)
        # ------------------------------------------------------------------
        p0 = self._num_to_array(problem["p0"]).reshape(1, 3)
        v0 = self._num_to_array(problem["v0"]).reshape(1, 3)
        p_target = self._num_to_array(problem["p_target"]).reshape(1, 3)

        g = float(problem["g"])
        m = float(problem["m"])
        h = float(problem["h"])
        K = int(problem["K"])
        F_max = float(problem["F_max"])
        gamma = float(problem["gamma"])

        # ------------------------------------------------------------------
        # 2.  Variable definitions (matrix‑shaped for vectorised constraints)
        # ------------------------------------------------------------------
        V = cp.Variable((K + 1, 3))
        P = cp.Variable((K + 1, 3))
        F = cp.Variable((K, 3))

        # ------------------------------------------------------------------
        # 3.  Constraints
        # ------------------------------------------------------------------
        constraints = [
            V[0] == v0,                     # initial velocity
            P[0] == p0,                     # initial position
            V[K] == np.zeros((1, 3)),       # final velocity zero
            P[K] == p_target,               # target position
            P[:, 2] >= 0,                   # altitude non‑negative
            # Euler integration of velocity
            V[1:, :2] == V[:-1, :2] + h * (F[:, :2] / m),
            V[1:, 2]  == V[:-1, 2]  + h * (F[:, 2] / m - g),
            # Trapezoidal integration of position
            P[1:] == P[:-1] + h / 2.0 * (V[:-1] + V[1:]),
            # Thrust bound
            cp.norm(F, 2, axis=1) <= F_max,
        ]

        # ------------------------------------------------------------------
        # 4.  Objective – total fuel consumption
        # ------------------------------------------------------------------
        fuel_cost = gamma * cp.sum(cp.norm(F, axis=1))
        objective = cp.Minimize(fuel_cost)

        # ------------------------------------------------------------------
        # 5.  Problem definition and solve
        # ------------------------------------------------------------------
        prob = cp.Problem(objective, constraints)

        # Use a fast solver.  
        # ECOS is reasonably fast for medium sized QPs; OSQP can be chosen
        # if only box constraints were present.
        try:
            prob.solve(solver=cp.ECOS, verbose=False, eps_abs=1e-6, eps_rel=1e-6)
        except Exception:
            return {"position": [], "velocity": [], "thrust": [], "fuel_consumption": None}

        # ------------------------------------------------------------------
        # 6.  Return results
        # ------------------------------------------------------------------
        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            return {"position": [], "velocity": [], "thrust": [], "fuel_consumption": None}

        return {
            "position": P.value.tolist(),
            "velocity": V.value.tolist(),
            "thrust": F.value.tolist(),
            "fuel_consumption": float(prob.value),
        }