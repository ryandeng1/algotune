from typing import Any, Dict
import numpy as np
import cvxpy as cp


class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve the optimal advertising problem using CVXPY.

        Parameters
        ----------
        problem : dict
            Dictionary with problem parameters.

        Returns
        -------
        dict
            Dictionary containing the solution status, optimal flag,
            displays matrix (list), clicks per ad, revenue per ad,
            total revenue, and the objective value.
        """
        # Convert inputs to numpy arrays (float) for vectorised arithmetic
        P = np.asarray(problem["P"], dtype=float)
        R = np.asarray(problem["R"], dtype=float)
        B = np.asarray(problem["B"], dtype=float)
        c = np.asarray(problem["c"], dtype=float)
        T = np.asarray(problem["T"], dtype=float)

        m, n = P.shape

        # Decision variable : displays of ad i in time slot j
        D = cp.Variable((m, n), nonneg=True)

        # Total clicks for each ad (dot product of P and displays)
        clicks = cp.sum(cp.multiply(P, D), axis=1)  # shape (m,)

        # Revenue for each ad (capped by budget)
        revenue_per_ad = cp.minimum(R * clicks, B)

        # Objective: maximise total revenue
        objective = cp.Maximize(cp.sum(revenue_per_ad))

        # Constraints
        constraints = [
            cp.sum(D, axis=0) <= T,  # traffic capacity per time slot
            cp.sum(D, axis=1) >= c,  # minimum display requirement per ad
        ]

        prob = cp.Problem(objective, constraints)

        try:
            prob.solve()
            if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return {"status": prob.status, "optimal": False}

            D_val = D.value
            clicks_val = np.sum(P * D_val, axis=1)

            revenue_val = np.minimum(R * clicks_val, B)

            return {
                "status": prob.status,
                "optimal": True,
                "displays": D_val.tolist(),
                "clicks": clicks_val.tolist(),
                "revenue_per_ad": revenue_val.tolist(),
                "total_revenue": float(np.sum(revenue_val)),
                "objective_value": float(prob.value),
            }

        except cp.SolverError as e:
            return {"status": "solver_error", "optimal": False, "error": str(e)}
        except Exception as e:
            return {"status": "error", "optimal": False, "error": str(e)}