from typing import Any
import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Solve the optimal advertising problem using CVXPY.
        """
        # Convert problem constants to NumPy arrays once
        P = np.asarray(problem["P"], dtype=float)
        R = np.asarray(problem["R"], dtype=float)
        B = np.asarray(problem["B"], dtype=float)
        c = np.asarray(problem["c"], dtype=float)
        T = np.asarray(problem["T"], dtype=float)

        m, n = P.shape

        # Decision variables
        D = cp.Variable((m, n))

        # Revenue per ad: min(R_i * (P_i @ D_i), B_i)
        revenue_per_ad = []
        for i in range(m):
            # P[i] @ D[i] computes expected clicks
            clicks = P[i, :] @ D[i, :]
            revenue_per_ad.append(cp.minimum(R[i] * clicks, B[i]))

        total_revenue = cp.sum(revenue_per_ad)

        # Constraints
        constraints = [
            D >= 0,
            cp.sum(D, axis=0) <= T,
            cp.sum(D, axis=1) >= c,
        ]

        # Formulate and solve the problem
        prob = cp.Problem(cp.Maximize(total_revenue), constraints)
        try:
            # Choose a fast solver; ECOS_BB handles the integer part
            prob.solve(solver=cp.ECOS_BB, verbose=False, max_iters=5000)

            if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                return {"status": prob.status, "optimal": False}

            # Extract solution
            D_val = D.value
            clicks = (P @ D_val.T).T  # shape (m, n) -> (m,)
            revenue = np.minimum(R * clicks, B)

            result = {
                "status": prob.status,
                "optimal": True,
                "displays": D_val.tolist(),
                "clicks": clicks.tolist(),
                "revenue_per_ad": revenue.tolist(),
                "total_revenue": float(np.sum(revenue)),
                "objective_value": float(prob.value),
            }
            return result

        except cp.SolverError as e:
            return {"status": "solver_error", "optimal": False, "error": str(e)}
        except Exception as e:
            return {"status": "error", "optimal": False, "error": str(e)}