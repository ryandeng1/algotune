from typing import Any
import cvxpy as cp
import numpy as np


class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Solve the optimal advertising problem using CVXPY.

        :param problem: Dictionary with problem parameters
        :return: Dictionary with optimal displays and revenue
        """
        # Extract problem parameters
        P = np.array(problem["P"])
        R = np.array(problem["R"])
        B = np.array(problem["B"])
        c = np.array(problem["c"])
        T = np.array(problem["T"])

        # Derive m and n from P matrix
        m, n = P.shape

        # Define variables
        D = cp.Variable((m, n))

        # Define objective: maximize total revenue using vectorized expressions
        clicks = cp.sum(cp.multiply(P, D), axis=1)
        revenue_per_ad = cp.minimum(R * clicks, B)
        total_revenue = cp.sum(revenue_per_ad)

        # Define constraints
        constraints = [
            D >= 0,  # Non-negative displays
            cp.sum(D, axis=0) <= T,  # Traffic capacity per time slot
            cp.sum(D, axis=1) >= c,  # Minimum display requirements
        ]

        # Define and solve the problem
        prob = cp.Problem(cp.Maximize(total_revenue), constraints)

        try:
            prob.solve()

            if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return {"status": prob.status, "optimal": False}

            # Calculate actual revenue using vectorized operations
            D_val = D.value
            clicks = np.sum(P * D_val, axis=1)
            revenue = np.minimum(R * clicks, B)

            # Return solution
            return {
                "status": prob.status,
                "optimal": True,
                "displays": D_val.tolist(),
                "clicks": clicks.tolist(),
                "revenue_per_ad": revenue.tolist(),
                "total_revenue": float(np.sum(revenue)),
                "objective_value": float(prob.value),
            }

        except cp.SolverError as e:
            return {"status": "solver_error", "optimal": False, "error": str(e)}
        except Exception as e:
            return {"status": "error", "optimal": False, "error": str(e)}