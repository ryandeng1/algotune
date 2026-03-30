# solver.py

from typing import Any, Dict, List
import cvxpy as cp
import numpy as np

class Solver:
    """
    Ad allocation optimizer using CVXPY with a linearised objective.
    The key performance improvement is to formulate the problem as a
    pure linear program (LP) instead of a DC program.  The LP can be
    solved extremely fast by the default ECOS solver.
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve the optimal advertising problem.

        Parameters
        ----------
        problem : dict
            Dictionary with keys
                P : 2‑D list or np.ndarray (m × n)  – probability matrix
                R : 1‑D list or np.ndarray (m,)    – revenue per click
                B : 1‑D list or np.ndarray (m,)    – budget cap per ad
                c : 1‑D list or np.ndarray (m,)    – minimum impressions per ad
                T : 1‑D list or np.ndarray (n,)    – total impressions per media

        Returns
        -------
        dict
            Solution status and values. Keys:
            status, optimal, displays, clicks,
            revenue_per_ad, total_revenue, objective_value.
            In case of failure the dict contains the error message.
        """
        try:
            # -------------------- 1️⃣  Pre‑process input --------------------
            P = np.asarray(problem["P"], dtype=np.float64)
            R = np.asarray(problem["R"], dtype=np.float64)
            B = np.asarray(problem["B"], dtype=np.float64)
            c = np.asarray(problem["c"], dtype=np.float64)
            T = np.asarray(problem["T"], dtype=np.float64)

            m, n = P.shape

            # -------------------- 2️⃣  Decision variables --------------------
            D = cp.Variable((m, n), nonneg=True)

            # -------------------- 3️⃣  Objective (LP) --------------------
            #  z_i <= R_i * (P_i · D_i)
            #  z_i <= B_i
            # maximize sum(z_i)
            z = cp.Variable(m)  # auxiliary revenue variables
            linear_expr = R[:, None] * P * D          # shape (m, n)
            rev_linear = cp.sum(linear_expr, axis=1)  # shape (m,)

            obj = cp.Maximize(cp.sum(z))

            constraints = [
                z <= rev_linear,
                z <= B,
                cp.sum(D, axis=0) <= T,
                cp.sum(D, axis=1) >= c
            ]

            prob = cp.Problem(obj, constraints)
            prob.solve(solver=cp.ECOS, verbose=False)

            # -------------------- 4️⃣  Post‑processing --------------------
            if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                return {"status": prob.status, "optimal": False}

            D_val = D.value
            # Compute clicks vector: P_i · D_i
            clicks = np.sum(P * D_val, axis=1)          # shape (m,)
            # Revenue per ad: min(R_i * clicks_i , B_i)
            revenue_per_ad = np.minimum(R * clicks, B)

            return {
                "status": prob.status,
                "optimal": True,
                "displays": D_val.tolist(),
                "clicks": clicks.tolist(),
                "revenue_per_ad": revenue_per_ad.tolist(),
                "total_revenue": float(revenue_per_ad.sum()),
                "objective_value": float(prob.value),
            }

        except cp.SolverError as e:   # pragma: no cover
            return {"status": "solver_error", "optimal": False, "error": str(e)}
        except Exception as e:        # pragma: no cover
            return {"status": "error", "optimal": False, "error": str(e)}