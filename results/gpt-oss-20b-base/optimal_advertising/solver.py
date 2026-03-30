# solver.py
import numpy as np
import cvxpy as cp

class Solver:
    """
    A highly‑optimised CVXPY solver for the optimal advertising problem.

    The problem is:

        maximize   Σ_i min( R[i] * (P[i] @ D[i]),  B[i] )
        subject to D >= 0,
                   Σ_i D[i, j]   ≤ T[j]          for all channels j
                   Σ_j D[i, j]   ≥ c[i]          for all ad slots i
    """

    # ------------------------------------------------------------------
    # Common optimisation parameters (change once per process)
    # ------------------------------------------------------------------
    _solve_kwargs = {"solver": cp.ECOS, "verbose": False, "warm_start": True}

    # ------------------------------------------------------------------
    def solve(self, problem: dict) -> dict:
        """
        Solve the problem defined by ``problem``.

        Parameters
        ----------
        problem : dict
            Expected keys: 'P', 'R', 'B', 'c', 'T',
            all entries must be convertible to NumPy arrays.

        Returns
        -------
        dict
            {
                'status': str,
                'optimal': bool,
                'displays': list,
                'clicks': list,
                'revenue_per_ad': list,
                'total_revenue': float,
                'objective_value': float
            }
        """
        # ------------------------------------------------------------------
        # Convert inputs to consistent NumPy arrays (fast, no deep copy)
        # ------------------------------------------------------------------
        P = np.asarray(problem["P"], dtype=np.float64)
        R = np.asarray(problem["R"], dtype=np.float64)
        B = np.asarray(problem["B"], dtype=np.float64)
        c = np.asarray(problem["c"], dtype=np.float64)
        T = np.asarray(problem["T"], dtype=np.float64)

        m, n = P.shape

        # ------------------------------------------------------------------
        # CVXPY variables
        # ------------------------------------------------------------------
        D = cp.Variable((m, n), nonneg=True)

        # ------------------------------------------------------------------
        # Constraints
        # ------------------------------------------------------------------
        constraints = [
            cp.sum(D, axis=0) <= T,  # channelwise budget
            cp.sum(D, axis=1) >= c   # minimum displays per ad slot
        ]

        # ------------------------------------------------------------------
        # Objective
        #   For each row i: min( R[i] * (P[i] @ D[i, :]), B[i] )
        # ------------------------------------------------------------------
        # Compute row‑wise dot products: R[i] * Σ_j P[i, j] * D[i, j]
        raw_revenue = R[:, None] * (P @ D.T)  # shape (m,)

        revenue_per_ad = cp.minimum(raw_revenue, B)
        objective = cp.Maximize(cp.sum(revenue_per_ad))

        # ------------------------------------------------------------------
        # Problem
        # ------------------------------------------------------------------
        prob = cp.Problem(objective, constraints)

        # ------------------------------------------------------------------
        # Solve
        # ------------------------------------------------------------------
        try:
            prob.solve(**self._solve_kwargs)
        except cp.SolverError as e:
            return {
                "status": "solver_error",
                "optimal": False,
                "error": str(e),
            }
        except Exception as e:
            return {
                "status": "error",
                "optimal": False,
                "error": str(e),
            }

        # ------------------------------------------------------------------
        # Collect results
        # ------------------------------------------------------------------
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return {"status": prob.status, "optimal": False}

        D_val = np.asarray(D.value, dtype=np.float64)

        # Compute clicks and revenue per ad (after solving)
        clicks = np.einsum("ij,ij->i", P, D_val)
        revenue = np.minimum(R * clicks, B)

        return {
            "status": prob.status,
            "optimal": True,
            "displays": D_val.tolist(),
            "clicks": clicks.tolist(),
            "revenue_per_ad": revenue.tolist(),
            "total_revenue": float(np.sum(revenue)),
            "objective_value": float(prob.value),
        }