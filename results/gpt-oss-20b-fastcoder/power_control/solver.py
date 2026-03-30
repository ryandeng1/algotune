# solver.py

import numpy as np
from scipy.optimize import linprog
from typing import Any, Dict, List


class Solver:
    """
    Fast linear‑programming solver using SciPy's `linprog` (HiGHS backend).
    The problem is a linear program derived from the original CVXPY model.
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parameters
        ----------
        problem : dict
            Dictionary containing the following keys:
            - 'G'   : (n × n) complex or real matrix per generator
            - 'σ'   : (n,) array of susceptances
            - 'P_min': (n,) lower bounds on power
            - 'P_max': (n,) upper bounds on power
            - 'S_min': scalar constraint parameter

        Returns
        -------
        dict
            Dictionary with keys:
            - 'P'        : list of optimal power values
            - 'objective': optimal objective value (sum of powers)
        """
        # Convert inputs to NumPy arrays of float type
        G = np.asarray(problem["G"], dtype=float)
        sigma = np.asarray(problem["σ"], dtype=float)
        P_min = np.asarray(problem["P_min"], dtype=float)
        P_max = np.asarray(problem["P_max"], dtype=float)
        S_min = float(problem["S_min"])

        n = G.shape[0]

        # Objective: minimize sum(P) -> c = ones
        c = np.ones(n, dtype=float)

        # bounds for each variable: (P_min[i], P_max[i])
        bounds: List[tuple[float, float]] = [(P_min[i], P_max[i]) for i in range(n)]

        # Build inequality constraints from
        #   Gii*(1+S_min)*Pi - S_min * Σj G[i,j]*Pj >= S_min*sigma[i]
        # Convert to A_ub * x <= b_ub form:
        rows = []
        rhs = []

        one_plus_Smin = 1.0 + S_min
        for i in range(n):
            a_i = G[i, i] * one_plus_Smin
            b_i = S_min * sigma[i]

            # coefficient for Pi
            row = np.empty(n, dtype=float)
            row[:] = S_min * G[i, :]          # S_min * G[i, j] for all j
            row[i] -= a_i                    # subtract a_i for j==i to get:
                                            # -a_i * Pi + S_min*sum_{j} G[i,j] Pj
            rows.append(row)
            rhs.append(-b_i)                 # multiply inequality by -1

        A_ub = np.vstack(rows)
        b_ub = np.array(rhs, dtype=float)

        # Solve with HiGHS for speed and robustness
        res = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method="highs",
            options={"presolve": True},
        )

        if not res.success:
            raise ValueError(f"Optimization failed: {res.message}")

        return {"P": res.x.tolist(), "objective": float(res.fun)}