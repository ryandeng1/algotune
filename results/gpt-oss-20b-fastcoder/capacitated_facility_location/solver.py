# solver.py
from __future__ import annotations
from typing import Any, Dict, List

import cvxpy as cp
import numpy as np


class Solver:
    """Fast Capacitated Facility Location solver (uses CVXPY + HIGHS)."""

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parameters
        ----------
        problem : dict
            Dictionary containing:
                * fixed_costs   : 1‑D array of facility opening costs
                * capacities   : 1‑D array of facility capacities
                * demands      : 1‑D array of customer demands
                * transportation_costs : 2‑D array of transport costs
                                      (facilities × customers)

        Returns
        -------
        dict
            * objective_value : optimum value
            * facility_status : list of booleans for each facility
            * assignments     : list of lists (N_fac × N_cust) of 0/1 assignments
        """

        # Constants
        fixed_costs = np.array(problem["fixed_costs"], dtype=np.float64)
        capacities = np.array(problem["capacities"], dtype=np.float64)
        demands = np.array(problem["demands"], dtype=np.float64)
        transport = np.array(problem["transportation_costs"], dtype=np.float64)

        n_fac, n_cust = fixed_costs.shape[0], demands.shape[0]

        # Decision variables
        y = cp.Variable(n_fac, boolean=True)
        x = cp.Variable((n_fac, n_cust), boolean=True)

        # Objective
        obj = cp.Minimize(fixed_costs @ y + cp.sum(cp.multiply(transport, x)))

        # Constraints – vectorised for speed
        constraints = [
            cp.sum(x, axis=0) == 1,                # each customer served exactly once
            demands @ x.T <= capacities[:, None] * y,  # capacity limits
            x <= cp.reshape(y, (-1, 1)),          # customers can only be served from open facilities
        ]

        prob = cp.Problem(obj, constraints)
        try:
            prob.solve(solver=cp.HIGHS, verbose=False)
        except Exception:
            # Infeasible / abort – return defaults
            return {
                "objective_value": float("inf"),
                "facility_status": [False] * n_fac,
                "assignments": [[0.0] * n_cust for _ in range(n_fac)],
            }

        # Handle solver status
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return {
                "objective_value": float("inf"),
                "facility_status": [False] * n_fac,
                "assignments": [[0.0] * n_cust for _ in range(n_fac)],
            }

        # Rounding and output conversion
        y_vals = np.rint(np.clip(y.value, 0, 1)).astype(int)
        x_vals = np.rint(np.clip(x.value, 0, 1)).astype(int)

        return {
            "objective_value": float(prob.value),
            "facility_status": y_vals.astype(bool).tolist(),
            "assignments": x_vals.tolist(),
        }