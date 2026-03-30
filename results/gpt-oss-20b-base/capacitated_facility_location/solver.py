# solver.py
# Optimised Solver for the Capacitated Facility Location Problem
# -------------------------------------------------------------------
# Python 3.10, CVXPY (HIGHS solver) with vectorised constraints.
# -------------------------------------------------------------------

from __future__ import annotations
from typing import Any, Dict, List

import cvxpy as cp
import numpy as np


class Solver:
    """
    Optimised solver for the Capacitated Facility Location Problem.
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parameters
        ----------
        problem : dict
            * 'fixed_costs'   : list/array of length #facilities
            * 'capacities'    : list/array of length #facilities
            * 'demands'       : list/array of length #customers
            * 'transportation_costs' : 2‑D array (#facilities × #customers)

        Returns
        -------
        dict
            * 'objective_value' : float
            * 'facility_status' : list[bool]
            * 'assignments'     : List[List[float]]
        """
        # ------------------------------------------------------------------
        # 1. Convert inputs to numpy arrays
        # ------------------------------------------------------------------
        fixed_costs = np.asarray(problem["fixed_costs"], dtype=float)
        capacities = np.asarray(problem["capacities"], dtype=float)
        demands = np.asarray(problem["demands"], dtype=float)
        transport_costs = np.asarray(problem["transportation_costs"], dtype=float)

        n_fac = fixed_costs.size
        n_cus = demanded_size = demands.size

        # ------------------------------------------------------------------
        # 2. CVXPY variables
        # ------------------------------------------------------------------
        y = cp.Variable(n_fac, boolean=True)                      # facility open
        x = cp.Variable((n_fac, n_cus), boolean=True)            # allocations

        # ------------------------------------------------------------------
        # 3. Objective
        # ------------------------------------------------------------------
        obj = cp.Minimize(fixed_costs @ y + cp.sum(cp.multiply(transport_costs, x)))

        # ------------------------------------------------------------------
        # 4. Constraints – fully vectorised
        # ------------------------------------------------------------------
        constraints = []

        # 4.1 Every customer served exactly once
        constraints.append(cp.sum(x, axis=0) == 1)

        # 4.2 Capacity constraints
        #     sum_j d_j * x_{ij} <= cap_i * y_i  for all i
        constraints.append(cp.sum(cp.multiply(x, demands), axis=1) <= capacities * y)

        # 4.3 Allocations only to open facilities
        #     x_{ij} <= y_i  ->  x <= y[:,None]
        constraints.append(x <= y[:, None])

        # ------------------------------------------------------------------
        # 5. Solve
        # ------------------------------------------------------------------
        prob = cp.Problem(obj, constraints)

        try:
            prob.solve(solver=cp.HIGHS, verbose=False)
        except Exception:
            # In case of any solver error return infeasible result
            return {
                "objective_value": float("inf"),
                "facility_status": [False] * n_fac,
                "assignments": [[0.0] * n_cus for _ in range(n_fac)],
            }

        # ------------------------------------------------------------------
        # 6. Check optimality
        # ------------------------------------------------------------------
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return {
                "objective_value": float("inf"),
                "facility_status": [False] * n_fac,
                "assignments": [[0.0] * n_cus for _ in range(n_fac)],
            }

        # ------------------------------------------------------------------
        # 7. Post‑processing – round to binary
        # ------------------------------------------------------------------
        y_vals = np.clip(np.round(y.value), 0, 1).astype(int)
        x_vals = np.clip(np.round(x.value), 0, 1).astype(int)

        facility_status: List[bool] = [bool(v) for v in y_vals.tolist()]
        assignments: List[List[int]] = x_vals.tolist()

        # ------------------------------------------------------------------
        # 8. Return results
        # ------------------------------------------------------------------
        return {
            "objective_value": float(prob.value),
            "facility_status": facility_status,
            "assignments": assignments,
        }