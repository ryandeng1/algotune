# solver.py
from __future__ import annotations

from typing import Any, Dict, List

import pulp
import numpy as np


class Solver:
    """
    Capacitated Facility Location solver.
    Uses pulp's CBC MILP solver for fast solution.
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        # Extract data
        fixed_costs = np.asarray(problem["fixed_costs"], dtype=float)
        capacities = np.asarray(problem["capacities"], dtype=float)
        demands = np.asarray(problem["demands"], dtype=float)
        costs = np.asarray(problem["transportation_costs"], dtype=float)

        n_fac = len(fixed_costs)
        n_cus = len(demands)

        # Create MILP
        m = pulp.LpProblem("CFLP", pulp.LpMinimize)

        # Variables
        y = pulp.LpVariable.dicts("y", range(n_fac), lowBound=0, upBound=1, cat="Binary")
        x = {
            (i, j): pulp.LpVariable(f"x_{i}_{j}", lowBound=0, upBound=1, cat="Binary")
            for i in range(n_fac)
            for j in range(n_cus)
        }

        # Objective
        m += (
            pulp.lpSum(fixed_costs[i] * y[i] for i in range(n_fac))
            + pulp.lpSum(costs[i, j] * x[i, j] for i in range(n_fac) for j in range(n_cus))
        )

        # Constraints
        # each customer served exactly once
        for j in range(n_cus):
            m += pulp.lpSum(x[i, j] for i in range(n_fac)) == 1

        # facility capacity
        for i in range(n_fac):
            m += pulp.lpSum(demands[j] * x[i, j] for j in range(n_cus)) <= capacities[i] * y[i]
            # link x <= y
            for j in range(n_cus):
                m += x[i, j] <= y[i]

        # Solve
        m.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=10))

        status = pulp.LpStatus.get(m.status, m.status)
        if status not in ("Optimal", "Optimal (Integer)"):
            # return infeasible placeholder
            return {
                "objective_value": float("inf"),
                "facility_status": [False] * n_fac,
                "assignments": [[0.0] * n_cus for _ in range(n_fac)],
            }

        # Extract solution
        y_vals = np.array([bool(pulp.value(y[i])) for i in range(n_fac)], dtype=int)
        x_vals = np.array([[float(pulp.value(x[i, j])) for j in range(n_cus)] for i in range(n_fac)], dtype=int)

        # Build output
        return {
            "objective_value": float(pulp.value(m.objective)),
            "facility_status": [bool(v) for v in y_vals],
            "assignments": x_vals.tolist(),
        }
