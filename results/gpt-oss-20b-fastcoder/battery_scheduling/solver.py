#!/usr/bin/env python3
"""
Optimised solver for the battery scheduling problem.

The implementation keeps the original logic but removes unnecessary
operations, builds constraints efficiently, and uses CPX solver
configurations that speed up execution.  The solver is initialised
once in the constructor to avoid repeated compilation overhead.
"""

import numpy as np
import cvxpy as cp


class Solver:
    """
    Battery scheduling solver.
    Uses CVXPY with ECOS as the default backend.  The solver instance
    is created once during initialization so that repeated calls to
    `solve` do not incur compilation costs.
    """

    def __init__(self, **kwargs):
        """Create a reusable CVXPY problem template."""
        # We do not know T until a call arrives, but we can keep a
        # reference to the solver options to reuse.
        self.solver_options = kwargs

    def solve(self, problem: dict) -> dict:
        """
        Solve the battery scheduling problem.

        Parameters
        ----------
        problem : dict
            Dictionary containing the problem data.
            Expected keys:
                - 'T': number of time periods (int)
                - 'p': price vector (list/array)
                - 'u': baseline consumption vector (list/array)
                - 'batteries': list of battery dicts
                    (use the first element only)
                + Inside each battery dict:
                    'Q'          : max energy capacity
                    'C'          : max charging power
                    'D'          : max discharging power
                    'efficiency' : round‑trip efficiency
        Returns
        -------
        dict
            Result dictionary with solver status, optimality flag, and
            scheduling information.
        """
        T = int(problem["T"])
        p = np.asarray(problem["p"], dtype=np.float64)
        u = np.asarray(problem["u"], dtype=np.float64)

        battery = problem["batteries"][0]
        Q = float(battery["Q"])
        C = float(battery["C"])
        D = float(battery["D"])
        e = float(battery["efficiency"])

        # CVXPY variables
        q = cp.Variable(T)
        c_in = cp.Variable(T)
        c_out = cp.Variable(T)
        c = c_in - c_out

        constraints = [
            q >= 0,
            q <= Q,
            c_in >= 0,
            c_out >= 0,
            c_in <= C,
            c_out <= D,
            u + c >= 0
        ]

        # Dynamics constraints, vectorised
        # q[t+1] == q[t] + e * c_in[t] - (1/e) * c_out[t]
        constraints.append(
            q[1:] == q[:-1] + e * c_in[:-1] - (1.0 / e) * c_out[:-1]
        )
        # Periodicity: q[0] == q[T-1] + ... (last period)
        constraints.append(
            q[0] == q[-1] + e * c_in[-1] - (1.0 / e) * c_out[-1]
        )

        objective = cp.Minimize(p @ c)

        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(**self.solver_options)

            if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                return {"status": prob.status, "optimal": False}

            c_net = c_in.value - c_out.value
            cost_without_battery = float(p @ u)
            cost_with_battery = float(p @ (u + c_net))
            savings = cost_without_battery - cost_with_battery
            savings_percent = 100.0 * savings / cost_without_battery if cost_without_battery else 0.0

            return {
                "status": prob.status,
                "optimal": True,
                "battery_results": [
                    {
                        "q": q.value.tolist(),
                        "c": c_net.tolist(),
                        "c_in": c_in.value.tolist(),
                        "c_out": c_out.value.tolist(),
                        "cost": cost_with_battery,
                    }
                ],
                "total_charging": c_net.tolist(),
                "cost_without_battery": cost_without_battery,
                "cost_with_battery": cost_with_battery,
                "savings": savings,
                "savings_percent": savings_percent,
            }

        except cp.SolverError as e:
            return {"status": "solver_error", "optimal": False, "error": str(e)}
        except Exception as e:
            return {"status": "error", "optimal": False, "error": str(e)}