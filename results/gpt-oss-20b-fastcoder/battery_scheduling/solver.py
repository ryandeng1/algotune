import numpy as np
import cvxpy as cp


class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Solve the battery scheduling problem using CVXPY and the OSQP solver.

        This finds the optimal charging schedule for a battery that minimizes
        the total electricity cost over the time horizon.

        :param problem: Dictionary with problem parameters
        :return: Dictionary with optimal schedules and costs
        """
        T = int(problem["T"])
        p = np.asarray(problem["p"], dtype=float)
        u = np.asarray(problem["u"], dtype=float)
        battery = problem["batteries"][0]
        Q, C, D, eff = (
            float(battery["Q"]),
            float(battery["C"]),
            float(battery["D"]),
            float(battery["efficiency"]),
        )

        # Decision variables
        q = cp.Variable(T, nonneg=True)          # state of charge
        c_in = cp.Variable(T, nonneg=True)       # charging
        c_out = cp.Variable(T, nonneg=True)      # discharging

        # Constraints
        constraints = [
            q <= Q,
            c_in <= C,
            c_out <= D,
            u + c_in - c_out >= 0,
        ]

        # Energy balance constraints
        eff_inv = 1.0 / eff
        for t in range(T - 1):
            constraints.append(
                q[t + 1]
                == q[t] + eff * c_in[t] - eff_inv * c_out[t]
            )
        constraints.append(
            q[0] == q[T - 1] + eff * c_in[T - 1] - eff_inv * c_out[T - 1]
        )

        # Objective
        objective = cp.Minimize(p @ (c_in - c_out))

        prob = cp.Problem(objective, constraints)

        try:
            # Use the OSQP solver for speed when dealing with pure QPs
            prob.solve(solver=cp.OSQP, verbose=False)

            if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                return {"status": prob.status, "optimal": False}

            c_net = (c_in.value - c_out.value).tolist()
            cost_without = float(p @ u)
            cost_with = float(p @ (u + c_net))
            savings = cost_without - cost_with

            return {
                "status": prob.status,
                "optimal": True,
                "battery_results": [
                    {
                        "q": q.value.tolist(),
                        "c": c_net,
                        "c_in": c_in.value.tolist(),
                        "c_out": c_out.value.tolist(),
                        "cost": cost_with,
                    }
                ],
                "total_charging": c_net,
                "cost_without_battery": cost_without,
                "cost_with_battery": cost_with,
                "savings": savings,
                "savings_percent": 100.0 * savings / cost_without,
            }

        except cp.SolverError as e:
            return {"status": "solver_error", "optimal": False, "error": str(e)}
        except Exception as e:
            return {"status": "error", "optimal": False, "error": str(e)}