from typing import Any
import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict) -> dict:
        T = int(problem["T"])
        p, u = np.array(problem["p"]), np.array(problem["u"])
        b = problem["batteries"][0]
        Q, C, D, eff = float(b["Q"]), float(b["C"]), float(b["D"]), float(b["efficiency"])

        q = cp.Variable(T)
        c_in = cp.Variable(T)
        c_out = cp.Variable(T)
        c = c_in - c_out

        constraints = [
            q >= 0, q <= Q,
            c_in >= 0, c_in <= C,
            c_out >= 0, c_out <= D,
            u + c >= 0
        ]

        for t in range(T - 1):
            constraints.append(q[t + 1] == q[t] + eff * c_in[t] - c_out[t] / eff)
        constraints.append(q[0] == q[T - 1] + eff * c_in[T - 1] - c_out[T - 1] / eff)

        prob = cp.Problem(cp.Minimize(p @ c), constraints)
        try:
            prob.solve(solver=cp.ECOS, verbose=False)
        except cp.SolverError as e:
            return {"status": "solver_error", "optimal": False, "error": str(e)}
        except Exception as e:
            return {"status": "error", "optimal": False, "error": str(e)}

        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            return {"status": prob.status, "optimal": False}

        c_net = c_in.value - c_out.value
        cost_without = float(p @ u)
        cost_with = float(p @ (u + c_net))
        savings = cost_without - cost_with

        return {
            "status": prob.status,
            "optimal": True,
            "battery_results": [
                {
                    "q": q.value.tolist(),
                    "c": c_net.tolist(),
                    "c_in": c_in.value.tolist(),
                    "c_out": c_out.value.tolist(),
                    "cost": cost_with,
                }
            ],
            "total_charging": c_net.tolist(),
            "cost_without_battery": cost_without,
            "cost_with_battery": cost_with,
            "savings": savings,
            "savings_percent": 100 * savings / cost_without if cost_without else 0.0,
        }