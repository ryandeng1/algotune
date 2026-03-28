from typing import Any
import cvxpy as cp
import numpy as np


class Solver:
    def solve(self, problem: dict) -> dict:
        T = int(problem["T"])
        p = np.array(problem["p"])
        u = np.array(problem["u"])
        battery = problem["batteries"][0]
        Q = float(battery["Q"])
        C = float(battery["C"])
        D = float(battery["D"])
        efficiency = float(battery["efficiency"])

        q = cp.Variable(T)
        c_in = cp.Variable(T)
        c_out = cp.Variable(T)
        c = c_in - c_out

        constraints = []
        constraints.append(q >= 0)
        constraints.append(q <= Q)
        constraints.append(c_in >= 0)
        constraints.append(c_out >= 0)
        constraints.append(c_in <= C)
        constraints.append(c_out <= D)

        e = efficiency * c_in - (1.0 / efficiency) * c_out
        constraints.append(cp.equality(q[1:], q[:-1] + e[:T-1]))
        constraints.append(cp.equality(q[0] - q[T-1], e[T-1]))
        constraints.append(u + c >= 0)

        objective = cp.Minimize(p @ c)
        prob = cp.Problem(objective, constraints)

        try:
            prob.solve()

            if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                return {"status": prob.status, "optimal": False}

            c_net = c.value
            cost_without_battery = float(p @ u)
            cost_with_battery = float(p @ (u + c_net))
            savings = cost_without_battery - cost_with_battery

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
                "savings_percent": float(100 * savings / cost_without_battery),
            }

        except cp.SolverError as e:
            return {"status": "solver_error", "optimal": False, "error": str(e)}
        except Exception as e:
            return {"status": "error", "optimal": False, "error": str(e)}